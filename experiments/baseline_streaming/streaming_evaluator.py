#!/usr/bin/env python3
"""
流式KWS评估框架

专门用于流式场景下的关键词检测评估，使用音频文件模拟在线流式输入。
评估指标包括：FAR、FRR、准确率、F1、RTF等。
"""

import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import soundfile as sf


# 正样本关键词
POSITIVE_KEYWORDS = [
    "你好真真",
    "你好珍珍",
    "你好甄甄",
    "你好臻臻",
    "你好桢桢",
]


@dataclass
class StreamConfig:
    """流式配置"""
    sample_rate: int = 16000
    frame_duration_ms: int = 30      # 每帧30ms
    buffer_duration_ms: int = 2000   # 缓冲区2秒
    

@dataclass
class DetectionResult:
    """单个样本的检测结果"""
    file: str
    text: str
    label: int                  # 真实标签 (0/1)
    detected: bool              # 是否检测到关键词
    detection_time_ms: float    # 检测时间点
    inference_time_ms: float    # 推理耗时
    audio_duration_ms: float    # 音频时长
    rtf: float                  # Real-Time Factor
    confidence: float = 0.0     # 置信度分数


@dataclass
class EvaluationMetrics:
    """评估指标"""
    total: int = 0
    positive: int = 0
    negative: int = 0
    tp: int = 0                 # True Positive
    tn: int = 0                 # True Negative
    fp: int = 0                 # False Positive (误报)
    fn: int = 0                 # False Negative (漏检)
    
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    far: float = 0.0            # False Accept Rate
    frr: float = 0.0            # False Reject Rate
    
    avg_rtf: float = 0.0        # 平均RTF
    max_rtf: float = 0.0        # 最大RTF
    avg_latency_ms: float = 0.0 # 平均延迟


class RingBuffer:
    """环形缓冲区"""
    
    def __init__(self, max_samples: int):
        self.buffer = np.zeros(max_samples, dtype=np.float32)
        self.max_samples = max_samples
        self.write_pos = 0
        self.filled = 0
    
    def append(self, samples: np.ndarray):
        n = len(samples)
        if n >= self.max_samples:
            self.buffer[:] = samples[-self.max_samples:]
            self.write_pos = 0
            self.filled = self.max_samples
        else:
            end_pos = (self.write_pos + n) % self.max_samples
            if end_pos > self.write_pos:
                self.buffer[self.write_pos:end_pos] = samples
            else:
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = samples[:first_part]
                self.buffer[:end_pos] = samples[first_part:]
            self.write_pos = end_pos
            self.filled = min(self.filled + n, self.max_samples)
    
    def get_all(self) -> np.ndarray:
        if self.filled < self.max_samples:
            return self.buffer[:self.filled].copy()
        else:
            return np.concatenate([
                self.buffer[self.write_pos:],
                self.buffer[:self.write_pos]
            ])
    
    def clear(self):
        self.buffer.fill(0)
        self.write_pos = 0
        self.filled = 0


class StreamingKWSEvaluator:
    """流式KWS评估器"""
    
    def __init__(
        self,
        model_dir: str,
        config: StreamConfig = None,
        keywords_threshold: float = 0.25,
        keywords_score: float = 1.5,
    ):
        """
        Args:
            model_dir: 模型目录（包含encoder/decoder/joiner ONNX文件和keywords.txt）
            config: 流式配置
            keywords_threshold: 关键词检测阈值
            keywords_score: 关键词加分
        """
        self.model_dir = Path(model_dir)
        self.config = config or StreamConfig()
        self.keywords_threshold = keywords_threshold
        self.keywords_score = keywords_score
        
        # 初始化缓冲区
        buffer_samples = int(self.config.buffer_duration_ms * self.config.sample_rate / 1000)
        self.buffer = RingBuffer(buffer_samples)
        
        # 帧大小
        self.frame_samples = int(self.config.frame_duration_ms * self.config.sample_rate / 1000)
        
        # 延迟加载模型
        self._kws_model = None
    
    def _load_model(self):
        """加载sherpa-onnx KWS模型"""
        if self._kws_model is not None:
            return
        
        import sherpa_onnx
        
        # 查找模型文件
        encoder_files = list(self.model_dir.glob("*encoder*.onnx"))
        decoder_files = list(self.model_dir.glob("*decoder*.onnx"))
        joiner_files = list(self.model_dir.glob("*joiner*.onnx"))
        
        if not encoder_files or not decoder_files or not joiner_files:
            raise FileNotFoundError(f"找不到模型文件: {self.model_dir}")
        
        # 选择模型文件：优先使用int8量化模型，但如果int8不可用则使用原始模型
        # 注意：某些ONNX运行时可能不支持INT8，此时回退到非量化版本
        def select_model(files, prefer_int8=True):
            int8_files = [f for f in files if "int8" in f.name]
            non_int8_files = [f for f in files if "int8" not in f.name]
            if prefer_int8 and int8_files:
                return str(sorted(int8_files)[0])
            elif non_int8_files:
                return str(sorted(non_int8_files)[0])
            else:
                return str(sorted(files)[0])
        
        # 对于V4模型，跳过INT8（兼容性问题）
        use_int8 = "exp_v4" not in str(self.model_dir)
        encoder = select_model(encoder_files, use_int8)
        decoder = select_model(decoder_files, use_int8)
        joiner = select_model(joiner_files, use_int8)
        
        # 查找tokens和keywords
        tokens = str(self.model_dir / "tokens.txt")
        keywords = str(self.model_dir / "keywords.txt")
        
        if not Path(tokens).exists():
            # 尝试从lang_partial_tone目录获取
            tokens = str(Path(__file__).parent.parent.parent / "data/lang_partial_tone/tokens.txt")
        
        print(f"  加载模型:")
        print(f"    Encoder: {encoder}")
        print(f"    Decoder: {decoder}")
        print(f"    Joiner: {joiner}")
        print(f"    Tokens: {tokens}")
        print(f"    Keywords: {keywords}")
        
        self._kws_model = sherpa_onnx.KeywordSpotter(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            keywords_file=keywords,
            keywords_threshold=self.keywords_threshold,
            keywords_score=self.keywords_score,
            num_threads=2,
            provider="cpu",
        )
    
    def _extract_text_from_filename(self, filename: str) -> str:
        """从文件名提取文本"""
        import re
        stem = Path(filename).stem
        if "_" in stem:
            return stem.split("_", 1)[1]
        else:
            return re.sub(r"^\d+", "", stem)
    
    def _is_positive(self, text: str) -> bool:
        """判断是否为正样本"""
        for kw in POSITIVE_KEYWORDS:
            if kw in text:
                return True
        return False
    
    def evaluate_file(self, audio_path: Path) -> DetectionResult:
        """评估单个音频文件（流式模拟）"""
        self._load_model()
        
        # 加载音频
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if sr != self.config.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
        
        audio_duration_ms = len(audio) / self.config.sample_rate * 1000
        text = self._extract_text_from_filename(audio_path.name)
        label = 1 if self._is_positive(text) else 0
        
        # 重置状态
        self.buffer.clear()
        stream = self._kws_model.create_stream()
        
        # 流式处理
        start_time = time.perf_counter()
        detected = False
        detection_time_ms = 0.0
        confidence = 0.0
        
        # 按帧处理
        for i in range(0, len(audio), self.frame_samples):
            frame = audio[i:i + self.frame_samples]
            if len(frame) < self.frame_samples:
                # 填充最后一帧
                frame = np.pad(frame, (0, self.frame_samples - len(frame)))
            
            self.buffer.append(frame)
            stream.accept_waveform(self.config.sample_rate, frame.tolist())
            
            while self._kws_model.is_ready(stream):
                self._kws_model.decode_stream(stream)
            
            result = self._kws_model.get_result(stream)
            if result:
                detected = True
                detection_time_ms = (i + self.frame_samples) / self.config.sample_rate * 1000
                confidence = 1.0
                break
        
        # 处理尾部（添加padding确保检测到结尾的关键词）
        if not detected:
            tail_padding = np.zeros(int(0.5 * self.config.sample_rate), dtype=np.float32)
            stream.accept_waveform(self.config.sample_rate, tail_padding.tolist())
            stream.input_finished()
            
            while self._kws_model.is_ready(stream):
                self._kws_model.decode_stream(stream)
            
            result = self._kws_model.get_result(stream)
            if result:
                detected = True
                detection_time_ms = audio_duration_ms
                confidence = 1.0
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        rtf = inference_time_ms / audio_duration_ms if audio_duration_ms > 0 else 0
        
        return DetectionResult(
            file=audio_path.name,
            text=text,
            label=label,
            detected=detected,
            detection_time_ms=detection_time_ms,
            inference_time_ms=inference_time_ms,
            audio_duration_ms=audio_duration_ms,
            rtf=rtf,
            confidence=confidence,
        )
    
    def evaluate_dataset(
        self,
        data_dir: str,
        verbose: bool = False,
    ) -> Tuple[EvaluationMetrics, List[DetectionResult]]:
        """评估整个数据集"""
        data_path = Path(data_dir)
        audio_files = list(data_path.glob("*.wav"))
        
        results = []
        metrics = EvaluationMetrics()
        
        rtf_list = []
        latency_list = []
        
        print(f"评估数据集: {data_path}")
        print(f"文件数量: {len(audio_files)}")
        print()
        
        for i, audio_path in enumerate(audio_files):
            result = self.evaluate_file(audio_path)
            results.append(result)
            
            # 统计
            metrics.total += 1
            if result.label == 1:
                metrics.positive += 1
                if result.detected:
                    metrics.tp += 1
                else:
                    metrics.fn += 1
            else:
                metrics.negative += 1
                if result.detected:
                    metrics.fp += 1
                else:
                    metrics.tn += 1
            
            rtf_list.append(result.rtf)
            if result.detected:
                latency_list.append(result.detection_time_ms)
            
            if verbose:
                status = "✓" if (result.label == 1) == result.detected else "✗"
                print(f"  [{i+1}/{len(audio_files)}] {status} {result.file[:50]} "
                      f"L={result.label} D={int(result.detected)} RTF={result.rtf:.3f}")
        
        # 计算指标
        if metrics.tp + metrics.fp > 0:
            metrics.precision = metrics.tp / (metrics.tp + metrics.fp)
        if metrics.tp + metrics.fn > 0:
            metrics.recall = metrics.tp / (metrics.tp + metrics.fn)
        if metrics.precision + metrics.recall > 0:
            metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        if metrics.total > 0:
            metrics.accuracy = (metrics.tp + metrics.tn) / metrics.total
        
        # FAR = FP / (FP + TN) = 误报率
        if metrics.fp + metrics.tn > 0:
            metrics.far = metrics.fp / (metrics.fp + metrics.tn)
        
        # FRR = FN / (FN + TP) = 漏检率
        if metrics.fn + metrics.tp > 0:
            metrics.frr = metrics.fn / (metrics.fn + metrics.tp)
        
        # RTF统计
        if rtf_list:
            metrics.avg_rtf = np.mean(rtf_list)
            metrics.max_rtf = np.max(rtf_list)
        
        if latency_list:
            metrics.avg_latency_ms = np.mean(latency_list)
        
        return metrics, results


def print_metrics(metrics: EvaluationMetrics, model_name: str = "Model"):
    """打印评估指标"""
    print()
    print("=" * 60)
    print(f"评估结果: {model_name}")
    print("=" * 60)
    print()
    print("【样本统计】")
    print(f"  总数: {metrics.total}")
    print(f"  正样本: {metrics.positive}")
    print(f"  负样本: {metrics.negative}")
    print()
    print("【混淆矩阵】")
    print(f"  TP (正确唤醒): {metrics.tp}")
    print(f"  TN (正确拒绝): {metrics.tn}")
    print(f"  FP (误报): {metrics.fp}")
    print(f"  FN (漏检): {metrics.fn}")
    print()
    print("【性能指标】")
    print(f"  准确率 (Accuracy): {metrics.accuracy * 100:.2f}%")
    print(f"  精确率 (Precision): {metrics.precision * 100:.2f}%")
    print(f"  召回率 (Recall): {metrics.recall * 100:.2f}%")
    print(f"  F1 Score: {metrics.f1 * 100:.2f}%")
    print()
    print("【错误率】")
    print(f"  FAR (误报率): {metrics.far * 100:.2f}%  {'✓' if metrics.far < 0.1 else '✗'} (目标 <10%)")
    print(f"  FRR (漏检率): {metrics.frr * 100:.2f}%  {'✓' if metrics.frr < 0.05 else '✗'} (目标 <5%)")
    print()
    print("【实时性能】")
    print(f"  平均 RTF: {metrics.avg_rtf:.4f}  {'✓' if metrics.avg_rtf < 1.0 else '✗'} (目标 <1.0)")
    print(f"  最大 RTF: {metrics.max_rtf:.4f}")
    print(f"  平均检测延迟: {metrics.avg_latency_ms:.1f}ms")
    print()
    
    # 综合评估
    passed = metrics.far < 0.1 and metrics.frr < 0.05 and metrics.avg_rtf < 1.0
    print("【综合评估】")
    print(f"  {'✓ 达标' if passed else '✗ 未达标'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="流式KWS评估")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="模型目录")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="测试数据目录")
    parser.add_argument("--model-name", type=str, default="Model",
                        help="模型名称（用于显示）")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="关键词检测阈值")
    parser.add_argument("--score", type=float, default=1.5,
                        help="关键词加分")
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    parser.add_argument("--output", type=str,
                        help="输出结果JSON文件")
    
    args = parser.parse_args()
    
    evaluator = StreamingKWSEvaluator(
        model_dir=args.model_dir,
        keywords_threshold=args.threshold,
        keywords_score=args.score,
    )
    
    metrics, results = evaluator.evaluate_dataset(
        args.data_dir,
        verbose=args.verbose,
    )
    
    print_metrics(metrics, args.model_name)
    
    # 保存结果
    if args.output:
        output_data = {
            "model_name": args.model_name,
            "model_dir": args.model_dir,
            "data_dir": args.data_dir,
            "config": {
                "threshold": args.threshold,
                "score": args.score,
            },
            "metrics": asdict(metrics),
            "results": [asdict(r) for r in results],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
