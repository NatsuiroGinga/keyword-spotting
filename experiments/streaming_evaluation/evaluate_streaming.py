#!/usr/bin/env python3
"""
流式 KWS 评估脚本

使用音频文件模拟流式输入，评估 V3 + MLP 两阶段检测方案的真实性能。

主要功能：
1. 逐帧流式处理音频
2. 两阶段检测：V3 触发 + MLP 验证
3. 计算检测指标、延迟、RTF
4. 生成评估报告
"""
import sys
import os

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

warnings.filterwarnings("ignore")

# 本地模块
from stream_simulator import StreamSimulator, StreamConfig, RingBuffer, BatchStreamSimulator
from metrics import (
    DetectionEvent, MetricsCalculator, EvaluationMetrics,
    format_metrics_table
)


class SherpaKWSWrapper:
    """
    Sherpa-ONNX KWS 封装
    
    封装 V3 模型的流式推理接口。
    """
    
    def __init__(
        self,
        model_dir: Path,
        keywords_threshold: float = 0.25,
        keywords_score: float = 1.5,
        num_threads: int = 2
    ):
        self.model_dir = Path(model_dir)
        self.keywords_threshold = keywords_threshold
        self.keywords_score = keywords_score
        self.num_threads = num_threads
        
        self._kws = None
        self._stream = None
        self._sample_rate = 16000
    
    def load(self) -> None:
        """加载模型"""
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError("需要安装 sherpa-onnx: pip install sherpa-onnx")
        
        # 查找 ONNX 文件
        encoder_files = list(self.model_dir.glob("encoder*.int8.onnx"))
        decoder_files = list(self.model_dir.glob("decoder*.int8.onnx"))
        joiner_files = list(self.model_dir.glob("joiner*.int8.onnx"))
        
        if not encoder_files or not decoder_files or not joiner_files:
            raise FileNotFoundError(f"ONNX 文件未找到: {self.model_dir}")
        
        encoder_path = str(encoder_files[0])
        decoder_path = str(decoder_files[0])
        joiner_path = str(joiner_files[0])
        tokens_path = str(self.model_dir / "tokens.txt")
        keywords_path = str(self.model_dir / "keywords.txt")
        
        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            keywords_file=keywords_path,
            num_threads=self.num_threads,
            keywords_score=self.keywords_score,
            keywords_threshold=self.keywords_threshold,
        )
        self._stream = self._kws.create_stream()
        
        print(f"V3 KWS 模型已加载: {self.model_dir.name}")
        print(f"  - threshold: {self.keywords_threshold}")
        print(f"  - score: {self.keywords_score}")
    
    def reset_stream(self) -> None:
        """重置流"""
        if self._stream is not None:
            self._kws.reset_stream(self._stream)
    
    def create_new_stream(self) -> None:
        """创建新流"""
        if self._kws is not None:
            self._stream = self._kws.create_stream()
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        处理音频块
        
        Args:
            audio_chunk: 音频数据 (float32)
            
        Returns:
            检测到的关键词，或 None
        """
        # 确保数据格式正确
        audio_list = audio_chunk.flatten().astype(np.float32).tolist()
        self._stream.accept_waveform(self._sample_rate, audio_list)
        
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)
            result = self._kws.get_result(self._stream)
            if result:
                return result.strip()
        
        return None
    
    def process_full_audio(self, audio: np.ndarray) -> Tuple[bool, str]:
        """
        处理完整音频（离线模式，带尾部填充）
        
        Args:
            audio: 完整音频数据
            
        Returns:
            (是否检测到, 关键词)
        """
        stream = self._kws.create_stream()
        audio_list = audio.flatten().astype(np.float32).tolist()
        stream.accept_waveform(self._sample_rate, audio_list)
        
        # 添加尾部填充
        tail_padding = [0.0] * int(0.3 * self._sample_rate)
        stream.accept_waveform(self._sample_rate, tail_padding)
        stream.input_finished()
        
        while self._kws.is_ready(stream):
            self._kws.decode_stream(stream)
            result = self._kws.get_result(stream)
            if result:
                return True, result.strip()
        
        return False, ""


class MLPVerifierWrapper:
    """
    MLP 验证器封装
    
    封装 MLP 验证器的推理接口。
    """
    
    def __init__(
        self,
        model_path: Path,
        threshold: float = 0.5,
        n_mfcc: int = 13,
        target_frames: int = 50
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.n_mfcc = n_mfcc
        self.target_frames = target_frames
        self.sample_rate = 16000
        
        self._model = None
        self._device = None
    
    def load(self) -> None:
        """加载模型"""
        import torch
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 检查模型类型
        model_path_str = str(self.model_path)
        
        if model_path_str.endswith(".onnx"):
            self._load_onnx()
        else:
            self._load_pytorch()
    
    def _load_pytorch(self) -> None:
        """加载 PyTorch 模型"""
        import torch
        import torch.nn as nn
        
        # 简单 MLP 结构（与训练时一致）
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim: int):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        input_dim = self.n_mfcc * self.target_frames
        self._model = SimpleMLP(input_dim)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self._device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            self._model.load_state_dict(checkpoint)
        else:
            self._model = checkpoint
        
        self._model = self._model.to(self._device)
        self._model.eval()
        
        print(f"MLP 验证器已加载 (PyTorch): {self.model_path.name}")
        print(f"  - threshold: {self.threshold}")
        print(f"  - device: {self._device}")
    
    def _load_onnx(self) -> None:
        """加载 ONNX 模型"""
        import onnxruntime as ort
        
        self._onnx_session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"]
        )
        self._input_name = self._onnx_session.get_inputs()[0].name
        self._output_name = self._onnx_session.get_outputs()[0].name
        self._is_onnx = True
        
        print(f"MLP 验证器已加载 (ONNX): {self.model_path.name}")
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """提取 MFCC 特征"""
        import librosa
        
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=160
        )
        
        # 填充或截断到目标帧数
        n_frames = mfcc.shape[1]
        if n_frames < self.target_frames:
            padding = np.zeros((self.n_mfcc, self.target_frames - n_frames))
            mfcc = np.concatenate([mfcc, padding], axis=1)
        elif n_frames > self.target_frames:
            mfcc = mfcc[:, :self.target_frames]
        
        # 归一化
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        
        return mfcc.flatten().astype(np.float32)
    
    def _extract_suffix(self, audio: np.ndarray) -> np.ndarray:
        """
        提取后缀音频（流式版本）
        
        核心逻辑：
        1. V3触发时，关键词刚刚说完
        2. 从缓冲区末尾提取约1秒作为关键词候选区域
        3. 从候选区域的40%位置提取后缀（与训练一致）
        
        这样既保持与训练的一致性，又适应流式场景。
        """
        # 关键词"你好真真"约800-1200ms
        keyword_duration_ms = 1000
        keyword_samples = int(keyword_duration_ms * self.sample_rate / 1000)
        
        # 从缓冲区末尾提取关键词候选区域
        if len(audio) > keyword_samples:
            keyword_region = audio[-keyword_samples:]
        else:
            keyword_region = audio
        
        # 与训练一致：从40%位置提取后缀
        total_samples = len(keyword_region)
        start_ratio = 0.4
        min_duration_ms = 200
        max_duration_ms = 800
        
        start_sample = int(total_samples * start_ratio)
        min_samples = int(min_duration_ms * self.sample_rate / 1000)
        max_samples = int(max_duration_ms * self.sample_rate / 1000)
        
        suffix = keyword_region[start_sample:]
        
        if len(suffix) < min_samples:
            new_start = max(0, total_samples - min_samples)
            suffix = keyword_region[new_start:]
        elif len(suffix) > max_samples:
            suffix = suffix[:max_samples]
        
        return suffix
    
    def verify(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        验证音频
        
        Args:
            audio: 完整唤醒词音频
            
        Returns:
            (是否通过验证, 置信度)
        """
        import torch
        
        # 提取后缀
        suffix = self._extract_suffix(audio)
        
        if len(suffix) < self.sample_rate * 0.1:  # 至少 100ms
            return True, 1.0  # 太短，跳过验证
        
        # 提取特征
        features = self._extract_mfcc(suffix)
        
        # 推理
        if hasattr(self, "_is_onnx") and self._is_onnx:
            # ONNX 推理
            inputs = features.reshape(1, -1)
            outputs = self._onnx_session.run(
                [self._output_name],
                {self._input_name: inputs}
            )
            confidence = float(outputs[0][0, 0])
        else:
            # PyTorch 推理
            with torch.no_grad():
                x = torch.from_numpy(features).float().unsqueeze(0).to(self._device)
                confidence = self._model(x).item()
        
        return confidence >= self.threshold, confidence


class StreamingKWSEvaluator:
    """
    流式 KWS 评估器
    
    整合流式模拟、两阶段检测、指标计算。
    """
    
    def __init__(
        self,
        kws_model: SherpaKWSWrapper,
        mlp_verifier: Optional[MLPVerifierWrapper],
        stream_config: StreamConfig,
        mlp_enabled: bool = True
    ):
        self.kws_model = kws_model
        self.mlp_verifier = mlp_verifier
        self.stream_config = stream_config
        self.mlp_enabled = mlp_enabled and mlp_verifier is not None
        
        self._simulator = StreamSimulator(stream_config)
        self._buffer = RingBuffer(stream_config.buffer_samples)
    
    def evaluate_file(self, audio_path: str, label: int) -> DetectionEvent:
        """
        流式评估单个文件
        
        Args:
            audio_path: 音频文件路径
            label: 真实标签 (1=正样本, 0=负样本)
            
        Returns:
            检测事件
        """
        # 加载音频
        audio = self._simulator.load_audio(audio_path)
        audio_duration_ms = len(audio) / self.stream_config.sample_rate * 1000
        
        # 重置状态
        self._buffer.clear()
        self.kws_model.create_new_stream()
        
        # 流式处理
        start_time = time.perf_counter()
        
        detected = False
        detection_time_ms = 0.0
        v3_triggered = False
        v3_score = 0.0
        mlp_score = 0.0
        max_mlp_score = 0.0  # 记录最高MLP分数
        confidence = 0.0
        
        for frame, frame_idx, timestamp_ms in self._simulator.stream_frames(audio):
            self._buffer.append(frame)
            
            # V3 检测
            result = self.kws_model.process_chunk(frame)
            
            if result:
                v3_triggered = True
                v3_score = 1.0
                
                if self.mlp_enabled:
                    # MLP 验证
                    buffer_audio = self._buffer.get_all()
                    verified, current_mlp_score = self.mlp_verifier.verify(buffer_audio)
                    
                    # 更新最高分数
                    if current_mlp_score > max_mlp_score:
                        max_mlp_score = current_mlp_score
                        mlp_score = current_mlp_score
                        detection_time_ms = timestamp_ms
                    
                    if verified:
                        detected = True
                        confidence = current_mlp_score
                        # 检测成功，立即返回
                        break
                else:
                    detected = True
                    confidence = 1.0
                    detection_time_ms = timestamp_ms
                    break
                
                # MLP验证失败，重置V3流继续检测
                self.kws_model.reset_stream()
        
        # 如果流式没检测到，尝试离线模式（添加尾部填充）
        if not v3_triggered:
            v3_detected, _ = self.kws_model.process_full_audio(audio)
            if v3_detected:
                v3_triggered = True
                v3_score = 1.0
                detection_time_ms = audio_duration_ms  # 在音频末尾检测到
                
                if self.mlp_enabled:
                    verified, mlp_score = self.mlp_verifier.verify(audio)
                    max_mlp_score = max(max_mlp_score, mlp_score)
                    if verified:
                        detected = True
                        confidence = mlp_score
                else:
                    detected = True
                    confidence = 1.0
        
        # 使用最高MLP分数
        mlp_score = max_mlp_score
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        return DetectionEvent(
            audio_path=audio_path,
            label=label,
            detected=detected,
            confidence=confidence,
            detection_time_ms=detection_time_ms,
            inference_time_ms=inference_time_ms,
            audio_duration_ms=audio_duration_ms,
            v3_triggered=v3_triggered,
            v3_score=v3_score,
            mlp_score=mlp_score
        )
    
    def evaluate_dataset(
        self,
        audio_paths: List[str],
        labels: List[int],
        verbose: bool = False
    ) -> Tuple[List[DetectionEvent], EvaluationMetrics]:
        """
        评估整个数据集
        
        Args:
            audio_paths: 音频文件路径列表
            labels: 标签列表
            verbose: 是否打印详细信息
            
        Returns:
            (检测事件列表, 评估指标)
        """
        events = []
        
        for i, (path, label) in enumerate(zip(audio_paths, labels)):
            event = self.evaluate_file(path, label)
            events.append(event)
            
            if verbose:
                filename = Path(path).name
                status = "✓" if event.detected == bool(label) else "✗"
                v3_str = "V3=1" if event.v3_triggered else "V3=0"
                mlp_str = f"MLP={event.mlp_score:.2f}" if self.mlp_enabled else ""
                print(f"{status} [{i+1:3d}/{len(audio_paths)}] {filename}: "
                      f"{v3_str} {mlp_str} det={int(event.detected)} label={label}")
        
        # 计算指标
        calculator = MetricsCalculator()
        metrics = calculator.calculate(events)
        
        return events, metrics


def load_dataset(
    data_dir: Path,
    positive_keywords: List[str] = None
) -> Tuple[List[str], List[int]]:
    """
    加载测试数据集
    
    Args:
        data_dir: 数据目录
        positive_keywords: 正样本关键词
        
    Returns:
        (paths, labels)
    """
    positive_keywords = positive_keywords or ["你好真真"]
    
    paths = []
    labels = []
    
    for wav_file in sorted(data_dir.glob("*.wav")):
        filename = wav_file.name
        # 注意：排除"你好珍珍"等相似词（应为负样本）
        is_positive = "你好真真" in filename
        
        paths.append(str(wav_file))
        labels.append(1 if is_positive else 0)
    
    return paths, labels


def main():
    parser = argparse.ArgumentParser(description="流式 KWS 评估")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="测试数据目录 (默认: data/all)")
    parser.add_argument("--kws-model-dir", type=str, default=None,
                        help="KWS 模型目录 (默认: exp/kws_finetune_v3)")
    parser.add_argument("--mlp-model", type=str, default=None,
                        help="MLP 验证器模型路径")
    parser.add_argument("--kws-threshold", type=float, default=0.25,
                        help="KWS 检测阈值 (默认: 0.25)")
    parser.add_argument("--kws-score", type=float, default=1.5,
                        help="KWS 关键词加分 (默认: 1.5)")
    parser.add_argument("--mlp-threshold", type=float, default=0.4,
                        help="MLP 验证阈值 (默认: 0.4，与离线评估一致)")
    parser.add_argument("--frame-ms", type=int, default=30,
                        help="帧大小 (毫秒, 默认: 30)")
    parser.add_argument("--buffer-ms", type=int, default=2000,
                        help="缓冲区大小 (毫秒, 默认: 2000，约2倍关键词时长)")
    parser.add_argument("--no-mlp", action="store_true",
                        help="禁用 MLP 验证器")
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录")
    args = parser.parse_args()
    
    # 设置路径
    project_root = Path(PROJECT_ROOT)
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data" / "all"
    kws_model_dir = Path(args.kws_model_dir) if args.kws_model_dir else project_root / "exp" / "kws_finetune_v3"
    
    # MLP 模型路径
    if args.mlp_model:
        mlp_model_path = Path(args.mlp_model)
    else:
        # 默认查找路径
        mlp_candidates = [
            project_root / "experiments" / "multi_stage_ablation" / "models" / "mlp_verifier.pt",
            project_root / "experiments" / "multi_stage_ablation" / "models" / "mlp_verifier.onnx",
        ]
        mlp_model_path = None
        for candidate in mlp_candidates:
            if candidate.exists():
                mlp_model_path = candidate
                break
    
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查文件
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    if not kws_model_dir.exists():
        print(f"错误: KWS 模型目录不存在: {kws_model_dir}")
        return
    
    mlp_enabled = not args.no_mlp
    if mlp_enabled and (mlp_model_path is None or not mlp_model_path.exists()):
        print(f"警告: MLP 模型不存在，禁用 MLP 验证")
        mlp_enabled = False
    
    print("=" * 60)
    print("流式 KWS 评估 (V3 + MLP)")
    print("=" * 60)
    print(f"测试数据: {data_dir}")
    print(f"KWS 模型: {kws_model_dir}")
    print(f"MLP 模型: {mlp_model_path if mlp_enabled else '禁用'}")
    print(f"帧大小: {args.frame_ms}ms")
    print(f"缓冲区: {args.buffer_ms}ms")
    print()
    
    # 加载数据
    audio_paths, labels = load_dataset(data_dir)
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    print(f"样本统计: 总计 {len(labels)} (正样本: {n_positive}, 负样本: {n_negative})")
    print()
    
    # 加载模型
    print("加载模型...")
    
    kws_model = SherpaKWSWrapper(
        model_dir=kws_model_dir,
        keywords_threshold=args.kws_threshold,
        keywords_score=args.kws_score
    )
    kws_model.load()
    
    mlp_verifier = None
    if mlp_enabled:
        mlp_verifier = MLPVerifierWrapper(
            model_path=mlp_model_path,
            threshold=args.mlp_threshold
        )
        mlp_verifier.load()
    
    # 流式配置
    stream_config = StreamConfig(
        sample_rate=16000,
        frame_duration_ms=args.frame_ms,
        buffer_duration_ms=args.buffer_ms
    )
    
    # 创建评估器
    evaluator = StreamingKWSEvaluator(
        kws_model=kws_model,
        mlp_verifier=mlp_verifier,
        stream_config=stream_config,
        mlp_enabled=mlp_enabled
    )
    
    # 模型预热
    print("\n模型预热...")
    if audio_paths:
        _ = evaluator.evaluate_file(audio_paths[0], labels[0])
    
    # 开始评估
    print("\n开始评估...")
    start_time = time.perf_counter()
    events, metrics = evaluator.evaluate_dataset(audio_paths, labels, verbose=args.verbose)
    total_time = time.perf_counter() - start_time
    
    # 打印结果
    print(format_metrics_table(metrics))
    print(f"\n总评估时间: {total_time:.2f}s")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    results = {
        "config": {
            "data_dir": str(data_dir),
            "kws_model_dir": str(kws_model_dir),
            "mlp_model": str(mlp_model_path) if mlp_enabled else None,
            "kws_threshold": args.kws_threshold,
            "kws_score": args.kws_score,
            "mlp_threshold": args.mlp_threshold,
            "mlp_enabled": mlp_enabled,
            "frame_ms": args.frame_ms,
            "buffer_ms": args.buffer_ms,
        },
        "metrics": metrics.to_dict(),
        "events": [
            {
                "audio_path": e.audio_path,
                "label": e.label,
                "detected": e.detected,
                "confidence": e.confidence,
                "detection_time_ms": e.detection_time_ms,
                "inference_time_ms": e.inference_time_ms,
                "audio_duration_ms": e.audio_duration_ms,
                "v3_triggered": e.v3_triggered,
                "v3_score": e.v3_score,
                "mlp_score": e.mlp_score,
            }
            for e in events
        ],
        "timestamp": timestamp,
        "total_time_s": total_time,
    }
    
    result_path = output_dir / f"streaming_eval_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {result_path}")
    
    # 保存文本报告
    report_path = output_dir / f"streaming_eval_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(format_metrics_table(metrics))
        f.write(f"\n\n总评估时间: {total_time:.2f}s\n")
        f.write(f"配置:\n")
        for k, v in results["config"].items():
            f.write(f"  {k}: {v}\n")
    
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
