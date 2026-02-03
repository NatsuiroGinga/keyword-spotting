#!/usr/bin/env python3
"""
流式评估脚本
使用音频文件模拟流式输入，评估 KWS + CNN 验证器的性能
支持 V1/V2/KWS 模型作为一阶段检测器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import time
import warnings
warnings.filterwarnings("ignore")

# sherpa-onnx
import sherpa_onnx

# 本地模块
from models.cnn_verifier import CNNVerifier, CNNConfig
from models.score_fusion import ScoreFusion, FusionConfig, find_optimal_threshold
from features.feature_extractor import FeatureExtractor, FeatureConfig, SuffixExtractor
from streaming.audio_simulator import AudioStreamSimulator, StreamConfig, RingBuffer


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


class KWSDetector:
    """KWS 检测器封装（支持 V1/V2/KWS 模型）"""
    
    def __init__(
        self,
        model_dir: Path,
        keywords_threshold: float = 0.25,
        keywords_score: float = 1.5,
        num_threads: int = 2
    ):
        self.model_dir = model_dir
        self.keywords_threshold = keywords_threshold
        self.keywords_score = keywords_score
        self.num_threads = num_threads
        
        self._kws = None
        self._stream = None
    
    def load(self):
        """加载模型"""
        # 查找 ONNX 文件（支持不同命名格式）
        encoder_files = list(self.model_dir.glob("encoder*.int8.onnx"))
        decoder_files = list(self.model_dir.glob("decoder*.int8.onnx"))
        joiner_files = list(self.model_dir.glob("joiner*.int8.onnx"))
        
        if not encoder_files or not decoder_files or not joiner_files:
            raise FileNotFoundError(f"ONNX files not found in {self.model_dir}")
        
        encoder_path = str(encoder_files[0])
        decoder_path = str(decoder_files[0])
        joiner_path = str(joiner_files[0])
        
        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=str(self.model_dir / "tokens.txt"),
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            keywords_file=str(self.model_dir / "keywords.txt"),
            num_threads=self.num_threads,
            keywords_score=self.keywords_score,
            keywords_threshold=self.keywords_threshold,
        )
        self._stream = self._kws.create_stream()
    
    def reset_stream(self):
        """重置流"""
        if self._stream is not None:
            self._kws.reset_stream(self._stream)
    
    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        处理音频块
        
        Returns:
            检测到的关键词，或 None
        """
        self._stream.accept_waveform(sample_rate, audio_chunk.tolist())
        
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)
            result = self._kws.get_result(self._stream)
            if result:
                return result.strip()
        
        return None
    
    def detect_full_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, str]:
        """
        处理完整音频（离线模式）
        
        Returns:
            (detected, keyword)
        """
        stream = self._kws.create_stream()
        stream.accept_waveform(sample_rate, audio.tolist())
        
        # 添加尾部填充
        tail_padding = [0.0] * int(0.3 * sample_rate)
        stream.accept_waveform(sample_rate, tail_padding)
        stream.input_finished()
        
        while self._kws.is_ready(stream):
            self._kws.decode_stream(stream)
            result = self._kws.get_result(stream)
            if result:
                return True, result.strip()
        
        return False, ""


class CNNVerifierWrapper:
    """CNN 验证器封装"""
    
    def __init__(self, model_path: Path, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        
        self._model = None
        self._config = None
        self._feature_extractor = None
        self._suffix_extractor = None
        self._device = None
    
    def load(self):
        """加载模型"""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self._device, weights_only=False)
        
        # 创建模型
        config = checkpoint.get("config", {})
        self._config = CNNConfig(
            n_mfcc=config.get("n_mfcc", 40),
            target_frames=config.get("target_frames", 50),
            hidden_channels=config.get("hidden_channels", [64, 128, 64]),
            kernel_size=config.get("kernel_size", 3),
            dropout=config.get("dropout", 0.3)
        )
        
        self._model = CNNVerifier(self._config)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self._device)
        self._model.eval()
        
        # 使用保存的最佳阈值
        if "best_threshold" in checkpoint:
            self.threshold = checkpoint["best_threshold"]
            print(f"使用保存的最佳阈值: {self.threshold:.2f}")
        
        # 特征提取器
        self._feature_extractor = FeatureExtractor(FeatureConfig(
            n_mfcc=self._config.n_mfcc,
            target_frames=self._config.target_frames
        ))
        self._suffix_extractor = SuffixExtractor()
    
    def verify(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        验证音频
        
        Args:
            audio: 音频数据（完整唤醒词音频）
        
        Returns:
            (verified, confidence)
        """
        # 提取后缀
        suffix = self._suffix_extractor.extract(audio)
        
        # 提取特征
        features = self._feature_extractor.extract_for_cnn(suffix)
        
        # 推理
        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0).to(self._device)
            confidence = self._model(x).item()
        
        return confidence >= self.threshold, confidence


class StreamingKWSEvaluator:
    """流式 KWS 评估器"""
    
    def __init__(
        self,
        kws_detector: KWSDetector,
        cnn_verifier: CNNVerifierWrapper,
        score_fusion: ScoreFusion,
        stream_config: StreamConfig
    ):
        self.kws_detector = kws_detector
        self.cnn_verifier = cnn_verifier
        self.score_fusion = score_fusion
        self.stream_config = stream_config
        
        self._audio_simulator = AudioStreamSimulator(stream_config)
    
    def evaluate_file_streaming(self, audio_path: str) -> Dict:
        """
        流式评估单个文件
        
        模拟真实的流式检测场景：
        1. 逐帧送入 KWS 检测器
        2. KWS 触发后提取缓冲区音频进行 CNN 验证
        3. 融合分数做最终判定
        """
        import librosa
        
        start_time = time.perf_counter()
        
        # 加载完整音频
        audio, _ = librosa.load(audio_path, sr=self.stream_config.sample_rate)
        
        # 环形缓冲区
        buffer = RingBuffer(self.stream_config.buffer_length)
        
        # 流式处理
        detected = False
        v3_score = 0.0
        cnn_score = 0.0
        final_score = 0.0
        
        frame_length = self.stream_config.frame_length
        
        for i in range(0, len(audio), frame_length):
            frame = audio[i:i + frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            
            buffer.append(frame)
            
            # V3 检测
            result = self.kws_detector.process_chunk(frame)
            
            if result:
                # V3 触发，进行 CNN 验证
                v3_score = 1.0
                
                # 获取缓冲区音频
                buffered_audio = buffer.get_all()
                
                # CNN 验证
                _, cnn_score = self.cnn_verifier.verify(buffered_audio)
                
                # 融合评分
                detected, final_score = self.score_fusion.decide(v3_score, cnn_score)
                
                # 重置 V3 流
                self.kws_detector.reset_stream()
                
                if detected:
                    break
        
        # 处理 V3 未触发的情况
        if v3_score == 0.0:
            # 尝试离线检测（添加尾部填充）
            v3_detected, _ = self.kws_detector.detect_full_audio(audio)
            if v3_detected:
                v3_score = 1.0
                _, cnn_score = self.cnn_verifier.verify(audio)
                detected, final_score = self.score_fusion.decide(v3_score, cnn_score)
        
        inference_time = time.perf_counter() - start_time
        audio_duration = len(audio) / self.stream_config.sample_rate
        
        return {
            "detected": detected,
            "v3_score": v3_score,
            "cnn_score": cnn_score,
            "final_score": final_score,
            "inference_time": inference_time,
            "audio_duration": audio_duration,
            "rtf": inference_time / audio_duration if audio_duration > 0 else 0,
        }
    
    def evaluate_dataset(
        self,
        audio_paths: List[str],
        labels: List[int],
        verbose: bool = False
    ) -> Dict:
        """评估整个数据集"""
        results = {
            "predictions": [],
            "v3_scores": [],
            "cnn_scores": [],
            "final_scores": [],
            "inference_times": [],
            "rtfs": [],
            "labels": labels,
        }
        
        for i, (path, label) in enumerate(zip(audio_paths, labels)):
            result = self.evaluate_file_streaming(path)
            
            results["predictions"].append(int(result["detected"]))
            results["v3_scores"].append(result["v3_score"])
            results["cnn_scores"].append(result["cnn_score"])
            results["final_scores"].append(result["final_score"])
            results["inference_times"].append(result["inference_time"])
            results["rtfs"].append(result["rtf"])
            
            if verbose:
                filename = Path(path).name
                status = "✓" if result["detected"] == label else "✗"
                print(f"{status} [{i+1:3d}/{len(audio_paths)}] {filename}: "
                      f"V3={result['v3_score']:.1f}, CNN={result['cnn_score']:.3f}, "
                      f"Final={result['final_score']:.3f}, Label={label}")
        
        # 计算指标
        predictions = np.array(results["predictions"])
        labels_arr = np.array(labels)
        
        tp = np.sum((predictions == 1) & (labels_arr == 1))
        tn = np.sum((predictions == 0) & (labels_arr == 0))
        fp = np.sum((predictions == 1) & (labels_arr == 0))
        fn = np.sum((predictions == 0) & (labels_arr == 1))
        
        results["metrics"] = {
            "accuracy": (tp + tn) / len(labels),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            "far": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "frr": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "avg_rtf": np.mean(results["rtfs"]),
            "avg_inference_time": np.mean(results["inference_times"]),
        }
        
        return results


def load_dataset(data_dir: Path, positive_keywords: List[str]) -> Tuple[List[str], List[int]]:
    """加载数据集"""
    audio_paths = []
    labels = []
    
    for wav_file in sorted(data_dir.glob("*.wav")):
        filename = wav_file.name
        is_positive = any(kw in filename for kw in positive_keywords)
        
        audio_paths.append(str(wav_file))
        labels.append(1 if is_positive else 0)
    
    return audio_paths, labels


def main():
    parser = argparse.ArgumentParser(description="流式 KWS 评估")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="测试数据目录")
    parser.add_argument("--kws-model-dir", type=str, default=None,
                        help="KWS 模型目录（默认 V2）")
    parser.add_argument("--cnn-model", type=str, default=None,
                        help="CNN 验证器模型路径")
    parser.add_argument("--kws-threshold", type=float, default=0.25,
                        help="KWS 检测阈值")
    parser.add_argument("--cnn-threshold", type=float, default=0.5,
                        help="CNN 验证阈值")
    parser.add_argument("--fusion-mode", type=str, default="cascade",
                        choices=["weighted", "cascade", "product"],
                        help="融合模式")
    parser.add_argument("--kws-weight", type=float, default=0.3,
                        help="KWS 权重（加权融合时）")
    parser.add_argument("--final-threshold", type=float, default=0.5,
                        help="最终判定阈值")
    parser.add_argument("--verbose", action="store_true",
                        help="详细输出")
    parser.add_argument("--search-threshold", action="store_true",
                        help="搜索最优阈值")
    args = parser.parse_args()
    
    # 设置路径
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "all"
    kws_model_dir = Path(args.kws_model_dir) if args.kws_model_dir else PROJECT_ROOT / "exp" / "kws_finetune_v2"
    cnn_model_path = Path(args.cnn_model) if args.cnn_model else Path(__file__).parent / "outputs" / "cnn_verifier_latest.pt"
    
    # 检查文件
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    if not kws_model_dir.exists():
        print(f"错误: KWS 模型目录不存在: {kws_model_dir}")
        return
    if not cnn_model_path.exists():
        print(f"错误: CNN 模型不存在: {cnn_model_path}")
        print("请先运行 train_cnn.py 训练模型")
        return
    
    print("=" * 60)
    print("流式 KWS 评估 (KWS + CNN)")
    print("=" * 60)
    print(f"测试集: {data_dir}")
    print(f"KWS 模型: {kws_model_dir}")
    print(f"CNN 模型: {cnn_model_path}")
    print(f"融合模式: {args.fusion_mode}")
    print()
    
    # 加载数据
    positive_keywords = ["你好真真", "你好珍珍"]
    audio_paths, labels = load_dataset(data_dir, positive_keywords)
    
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    print(f"样本统计: 总计 {len(labels)} (正样本: {n_positive}, 负样本: {n_negative})")
    print()
    
    # 创建组件
    print("加载模型...")
    
    # KWS 检测器（默认使用 V2 模型）
    kws_detector = KWSDetector(
        model_dir=kws_model_dir,
        keywords_threshold=args.kws_threshold,
        keywords_score=1.5
    )
    kws_detector.load()
    print("  KWS 检测器已加载")
    
    # CNN 验证器
    cnn_verifier = CNNVerifierWrapper(
        model_path=cnn_model_path,
        threshold=args.cnn_threshold
    )
    cnn_verifier.load()
    print("  CNN 验证器已加载")
    
    # 评分融合
    fusion_config = FusionConfig(
        mode=args.fusion_mode,
        v3_weight=args.kws_weight,
        cnn_weight=1 - args.kws_weight,
        v3_threshold=args.kws_threshold,
        final_threshold=args.final_threshold
    )
    score_fusion = ScoreFusion(fusion_config)
    print(f"  评分融合: {args.fusion_mode}")
    
    # 流式配置
    stream_config = StreamConfig(
        sample_rate=16000,
        frame_duration_ms=30,
        buffer_duration_s=1.5
    )
    
    # 创建评估器
    evaluator = StreamingKWSEvaluator(
        kws_detector=kws_detector,
        cnn_verifier=cnn_verifier,
        score_fusion=score_fusion,
        stream_config=stream_config
    )
    
    print("\n开始评估...")
    results = evaluator.evaluate_dataset(audio_paths, labels, verbose=args.verbose)
    
    # 打印结果
    metrics = results["metrics"]
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"准确率: {metrics['accuracy']*100:.2f}%")
    print(f"精确率: {metrics['precision']*100:.2f}%")
    print(f"召回率: {metrics['recall']*100:.2f}%")
    print(f"F1 分数: {metrics['f1']*100:.2f}%")
    print()
    print(f"FAR (误报率): {metrics['far']*100:.2f}%")
    print(f"FRR (漏报率): {metrics['frr']*100:.2f}%")
    print()
    print(f"混淆矩阵:")
    print(f"  TP: {metrics['tp']}, FN: {metrics['fn']}")
    print(f"  FP: {metrics['fp']}, TN: {metrics['tn']}")
    print()
    print(f"平均 RTF: {metrics['avg_rtf']:.4f}")
    print(f"平均推理时间: {metrics['avg_inference_time']*1000:.1f}ms")
    
    # 搜索最优阈值
    if args.search_threshold:
        print("\n" + "=" * 60)
        print("阈值搜索 (目标 FAR < 10%)")
        print("=" * 60)
        
        final_scores = np.array(results["final_scores"])
        labels_arr = np.array(labels)
        
        best_th, best_metrics = find_optimal_threshold(final_scores, labels_arr, target_far=0.1)
        
        print(f"最优阈值: {best_th:.2f}")
        print(f"  FAR: {best_metrics['far']*100:.2f}%")
        print(f"  FRR: {best_metrics['frr']*100:.2f}%")
        print(f"  准确率: {best_metrics['accuracy']*100:.2f}%")
    
    # 保存结果
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"eval_results_{timestamp}.json"
    
    save_results = {
        "config": {
            "data_dir": str(data_dir),
            "kws_model_dir": str(kws_model_dir),
            "cnn_model": str(cnn_model_path),
            "kws_threshold": args.kws_threshold,
            "cnn_threshold": cnn_verifier.threshold,
            "fusion_mode": args.fusion_mode,
            "final_threshold": args.final_threshold,
        },
        "metrics": metrics,
        "sample_stats": {
            "total": len(labels),
            "positive": n_positive,
            "negative": n_negative,
        },
    }
    
    with open(result_path, "w") as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n结果已保存: {result_path}")
    
    # 判断是否达标
    print("\n" + "=" * 60)
    print("达标判断")
    print("=" * 60)
    
    far_ok = metrics["far"] < 0.10
    frr_ok = metrics["frr"] < 0.05
    rtf_ok = metrics["avg_rtf"] < 1.0
    
    print(f"FAR < 10%: {'✓' if far_ok else '✗'} ({metrics['far']*100:.2f}%)")
    print(f"FRR < 5%:  {'✓' if frr_ok else '✗'} ({metrics['frr']*100:.2f}%)")
    print(f"RTF < 1.0: {'✓' if rtf_ok else '✗'} ({metrics['avg_rtf']:.4f})")
    
    if far_ok and frr_ok and rtf_ok:
        print("\n✓ 所有指标达标！")
    else:
        print("\n✗ 部分指标未达标")


if __name__ == "__main__":
    main()
