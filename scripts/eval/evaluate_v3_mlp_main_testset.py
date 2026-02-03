#!/usr/bin/env python3
"""
V3 + MLP 多阶段关键词检测评估脚本

在主要测试集 data/all/ (406个文件) 上评估 V3 + MLP 方案的性能。

正样本判断：文件名包含"你好真真"或"你好珍珍"（发音相同）
负样本：其他所有文件

指标：FAR（误报率）、FRR（漏报率）、RTF（实时因子）
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "multi_stage_ablation"))

import sherpa_onnx
import librosa


# ========================= 配置 =========================
DEFAULT_CONFIG = {
    "test_dir": "data/all",
    "v3_model_dir": "exp/kws_finetune_v3",
    "mlp_model_path": "experiments/multi_stage_ablation/models/mlp_verifier_real_voice.pt",
    
    # 正样本关键词（发音相同的词都算正样本）
    "positive_keywords": ["你好真真", "你好珍珍"],
    
    # 阶段1配置
    "stage1_threshold": 0.3,
    "num_threads": 4,
    
    # MLP配置
    "mlp_threshold": 0.4,
    "n_mfcc": 13,
    "target_frames": 50,
    
    # 后缀提取配置
    "suffix_start_ratio": 0.4,
    "suffix_min_duration_ms": 200,
    "suffix_max_duration_ms": 800,
    
    "sample_rate": 16000,
}


# ========================= MLP 模型定义 =========================
class SimpleMLP(nn.Module):
    """MLP分类器网络结构"""
    def __init__(self, input_dim: int = 650):
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


# ========================= V3 检测器 =========================
class V3KWSDetector:
    """V3 模型加载和推理"""
    
    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.3,
        num_threads: int = 4
    ):
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.num_threads = num_threads
        self.spotter = None
    
    def load_model(self) -> None:
        """加载 sherpa-onnx KWS 模型"""
        suffix = ".int8.onnx"
        
        # 查找模型文件
        encoder_files = list(self.model_dir.glob(f"encoder-*{suffix}"))
        decoder_files = list(self.model_dir.glob(f"decoder-*{suffix}"))
        joiner_files = list(self.model_dir.glob(f"joiner-*{suffix}"))
        
        if not encoder_files or not decoder_files or not joiner_files:
            raise FileNotFoundError(f"找不到 INT8 ONNX 模型文件: {self.model_dir}")
        
        encoder = str(encoder_files[0])
        decoder = str(decoder_files[0])
        joiner = str(joiner_files[0])
        tokens = str(self.model_dir / "tokens.txt")
        keywords = str(self.model_dir / "keywords.txt")
        
        # 创建 KeywordSpotter
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords,
            num_threads=self.num_threads,
            keywords_threshold=self.threshold,
            provider="cpu",
        )
        print(f"[V3] 模型加载完成: {self.model_dir}")
    
    def detect(self, samples: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, str, float]:
        """
        检测音频中是否包含关键词
        
        Returns:
            (detected, keyword_text, confidence)
        """
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        # 创建流并输入音频
        stream = self.spotter.create_stream()
        stream.accept_waveform(sample_rate, samples.tolist())
        
        # 添加 300ms 尾部填充
        tail_padding = [0.0] * int(0.3 * sample_rate)
        stream.accept_waveform(sample_rate, tail_padding)
        stream.input_finished()
        
        # 解码
        detected, keyword_text = False, ""
        while self.spotter.is_ready(stream):
            self.spotter.decode_stream(stream)
            result = self.spotter.get_result(stream)
            if result:
                detected = True
                keyword_text = result.strip()
                break
        
        return detected, keyword_text, 1.0 if detected else 0.0


# ========================= MLP 验证器 =========================
class MLPVerifier:
    """MLP 验证器加载和推理"""
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        n_mfcc: int = 13,
        target_frames: int = 50
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.n_mfcc = n_mfcc
        self.target_frames = target_frames
        self.input_dim = n_mfcc * target_frames
        self.model = None
    
    def load_model(self) -> None:
        """加载模型"""
        self.model = SimpleMLP(self.input_dim)
        if self.model_path and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            print(f"[MLP] 模型加载完成: {self.model_path}")
        else:
            raise FileNotFoundError(f"找不到 MLP 模型: {self.model_path}")
        self.model.eval()
    
    def _extract_mfcc(self, samples: np.ndarray, sr: int = 16000) -> np.ndarray:
        """提取 MFCC 特征"""
        mfcc = librosa.feature.mfcc(
            y=samples, sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=160
        )
        return mfcc  # (13, n_frames)
    
    def _pad_or_trim(self, features: np.ndarray) -> np.ndarray:
        """填充或截断到目标帧数"""
        n_mfcc, n_frames = features.shape
        if n_frames < self.target_frames:
            padding = np.zeros((n_mfcc, self.target_frames - n_frames))
            features = np.concatenate([features, padding], axis=1)
        elif n_frames > self.target_frames:
            features = features[:, :self.target_frames]
        return features
    
    def verify(self, audio_segment: np.ndarray, sr: int = 16000) -> Tuple[bool, float]:
        """
        验证后缀音频
        
        Returns:
            (is_accepted, confidence)
        """
        if self.model is None:
            self.load_model()
        
        # 1. 提取 MFCC
        mfcc = self._extract_mfcc(audio_segment, sr)
        
        # 2. 填充/截断到目标帧数
        mfcc = self._pad_or_trim(mfcc)
        
        # 3. 归一化
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        
        # 4. 展平为向量
        features = mfcc.flatten()
        
        # 5. 推理
        x = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            confidence = self.model(x).item()
        
        return confidence >= self.threshold, confidence


# ========================= 工具函数 =========================
def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """加载音频文件"""
    samples, sr = sf.read(audio_path, dtype="float32")
    if len(samples.shape) > 1:
        samples = samples[:, 0]  # 转单声道
    if sr != target_sr:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
    return samples, target_sr


def extract_suffix(
    samples: np.ndarray,
    sr: int,
    start_ratio: float = 0.4,
    min_duration_ms: int = 200,
    max_duration_ms: int = 800
) -> np.ndarray:
    """
    提取后缀音频（"真真"部分）
    """
    total_samples = len(samples)
    start_sample = int(total_samples * start_ratio)
    min_samples = int(min_duration_ms * sr / 1000)
    max_samples = int(max_duration_ms * sr / 1000)
    
    suffix = samples[start_sample:]
    
    if len(suffix) < min_samples:
        new_start = max(0, total_samples - min_samples)
        suffix = samples[new_start:]
    elif len(suffix) > max_samples:
        suffix = suffix[:max_samples]
    
    return suffix


def is_positive_sample(filename: str, positive_keywords: List[str]) -> bool:
    """
    判断是否为正样本
    
    正样本：文件名中包含任一正样本关键词
    """
    for keyword in positive_keywords:
        if keyword in filename:
            return True
    return False


def get_audio_duration(samples: np.ndarray, sr: int) -> float:
    """获取音频时长（秒）"""
    return len(samples) / sr


# ========================= 两阶段检测系统 =========================
class TwoStageKWS:
    """两阶段关键词检测系统"""
    
    def __init__(self, config: Dict, stage1_only: bool = False):
        self.config = config
        self.stage1_detector = None
        self.mlp_verifier = None
        self.stage1_only = stage1_only
    
    def load(self, project_root: Path):
        """加载模型"""
        # 阶段1: V3 KWS
        v3_model_dir = project_root / self.config["v3_model_dir"]
        self.stage1_detector = V3KWSDetector(
            model_dir=str(v3_model_dir),
            threshold=self.config["stage1_threshold"],
            num_threads=self.config["num_threads"]
        )
        self.stage1_detector.load_model()
        
        # 阶段2: MLP 验证器（仅在非 stage1_only 模式加载）
        if not self.stage1_only:
            mlp_model_path = project_root / self.config["mlp_model_path"]
            self.mlp_verifier = MLPVerifier(
                model_path=str(mlp_model_path),
                threshold=self.config["mlp_threshold"],
                n_mfcc=self.config["n_mfcc"],
                target_frames=self.config["target_frames"]
            )
            self.mlp_verifier.load_model()
        else:
            print("[INFO] 仅阶段1模式，跳过 MLP 验证器加载")
    
    def detect(self, audio_path: str) -> Dict:
        """
        两阶段检测
        
        判断逻辑：
        1. 阶段1通过 → 进入阶段2
        2. 阶段1不通过 → 直接拒绝
        3. 阶段2通过 → 最终接受
        4. 阶段2不通过 → 最终拒绝（过滤误报）
        """
        samples, sr = load_audio(audio_path, self.config["sample_rate"])
        audio_duration = get_audio_duration(samples, sr)
        
        result = {
            "audio_path": audio_path,
            "audio_duration": audio_duration,
            "stage1_passed": False,
            "stage1_keyword": "",
            "stage1_confidence": 0.0,
            "stage2_passed": False,
            "stage2_confidence": 0.0,
            "final_accepted": False,
            "inference_time": 0.0
        }
        
        start_time = time.perf_counter()
        
        # ===== 阶段1: sherpa-onnx KWS =====
        detected, keyword, confidence = self.stage1_detector.detect(samples, sr)
        result["stage1_passed"] = detected
        result["stage1_keyword"] = keyword
        result["stage1_confidence"] = confidence
        
        if not detected:
            # 阶段1未检测到关键词，直接拒绝
            result["inference_time"] = time.perf_counter() - start_time
            return result
        
        # 如果是仅阶段1模式，直接返回阶段1结果
        if self.stage1_only:
            result["stage2_passed"] = True  # 跳过阶段2，默认通过
            result["stage2_confidence"] = 1.0
            result["final_accepted"] = detected
            result["inference_time"] = time.perf_counter() - start_time
            return result
        
        # ===== 阶段2: MLP 后缀验证 =====
        # 提取后缀音频
        suffix = extract_suffix(
            samples, sr,
            start_ratio=self.config["suffix_start_ratio"],
            min_duration_ms=self.config["suffix_min_duration_ms"],
            max_duration_ms=self.config["suffix_max_duration_ms"]
        )
        
        # MLP 验证
        accepted, mlp_conf = self.mlp_verifier.verify(suffix, sr)
        result["stage2_passed"] = accepted
        result["stage2_confidence"] = mlp_conf
        
        # ===== 最终判断 =====
        result["final_accepted"] = detected and accepted
        result["inference_time"] = time.perf_counter() - start_time
        
        return result


# ========================= 指标计算 =========================
def calculate_metrics(results: List[Dict], positive_keywords: List[str]) -> Dict:
    """计算评估指标"""
    # 分类统计
    total_positive = 0  # 正样本总数
    total_negative = 0  # 负样本总数
    
    true_positive = 0   # 正样本被正确检测
    false_negative = 0  # 正样本被漏报 (FN)
    false_positive = 0  # 负样本被误报 (FP)
    true_negative = 0   # 负样本被正确拒绝
    
    total_inference_time = 0.0
    total_audio_duration = 0.0
    
    # 错误详情
    false_negatives_list = []  # 漏报列表
    false_positives_list = []  # 误报列表
    
    for r in results:
        filename = Path(r["audio_path"]).name
        is_positive = is_positive_sample(filename, positive_keywords)
        detected = r["final_accepted"]
        
        total_inference_time += r["inference_time"]
        total_audio_duration += r["audio_duration"]
        
        if is_positive:
            total_positive += 1
            if detected:
                true_positive += 1
            else:
                false_negative += 1
                false_negatives_list.append({
                    "file": filename,
                    "stage1_passed": r["stage1_passed"],
                    "stage2_passed": r["stage2_passed"],
                    "stage2_confidence": r["stage2_confidence"]
                })
        else:
            total_negative += 1
            if detected:
                false_positive += 1
                false_positives_list.append({
                    "file": filename,
                    "stage1_keyword": r["stage1_keyword"],
                    "stage2_confidence": r["stage2_confidence"]
                })
            else:
                true_negative += 1
    
    # 计算指标
    frr = false_negative / total_positive if total_positive > 0 else 0.0
    far = false_positive / total_negative if total_negative > 0 else 0.0
    recall = true_positive / total_positive if total_positive > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    accuracy = (true_positive + true_negative) / len(results) if results else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0.0
    
    return {
        "total_samples": len(results),
        "total_positive": total_positive,
        "total_negative": total_negative,
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "frr": frr,
        "far": far,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "f1": f1,
        "rtf": rtf,
        "total_inference_time": total_inference_time,
        "total_audio_duration": total_audio_duration,
        "avg_inference_time_ms": total_inference_time / len(results) * 1000 if results else 0.0,
        "false_negatives": false_negatives_list,
        "false_positives": false_positives_list
    }


# ========================= 主函数 =========================
def main():
    parser = argparse.ArgumentParser(description="V3 + MLP 多阶段 KWS 评估")
    parser.add_argument("--test-dir", type=str, default=None, help="测试集目录")
    parser.add_argument("--verbose", action="store_true", help="显示详细输出")
    parser.add_argument("--save-results", action="store_true", default=True, help="保存结果到 JSON")
    parser.add_argument("--stage1-only", action="store_true", help="仅评估阶段1（跳过MLP验证）")
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    
    # 设置路径
    test_dir = Path(args.test_dir) if args.test_dir else PROJECT_ROOT / config["test_dir"]
    
    print("=" * 60)
    if args.stage1_only:
        print("V3 模型评估（仅阶段1）")
    else:
        print("V3 + MLP 多阶段关键词检测评估")
    print("=" * 60)
    print(f"测试集: {test_dir}")
    print(f"正样本关键词: {config['positive_keywords']}")
    print(f"阶段1阈值: {config['stage1_threshold']}")
    if not args.stage1_only:
        print(f"MLP阈值: {config['mlp_threshold']}")
    print()
    
    # 加载模型
    print("加载模型...")
    kws = TwoStageKWS(config, stage1_only=args.stage1_only)
    kws.load(PROJECT_ROOT)
    print()
    
    # 获取测试文件
    audio_files = sorted(test_dir.glob("*.wav"))
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 统计正负样本
    positive_count = sum(1 for f in audio_files if is_positive_sample(f.name, config["positive_keywords"]))
    negative_count = len(audio_files) - positive_count
    print(f"正样本: {positive_count}, 负样本: {negative_count}")
    print()
    
    # 批量评估
    print("开始评估...")
    results = []
    for i, audio_file in enumerate(audio_files):
        result = kws.detect(str(audio_file))
        results.append(result)
        
        if args.verbose:
            filename = audio_file.name
            is_pos = is_positive_sample(filename, config["positive_keywords"])
            label = "正" if is_pos else "负"
            pred = "✓" if result["final_accepted"] else "✗"
            s1 = "✓" if result["stage1_passed"] else "✗"
            s2_conf = f"{result['stage2_confidence']:.3f}" if result["stage1_passed"] else "-"
            print(f"[{i+1:3d}/{len(audio_files)}] [{label}] {pred} S1:{s1} S2:{s2_conf} {filename[:50]}")
        else:
            if (i + 1) % 50 == 0:
                print(f"进度: {i+1}/{len(audio_files)}")
    
    print()
    
    # 计算指标
    metrics = calculate_metrics(results, config["positive_keywords"])
    
    # 输出结果
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"样本统计:")
    print(f"  总样本: {metrics['total_samples']}")
    print(f"  正样本: {metrics['total_positive']} (检测成功: {metrics['true_positive']}, 漏报: {metrics['false_negative']})")
    print(f"  负样本: {metrics['total_negative']} (正确拒绝: {metrics['true_negative']}, 误报: {metrics['false_positive']})")
    print()
    print(f"性能指标:")
    print(f"  FRR (漏报率): {metrics['frr']*100:.2f}%")
    print(f"  FAR (误报率): {metrics['far']*100:.2f}%")
    print(f"  Recall: {metrics['recall']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score: {metrics['f1']*100:.2f}")
    print()
    print(f"时间性能:")
    print(f"  总推理时间: {metrics['total_inference_time']:.2f}s")
    print(f"  总音频时长: {metrics['total_audio_duration']:.2f}s")
    print(f"  RTF: {metrics['rtf']:.4f}")
    print(f"  平均推理时间: {metrics['avg_inference_time_ms']:.2f}ms/样本")
    print()
    
    # 显示漏报详情
    if metrics['false_negatives']:
        print(f"漏报详情 ({len(metrics['false_negatives'])}个):")
        for fn in metrics['false_negatives'][:10]:
            s1 = "✓" if fn['stage1_passed'] else "✗"
            s2 = "✓" if fn['stage2_passed'] else "✗"
            print(f"  S1:{s1} S2:{s2} (conf={fn['stage2_confidence']:.3f}) {fn['file']}")
        if len(metrics['false_negatives']) > 10:
            print(f"  ... 共 {len(metrics['false_negatives'])} 个漏报")
        print()
    
    # 显示误报详情
    if metrics['false_positives']:
        print(f"误报详情 ({len(metrics['false_positives'])}个):")
        for fp in metrics['false_positives'][:10]:
            print(f"  S2 conf={fp['stage2_confidence']:.3f} {fp['file']}")
        if len(metrics['false_positives']) > 10:
            print(f"  ... 共 {len(metrics['false_positives'])} 个误报")
        print()
    
    # 保存结果
    if args.save_results:
        log_dir = PROJECT_ROOT / "log" / "evaluation"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = log_dir / f"v3_mlp_eval_{timestamp}.json"
        
        output_data = {
            "config": config,
            "metrics": {k: v for k, v in metrics.items() if k not in ["false_negatives", "false_positives"]},
            "false_negatives": metrics["false_negatives"],
            "false_positives": metrics["false_positives"],
            "timestamp": timestamp
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {output_file}")
    
    print()
    print("=" * 60)
    print("达标检查")
    print("=" * 60)
    frr_pass = metrics['frr'] < 0.05
    far_pass = metrics['far'] < 0.10
    rtf_pass = metrics['rtf'] < 1.0
    print(f"FRR < 5%:  {'✓ 达标' if frr_pass else '✗ 未达标'} ({metrics['frr']*100:.2f}%)")
    print(f"FAR < 10%: {'✓ 达标' if far_pass else '✗ 未达标'} ({metrics['far']*100:.2f}%)")
    print(f"RTF < 1.0: {'✓ 达标' if rtf_pass else '✗ 未达标'} ({metrics['rtf']:.4f})")
    print(f"总体: {'✓ 全部达标' if (frr_pass and far_pass and rtf_pass) else '✗ 未达标'}")


if __name__ == "__main__":
    main()
