#!/usr/bin/env python3
"""
基于声学规则的验证器
分析"真真"后缀的声学特征，设计规则来区分正负样本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple, Dict
import json


PROJECT_ROOT = Path(__file__).parent.parent.parent


def extract_suffix_features(audio: np.ndarray, sr: int = 16000) -> Dict:
    """
    提取后缀音频的声学特征
    """
    # 确保足够长度
    if len(audio) < sr * 0.2:  # 至少 200ms
        return None
    
    # 能量特征
    rms = librosa.feature.rms(y=audio, hop_length=160)[0]
    
    # 过零率
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=160)[0]
    
    # 基频估计（用于检测声调）
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )
    f0_valid = f0[~np.isnan(f0)]
    
    # 音频时长
    duration = len(audio) / sr
    
    # MFCC 统计
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    features = {
        "duration": duration,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_max": float(np.max(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "f0_mean": float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0,
        "f0_std": float(np.std(f0_valid)) if len(f0_valid) > 0 else 0.0,
        "voiced_ratio": float(np.mean(voiced_probs > 0.5)) if len(voiced_probs) > 0 else 0.0,
        "mfcc1_mean": float(np.mean(mfcc[0])),
        "mfcc1_std": float(np.std(mfcc[0])),
    }
    
    return features


def extract_suffix(audio: np.ndarray, start_ratio: float = 0.4, 
                   min_ms: int = 200, max_ms: int = 800, sr: int = 16000) -> np.ndarray:
    """提取后缀音频"""
    total_samples = len(audio)
    start_sample = int(total_samples * start_ratio)
    
    suffix = audio[start_sample:]
    
    min_samples = int(min_ms * sr / 1000)
    max_samples = int(max_ms * sr / 1000)
    
    if len(suffix) < min_samples:
        suffix = audio[max(0, total_samples - min_samples):]
    elif len(suffix) > max_samples:
        suffix = suffix[:max_samples]
    
    return suffix


def analyze_dataset(data_dir: Path, positive_keywords: List[str]):
    """分析数据集的声学特征分布"""
    
    positive_features = []
    negative_features = []
    
    for wav_file in sorted(data_dir.glob("*.wav")):
        filename = wav_file.name
        is_positive = any(kw in filename for kw in positive_keywords)
        
        # 加载音频
        audio, sr = librosa.load(str(wav_file), sr=16000)
        
        # 提取后缀
        suffix = extract_suffix(audio)
        
        # 提取特征
        features = extract_suffix_features(suffix)
        if features is None:
            continue
        
        features["filename"] = filename
        
        if is_positive:
            positive_features.append(features)
        else:
            negative_features.append(features)
    
    return positive_features, negative_features


def print_feature_stats(features: List[Dict], label: str):
    """打印特征统计"""
    print(f"\n{label} ({len(features)} 样本):")
    
    if not features:
        return
    
    keys = [k for k in features[0].keys() if k != "filename"]
    
    for key in keys:
        values = [f[key] for f in features]
        print(f"  {key}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
              f"min={np.min(values):.4f}, max={np.max(values):.4f}")


def find_discriminative_rules(pos_features: List[Dict], neg_features: List[Dict]) -> List[Dict]:
    """寻找区分性规则"""
    rules = []
    
    feature_keys = [k for k in pos_features[0].keys() if k != "filename"]
    
    for key in feature_keys:
        pos_values = np.array([f[key] for f in pos_features])
        neg_values = np.array([f[key] for f in neg_features])
        
        # 计算分离度
        pos_mean, pos_std = np.mean(pos_values), np.std(pos_values)
        neg_mean, neg_std = np.mean(neg_values), np.std(neg_values)
        
        # 效应大小 (Cohen's d)
        pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
        if pooled_std > 0:
            cohens_d = abs(pos_mean - neg_mean) / pooled_std
        else:
            cohens_d = 0
        
        if cohens_d > 0.3:  # 中等效应
            # 确定方向
            direction = ">" if pos_mean > neg_mean else "<"
            threshold = (pos_mean + neg_mean) / 2
            
            # 计算 FAR 和 FRR
            if direction == ">":
                tp = np.sum(pos_values >= threshold)
                fn = np.sum(pos_values < threshold)
                fp = np.sum(neg_values >= threshold)
                tn = np.sum(neg_values < threshold)
            else:
                tp = np.sum(pos_values <= threshold)
                fn = np.sum(pos_values > threshold)
                fp = np.sum(neg_values <= threshold)
                tn = np.sum(neg_values > threshold)
            
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            rules.append({
                "feature": key,
                "direction": direction,
                "threshold": threshold,
                "cohens_d": cohens_d,
                "far": far,
                "frr": frr,
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
            })
    
    # 按效应大小排序
    rules.sort(key=lambda x: x["cohens_d"], reverse=True)
    
    return rules


def main():
    data_dir = PROJECT_ROOT / "data" / "all"
    positive_keywords = ["你好真真", "你好珍珍"]
    
    print("=" * 60)
    print("声学特征分析")
    print("=" * 60)
    
    pos_features, neg_features = analyze_dataset(data_dir, positive_keywords)
    
    print_feature_stats(pos_features, "正样本")
    print_feature_stats(neg_features, "负样本")
    
    print("\n" + "=" * 60)
    print("区分性规则（按 Cohen's d 排序）")
    print("=" * 60)
    
    rules = find_discriminative_rules(pos_features, neg_features)
    
    for rule in rules[:10]:
        print(f"\n{rule['feature']}:")
        print(f"  规则: value {rule['direction']} {rule['threshold']:.4f}")
        print(f"  Cohen's d: {rule['cohens_d']:.3f}")
        print(f"  正样本均值: {rule['pos_mean']:.4f}, 负样本均值: {rule['neg_mean']:.4f}")
        print(f"  FAR: {rule['far']*100:.2f}%, FRR: {rule['frr']*100:.2f}%")


if __name__ == "__main__":
    main()
