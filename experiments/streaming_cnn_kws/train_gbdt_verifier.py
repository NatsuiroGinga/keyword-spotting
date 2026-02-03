#!/usr/bin/env python3
"""
组合声学规则验证器
组合多个弱特征创建更强的验证规则
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import joblib


PROJECT_ROOT = Path(__file__).parent.parent.parent


def extract_extended_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    提取扩展声学特征向量
    """
    if len(audio) < sr * 0.2:
        return None
    
    features = []
    
    # RMS 能量
    rms = librosa.feature.rms(y=audio, hop_length=160)[0]
    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])
    
    # 过零率
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=160)[0]
    features.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])
    
    # 频谱质心
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=160)[0]
    features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
    
    # 频谱带宽
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=160)[0]
    features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
    
    # 频谱滚降点
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=160)[0]
    features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
    
    # MFCC（前 13 个系数的统计量）
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=160)
    for i in range(13):
        features.extend([np.mean(mfcc[i]), np.std(mfcc[i])])
    
    # 色度特征
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=160)
    features.extend([np.mean(chroma), np.std(chroma)])
    
    # 时长
    features.append(len(audio) / sr)
    
    return np.array(features, dtype=np.float32)


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


def load_dataset_features(data_dir: Path, positive_keywords: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载数据集并提取特征"""
    
    features_list = []
    labels = []
    filenames = []
    
    for wav_file in sorted(data_dir.glob("*.wav")):
        filename = wav_file.name
        is_positive = any(kw in filename for kw in positive_keywords)
        
        audio, sr = librosa.load(str(wav_file), sr=16000)
        suffix = extract_suffix(audio)
        
        feat = extract_extended_features(suffix)
        if feat is None:
            continue
        
        features_list.append(feat)
        labels.append(1 if is_positive else 0)
        filenames.append(filename)
    
    return np.array(features_list), np.array(labels), filenames


def train_gbdt_verifier(features: np.ndarray, labels: np.ndarray) -> Tuple:
    """训练 GBDT 验证器"""
    
    # 交叉验证预测
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=3,
        learning_rate=0.1,
        random_state=42
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = cross_val_predict(clf, features, labels, cv=cv, method='predict_proba')[:, 1]
    
    # 计算 AUC
    auc = roc_auc_score(labels, probs)
    print(f"CV AUC: {auc:.4f}")
    
    # 搜索最优阈值
    best_th, best_far, best_frr = 0.5, 1.0, 1.0
    for th in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= th).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        if far <= 0.10 and frr < best_frr:
            best_th, best_far, best_frr = th, far, frr
    
    print(f"最佳阈值 (FAR<10%): {best_th:.2f}")
    print(f"  FAR: {best_far*100:.2f}%")
    print(f"  FRR: {best_frr*100:.2f}%")
    
    # 在全部数据上训练最终模型
    clf.fit(features, labels)
    
    # 特征重要性
    print("\n特征重要性 Top 10:")
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1][:10]
    feature_names = (
        ["rms_mean", "rms_std", "rms_max", "rms_min"] +
        ["zcr_mean", "zcr_std", "zcr_max"] +
        ["centroid_mean", "centroid_std"] +
        ["bandwidth_mean", "bandwidth_std"] +
        ["rolloff_mean", "rolloff_std"] +
        [f"mfcc{i}_mean" for i in range(13)] +
        [f"mfcc{i}_std" for i in range(13)] +
        ["chroma_mean", "chroma_std", "duration"]
    )
    
    for i in indices:
        if i < len(feature_names):
            print(f"  {feature_names[i]}: {importance[i]:.4f}")
    
    return clf, best_th, {"auc": auc, "far": best_far, "frr": best_frr}


def main():
    data_dir = PROJECT_ROOT / "data" / "all"
    positive_keywords = ["你好真真", "你好珍珍"]
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GBDT 声学验证器训练")
    print("=" * 60)
    
    print("\n加载数据集...")
    features, labels, filenames = load_dataset_features(data_dir, positive_keywords)
    print(f"样本数: {len(labels)} (正:{sum(labels)}, 负:{len(labels)-sum(labels)})")
    print(f"特征维度: {features.shape[1]}")
    
    print("\n训练 GBDT 验证器...")
    clf, threshold, metrics = train_gbdt_verifier(features, labels)
    
    # 保存模型
    model_path = output_dir / "gbdt_verifier.joblib"
    save_dict = {
        "model": clf,
        "threshold": threshold,
        "metrics": metrics,
    }
    joblib.dump(save_dict, model_path)
    print(f"\n模型已保存: {model_path}")
    
    # 详细分析错误样本
    print("\n" + "=" * 60)
    print("错误分析")
    print("=" * 60)
    
    probs = clf.predict_proba(features)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    # 假阳性（FP）
    fp_indices = np.where((preds == 1) & (labels == 0))[0]
    print(f"\n假阳性 (FP): {len(fp_indices)} 个")
    for i in fp_indices[:10]:
        print(f"  {filenames[i]}: prob={probs[i]:.3f}")
    
    # 假阴性（FN）
    fn_indices = np.where((preds == 0) & (labels == 1))[0]
    print(f"\n假阴性 (FN): {len(fn_indices)} 个")
    for i in fn_indices[:10]:
        print(f"  {filenames[i]}: prob={probs[i]:.3f}")


if __name__ == "__main__":
    main()
