#!/usr/bin/env python3
"""
在真实人声数据集上训练 MLP 验证器

数据集: data/all/ (406个文件)
- 正样本: 文件名包含"你好真真"或"你好珍珍" (87个)
- 负样本: 其他文件 (319个)

特征: 从音频后缀提取 MFCC 特征 (13维 x 50帧 = 650维)
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import random

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ========================= 配置 =========================
DEFAULT_CONFIG = {
    "data_dir": "data/all",
    "output_dir": "experiments/multi_stage_ablation/models",
    "positive_keywords": ["你好真真", "你好珍珍"],
    
    # 特征提取配置
    "n_mfcc": 13,
    "target_frames": 50,
    "n_fft": 512,
    "hop_length": 160,
    "sample_rate": 16000,
    
    # 后缀提取配置
    "suffix_start_ratio": 0.4,
    "suffix_min_duration_ms": 200,
    "suffix_max_duration_ms": 800,
    
    # 训练配置
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.001,
    "val_split": 0.2,
    "random_seed": 42,
    
    # 类别平衡
    "use_class_weight": True,
}


# ========================= MLP 模型 =========================
class SimpleMLP(nn.Module):
    """简单的 MLP 分类器"""
    
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


# ========================= 数据加载 =========================
def is_positive_sample(filename: str, positive_keywords: List[str]) -> bool:
    """判断是否为正样本"""
    for keyword in positive_keywords:
        if keyword in filename:
            return True
    return False


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
    """提取后缀音频（"真真"部分）"""
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


def extract_mfcc(
    samples: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160
) -> np.ndarray:
    """提取 MFCC 特征"""
    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfcc  # (n_mfcc, n_frames)


def pad_or_trim(features: np.ndarray, target_frames: int = 50) -> np.ndarray:
    """填充或截断到目标帧数"""
    n_mfcc, n_frames = features.shape
    
    if n_frames < target_frames:
        padding = np.zeros((n_mfcc, target_frames - n_frames))
        features = np.concatenate([features, padding], axis=1)
    elif n_frames > target_frames:
        features = features[:, :target_frames]
    
    return features


def load_dataset(config: Dict) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    """
    加载数据集
    
    Returns:
        (positive_samples, negative_samples)
        每个元素是 (suffix_audio, sample_rate) 的元组
    """
    data_dir = PROJECT_ROOT / config["data_dir"]
    audio_files = sorted(data_dir.glob("*.wav"))
    
    positive_samples = []
    negative_samples = []
    
    print(f"加载数据集: {data_dir}")
    print(f"找到 {len(audio_files)} 个音频文件")
    
    for audio_path in audio_files:
        try:
            # 加载音频
            samples, sr = load_audio(str(audio_path), config["sample_rate"])
            
            # 提取后缀
            suffix = extract_suffix(
                samples, sr,
                start_ratio=config["suffix_start_ratio"],
                min_duration_ms=config["suffix_min_duration_ms"],
                max_duration_ms=config["suffix_max_duration_ms"]
            )
            
            # 根据文件名判断标签
            if is_positive_sample(audio_path.name, config["positive_keywords"]):
                positive_samples.append((suffix, sr))
            else:
                negative_samples.append((suffix, sr))
                
        except Exception as e:
            print(f"  跳过 {audio_path.name}: {e}")
    
    print(f"正样本: {len(positive_samples)}, 负样本: {len(negative_samples)}")
    
    return positive_samples, negative_samples


def prepare_features(
    positive_samples: List[Tuple[np.ndarray, int]],
    negative_samples: List[Tuple[np.ndarray, int]],
    config: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备特征和标签
    
    Returns:
        (X, y) - 特征矩阵和标签向量
    """
    X, y = [], []
    
    print("提取 MFCC 特征...")
    
    # 正样本
    for samples, sr in positive_samples:
        mfcc = extract_mfcc(
            samples, sr,
            n_mfcc=config["n_mfcc"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"]
        )
        mfcc = pad_or_trim(mfcc, config["target_frames"])
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)  # 归一化
        X.append(mfcc.flatten())
        y.append(1)
    
    # 负样本
    for samples, sr in negative_samples:
        mfcc = extract_mfcc(
            samples, sr,
            n_mfcc=config["n_mfcc"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"]
        )
        mfcc = pad_or_trim(mfcc, config["target_frames"])
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)  # 归一化
        X.append(mfcc.flatten())
        y.append(0)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"特征维度: {X.shape}")
    
    return X, y


# ========================= 训练 =========================
def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict
) -> Tuple[SimpleMLP, Dict]:
    """
    训练 MLP 模型
    
    Returns:
        (model, history)
    """
    # 转换为 tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    
    # 数据加载器
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # 初始化模型
    input_dim = config["n_mfcc"] * config["target_frames"]
    model = SimpleMLP(input_dim)
    
    # 计算类别权重（处理类别不平衡）
    if config["use_class_weight"]:
        n_positive = y_train.sum()
        n_negative = len(y_train) - n_positive
        pos_weight = torch.tensor([n_negative / n_positive])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # 由于模型输出已经是 sigmoid，需要调整
        # 改用带权重的 BCELoss
        weight_pos = n_negative / (n_positive + n_negative)
        weight_neg = n_positive / (n_positive + n_negative)
        print(f"类别权重: 正样本={weight_pos:.3f}, 负样本={weight_neg:.3f}")
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )
    
    # 训练历史
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\n开始训练 ({config['epochs']} epochs)...")
    print("-" * 60)
    
    for epoch in range(config["epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            
            # 如果使用类别权重，手动计算加权损失
            if config["use_class_weight"]:
                weights = torch.where(batch_y == 1, weight_pos, weight_neg)
                loss = (weights * nn.functional.binary_cross_entropy(outputs, batch_y, reduction='none')).mean()
            else:
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_predicted = (val_outputs >= 0.5).float()
            val_correct = (val_predicted == y_val_t).sum().item()
            val_acc = val_correct / len(y_val)
            
            # 计算精确率、召回率、F1
            val_pred_np = val_predicted.numpy().flatten()
            tp = ((val_pred_np == 1) & (y_val == 1)).sum()
            fp = ((val_pred_np == 1) & (y_val == 0)).sum()
            fn = ((val_pred_np == 0) & (y_val == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录历史
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # 打印进度
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                  f"P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n加载最佳模型 (Val Acc: {best_val_acc:.4f})")
    
    return model, history


# ========================= 评估 =========================
def evaluate_model(
    model: SimpleMLP,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """评估模型性能"""
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        outputs = model(X_test_t)
        predictions = (outputs >= 0.5).float().numpy().flatten()
    
    # 分类报告
    report = classification_report(y_test, predictions, target_names=["负样本", "正样本"], output_dict=True)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, predictions)
    
    # 计算 FAR 和 FRR
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # 误报率
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # 漏报率
    
    return {
        "accuracy": report["accuracy"],
        "precision": report["正样本"]["precision"],
        "recall": report["正样本"]["recall"],
        "f1": report["正样本"]["f1-score"],
        "far": far,
        "frr": frr,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


# ========================= 主函数 =========================
def main():
    parser = argparse.ArgumentParser(description="在真实人声数据集上训练 MLP 验证器")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--no-class-weight", action="store_true", help="不使用类别权重")
    parser.add_argument("--output-name", type=str, default="mlp_verifier_real_voice.pt", help="输出模型名称")
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["use_class_weight"] = not args.no_class_weight
    
    # 设置随机种子
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    print("=" * 60)
    print("在真实人声数据集上训练 MLP 验证器")
    print("=" * 60)
    print(f"正样本关键词: {config['positive_keywords']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"使用类别权重: {config['use_class_weight']}")
    print()
    
    # 1. 加载数据集
    positive_samples, negative_samples = load_dataset(config)
    
    # 2. 提取特征
    X, y = prepare_features(positive_samples, negative_samples, config)
    
    # 3. 划分训练集/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config["val_split"],
        random_state=config["random_seed"],
        stratify=y
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} (正: {y_train.sum()}, 负: {len(y_train) - y_train.sum()})")
    print(f"  验证集: {len(X_val)} (正: {y_val.sum()}, 负: {len(y_val) - y_val.sum()})")
    
    # 4. 训练模型
    model, history = train_mlp(X_train, y_train, X_val, y_val, config)
    
    # 5. 评估模型
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    
    metrics = evaluate_model(model, X_val, y_val)
    
    print(f"\n验证集性能:")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall: {metrics['recall']*100:.2f}%")
    print(f"  F1 Score: {metrics['f1']*100:.2f}%")
    print(f"  FAR (误报率): {metrics['far']*100:.2f}%")
    print(f"  FRR (漏报率): {metrics['frr']*100:.2f}%")
    
    print(f"\n混淆矩阵:")
    cm = metrics["confusion_matrix"]
    print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}, TP={cm[1][1]}")
    
    # 6. 保存模型
    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / args.output_name
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存: {model_path}")
    
    # 保存训练日志
    log_dir = PROJECT_ROOT / "log" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"mlp_real_voice_train_{timestamp}.json"
    
    log_data = {
        "config": config,
        "metrics": metrics,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "model_path": str(model_path),
        "timestamp": timestamp
    }
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"训练日志: {log_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
