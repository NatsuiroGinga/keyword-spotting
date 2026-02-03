#!/usr/bin/env python3
"""
CNN 验证器训练脚本
在真实人声数据集上训练 1D CNN 验证器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from models.cnn_verifier import CNNVerifier, CNNConfig, create_cnn_verifier
from features.feature_extractor import FeatureExtractor, FeatureConfig, SuffixExtractor


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


class KWSDataset(Dataset):
    """唤醒词数据集"""
    
    def __init__(
        self,
        audio_paths: List[str],
        labels: List[int],
        feature_config: FeatureConfig,
        suffix_extractor: SuffixExtractor,
        augment: bool = False
    ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = FeatureExtractor(feature_config)
        self.suffix_extractor = suffix_extractor
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import librosa
        
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # 加载音频
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # 提取后缀
        suffix = self.suffix_extractor.extract(audio)
        
        # 数据增强
        if self.augment:
            suffix = self._augment(suffix)
        
        # 提取特征
        features = self.feature_extractor.extract_for_cnn(suffix)
        
        return (
            torch.from_numpy(features).float(),
            torch.tensor([label], dtype=torch.float32)
        )
    
    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """增强的数据增强"""
        # 随机时间偏移（更强）
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-0.15, 0.15) * len(audio))
            audio = np.roll(audio, shift)
        
        # 随机增益（更大范围）
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.7, 1.3)
            audio = audio * gain
        
        # 添加噪声（更强）
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.002, 0.02)
            noise = np.random.randn(len(audio)) * noise_level
            audio = audio + noise
        
        # 随机速度变换（通过重采样模拟）
        if np.random.random() < 0.3:
            speed_factor = np.random.uniform(0.9, 1.1)
            new_length = int(len(audio) / speed_factor)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio
            )
            # 填充或截断
            if len(audio) < self.suffix_extractor.min_duration_ms * 16:
                audio = np.pad(audio, (0, self.suffix_extractor.min_duration_ms * 16 - len(audio)))
        
        # SpecAugment-like: 随机零掩码
        if np.random.random() < 0.3:
            mask_len = int(np.random.uniform(0.05, 0.15) * len(audio))
            mask_start = np.random.randint(0, max(1, len(audio) - mask_len))
            audio[mask_start:mask_start + mask_len] = 0
        
        return audio


def load_dataset(data_dir: Path, positive_keywords: List[str]) -> Tuple[List[str], List[int]]:
    """加载数据集"""
    audio_paths = []
    labels = []
    
    for wav_file in sorted(data_dir.glob("*.wav")):
        filename = wav_file.name
        
        # 判断是否为正样本
        is_positive = any(kw in filename for kw in positive_keywords)
        
        audio_paths.append(str(wav_file))
        labels.append(1 if is_positive else 0)
    
    return audio_paths, labels


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """创建加权采样器，处理类别不平衡"""
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> dict:
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            if probs.ndim == 0:
                probs = [probs.item()]
                preds = [preds.item()]
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "frr": frr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "probs": all_probs,
        "labels": all_labels,
    }


def find_best_threshold(probs: np.ndarray, labels: np.ndarray, target_far: float = 0.1) -> Tuple[float, dict]:
    """寻找最佳阈值"""
    best_th = 0.5
    best_metrics = None
    
    for th in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= th).astype(int)
        
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        if far <= target_far:
            if best_metrics is None or frr < best_metrics["frr"]:
                best_th = th
                best_metrics = {"threshold": th, "far": far, "frr": frr}
    
    if best_metrics is None:
        best_metrics = {"threshold": 0.5, "far": 1.0, "frr": 1.0}
    
    return best_th, best_metrics


def main():
    parser = argparse.ArgumentParser(description="训练 CNN 验证器")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="数据目录")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--model-type", type=str, default="cnn",
                        choices=["cnn", "cnn_attention"],
                        help="模型类型")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--n-mfcc", type=int, default=40,
                        help="MFCC 系数数量")
    parser.add_argument("--target-frames", type=int, default=50,
                        help="目标帧数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置路径
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "all"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 正样本关键词
    positive_keywords = ["你好真真", "你好珍珍"]
    
    # 加载数据
    print(f"\n加载数据集: {data_dir}")
    audio_paths, labels = load_dataset(data_dir, positive_keywords)
    
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    print(f"总样本数: {len(labels)}")
    print(f"正样本: {n_positive}, 负样本: {n_negative}")
    print(f"类别比例: 1:{n_negative/n_positive:.1f}")
    
    # 划分训练集和验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        audio_paths, labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels
    )
    
    print(f"\n训练集: {len(train_paths)} (正:{sum(train_labels)}, 负:{len(train_labels)-sum(train_labels)})")
    print(f"验证集: {len(val_paths)} (正:{sum(val_labels)}, 负:{len(val_labels)-sum(val_labels)})")
    
    # 配置
    feature_config = FeatureConfig(
        n_mfcc=args.n_mfcc,
        target_frames=args.target_frames
    )
    suffix_extractor = SuffixExtractor()
    
    # 创建数据集
    train_dataset = KWSDataset(
        train_paths, train_labels,
        feature_config, suffix_extractor,
        augment=True
    )
    val_dataset = KWSDataset(
        val_paths, val_labels,
        feature_config, suffix_extractor,
        augment=False
    )
    
    # 创建数据加载器（使用加权采样器处理类别不平衡）
    train_sampler = create_weighted_sampler(train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型
    cnn_config = CNNConfig(
        n_mfcc=args.n_mfcc,
        target_frames=args.target_frames,
        hidden_channels=[64, 128, 64],
        kernel_size=3,
        dropout=0.3
    )
    model = create_cnn_verifier(args.model_type, cnn_config)
    model = model.to(device)
    
    # 打印模型信息
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型: {args.model_type}")
    print(f"参数量: {n_params:,}")
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 训练
    print(f"\n开始训练 (epochs={args.epochs})...")
    best_val_f1 = 0.0
    best_model_state = None
    history = []
    
    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_metrics["loss"])
        
        # 记录历史
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_far": val_metrics["far"],
            "val_frr": val_metrics["frr"],
        })
        
        # 保存最佳模型
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = model.state_dict().copy()
            
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}, "
                  f"val_f1={val_metrics['f1']:.4f} *, "
                  f"FAR={val_metrics['far']:.4f}, FRR={val_metrics['frr']:.4f}")
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}, "
                  f"val_f1={val_metrics['f1']:.4f}, "
                  f"FAR={val_metrics['far']:.4f}, FRR={val_metrics['frr']:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估（最佳模型）")
    print("=" * 60)
    
    final_metrics = evaluate(model, val_loader, criterion, device)
    
    # 寻找最佳阈值
    best_th, th_metrics = find_best_threshold(
        final_metrics["probs"],
        final_metrics["labels"],
        target_far=0.1
    )
    
    print(f"\n默认阈值 (0.5):")
    print(f"  准确率: {final_metrics['accuracy']:.4f}")
    print(f"  F1: {final_metrics['f1']:.4f}")
    print(f"  FAR: {final_metrics['far']:.4f}")
    print(f"  FRR: {final_metrics['frr']:.4f}")
    
    print(f"\n最佳阈值 ({best_th:.2f}, 目标 FAR<10%):")
    print(f"  FAR: {th_metrics['far']:.4f}")
    print(f"  FRR: {th_metrics['frr']:.4f}")
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"cnn_verifier_{timestamp}.pt"
    
    save_dict = {
        "model_state_dict": best_model_state,
        "config": {
            "model_type": args.model_type,
            "n_mfcc": args.n_mfcc,
            "target_frames": args.target_frames,
            "hidden_channels": [64, 128, 64],
            "kernel_size": 3,
            "dropout": 0.3,
        },
        "best_threshold": best_th,
        "metrics": {
            "accuracy": final_metrics["accuracy"],
            "f1": final_metrics["f1"],
            "far": final_metrics["far"],
            "frr": final_metrics["frr"],
        },
        "training_args": vars(args),
    }
    
    torch.save(save_dict, model_path)
    print(f"\n模型已保存: {model_path}")
    
    # 同时保存一份固定名称的
    latest_path = output_dir / "cnn_verifier_latest.pt"
    torch.save(save_dict, latest_path)
    print(f"最新模型: {latest_path}")
    
    # 保存训练历史
    history_path = output_dir / f"training_history_{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"训练历史: {history_path}")


if __name__ == "__main__":
    main()
