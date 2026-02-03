#!/usr/bin/env python3
"""
改进版 CNN 训练脚本
- 使用 Focal Loss 处理类别不平衡
- 更强的正则化
- 交叉验证
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
from typing import List, Tuple
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

from models.cnn_verifier import CNNVerifier, CNNConfig
from features.feature_extractor import FeatureExtractor, FeatureConfig, SuffixExtractor


PROJECT_ROOT = Path(__file__).parent.parent.parent


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class KWSDataset(Dataset):
    """唤醒词数据集"""
    
    def __init__(
        self,
        audio_paths: List[str],
        labels: List[int],
        feature_config: FeatureConfig,
        suffix_extractor: SuffixExtractor,
        augment: bool = False,
        mixup_alpha: float = 0.0
    ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = FeatureExtractor(feature_config)
        self.suffix_extractor = suffix_extractor
        self.augment = augment
        self.mixup_alpha = mixup_alpha
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import librosa
        
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        audio, _ = librosa.load(audio_path, sr=16000)
        suffix = self.suffix_extractor.extract(audio)
        
        if self.augment:
            suffix = self._augment(suffix)
        
        features = self.feature_extractor.extract_for_cnn(suffix)
        
        return (
            torch.from_numpy(features).float(),
            torch.tensor([label], dtype=torch.float32)
        )
    
    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """数据增强"""
        # 时间偏移
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-0.15, 0.15) * len(audio))
            audio = np.roll(audio, shift)
        
        # 增益
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.7, 1.3)
            audio = audio * gain
        
        # 噪声
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.002, 0.02)
            audio = audio + np.random.randn(len(audio)) * noise_level
        
        return audio


def load_dataset(data_dir: Path, positive_keywords: List[str]) -> Tuple[List[str], List[int]]:
    audio_paths, labels = [], []
    for wav_file in sorted(data_dir.glob("*.wav")):
        is_positive = any(kw in wav_file.name for kw in positive_keywords)
        audio_paths.append(str(wav_file))
        labels.append(1 if is_positive else 0)
    return audio_paths, labels


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int = 20
) -> Tuple[dict, dict]:
    """训练单个 fold"""
    best_val_auc = 0.0
    best_model_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                outputs = model(features).squeeze().cpu().numpy()
                if outputs.ndim == 0:
                    outputs = np.array([outputs.item()])
                all_probs.extend(outputs.tolist())
                
                labels_np = labels.squeeze().numpy()
                if labels_np.ndim == 0:
                    labels_np = np.array([labels_np.item()])
                all_labels.extend(labels_np.tolist())
        
        # 计算 AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.5
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            break
    
    # 使用最佳模型评估
    model.load_state_dict(best_model_state)
    model.eval()
    
    all_probs, all_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features).squeeze().cpu().numpy()
            if outputs.ndim == 0:
                outputs = np.array([outputs.item()])
            all_probs.extend(outputs.tolist())
            
            labels_np = labels.squeeze().numpy()
            if labels_np.ndim == 0:
                labels_np = np.array([labels_np.item()])
            all_labels.extend(labels_np.tolist())
    
    return best_model_state, {"probs": np.array(all_probs), "labels": np.array(all_labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "all"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    positive_keywords = ["你好真真", "你好珍珍"]
    audio_paths, labels = load_dataset(data_dir, positive_keywords)
    
    print(f"总样本: {len(labels)} (正:{sum(labels)}, 负:{len(labels)-sum(labels)})")
    
    # 配置
    feature_config = FeatureConfig(n_mfcc=40, target_frames=50)
    suffix_extractor = SuffixExtractor()
    cnn_config = CNNConfig(
        n_mfcc=40, target_frames=50,
        hidden_channels=[64, 128, 64],
        dropout=0.4  # 更强的 dropout
    )
    
    # K-Fold 交叉验证
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    all_probs = np.zeros(len(labels))
    all_labels = np.array(labels)
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(audio_paths, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{args.n_folds}")
        print(f"{'='*60}")
        
        train_paths = [audio_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [audio_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print(f"训练: {len(train_paths)} (正:{sum(train_labels)})")
        print(f"验证: {len(val_paths)} (正:{sum(val_labels)})")
        
        # 数据集
        train_dataset = KWSDataset(train_paths, train_labels, feature_config, suffix_extractor, augment=True)
        val_dataset = KWSDataset(val_paths, val_labels, feature_config, suffix_extractor, augment=False)
        
        # 加权采样
        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts
        sample_weights = [weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # 模型
        model = CNNVerifier(cnn_config).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        
        # 训练
        best_state, results = train_fold(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)
        
        # 记录预测
        for i, idx in enumerate(val_idx):
            all_probs[idx] = results["probs"][i]
        
        best_models.append(best_state)
        
        # 计算 fold 指标
        from sklearn.metrics import roc_auc_score
        try:
            fold_auc = roc_auc_score(results["labels"], results["probs"])
        except:
            fold_auc = 0.5
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")
    
    # 整体评估
    print("\n" + "=" * 60)
    print("整体交叉验证结果")
    print("=" * 60)
    
    from sklearn.metrics import roc_auc_score
    overall_auc = roc_auc_score(all_labels, all_probs)
    print(f"Overall AUC: {overall_auc:.4f}")
    
    # 搜索最优阈值
    best_th, best_far, best_frr = 0.5, 1.0, 1.0
    for th in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs >= th).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        if far <= 0.10 and frr < best_frr:
            best_th, best_far, best_frr = th, far, frr
    
    print(f"\n最佳阈值 (目标 FAR<10%): {best_th:.2f}")
    print(f"  FAR: {best_far*100:.2f}%")
    print(f"  FRR: {best_frr*100:.2f}%")
    
    # 保存 ensemble 模型（取第一个 fold 的最佳模型）
    model = CNNVerifier(cnn_config)
    model.load_state_dict(best_models[0])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dict = {
        "model_state_dict": best_models[0],
        "config": {
            "n_mfcc": 40,
            "target_frames": 50,
            "hidden_channels": [64, 128, 64],
            "dropout": 0.4,
        },
        "best_threshold": best_th,
        "cv_auc": overall_auc,
        "metrics": {"far": best_far, "frr": best_frr},
    }
    
    model_path = output_dir / f"cnn_verifier_cv_{timestamp}.pt"
    torch.save(save_dict, model_path)
    print(f"\n模型已保存: {model_path}")
    
    # 更新 latest
    latest_path = output_dir / "cnn_verifier_latest.pt"
    torch.save(save_dict, latest_path)


if __name__ == "__main__":
    main()
