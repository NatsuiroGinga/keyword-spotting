#!/usr/bin/env python3
"""
联合评分融合模块
将 V3 检测分数与 CNN 验证分数融合为单一置信度
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FusionMode(Enum):
    """融合模式"""
    WEIGHTED = "weighted"      # 加权平均
    CASCADE = "cascade"        # 级联（先 V3 后 CNN）
    LEARNED = "learned"        # 可学习融合
    PRODUCT = "product"        # 分数相乘


@dataclass
class FusionConfig:
    """融合配置"""
    mode: str = "weighted"
    
    # 加权融合参数
    v3_weight: float = 0.3
    cnn_weight: float = 0.7
    
    # 级联融合参数
    v3_threshold: float = 0.25  # V3 通过阈值
    
    # 最终判定阈值
    final_threshold: float = 0.5
    
    # 可学习融合的隐藏层大小
    hidden_size: int = 8


class WeightedFusion(nn.Module):
    """加权平均融合"""
    
    def __init__(self, v3_weight: float = 0.3, cnn_weight: float = 0.7):
        super().__init__()
        # 使用 softmax 归一化的权重
        self.weights = nn.Parameter(torch.tensor([v3_weight, cnn_weight]))
    
    def forward(self, score_v3: torch.Tensor, score_cnn: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.weights, dim=0)
        return weights[0] * score_v3 + weights[1] * score_cnn


class LearnedFusion(nn.Module):
    """可学习的融合网络"""
    
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.fusion_net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, score_v3: torch.Tensor, score_cnn: torch.Tensor) -> torch.Tensor:
        # 确保输入是正确的形状
        if score_v3.dim() == 0:
            score_v3 = score_v3.unsqueeze(0)
        if score_cnn.dim() == 0:
            score_cnn = score_cnn.unsqueeze(0)
        
        x = torch.stack([score_v3, score_cnn], dim=-1)
        return self.fusion_net(x).squeeze(-1)


import torch.nn.functional as F


class ScoreFusion:
    """
    评分融合器 - 统一接口
    支持多种融合策略，将 V3 分数和 CNN 分数融合为最终置信度
    """
    
    def __init__(self, config: FusionConfig = None):
        if config is None:
            config = FusionConfig()
        self.config = config
        self.mode = FusionMode(config.mode)
        
        # 初始化融合模块（如果是可学习的）
        self._fusion_module = None
        if self.mode == FusionMode.LEARNED:
            self._fusion_module = LearnedFusion(config.hidden_size)
    
    def fuse(self, score_v3: float, score_cnn: float) -> float:
        """
        融合两个分数
        
        Args:
            score_v3: V3 检测分数 [0, 1]（实际上 sherpa-onnx 只返回 0 或 1）
            score_cnn: CNN 验证分数 [0, 1]
        
        Returns:
            融合后的最终置信度 [0, 1]
        """
        if self.mode == FusionMode.WEIGHTED:
            return self._weighted_fusion(score_v3, score_cnn)
        elif self.mode == FusionMode.CASCADE:
            return self._cascade_fusion(score_v3, score_cnn)
        elif self.mode == FusionMode.PRODUCT:
            return self._product_fusion(score_v3, score_cnn)
        elif self.mode == FusionMode.LEARNED:
            return self._learned_fusion(score_v3, score_cnn)
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")
    
    def _weighted_fusion(self, score_v3: float, score_cnn: float) -> float:
        """加权平均"""
        return self.config.v3_weight * score_v3 + self.config.cnn_weight * score_cnn
    
    def _cascade_fusion(self, score_v3: float, score_cnn: float) -> float:
        """
        级联融合：V3 作为门控，CNN 决定最终分数
        如果 V3 通过阈值，返回 CNN 分数；否则返回 0
        """
        if score_v3 >= self.config.v3_threshold:
            return score_cnn
        return 0.0
    
    def _product_fusion(self, score_v3: float, score_cnn: float) -> float:
        """分数相乘（更严格）"""
        return score_v3 * score_cnn
    
    def _learned_fusion(self, score_v3: float, score_cnn: float) -> float:
        """使用学习到的融合网络"""
        if self._fusion_module is None:
            raise RuntimeError("Learned fusion module not initialized")
        
        with torch.no_grad():
            v3_t = torch.tensor(score_v3, dtype=torch.float32)
            cnn_t = torch.tensor(score_cnn, dtype=torch.float32)
            return self._fusion_module(v3_t, cnn_t).item()
    
    def decide(self, score_v3: float, score_cnn: float) -> Tuple[bool, float]:
        """
        做出最终判定
        
        Returns:
            (is_accepted, final_score): 是否接受，最终分数
        """
        final_score = self.fuse(score_v3, score_cnn)
        is_accepted = final_score >= self.config.final_threshold
        return is_accepted, final_score
    
    def get_fusion_module(self) -> Optional[nn.Module]:
        """获取可学习的融合模块（用于训练）"""
        return self._fusion_module
    
    def load_fusion_weights(self, state_dict: dict):
        """加载融合模块权重"""
        if self._fusion_module is not None:
            self._fusion_module.load_state_dict(state_dict)


class AdaptiveScoreFusion:
    """
    自适应评分融合
    根据 V3 分数动态调整 CNN 权重
    """
    
    def __init__(self, base_v3_weight: float = 0.3, 
                 v3_high_threshold: float = 0.8,
                 v3_low_threshold: float = 0.3):
        self.base_v3_weight = base_v3_weight
        self.v3_high_threshold = v3_high_threshold
        self.v3_low_threshold = v3_low_threshold
    
    def fuse(self, score_v3: float, score_cnn: float) -> float:
        """
        自适应融合：
        - V3 分数高时，更信任 V3
        - V3 分数低时，更依赖 CNN 验证
        """
        if score_v3 >= self.v3_high_threshold:
            # V3 高置信度，增加 V3 权重
            v3_weight = min(0.6, self.base_v3_weight + 0.2)
        elif score_v3 <= self.v3_low_threshold:
            # V3 低置信度，降低 V3 权重
            v3_weight = max(0.1, self.base_v3_weight - 0.2)
        else:
            v3_weight = self.base_v3_weight
        
        cnn_weight = 1.0 - v3_weight
        return v3_weight * score_v3 + cnn_weight * score_cnn


def find_optimal_threshold(
    fused_scores: np.ndarray,
    labels: np.ndarray,
    target_far: float = 0.1
) -> Tuple[float, dict]:
    """
    寻找最优阈值
    
    Args:
        fused_scores: 融合后的分数数组
        labels: 真实标签数组 (1=正样本, 0=负样本)
        target_far: 目标误报率
    
    Returns:
        (optimal_threshold, metrics): 最优阈值和对应指标
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_metrics = None
    
    for th in thresholds:
        predictions = (fused_scores >= th).astype(int)
        
        # 计算指标
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
        
        # 目标：FAR <= target_far 时，FRR 最小
        if far <= target_far:
            if best_metrics is None or frr < best_metrics["frr"]:
                best_threshold = th
                best_metrics = {
                    "threshold": th,
                    "far": far,
                    "frr": frr,
                    "accuracy": accuracy,
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn
                }
    
    # 如果没有找到满足 FAR 目标的阈值，返回 FAR 最接近目标的
    if best_metrics is None:
        min_diff = float("inf")
        for th in thresholds:
            predictions = (fused_scores >= th).astype(int)
            fp = np.sum((predictions == 1) & (labels == 0))
            tn = np.sum((predictions == 0) & (labels == 0))
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            diff = abs(far - target_far)
            if diff < min_diff:
                min_diff = diff
                
                tp = np.sum((predictions == 1) & (labels == 1))
                fn = np.sum((predictions == 0) & (labels == 1))
                frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                accuracy = (tp + tn) / len(labels)
                
                best_threshold = th
                best_metrics = {
                    "threshold": th,
                    "far": far,
                    "frr": frr,
                    "accuracy": accuracy,
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn
                }
    
    return best_threshold, best_metrics


if __name__ == "__main__":
    # 测试融合模块
    config = FusionConfig(mode="weighted", v3_weight=0.3, cnn_weight=0.7)
    fusion = ScoreFusion(config)
    
    # 测试用例
    test_cases = [
        (1.0, 0.8),  # V3 检测到，CNN 高置信度
        (1.0, 0.3),  # V3 检测到，CNN 低置信度
        (0.0, 0.9),  # V3 未检测到，CNN 高置信度
    ]
    
    print("Weighted Fusion Test:")
    for v3, cnn in test_cases:
        accepted, score = fusion.decide(v3, cnn)
        print(f"  V3={v3:.1f}, CNN={cnn:.1f} -> Score={score:.3f}, Accepted={accepted}")
    
    # 测试级联融合
    config_cascade = FusionConfig(mode="cascade", v3_threshold=0.5)
    fusion_cascade = ScoreFusion(config_cascade)
    
    print("\nCascade Fusion Test:")
    for v3, cnn in test_cases:
        accepted, score = fusion_cascade.decide(v3, cnn)
        print(f"  V3={v3:.1f}, CNN={cnn:.1f} -> Score={score:.3f}, Accepted={accepted}")
