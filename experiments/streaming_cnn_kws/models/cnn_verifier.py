#!/usr/bin/env python3
"""
1D CNN 验证器模块 - 替代 MLP 验证器
使用一维卷积网络处理时序特征，提升唤醒词后缀识别能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CNNConfig:
    """CNN 验证器配置"""
    # 特征配置
    n_mfcc: int = 40  # 使用更多 MFCC 系数
    target_frames: int = 50  # 目标帧数
    
    # 网络结构
    hidden_channels: list = None  # [64, 128, 64]
    kernel_size: int = 3
    dropout: float = 0.3
    
    # 训练配置
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    def __post_init__(self):
        if self.hidden_channels is None:
            self.hidden_channels = [64, 128, 64]
    
    @property
    def input_dim(self) -> int:
        return self.n_mfcc


class Conv1DBlock(nn.Module):
    """1D 卷积块：Conv1d + BatchNorm + ReLU + Dropout"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class CNNVerifier(nn.Module):
    """
    1D CNN 唤醒词验证器
    
    输入: (batch, n_mfcc, n_frames) - 时序 MFCC 特征
    输出: (batch, 1) - 唤醒词置信度 [0, 1]
    
    网络结构:
    - 多层 1D 卷积提取时序局部特征
    - 全局平均池化聚合时序信息
    - 全连接分类头输出置信度
    """
    
    def __init__(self, config: CNNConfig = None):
        super().__init__()
        
        if config is None:
            config = CNNConfig()
        self.config = config
        
        # 构建卷积层
        channels = [config.input_dim] + config.hidden_channels
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers.append(
                Conv1DBlock(
                    channels[i], channels[i + 1],
                    kernel_size=config.kernel_size,
                    dropout=config.dropout
                )
            )
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mfcc, n_frames) 或 (batch, n_frames, n_mfcc)
        Returns:
            confidence: (batch, 1) 唤醒词置信度
        """
        # 确保输入形状为 (batch, features, time)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # 如果输入是 (batch, time, features)，转置
        if x.shape[1] > x.shape[2] and x.shape[2] == self.config.input_dim:
            x = x.transpose(1, 2)
        
        # 卷积特征提取
        x = self.conv_layers(x)  # (batch, hidden[-1], time)
        
        # 全局池化
        x = self.global_pool(x).squeeze(-1)  # (batch, hidden[-1])
        
        # 分类
        return self.classifier(x)  # (batch, 1)
    
    def predict(self, features: np.ndarray) -> float:
        """单样本预测"""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            confidence = self.forward(x)
            return confidence.item()


class CNNVerifierWithAttention(nn.Module):
    """
    带注意力机制的 CNN 验证器
    在卷积后添加自注意力层，更好地捕获全局时序依赖
    """
    
    def __init__(self, config: CNNConfig = None):
        super().__init__()
        
        if config is None:
            config = CNNConfig()
        self.config = config
        
        # 第一层卷积
        self.conv1 = Conv1DBlock(
            config.input_dim, config.hidden_channels[0],
            kernel_size=config.kernel_size, dropout=config.dropout
        )
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_channels[0],
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(config.hidden_channels[0])
        
        # 后续卷积层
        self.conv2 = Conv1DBlock(
            config.hidden_channels[0], config.hidden_channels[1],
            kernel_size=config.kernel_size, dropout=config.dropout
        )
        self.conv3 = Conv1DBlock(
            config.hidden_channels[1], config.hidden_channels[2],
            kernel_size=config.kernel_size, dropout=config.dropout
        )
        
        # 全局池化和分类头
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入形状
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[1] > x.shape[2] and x.shape[2] == self.config.input_dim:
            x = x.transpose(1, 2)
        
        # 第一层卷积
        x = self.conv1(x)  # (batch, hidden[0], time)
        
        # 自注意力（需要转置）
        x_attn = x.transpose(1, 2)  # (batch, time, hidden[0])
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        x_attn = self.attn_norm(x_attn + attn_out)
        x = x_attn.transpose(1, 2)  # (batch, hidden[0], time)
        
        # 后续卷积
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 池化和分类
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)
    
    def predict(self, features: np.ndarray) -> float:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            return self.forward(x).item()


def create_cnn_verifier(model_type: str = "cnn", config: CNNConfig = None) -> nn.Module:
    """工厂函数创建验证器"""
    if config is None:
        config = CNNConfig()
    
    if model_type == "cnn":
        return CNNVerifier(config)
    elif model_type == "cnn_attention":
        return CNNVerifierWithAttention(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试模型
    config = CNNConfig(n_mfcc=40, target_frames=50)
    model = CNNVerifier(config)
    
    # 测试输入
    x = torch.randn(4, 40, 50)  # (batch, features, time)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output: {y.squeeze()}")
    
    # 参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
