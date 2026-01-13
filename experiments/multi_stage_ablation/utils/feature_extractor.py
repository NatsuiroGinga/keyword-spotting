"""
特征提取工具
"""
import numpy as np
from typing import Optional


def extract_mfcc(
    samples: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80
) -> np.ndarray:
    """
    提取MFCC特征
    
    Args:
        samples: 音频样本
        sample_rate: 采样率
        n_mfcc: MFCC系数数量
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_mels: Mel滤波器数量
        
    Returns:
        MFCC特征 (n_mfcc, n_frames)
    """
    try:
        import librosa
        mfcc = librosa.feature.mfcc(
            y=samples,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        return mfcc
    except ImportError:
        raise ImportError("需要安装 librosa: pip install librosa")


def extract_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80
) -> np.ndarray:
    """
    提取Mel频谱图
    
    Args:
        samples: 音频样本
        sample_rate: 采样率
        n_fft: FFT窗口大小
        hop_length: 帧移
        n_mels: Mel滤波器数量
        
    Returns:
        Mel频谱图 (n_mels, n_frames)
    """
    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=samples,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        # 转换为对数刻度
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    except ImportError:
        raise ImportError("需要安装 librosa: pip install librosa")


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    特征归一化
    
    Args:
        features: 特征矩阵
        mean: 均值（如果为None则计算）
        std: 标准差（如果为None则计算）
        
    Returns:
        归一化后的特征
    """
    if mean is None:
        mean = np.mean(features, axis=-1, keepdims=True)
    if std is None:
        std = np.std(features, axis=-1, keepdims=True)
        std = np.where(std == 0, 1, std)  # 避免除零
    
    return (features - mean) / std


def pad_features(
    features: np.ndarray,
    target_frames: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    填充特征到指定帧数
    
    Args:
        features: 特征矩阵 (n_features, n_frames)
        target_frames: 目标帧数
        pad_value: 填充值
        
    Returns:
        填充后的特征
    """
    n_features, n_frames = features.shape
    
    if n_frames < target_frames:
        padding = np.full(
            (n_features, target_frames - n_frames),
            pad_value,
            dtype=features.dtype
        )
        features = np.concatenate([features, padding], axis=1)
    elif n_frames > target_frames:
        features = features[:, :target_frames]
    
    return features
