"""
音频处理工具
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import soundfile as sf


def load_audio(
    audio_path: str,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件
    
    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率
        
    Returns:
        (samples, sample_rate)
    """
    samples, sr = sf.read(audio_path, dtype="float32")
    
    # 转换为单声道
    if len(samples.shape) > 1:
        samples = samples[:, 0]
    
    # 重采样
    if sr != target_sr:
        try:
            import librosa
            samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            raise ImportError("需要安装 librosa 进行重采样: pip install librosa")
    
    return samples, sr


def get_audio_duration(audio_path: str) -> float:
    """获取音频时长（秒）"""
    info = sf.info(audio_path)
    return info.duration


def extract_suffix(
    samples: np.ndarray,
    sample_rate: int,
    start_ratio: float = 0.4,
    min_duration_ms: int = 200,
    max_duration_ms: int = 800
) -> np.ndarray:
    """
    提取音频后缀片段（用于"真真"验证）
    
    Args:
        samples: 音频样本
        sample_rate: 采样率
        start_ratio: 从音频的哪个位置开始提取（0.4表示40%位置）
        min_duration_ms: 最小时长（毫秒）
        max_duration_ms: 最大时长（毫秒）
        
    Returns:
        后缀音频片段
    """
    total_samples = len(samples)
    
    # 计算起始位置
    start_sample = int(total_samples * start_ratio)
    
    # 计算最小和最大样本数
    min_samples = int(min_duration_ms * sample_rate / 1000)
    max_samples = int(max_duration_ms * sample_rate / 1000)
    
    # 提取后缀
    suffix = samples[start_sample:]
    
    # 确保长度在范围内
    if len(suffix) < min_samples:
        # 如果太短，向前扩展
        new_start = max(0, total_samples - min_samples)
        suffix = samples[new_start:]
    elif len(suffix) > max_samples:
        # 如果太长，截断
        suffix = suffix[:max_samples]
    
    return suffix


def extract_suffix_with_vad(
    samples: np.ndarray,
    sample_rate: int,
    energy_threshold: float = 0.01
) -> Optional[np.ndarray]:
    """
    使用能量检测提取后缀（更精确的分割）
    
    Args:
        samples: 音频样本
        sample_rate: 采样率
        energy_threshold: 能量阈值
        
    Returns:
        后缀音频片段，如果无法检测则返回None
    """
    # 计算短时能量
    frame_size = int(0.025 * sample_rate)  # 25ms
    hop_size = int(0.010 * sample_rate)    # 10ms
    
    energies = []
    for i in range(0, len(samples) - frame_size, hop_size):
        frame = samples[i:i + frame_size]
        energy = np.mean(frame ** 2)
        energies.append(energy)
    
    energies = np.array(energies)
    
    # 归一化能量
    if energies.max() > 0:
        energies = energies / energies.max()
    
    # 找到能量峰值（可能对应音节边界）
    # 假设"你好"和"真真"之间有短暂停顿
    mid_point = len(energies) // 2
    
    # 在中间区域寻找能量谷值
    search_start = int(len(energies) * 0.3)
    search_end = int(len(energies) * 0.6)
    
    if search_end <= search_start:
        return None
    
    search_region = energies[search_start:search_end]
    min_idx = np.argmin(search_region) + search_start
    
    # 转换回样本索引
    split_sample = min_idx * hop_size
    
    # 提取后缀
    suffix = samples[split_sample:]
    
    # 确保后缀有足够长度
    min_samples = int(0.2 * sample_rate)  # 至少200ms
    if len(suffix) < min_samples:
        return None
    
    return suffix


def pad_or_trim(
    samples: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    填充或截断音频到指定长度
    
    Args:
        samples: 音频样本
        target_length: 目标长度
        
    Returns:
        调整后的音频
    """
    if len(samples) < target_length:
        # 填充
        padding = np.zeros(target_length - len(samples), dtype=samples.dtype)
        samples = np.concatenate([samples, padding])
    elif len(samples) > target_length:
        # 截断
        samples = samples[:target_length]
    
    return samples
