#!/usr/bin/env python3
"""
特征提取模块
支持 MFCC 和 Mel 频谱图特征提取
"""

import numpy as np
import librosa
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """特征提取配置"""
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_mels: int = 80
    n_fft: int = 512
    hop_length: int = 160  # 10ms @ 16kHz
    win_length: int = 400  # 25ms @ 16kHz
    target_frames: int = 50  # 约 500ms
    normalize: bool = True
    
    @property
    def feature_dim(self) -> int:
        """特征维度（用于 CNN）"""
        return self.n_mfcc


class FeatureExtractor:
    """
    特征提取器
    支持 MFCC 和 Mel 频谱图
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig()
        self.config = config
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        提取 MFCC 特征
        
        Args:
            audio: 音频数据 (n_samples,)
        
        Returns:
            mfcc: (n_mfcc, n_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length
        )
        
        return mfcc
    
    def extract_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        提取 Mel 频谱图
        
        Args:
            audio: 音频数据 (n_samples,)
        
        Returns:
            mel: (n_mels, n_frames)
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels
        )
        
        # 转换为 dB 刻度
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        return mel_db
    
    def extract_mfcc_fixed(self, audio: np.ndarray) -> np.ndarray:
        """
        提取固定长度的 MFCC 特征（用于 CNN 验证器）
        
        Args:
            audio: 音频数据 (n_samples,)
        
        Returns:
            mfcc: (n_mfcc, target_frames)
        """
        mfcc = self.extract_mfcc(audio)
        mfcc = self._pad_or_truncate(mfcc, self.config.target_frames)
        
        if self.config.normalize:
            mfcc = self._normalize(mfcc)
        
        return mfcc
    
    def extract_for_cnn(self, audio: np.ndarray) -> np.ndarray:
        """
        提取 CNN 验证器所需的特征
        
        Args:
            audio: 音频数据 (n_samples,)
        
        Returns:
            features: (n_mfcc, target_frames)
        """
        return self.extract_mfcc_fixed(audio)
    
    def _pad_or_truncate(self, features: np.ndarray, target_frames: int) -> np.ndarray:
        """填充或截断到目标帧数"""
        n_frames = features.shape[1]
        
        if n_frames < target_frames:
            # 填充
            pad_width = target_frames - n_frames
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        elif n_frames > target_frames:
            # 截断
            features = features[:, :target_frames]
        
        return features
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """特征归一化"""
        mean = np.mean(features)
        std = np.std(features)
        if std > 1e-8:
            features = (features - mean) / std
        return features
    
    def get_audio_frames_needed(self) -> int:
        """计算生成 target_frames 帧特征所需的音频样本数"""
        # n_frames = 1 + (n_samples - win_length) / hop_length
        # n_samples = (n_frames - 1) * hop_length + win_length
        n_samples = (self.config.target_frames - 1) * self.config.hop_length + self.config.win_length
        return n_samples
    
    def get_duration_needed(self) -> float:
        """计算所需音频时长（秒）"""
        return self.get_audio_frames_needed() / self.config.sample_rate


class SuffixExtractor:
    """
    后缀音频提取器
    从完整音频中提取后缀部分（如"真真"）
    """
    
    def __init__(
        self,
        start_ratio: float = 0.4,
        min_duration_ms: int = 200,
        max_duration_ms: int = 800,
        sample_rate: int = 16000
    ):
        self.start_ratio = start_ratio
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.sample_rate = sample_rate
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        提取后缀音频
        
        Args:
            audio: 完整音频 (n_samples,)
        
        Returns:
            suffix: 后缀音频
        """
        total_samples = len(audio)
        
        # 计算起始位置
        start_sample = int(total_samples * self.start_ratio)
        
        # 计算长度限制
        min_samples = int(self.min_duration_ms * self.sample_rate / 1000)
        max_samples = int(self.max_duration_ms * self.sample_rate / 1000)
        
        # 提取后缀
        suffix = audio[start_sample:]
        
        # 长度约束
        if len(suffix) < min_samples:
            # 后缀太短，从更早位置开始
            new_start = max(0, total_samples - min_samples)
            suffix = audio[new_start:]
        elif len(suffix) > max_samples:
            # 后缀太长，截断
            suffix = suffix[:max_samples]
        
        return suffix


class StreamingFeatureExtractor:
    """
    流式特征提取器
    维护帧缓冲区，支持流式特征计算
    """
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig()
        self.config = config
        self._feature_extractor = FeatureExtractor(config)
        
        # 特征缓冲区
        self._feature_buffer = None
        self._buffer_frames = 0
    
    def process_audio_buffer(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        处理音频缓冲区，返回特征
        
        Args:
            audio: 音频缓冲区数据
        
        Returns:
            features: (n_mfcc, target_frames) 或 None（数据不足）
        """
        needed_samples = self._feature_extractor.get_audio_frames_needed()
        
        if len(audio) < needed_samples:
            return None
        
        # 取最后所需长度的音频
        audio_segment = audio[-needed_samples:]
        
        return self._feature_extractor.extract_for_cnn(audio_segment)
    
    def reset(self):
        """重置状态"""
        self._feature_buffer = None
        self._buffer_frames = 0


if __name__ == "__main__":
    # 测试特征提取
    config = FeatureConfig(n_mfcc=40, target_frames=50)
    extractor = FeatureExtractor(config)
    
    # 生成测试音频
    duration = 0.5  # 500ms
    audio = np.random.randn(int(16000 * duration)).astype(np.float32)
    
    # 提取 MFCC
    mfcc = extractor.extract_mfcc_fixed(audio)
    print(f"Audio shape: {audio.shape}")
    print(f"MFCC shape: {mfcc.shape}")
    print(f"Needed samples: {extractor.get_audio_frames_needed()}")
    print(f"Needed duration: {extractor.get_duration_needed():.3f}s")
    
    # 测试后缀提取
    suffix_extractor = SuffixExtractor()
    suffix = suffix_extractor.extract(audio)
    print(f"Suffix shape: {suffix.shape}")
    print(f"Suffix duration: {len(suffix)/16000:.3f}s")
