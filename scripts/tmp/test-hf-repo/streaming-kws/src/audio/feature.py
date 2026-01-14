"""
音频特征提取模块

提供MFCC特征提取功能，用于MLP验证器的输入。
"""
import numpy as np
from typing import Optional


class FeatureExtractor:
    """
    MFCC特征提取器
    
    用于从音频中提取MFCC特征，作为MLP验证器的输入。
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160,
        target_frames: int = 50
    ):
        """
        初始化特征提取器
        
        Args:
            sample_rate: 采样率
            n_mfcc: MFCC系数数量
            n_fft: FFT窗口大小
            hop_length: 帧移（采样点数）
            target_frames: 目标帧数
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_frames = target_frames
        
        # 计算输入维度
        self.input_dim = n_mfcc * target_frames
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            audio: 音频数据 (float32, 范围[-1, 1])
            
        Returns:
            MFCC特征 (n_mfcc, n_frames)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError("需要安装 librosa: pip install librosa")
        
        # 确保音频是float32类型
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def pad_or_trim(self, features: np.ndarray) -> np.ndarray:
        """
        填充或截断特征到目标帧数
        
        Args:
            features: MFCC特征 (n_mfcc, n_frames)
            
        Returns:
            处理后的特征 (n_mfcc, target_frames)
        """
        n_mfcc, n_frames = features.shape
        
        if n_frames < self.target_frames:
            # 填充零
            padding = np.zeros((n_mfcc, self.target_frames - n_frames), dtype=np.float32)
            features = np.concatenate([features, padding], axis=1)
        elif n_frames > self.target_frames:
            # 截断
            features = features[:, :self.target_frames]
        
        return features
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        归一化特征
        
        Args:
            features: MFCC特征
            
        Returns:
            归一化后的特征
        """
        mean = features.mean()
        std = features.std()
        return (features - mean) / (std + 1e-8)
    
    def extract_for_mlp(self, audio: np.ndarray) -> np.ndarray:
        """
        提取用于MLP验证器的特征向量
        
        Args:
            audio: 音频数据
            
        Returns:
            展平的特征向量 (input_dim,)
        """
        # 提取MFCC
        mfcc = self.extract_mfcc(audio)
        
        # 填充或截断
        mfcc = self.pad_or_trim(mfcc)
        
        # 归一化
        mfcc = self.normalize(mfcc)
        
        # 展平
        return mfcc.flatten().astype(np.float32)
    
    def get_audio_duration_for_frames(self, n_frames: int) -> float:
        """
        计算指定帧数对应的音频时长
        
        Args:
            n_frames: 帧数
            
        Returns:
            时长（秒）
        """
        # 帧数 = (音频长度 - n_fft) / hop_length + 1
        # 音频长度 = (帧数 - 1) * hop_length + n_fft
        n_samples = (n_frames - 1) * self.hop_length + self.n_fft
        return n_samples / self.sample_rate
    
    def get_target_audio_duration(self) -> float:
        """
        获取目标帧数对应的音频时长
        
        Returns:
            时长（秒）
        """
        return self.get_audio_duration_for_frames(self.target_frames)
