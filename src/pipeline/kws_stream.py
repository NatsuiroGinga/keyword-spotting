"""
流式关键词识别管道

基于sherpa-onnx实现两阶段流式KWS：
1. 第一阶段：Zipformer流式ASR进行关键词检测
2. 第二阶段：MLP验证器进行二次确认
"""
import time
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path

import numpy as np

from ..audio.capture import AudioBuffer
from ..audio.feature import FeatureExtractor
from ..models.mlp_verifier import MLPVerifierONNX
from ..utils.config import KWSConfig


@dataclass
class DetectionResult:
    """检测结果"""
    keyword: str
    confidence: float
    timestamp: float
    verified: bool = True
    mlp_confidence: Optional[float] = None


class StreamingKWSPipeline:
    """
    流式关键词识别管道
    
    整合sherpa-onnx KWS和MLP验证器，实现两阶段检测。
    """
    
    def __init__(self, config: KWSConfig):
        """
        初始化流式KWS管道
        
        Args:
            config: KWS配置
        """
        self.config = config
        
        # 组件
        self._kws = None  # sherpa-onnx KeywordSpotter
        self._stream = None  # sherpa-onnx Stream
        self._mlp_verifier: Optional[MLPVerifierONNX] = None
        self._feature_extractor: Optional[FeatureExtractor] = None
        self._audio_buffer: Optional[AudioBuffer] = None
        
        # 状态
        self._is_loaded = False
        self._detection_count = 0
        self._start_time = None
        self._chunk_count = 0  # 处理块计数器，用于定期重置
        self._total_samples_received = 0  # 记录实际接收的音频样本数（不含padding）
        
        # 回调
        self._on_detection: Optional[Callable[[DetectionResult], None]] = None
    
    def load(self) -> None:
        """加载所有模型"""
        self._load_sherpa_kws()
        
        if self.config.mlp_enabled:
            self._load_mlp_verifier()
        
        # 初始化音频缓冲区
        self._audio_buffer = AudioBuffer(
            max_duration=self.config.buffer_duration,
            sample_rate=self.config.sample_rate
        )
        
        self._is_loaded = True
        self._start_time = time.time()
        print("流式KWS管道已加载完成")
    
    def _load_sherpa_kws(self) -> None:
        """加载sherpa-onnx关键词识别器"""
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError("需要安装 sherpa-onnx: pip install sherpa-onnx")
        
        # 检查模型文件
        required_files = [
            self.config.encoder_path,
            self.config.decoder_path,
            self.config.joiner_path,
            self.config.tokens_path,
        ]
        
        for f in required_files:
            if not Path(f).exists():
                raise FileNotFoundError(f"模型文件不存在: {f}")
        
        # 创建关键词识别器配置
        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=self.config.tokens_path,
            encoder=self.config.encoder_path,
            decoder=self.config.decoder_path,
            joiner=self.config.joiner_path,
            keywords_file=self.config.keywords_file if Path(self.config.keywords_file).exists() else "",
            num_threads=self.config.num_threads,
            provider=self.config.provider,
            keywords_score=self.config.keywords_score,
            keywords_threshold=self.config.keywords_threshold,
        )
        
        # 创建流
        self._stream = self._kws.create_stream()
        
        print(f"Sherpa-ONNX KWS已加载")
        print(f"  - 关键词: {self.config.keywords}")
        print(f"  - 加分权重: {self.config.keywords_score}")
        print(f"  - 触发阈值: {self.config.keywords_threshold}")
    
    def _load_mlp_verifier(self) -> None:
        """加载MLP验证器"""
        if not Path(self.config.mlp_model_path).exists():
            print(f"警告: MLP模型不存在，禁用二阶段验证: {self.config.mlp_model_path}")
            self.config.mlp_enabled = False
            return
        
        self._mlp_verifier = MLPVerifierONNX(
            model_path=self.config.mlp_model_path,
            threshold=self.config.mlp_threshold
        )
        self._mlp_verifier.load()
        
        self._feature_extractor = FeatureExtractor(
            sample_rate=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            target_frames=self.config.target_frames
        )
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[DetectionResult]:
        """
        处理单个音频块
        
        Args:
            audio_chunk: 音频数据块 (float32, 范围[-1, 1])
            
        Returns:
            检测结果，如果没有检测到关键词返回None
        """
        if not self._is_loaded:
            self.load()
        
        # 添加到缓冲区
        self._audio_buffer.append(audio_chunk)
        
        # 送入sherpa-onnx流
        self._stream.accept_waveform(self.config.sample_rate, audio_chunk)
        
        # 增加块计数，定期重建流以防止内部状态累积
        self._chunk_count += 1
        if self._chunk_count >= self.config.stream_reset_interval:
            self._rebuild_stream()
            self._chunk_count = 0
        
        # 解码
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)
            
            result = self._kws.get_result(self._stream)
            
            if result:
                # 第一阶段检测到关键词
                timestamp = time.time() - self._start_time
                
                # 第二阶段MLP验证
                verified = True
                mlp_confidence = None
                
                if self.config.mlp_enabled and self._mlp_verifier is not None:
                    verified, mlp_confidence = self._verify_with_mlp()
                
                if verified:
                    self._detection_count += 1
                    detection = DetectionResult(
                        keyword=result,
                        confidence=1.0,  # sherpa-onnx不直接返回置信度
                        timestamp=timestamp,
                        verified=verified,
                        mlp_confidence=mlp_confidence
                    )
                    
                    # 重置流
                    self._kws.reset_stream(self._stream)
                    
                    # 触发回调
                    if self._on_detection is not None:
                        self._on_detection(detection)
                    
                    return detection
                else:
                    # MLP验证失败，重置流继续检测
                    self._kws.reset_stream(self._stream)
        
        return None
    
    def _verify_with_mlp(self) -> tuple:
        """
        使用MLP验证器进行二次验证
        
        后缀提取策略与训练时保持一致：
        - 从音频40%位置开始提取后缀
        - 后缀长度限制在200-800ms范围内
        
        Returns:
            (是否通过验证, MLP置信度)
        """
        # 获取完整的关键词音频（约1.5秒，覆盖"你好真真"完整发音）
        full_audio = self._audio_buffer.get_last(self.config.buffer_duration)
        
        if len(full_audio) < self.config.sample_rate * 0.3:  # 至少300ms
            return True, None  # 音频太短，跳过验证
        
        # 与训练时一致：从音频40%位置开始提取后缀
        total_samples = len(full_audio)
        start_ratio = 0.4
        min_duration_ms = 200
        max_duration_ms = 800
        
        start_sample = int(total_samples * start_ratio)
        min_samples = int(min_duration_ms * self.config.sample_rate / 1000)
        max_samples = int(max_duration_ms * self.config.sample_rate / 1000)
        
        # 提取后缀
        suffix_audio = full_audio[start_sample:]
        
        # 确保长度在范围内
        if len(suffix_audio) < min_samples:
            # 如果太短，向前扩展
            new_start = max(0, total_samples - min_samples)
            suffix_audio = full_audio[new_start:]
        elif len(suffix_audio) > max_samples:
            # 如果太长，截断
            suffix_audio = suffix_audio[:max_samples]
        
        if len(suffix_audio) < self.config.sample_rate * 0.1:  # 至少100ms
            return True, None  # 音频太短，跳过验证
        
        # 提取特征
        features = self._feature_extractor.extract_for_mlp(suffix_audio)
        
        # MLP验证
        verified, confidence = self._mlp_verifier.verify(features)
        
        return verified, confidence
    
    def reset(self) -> None:
        """重置流式状态"""
        if self._stream is not None:
            self._kws.reset_stream(self._stream)
        
        if self._audio_buffer is not None:
            self._audio_buffer.clear()
        
        self._detection_count = 0
        self._chunk_count = 0
        self._start_time = time.time()
    
    def _rebuild_stream(self) -> None:
        """重建sherpa-onnx流，释放内部累积状态"""
        if self._kws is not None:
            self._stream = self._kws.create_stream()
    
    def set_on_detection(self, callback: Callable[[DetectionResult], None]) -> None:
        """
        设置检测回调函数
        
        Args:
            callback: 检测到关键词时调用的函数
        """
        self._on_detection = callback
    
    @property
    def is_loaded(self) -> bool:
        """检查是否已加载"""
        return self._is_loaded
    
    @property
    def detection_count(self) -> int:
        """获取检测次数"""
        return self._detection_count
