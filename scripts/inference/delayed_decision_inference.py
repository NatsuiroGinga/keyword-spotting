#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delayed Decision Inference Logic for KWS
实现延迟决策状态机，解决前缀触发问题

基于 implementation_plan.md 阶段4设计：
- 状态机: IDLE -> PREFIX_DETECTED -> WAITING_SUFFIX -> TRIGGERED/REJECTED
- 两阶段检测: "你好" -> 600ms内验证"真真"
- 音频缓冲和回放机制

Usage:
python scripts/delayed_decision_inference.py --model-path models/model.onnx --test-audio test.wav
"""

import argparse
import logging
import time
import threading
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import numpy as np
import soundfile as sf

# 假设我们有sherpa-onnx可用
try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    logging.warning("sherpa_onnx not available, using mock implementation")


class KWSState(Enum):
    """KWS状态机状态"""
    IDLE = "idle"
    PREFIX_DETECTED = "prefix_detected"
    WAITING_SUFFIX = "waiting_suffix"
    TRIGGERED = "triggered"
    REJECTED = "rejected"


class AudioBuffer:
    """音频缓冲区，用于延迟决策"""
    
    def __init__(self, max_duration_ms: int = 1000, sample_rate: int = 16000):
        self.max_duration_ms = max_duration_ms
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_ms * sample_rate / 1000)
        self.buffer = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()
    
    def add_samples(self, samples: np.ndarray):
        """添加音频样本到缓冲区"""
        with self.lock:
            self.buffer.extend(samples.flatten())
    
    def get_recent_audio(self, duration_ms: int) -> np.ndarray:
        """获取最近duration_ms毫秒的音频"""
        num_samples = int(duration_ms * self.sample_rate / 1000)
        with self.lock:
            if len(self.buffer) < num_samples:
                # 不够的话用零填充
                audio = np.zeros(num_samples, dtype=np.float32)
                available = len(self.buffer)
                if available > 0:
                    audio[-available:] = np.array(list(self.buffer))
                return audio
            else:
                return np.array(list(self.buffer)[-num_samples:])
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()


class DelayedDecisionKWS:
    """延迟决策关键词识别器"""
    
    def __init__(
        self,
        model_path: str,
        tokens_path: str = None,
        sample_rate: int = 16000,
        prefix_timeout_ms: int = 600,
        confidence_threshold: float = 0.5,
        on_triggered: Optional[Callable] = None,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.prefix_timeout_ms = prefix_timeout_ms
        self.confidence_threshold = confidence_threshold
        self.on_triggered = on_triggered or self._default_trigger_callback
        
        # 状态机
        self.state = KWSState.IDLE
        self.state_lock = threading.Lock()
        
        # 计时器
        self.prefix_timer = None
        
        # 音频缓冲
        self.audio_buffer = AudioBuffer(max_duration_ms=1000, sample_rate=sample_rate)
        
        # 初始化模型
        self._init_model()
        
        # 统计信息
        self.stats = {
            "total_detections": 0,
            "prefix_detections": 0,
            "full_keyword_detections": 0,
            "false_positives": 0,
            "timeouts": 0,
        }
        
        logging.info(f"DelayedDecisionKWS initialized with model: {model_path}")
        logging.info(f"Prefix timeout: {prefix_timeout_ms}ms")
        logging.info(f"Confidence threshold: {confidence_threshold}")
    
    def _init_model(self):
        """初始化KWS模型"""
        if SHERPA_AVAILABLE:
            try:
                # 配置sherpa-onnx
                config = sherpa_onnx.KeywordSpotterConfig(
                    model=sherpa_onnx.OnlineModelConfig(
                        transducer=sherpa_onnx.OnlineTransducerModelConfig(
                            encoder=self.model_path,
                            decoder="",
                            joiner="",
                        ),
                        tokens="",
                        modeling_unit="cjkchar",
                        bpe_vocab="",
                    ),
                    max_active_paths=4,
                    num_trailing_blanks=1,
                    keywords_score=1.0,
                    keywords_threshold=self.confidence_threshold,
                    keywords_file="",
                )
                
                self.recognizer = sherpa_onnx.KeywordSpotter(config)
                logging.info("Sherpa-ONNX model loaded successfully")
                
            except Exception as e:
                logging.error(f"Failed to load sherpa-onnx model: {e}")
                self.recognizer = None
        else:
            # Mock实现用于测试
            self.recognizer = None
            logging.info("Using mock KWS implementation")
    
    def _default_trigger_callback(self):
        """默认触发回调"""
        logging.info("🎯 KEYWORD TRIGGERED: 你好真真")
        print("🎯 关键词触发: 你好真真")
    
    def _detect_keyword(self, audio: np.ndarray) -> Tuple[bool, str, float]:
        """
        检测关键词
        Returns: (detected, keyword, confidence)
        """
        if self.recognizer is not None:
            # 使用真实的sherpa-onnx模型
            try:
                stream = self.recognizer.create_stream()
                stream.accept_waveform(self.sample_rate, audio)
                
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)
                
                result = self.recognizer.get_result(stream)
                if result.keyword:
                    return True, result.keyword, 1.0  # sherpa-onnx doesn't provide confidence
                else:
                    return False, "", 0.0
                    
            except Exception as e:
                logging.error(f"Recognition error: {e}")
                return False, "", 0.0
        else:
            # Mock实现 - 基于简单的音频特征
            return self._mock_detection(audio)
    
    def _mock_detection(self, audio: np.ndarray) -> Tuple[bool, str, float]:
        """Mock检测实现，用于测试"""
        # 简单的基于能量和长度的检测
        energy = np.mean(audio ** 2)
        duration = len(audio) / self.sample_rate
        
        # 模拟检测逻辑
        if energy > 0.01 and 0.5 < duration < 2.0:
            # 根据音频特征模拟不同的检测结果
            if duration > 1.5:
                return True, "你好真真", 0.8
            elif duration > 0.8:
                return True, "你好", 0.7
            else:
                return True, "真真", 0.6
        
        return False, "", 0.0
    
    def _start_prefix_timer(self):
        """启动前缀超时计时器"""
        if self.prefix_timer is not None:
            self.prefix_timer.cancel()
        
        self.prefix_timer = threading.Timer(
            self.prefix_timeout_ms / 1000.0,
            self._on_prefix_timeout
        )
        self.prefix_timer.start()
        logging.debug(f"Started prefix timer: {self.prefix_timeout_ms}ms")
    
    def _cancel_prefix_timer(self):
        """取消前缀计时器"""
        if self.prefix_timer is not None:
            self.prefix_timer.cancel()
            self.prefix_timer = None
            logging.debug("Cancelled prefix timer")
    
    def _on_prefix_timeout(self):
        """前缀超时处理"""
        with self.state_lock:
            if self.state == KWSState.WAITING_SUFFIX:
                logging.debug("Prefix timeout - rejecting detection")
                self.state = KWSState.REJECTED
                self.stats["timeouts"] += 1
                self._reset_to_idle()
    
    def _reset_to_idle(self):
        """重置到IDLE状态"""
        self.state = KWSState.IDLE
        self._cancel_prefix_timer()
        # 不清空音频缓冲，保持连续性
        logging.debug("Reset to IDLE state")
    
    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        处理音频块
        
        Args:
            audio_chunk: 音频数据 (float32, mono)
        """
        # 添加到音频缓冲
        self.audio_buffer.add_samples(audio_chunk)
        
        # 获取用于检测的音频窗口
        detection_window_ms = 800  # 800ms检测窗口
        detection_audio = self.audio_buffer.get_recent_audio(detection_window_ms)
        
        # 执行关键词检测
        detected, keyword, confidence = self._detect_keyword(detection_audio)
        
        if detected and confidence >= self.confidence_threshold:
            self._handle_detection(keyword, confidence)
    
    def _handle_detection(self, keyword: str, confidence: float):
        """处理检测结果"""
        with self.state_lock:
            self.stats["total_detections"] += 1
            
            logging.debug(f"Detection: '{keyword}' (confidence: {confidence:.3f}) in state {self.state.value}")
            
            if self.state == KWSState.IDLE:
                if keyword == "你好真真":
                    # 直接检测到完整关键词
                    logging.info(f"✅ Full keyword detected: {keyword}")
                    self.state = KWSState.TRIGGERED
                    self.stats["full_keyword_detections"] += 1
                    self.on_triggered()
                    self._reset_to_idle()
                    
                elif keyword == "你好":
                    # 检测到前缀
                    logging.info(f"🔍 Prefix detected: {keyword}")
                    self.state = KWSState.PREFIX_DETECTED
                    self.stats["prefix_detections"] += 1
                    self._transition_to_waiting_suffix()
                    
            elif self.state == KWSState.PREFIX_DETECTED:
                if keyword == "你好真真":
                    # 在前缀检测后又检测到完整关键词
                    logging.info(f"✅ Full keyword confirmed: {keyword}")
                    self.state = KWSState.TRIGGERED
                    self.stats["full_keyword_detections"] += 1
                    self.on_triggered()
                    self._reset_to_idle()
                else:
                    # 进入等待后缀状态
                    self._transition_to_waiting_suffix()
                    
            elif self.state == KWSState.WAITING_SUFFIX:
                if keyword == "你好真真":
                    # 在等待期间检测到完整关键词
                    logging.info(f"✅ Full keyword in waiting: {keyword}")
                    self.state = KWSState.TRIGGERED
                    self.stats["full_keyword_detections"] += 1
                    self.on_triggered()
                    self._reset_to_idle()
                    
                elif keyword == "真真":
                    # 检测到后缀
                    logging.info(f"✅ Suffix detected: {keyword}")
                    self.state = KWSState.TRIGGERED
                    self.stats["full_keyword_detections"] += 1
                    self.on_triggered()
                    self._reset_to_idle()
    
    def _transition_to_waiting_suffix(self):
        """转换到等待后缀状态"""
        self.state = KWSState.WAITING_SUFFIX
        self._start_prefix_timer()
        logging.debug("Transitioned to WAITING_SUFFIX state")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {key: 0 for key in self.stats}
    
    def cleanup(self):
        """清理资源"""
        self._cancel_prefix_timer()
        self.audio_buffer.clear()
        logging.info("DelayedDecisionKWS cleaned up")


def test_with_audio_file(kws: DelayedDecisionKWS, audio_file: str, chunk_size_ms: int = 100):
    """使用音频文件测试KWS"""
    logging.info(f"Testing with audio file: {audio_file}")
    
    # 读取音频文件
    audio, sr = sf.read(audio_file)
    if sr != kws.sample_rate:
        logging.warning(f"Audio sample rate {sr} != expected {kws.sample_rate}")
    
    # 转换为mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # 分块处理
    chunk_samples = int(chunk_size_ms * kws.sample_rate / 1000)
    total_chunks = len(audio) // chunk_samples + 1
    
    logging.info(f"Processing {total_chunks} chunks of {chunk_size_ms}ms each")
    
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        if len(chunk) < chunk_samples:
            # 最后一块，用零填充
            padded_chunk = np.zeros(chunk_samples, dtype=np.float32)
            padded_chunk[:len(chunk)] = chunk
            chunk = padded_chunk
        
        kws.process_audio_chunk(chunk.astype(np.float32))
        
        # 模拟实时处理
        time.sleep(chunk_size_ms / 1000.0)
    
    # 显示统计信息
    stats = kws.get_stats()
    logging.info("Test completed. Statistics:")
    for key, value in stats.items():
        logging.info(f"  {key}: {value}")


def create_test_audio(output_path: str, sample_rate: int = 16000):
    """创建测试音频文件"""
    # 创建一个包含不同情况的测试音频
    duration = 10.0  # 10秒
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # 基础噪声
    audio = np.random.normal(0, 0.01, len(t))
    
    # 添加一些"事件"来模拟不同的检测情况
    # 1. 在2秒处添加"你好"事件
    start_idx = int(2.0 * sample_rate)
    event_duration = int(0.5 * sample_rate)
    audio[start_idx:start_idx + event_duration] += 0.1 * np.sin(2 * np.pi * 440 * t[start_idx:start_idx + event_duration])
    
    # 2. 在5秒处添加"你好真真"事件
    start_idx = int(5.0 * sample_rate)
    event_duration = int(1.0 * sample_rate)
    audio[start_idx:start_idx + event_duration] += 0.15 * np.sin(2 * np.pi * 880 * t[start_idx:start_idx + event_duration])
    
    # 3. 在8秒处添加另一个"你好"事件（应该超时）
    start_idx = int(8.0 * sample_rate)
    event_duration = int(0.5 * sample_rate)
    audio[start_idx:start_idx + event_duration] += 0.1 * np.sin(2 * np.pi * 440 * t[start_idx:start_idx + event_duration])
    
    # 保存音频
    sf.write(output_path, audio, sample_rate)
    logging.info(f"Created test audio: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Delayed Decision KWS Inference")
    parser.add_argument("--model-path", type=str, 
                       default="/data/workspace/llm/keyword-spotting/models/model.onnx",
                       help="Path to ONNX model file")
    parser.add_argument("--tokens-path", type=str, help="Path to tokens file")
    parser.add_argument("--test-audio", type=str, help="Path to test audio file")
    parser.add_argument("--create-test-audio", action="store_true",
                       help="Create a test audio file")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for detection")
    parser.add_argument("--prefix-timeout", type=int, default=600,
                       help="Prefix timeout in milliseconds")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("=" * 60)
    logging.info("Delayed Decision KWS Inference")
    logging.info("=" * 60)
    
    # 创建测试音频
    if args.create_test_audio:
        test_audio_path = "/tmp/test_kws_audio.wav"
        create_test_audio(test_audio_path)
        args.test_audio = test_audio_path
    
    # 初始化KWS
    kws = DelayedDecisionKWS(
        model_path=args.model_path,
        tokens_path=args.tokens_path,
        confidence_threshold=args.confidence_threshold,
        prefix_timeout_ms=args.prefix_timeout,
    )
    
    try:
        if args.test_audio:
            # 测试音频文件
            if Path(args.test_audio).exists():
                test_with_audio_file(kws, args.test_audio)
            else:
                logging.error(f"Test audio file not found: {args.test_audio}")
        else:
            logging.info("No test audio specified. Use --test-audio or --create-test-audio")
            logging.info("KWS system initialized and ready for real-time processing")
    
    finally:
        kws.cleanup()
    
    logging.info("Delayed Decision KWS test completed")


if __name__ == "__main__":
    main()