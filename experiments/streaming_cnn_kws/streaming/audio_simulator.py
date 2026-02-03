#!/usr/bin/env python3
"""
流式音频模拟器
从音频文件模拟实时流式输入，用于测试和评估
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
from dataclasses import dataclass
import time


@dataclass
class StreamConfig:
    """流式配置"""
    sample_rate: int = 16000
    frame_duration_ms: int = 30  # 每帧时长（毫秒）
    buffer_duration_s: float = 1.5  # 缓冲区时长（秒）
    
    @property
    def frame_length(self) -> int:
        """每帧样本数"""
        return int(self.sample_rate * self.frame_duration_ms / 1000)
    
    @property
    def buffer_length(self) -> int:
        """缓冲区样本数"""
        return int(self.sample_rate * self.buffer_duration_s)


class AudioStreamSimulator:
    """
    音频流模拟器
    将音频文件切分为帧，模拟实时流式输入
    """
    
    def __init__(self, config: StreamConfig = None):
        if config is None:
            config = StreamConfig()
        self.config = config
        self._current_audio = None
        self._current_position = 0
    
    def stream_file(self, audio_path: str) -> Iterator[np.ndarray]:
        """
        流式读取音频文件
        
        Args:
            audio_path: 音频文件路径
        
        Yields:
            np.ndarray: 每帧音频数据 (frame_length,)
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        
        # 按帧长切分
        frame_length = self.config.frame_length
        for i in range(0, len(audio), frame_length):
            frame = audio[i:i + frame_length]
            
            # 最后一帧可能不足，补零
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            
            yield frame.astype(np.float32)
    
    def stream_file_with_buffer(self, audio_path: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        流式读取，同时维护滑动窗口缓冲区
        
        Yields:
            (frame, buffer): 当前帧和缓冲区内容
        """
        buffer = np.zeros(self.config.buffer_length, dtype=np.float32)
        
        for frame in self.stream_file(audio_path):
            # 更新缓冲区（滑动窗口）
            buffer = np.roll(buffer, -len(frame))
            buffer[-len(frame):] = frame
            
            yield frame, buffer.copy()
    
    def get_full_audio(self, audio_path: str) -> np.ndarray:
        """加载完整音频"""
        audio, _ = librosa.load(audio_path, sr=self.config.sample_rate)
        return audio.astype(np.float32)
    
    def get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        duration = librosa.get_duration(path=audio_path)
        return duration


class RingBuffer:
    """环形缓冲区 - 高效的滑动窗口实现"""
    
    def __init__(self, max_samples: int):
        self.max_samples = max_samples
        self._buffer = np.zeros(max_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
    
    def append(self, audio: np.ndarray):
        """添加音频到缓冲区"""
        n = len(audio)
        
        if n >= self.max_samples:
            # 音频长度超过缓冲区，只保留最后部分
            self._buffer[:] = audio[-self.max_samples:]
            self._write_pos = 0
            self._total_written = self.max_samples
        else:
            # 写入位置
            end_pos = self._write_pos + n
            
            if end_pos <= self.max_samples:
                self._buffer[self._write_pos:end_pos] = audio
            else:
                # 需要环绕
                first_part = self.max_samples - self._write_pos
                self._buffer[self._write_pos:] = audio[:first_part]
                self._buffer[:n - first_part] = audio[first_part:]
            
            self._write_pos = end_pos % self.max_samples
            self._total_written += n
    
    def get_last(self, n_samples: int) -> np.ndarray:
        """获取最近 n 个样本"""
        n_samples = min(n_samples, self.max_samples, self._total_written)
        
        if n_samples == 0:
            return np.array([], dtype=np.float32)
        
        result = np.zeros(n_samples, dtype=np.float32)
        
        # 计算读取位置
        if self._total_written < self.max_samples:
            # 缓冲区未满
            start = max(0, self._total_written - n_samples)
            result[:] = self._buffer[start:start + n_samples]
        else:
            # 缓冲区已满，需要处理环绕
            read_start = (self._write_pos - n_samples) % self.max_samples
            
            if read_start + n_samples <= self.max_samples:
                result[:] = self._buffer[read_start:read_start + n_samples]
            else:
                first_part = self.max_samples - read_start
                result[:first_part] = self._buffer[read_start:]
                result[first_part:] = self._buffer[:n_samples - first_part]
        
        return result
    
    def get_all(self) -> np.ndarray:
        """获取缓冲区所有有效数据"""
        n = min(self._total_written, self.max_samples)
        return self.get_last(n)
    
    def clear(self):
        """清空缓冲区"""
        self._buffer.fill(0)
        self._write_pos = 0
        self._total_written = 0


class BatchStreamSimulator:
    """
    批量流式模拟器
    处理多个音频文件的流式评估
    """
    
    def __init__(self, config: StreamConfig = None):
        if config is None:
            config = StreamConfig()
        self.config = config
        self.simulator = AudioStreamSimulator(config)
    
    def evaluate_files(
        self,
        audio_paths: List[str],
        labels: List[int],
        detector_fn,
        verbose: bool = False
    ) -> dict:
        """
        批量评估音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            labels: 标签列表 (1=正样本, 0=负样本)
            detector_fn: 检测函数，接收音频返回 (detected, confidence)
            verbose: 是否打印详细信息
        
        Returns:
            评估结果字典
        """
        results = {
            "predictions": [],
            "confidences": [],
            "labels": labels,
            "detection_times": [],
        }
        
        for i, (path, label) in enumerate(zip(audio_paths, labels)):
            start_time = time.perf_counter()
            
            # 模拟流式检测
            detected, confidence = self._stream_detect(path, detector_fn)
            
            detection_time = time.perf_counter() - start_time
            
            results["predictions"].append(int(detected))
            results["confidences"].append(confidence)
            results["detection_times"].append(detection_time)
            
            if verbose:
                status = "✓" if detected == label else "✗"
                print(f"{status} [{i+1}/{len(audio_paths)}] {Path(path).name}: "
                      f"pred={detected}, label={label}, conf={confidence:.3f}")
        
        # 计算指标
        results["metrics"] = self._calculate_metrics(
            np.array(results["predictions"]),
            np.array(labels)
        )
        
        return results
    
    def _stream_detect(self, audio_path: str, detector_fn) -> Tuple[bool, float]:
        """流式检测单个文件"""
        buffer = RingBuffer(self.config.buffer_length)
        
        detected = False
        max_confidence = 0.0
        
        for frame in self.simulator.stream_file(audio_path):
            buffer.append(frame)
            
            # 调用检测器
            result = detector_fn(frame, buffer.get_all())
            
            if result is not None:
                det, conf = result
                if det:
                    detected = True
                    max_confidence = max(max_confidence, conf)
        
        return detected, max_confidence
    
    def _calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> dict:
        """计算评估指标"""
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
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
        }


if __name__ == "__main__":
    # 测试流式模拟器
    config = StreamConfig(
        sample_rate=16000,
        frame_duration_ms=30,
        buffer_duration_s=1.5
    )
    
    simulator = AudioStreamSimulator(config)
    
    print(f"Frame length: {config.frame_length} samples ({config.frame_duration_ms}ms)")
    print(f"Buffer length: {config.buffer_length} samples ({config.buffer_duration_s}s)")
    
    # 测试环形缓冲区
    buffer = RingBuffer(max_samples=4800)  # 300ms
    
    # 模拟添加数据
    for i in range(10):
        chunk = np.random.randn(480).astype(np.float32)  # 30ms
        buffer.append(chunk)
        last_100ms = buffer.get_last(1600)
        print(f"After chunk {i+1}: buffer has {len(buffer.get_all())} samples, "
              f"last 100ms: {len(last_100ms)} samples")
