#!/usr/bin/env python3
"""
流式音频模拟器

将音频文件按固定帧大小分块输出，模拟真实的流式输入场景。
"""
import numpy as np
import librosa
from pathlib import Path
from typing import Iterator, Tuple, Optional, List, Generator
from dataclasses import dataclass
import time


@dataclass
class StreamConfig:
    """流式配置"""
    sample_rate: int = 16000
    frame_duration_ms: int = 30      # 每帧时长（毫秒）
    buffer_duration_ms: int = 2000   # 缓冲区时长（毫秒）- 约2倍关键词时长，覆盖检测延迟
    simulate_realtime: bool = False  # 是否模拟实时延迟
    
    @property
    def frame_samples(self) -> int:
        """每帧样本数"""
        return int(self.sample_rate * self.frame_duration_ms / 1000)
    
    @property
    def buffer_samples(self) -> int:
        """缓冲区样本数"""
        return int(self.sample_rate * self.buffer_duration_ms / 1000)


class RingBuffer:
    """
    环形缓冲区 - 高效的滑动窗口实现
    
    用于维护最近 N 个样本的音频数据。
    """
    
    def __init__(self, max_samples: int):
        """
        初始化环形缓冲区
        
        Args:
            max_samples: 最大样本数
        """
        self.max_samples = max_samples
        self._buffer = np.zeros(max_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
    
    def append(self, audio: np.ndarray) -> None:
        """
        添加音频到缓冲区
        
        Args:
            audio: 音频数据 (float32)
        """
        audio = audio.flatten().astype(np.float32)
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
        """
        获取最近 n 个样本
        
        Args:
            n_samples: 要获取的样本数
            
        Returns:
            音频数据
        """
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
    
    def clear(self) -> None:
        """清空缓冲区"""
        self._buffer.fill(0)
        self._write_pos = 0
        self._total_written = 0
    
    @property
    def length(self) -> int:
        """当前缓冲区有效数据长度"""
        return min(self._total_written, self.max_samples)


class StreamSimulator:
    """
    流式音频模拟器
    
    将音频文件按帧分块输出，模拟真实流式输入场景。
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """
        初始化流式模拟器
        
        Args:
            config: 流式配置
        """
        self.config = config or StreamConfig()
        self._buffer = RingBuffer(self.config.buffer_samples)
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频数据 (float32, 范围 [-1, 1])
        """
        audio, _ = librosa.load(audio_path, sr=self.config.sample_rate)
        return audio.astype(np.float32)
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频时长（秒）
        
        Args:
            audio_path: 音频文件路径
        """
        return librosa.get_duration(path=audio_path)
    
    def stream_frames(self, audio: np.ndarray) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        将音频分帧输出
        
        Args:
            audio: 音频数据
            
        Yields:
            (frame, frame_index, timestamp_ms): 帧数据、帧索引、时间戳（毫秒）
        """
        frame_samples = self.config.frame_samples
        total_samples = len(audio)
        
        frame_index = 0
        for start in range(0, total_samples, frame_samples):
            end = min(start + frame_samples, total_samples)
            frame = audio[start:end]
            
            # 最后一帧不足时补零
            if len(frame) < frame_samples:
                frame = np.pad(frame, (0, frame_samples - len(frame)))
            
            # 计算时间戳
            timestamp_ms = start / self.config.sample_rate * 1000
            
            # 模拟实时延迟
            if self.config.simulate_realtime:
                time.sleep(self.config.frame_duration_ms / 1000)
            
            yield frame, frame_index, timestamp_ms
            frame_index += 1
    
    def stream_file(self, audio_path: str) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        从文件流式读取音频
        
        Args:
            audio_path: 音频文件路径
            
        Yields:
            (frame, frame_index, timestamp_ms): 帧数据、帧索引、时间戳（毫秒）
        """
        audio = self.load_audio(audio_path)
        yield from self.stream_frames(audio)
    
    def stream_file_with_buffer(
        self, 
        audio_path: str
    ) -> Generator[Tuple[np.ndarray, np.ndarray, int, float], None, None]:
        """
        从文件流式读取，同时维护缓冲区
        
        Args:
            audio_path: 音频文件路径
            
        Yields:
            (frame, buffer, frame_index, timestamp_ms): 帧数据、缓冲区、帧索引、时间戳
        """
        self._buffer.clear()
        audio = self.load_audio(audio_path)
        
        for frame, frame_index, timestamp_ms in self.stream_frames(audio):
            self._buffer.append(frame)
            yield frame, self._buffer.get_all().copy(), frame_index, timestamp_ms
    
    def get_buffer(self) -> np.ndarray:
        """获取当前缓冲区内容"""
        return self._buffer.get_all()
    
    def clear_buffer(self) -> None:
        """清空缓冲区"""
        self._buffer.clear()


@dataclass
class StreamedAudioInfo:
    """流式音频信息"""
    audio_path: str
    label: int  # 1=正样本, 0=负样本
    duration_ms: float
    total_frames: int
    
    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000


class BatchStreamSimulator:
    """
    批量流式模拟器
    
    处理多个音频文件的流式评估。
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """
        初始化批量模拟器
        
        Args:
            config: 流式配置
        """
        self.config = config or StreamConfig()
        self.simulator = StreamSimulator(config)
    
    def prepare_dataset(
        self,
        audio_dir: str,
        positive_keywords: List[str] = None
    ) -> List[StreamedAudioInfo]:
        """
        准备数据集信息
        
        Args:
            audio_dir: 音频目录
            positive_keywords: 正样本关键词列表（文件名包含则为正样本）
            
        Returns:
            音频信息列表
        """
        audio_dir = Path(audio_dir)
        positive_keywords = positive_keywords or ["你好真真"]
        
        infos = []
        for wav_file in sorted(audio_dir.glob("*.wav")):
            # 判断是否为正样本
            filename = wav_file.name
            is_positive = any(kw in filename for kw in positive_keywords)
            
            # 获取时长
            duration_s = self.simulator.get_audio_duration(str(wav_file))
            duration_ms = duration_s * 1000
            
            # 计算帧数
            total_frames = int(np.ceil(duration_s * self.config.sample_rate / self.config.frame_samples))
            
            infos.append(StreamedAudioInfo(
                audio_path=str(wav_file),
                label=1 if is_positive else 0,
                duration_ms=duration_ms,
                total_frames=total_frames
            ))
        
        return infos
    
    def get_audio_paths_and_labels(
        self,
        audio_dir: str,
        positive_keywords: List[str] = None
    ) -> Tuple[List[str], List[int]]:
        """
        获取音频路径和标签列表
        
        Args:
            audio_dir: 音频目录
            positive_keywords: 正样本关键词列表
            
        Returns:
            (paths, labels): 路径列表、标签列表
        """
        infos = self.prepare_dataset(audio_dir, positive_keywords)
        paths = [info.audio_path for info in infos]
        labels = [info.label for info in infos]
        return paths, labels


if __name__ == "__main__":
    # 测试流式模拟器
    config = StreamConfig(
        sample_rate=16000,
        frame_duration_ms=30,
        buffer_duration_ms=1500
    )
    
    print(f"帧大小: {config.frame_samples} 样本 ({config.frame_duration_ms}ms)")
    print(f"缓冲区: {config.buffer_samples} 样本 ({config.buffer_duration_ms}ms)")
    
    # 测试环形缓冲区
    buffer = RingBuffer(max_samples=4800)  # 300ms
    
    for i in range(10):
        chunk = np.random.randn(480).astype(np.float32)  # 30ms
        buffer.append(chunk)
        print(f"块 {i+1}: 缓冲区 {buffer.length} 样本")
    
    print(f"\n最后 100ms: {len(buffer.get_last(1600))} 样本")
