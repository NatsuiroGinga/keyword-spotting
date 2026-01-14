"""
麦克风音频采集模块

支持跨平台（Windows/Linux/macOS）的实时音频采集。
"""
import queue
import threading
from typing import Callable, Optional

import numpy as np


class AudioCapture:
    """
    麦克风音频采集器
    
    使用sounddevice库实现跨平台音频采集，支持回调模式处理实时音频数据。
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        channels: int = 1,
        device: Optional[int] = None
    ):
        """
        初始化音频采集器
        
        Args:
            sample_rate: 采样率（默认16000Hz）
            chunk_duration_ms: 每个音频块的时长（毫秒）
            channels: 声道数（默认单声道）
            device: 音频设备索引（None表示使用默认设备）
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.channels = channels
        self.device = device
        
        # 计算每个块的采样点数
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        # 音频队列
        self._audio_queue: queue.Queue = queue.Queue()
        self._stream = None
        self._is_running = False
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        
    def _audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if status:
            print(f"音频状态: {status}")
        
        # 将音频数据放入队列
        audio_data = indata[:, 0].copy() if self.channels == 1 else indata.copy()
        self._audio_queue.put(audio_data)
        
        # 如果有回调函数，直接调用
        if self._callback is not None:
            self._callback(audio_data)
    
    def start(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> None:
        """
        启动音频采集
        
        Args:
            callback: 可选的回调函数，接收音频数据块
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("需要安装 sounddevice: pip install sounddevice")
        
        if self._is_running:
            print("音频采集已在运行中")
            return
        
        self._callback = callback
        self._is_running = True
        
        # 创建输入流
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._audio_callback
        )
        
        self._stream.start()
        print(f"音频采集已启动: {self.sample_rate}Hz, {self.chunk_duration_ms}ms/块")
    
    def stop(self) -> None:
        """停止音频采集"""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        self._is_running = False
        self._callback = None
        print("音频采集已停止")
    
    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        从队列读取音频数据
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            音频数据数组，如果超时返回None
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def read_seconds(self, seconds: float) -> np.ndarray:
        """
        读取指定秒数的音频
        
        Args:
            seconds: 要读取的秒数
            
        Returns:
            音频数据数组
        """
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("需要安装 sounddevice: pip install sounddevice")
        
        samples = int(self.sample_rate * seconds)
        audio = sd.rec(
            samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            device=self.device
        )
        sd.wait()
        return audio.flatten()
    
    @property
    def is_running(self) -> bool:
        """返回采集器是否正在运行"""
        return self._is_running
    
    @staticmethod
    def list_devices() -> None:
        """列出所有可用的音频设备"""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("需要安装 sounddevice: pip install sounddevice")
        
        print("可用音频设备:")
        print(sd.query_devices())
    
    @staticmethod
    def get_default_device() -> dict:
        """获取默认输入设备信息"""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("需要安装 sounddevice: pip install sounddevice")
        
        device_id = sd.default.device[0]
        if device_id is None:
            device_id = 0
        return sd.query_devices(device_id)


class AudioBuffer:
    """
    环形音频缓冲区
    
    用于存储最近的音频数据，支持提取指定时长的历史音频。
    优化：使用预分配缓冲区减少内存分配。
    """
    
    def __init__(self, max_duration: float = 5.0, sample_rate: int = 16000):
        """
        初始化缓冲区
        
        Args:
            max_duration: 最大缓冲时长（秒）
            sample_rate: 采样率
        """
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self._buffer = np.zeros(self.max_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_samples = 0
        # 预分配输出缓冲区，避免频繁内存分配
        self._output_buffer = np.zeros(self.max_samples, dtype=np.float32)
    
    def append(self, audio: np.ndarray) -> None:
        """
        添加音频数据到缓冲区
        
        Args:
            audio: 音频数据
        """
        audio = audio.flatten()
        n_samples = len(audio)
        
        if n_samples >= self.max_samples:
            # 如果新数据超过缓冲区大小，只保留最后部分
            self._buffer[:] = audio[-self.max_samples:]
            self._write_pos = 0
            self._total_samples = self.max_samples
        else:
            # 计算写入位置
            end_pos = self._write_pos + n_samples
            
            if end_pos <= self.max_samples:
                self._buffer[self._write_pos:end_pos] = audio
            else:
                # 需要环绕写入
                first_part = self.max_samples - self._write_pos
                self._buffer[self._write_pos:] = audio[:first_part]
                self._buffer[:n_samples - first_part] = audio[first_part:]
            
            self._write_pos = end_pos % self.max_samples
            self._total_samples = min(self._total_samples + n_samples, self.max_samples)
    
    def get_last(self, duration: float) -> np.ndarray:
        """
        获取最近指定时长的音频
        
        Args:
            duration: 时长（秒）
            
        Returns:
            音频数据（使用预分配缓冲区的视图）
        """
        n_samples = min(int(duration * self.sample_rate), self._total_samples)
        
        if n_samples == 0:
            return self._output_buffer[:0]  # 返回空视图而非新数组
        
        # 计算起始位置
        start_pos = (self._write_pos - n_samples) % self.max_samples
        
        # 使用预分配缓冲区
        output = self._output_buffer[:n_samples]
        
        if start_pos < self._write_pos:
            # 连续读取：直接拷贝
            np.copyto(output, self._buffer[start_pos:self._write_pos])
        else:
            # 环绕读取：分段拷贝到预分配缓冲区
            first_part = self.max_samples - start_pos
            np.copyto(output[:first_part], self._buffer[start_pos:])
            np.copyto(output[first_part:], self._buffer[:self._write_pos])
        
        return output
    
    def clear(self) -> None:
        """清空缓冲区"""
        self._buffer.fill(0)
        self._output_buffer.fill(0)
        self._write_pos = 0
        self._total_samples = 0
