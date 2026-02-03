#!/usr/bin/env python3
"""
你好真真 - 实时麦克风关键词检测

使用方法:
    python realtime_detection.py

依赖:
    pip install sherpa-onnx sounddevice numpy

按 Ctrl+C 停止检测
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import sherpa_onnx
except ImportError:
    print("错误: 请安装 sherpa-onnx")
    print("  pip install sherpa-onnx")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("错误: 请安装 sounddevice")
    print("  pip install sounddevice")
    sys.exit(1)


class KeywordDetector:
    """关键词检测器"""
    
    def __init__(self, model_dir: str = None):
        """
        初始化检测器
        
        Args:
            model_dir: 模型目录路径，默认为当前脚本所在目录的父目录
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent
        else:
            model_dir = Path(model_dir)
        
        # 加载配置
        config_path = model_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        # 模型文件路径
        model_path = model_dir / "model"
        encoder = str(model_path / "encoder.onnx")
        decoder = str(model_path / "decoder.onnx")
        joiner = str(model_path / "joiner.onnx")
        tokens = str(model_path / "tokens.txt")
        keywords = str(model_path / "keywords.txt")
        
        # 推理参数
        inference_config = self.config["inference"]
        
        print(f"加载模型: {self.config['model']['name']}")
        print(f"关键词: {', '.join(self.config['keywords'])}")
        print(f"阈值: {inference_config['keywords_threshold']}")
        print()
        
        # 创建关键词检测器
        self.kws = sherpa_onnx.KeywordSpotter(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            keywords_file=keywords,
            keywords_threshold=inference_config["keywords_threshold"],
            keywords_score=inference_config["keywords_score"],
            num_threads=inference_config["num_threads"],
            provider=inference_config["provider"],
        )
        
        # 音频参数
        self.sample_rate = self.config["audio"]["sample_rate"]
        self.chunk_duration_ms = self.config["audio"]["chunk_duration_ms"]
        self.chunk_samples = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # 创建流
        self.stream = self.kws.create_stream()
        
        # 检测计数
        self.detection_count = 0
    
    def process_audio(self, audio_data: np.ndarray) -> str:
        """
        处理音频数据
        
        Args:
            audio_data: 音频数据 (float32, 单声道)
            
        Returns:
            检测到的关键词，如果没有检测到返回空字符串
        """
        # 确保是float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 确保是一维数组
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        # 送入流
        self.stream.accept_waveform(self.sample_rate, audio_data.tolist())
        
        # 解码
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
        
        # 获取结果
        result = self.kws.get_result(self.stream)
        
        # 处理结果
        if isinstance(result, str):
            keyword = result.strip()
        elif hasattr(result, "keyword"):
            keyword = result.keyword.strip() if result.keyword else ""
        else:
            keyword = ""
        
        return keyword
    
    def on_keyword_detected(self, keyword: str):
        """关键词检测回调"""
        self.detection_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] 🎯 检测到关键词: {keyword} (第{self.detection_count}次)")


def audio_callback(detector: KeywordDetector):
    """创建音频回调函数"""
    def callback(indata, frames, time_info, status):
        if status:
            print(f"音频状态: {status}", file=sys.stderr)
        
        # 转换为float32
        audio = indata[:, 0].astype(np.float32)
        
        # 处理音频
        keyword = detector.process_audio(audio)
        
        if keyword:
            detector.on_keyword_detected(keyword)
    
    return callback


def main():
    """主函数"""
    print("=" * 60)
    print("你好真真 - 实时关键词检测")
    print("=" * 60)
    print()
    
    # 创建检测器
    detector = KeywordDetector()
    
    # 音频参数
    sample_rate = detector.sample_rate
    chunk_samples = detector.chunk_samples
    
    print(f"采样率: {sample_rate} Hz")
    print(f"块大小: {chunk_samples} 样本 ({detector.chunk_duration_ms}ms)")
    print()
    print("开始监听麦克风...")
    print("说 '你好真真' 或 '你好珍珍' 来触发检测")
    print("按 Ctrl+C 停止")
    print("-" * 60)
    print()
    
    try:
        # 开始音频流
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples,
            callback=audio_callback(detector),
        ):
            # 保持运行
            while True:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print()
        print("-" * 60)
        print(f"停止监听。共检测到 {detector.detection_count} 次关键词。")
    
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
