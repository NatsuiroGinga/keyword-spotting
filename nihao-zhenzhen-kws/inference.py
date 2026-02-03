#!/usr/bin/env python3
"""
你好真真 - 关键词检测推理接口

提供简单易用的API进行关键词检测，支持FP32和INT8模型
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import sherpa_onnx
except ImportError:
    raise ImportError("请安装 sherpa-onnx: pip install sherpa-onnx")


class KeywordSpotter:
    """关键词检测器"""
    
    def __init__(self, model_dir: Optional[str] = None, variant: str = None):
        """
        初始化关键词检测器
        
        Args:
            model_dir: 模型目录路径，默认为当前文件所在目录
            variant: 模型变体，可选 "fp32"(默认) 或 "int8"
        """
        if model_dir is None:
            model_dir = Path(__file__).parent
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        
        # 加载配置
        config_path = model_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        # 选择模型变体
        if variant is None:
            variant = self.config.get("default_variant", "fp32")
        
        if variant not in self.config.get("variants", {}):
            # 兼容旧版配置
            variant_config = {
                "files": self.config.get("files", {}),
                "inference": self.config.get("inference", {})
            }
        else:
            variant_config = self.config["variants"][variant]
        
        self.variant = variant
        
        # 模型文件路径
        files = variant_config["files"]
        inference_cfg = variant_config.get("inference", {})
        global_inference = self.config.get("inference", {})
        
        # 合并推理参数
        threshold = inference_cfg.get("keywords_threshold", global_inference.get("keywords_threshold", 0.5))
        score = inference_cfg.get("keywords_score", global_inference.get("keywords_score", 1.5))
        num_threads = global_inference.get("num_threads", 2)
        provider = global_inference.get("provider", "cpu")
        
        # 创建关键词检测器
        self._kws = sherpa_onnx.KeywordSpotter(
            encoder=str(model_dir / files["encoder"]),
            decoder=str(model_dir / files["decoder"]),
            joiner=str(model_dir / files["joiner"]),
            tokens=str(model_dir / files["tokens"]),
            keywords_file=str(model_dir / files["keywords"]),
            keywords_threshold=threshold,
            keywords_score=score,
            num_threads=num_threads,
            provider=provider,
        )
        
        self._threshold = threshold
        self.sample_rate = self.config["audio"]["sample_rate"]
    
    def create_stream(self):
        """创建音频流"""
        return self._kws.create_stream()
    
    def detect(self, audio: Union[np.ndarray, str], sample_rate: int = None) -> dict:
        """
        检测音频中的关键词
        
        Args:
            audio: 音频数据 (numpy数组) 或音频文件路径
            sample_rate: 采样率 (如果audio是数组则需要提供)
            
        Returns:
            dict: {
                "detected": bool,  # 是否检测到关键词
                "keyword": str,    # 检测到的关键词
            }
        """
        # 如果是文件路径，加载音频
        if isinstance(audio, str):
            import soundfile as sf
            audio, sample_rate = sf.read(audio, dtype="float32")
        
        # 重采样
        if sample_rate != self.sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            except ImportError:
                raise ImportError("需要安装 librosa 进行重采样: pip install librosa")
        
        # 确保是单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # 确保是float32
        audio = audio.astype(np.float32)
        
        # 创建流
        stream = self._kws.create_stream()
        
        # 分块处理
        chunk_size = int(0.1 * self.sample_rate)  # 100ms
        detected = False
        keyword = ""
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            stream.accept_waveform(self.sample_rate, chunk.tolist())
            
            while self._kws.is_ready(stream):
                self._kws.decode_stream(stream)
            
            result = self._kws.get_result(stream)
            
            if isinstance(result, str) and result.strip():
                detected = True
                keyword = result.strip()
                break
            elif hasattr(result, "keyword") and result.keyword:
                detected = True
                keyword = result.keyword
                break
        
        return {
            "detected": detected,
            "keyword": keyword,
        }
    
    @property
    def keywords(self):
        """返回支持的关键词列表"""
        return self.config["keywords"]
    
    @property
    def threshold(self):
        """返回检测阈值"""
        return self._threshold


# 便捷函数
def load_model(model_dir: str = None, variant: str = None) -> KeywordSpotter:
    """
    加载关键词检测模型
    
    Args:
        model_dir: 模型目录路径
        variant: 模型变体，可选 "fp32"(默认) 或 "int8"
        
    Returns:
        KeywordSpotter: 检测器实例
    """
    return KeywordSpotter(model_dir, variant)
