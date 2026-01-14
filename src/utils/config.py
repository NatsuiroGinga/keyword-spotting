"""
配置管理模块
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class KWSConfig:
    """
    KWS系统配置
    """
    # 音频配置
    sample_rate: int = 16000
    chunk_duration_ms: int = 100
    
    # Sherpa-ONNX KWS配置
    encoder_path: str = ""
    decoder_path: str = ""
    joiner_path: str = ""
    tokens_path: str = ""
    keywords_file: str = ""
    
    # 关键词配置
    keywords: List[str] = field(default_factory=lambda: ["你好真真"])
    keywords_score: float = 1.5
    keywords_threshold: float = 0.25
    
    # MLP验证器配置
    mlp_model_path: str = ""
    mlp_threshold: float = 0.5
    mlp_enabled: bool = False  # 默认禁用MLP验证器（流式场景下效果不佳）
    
    # 特征提取配置
    n_mfcc: int = 13
    target_frames: int = 50
    
    # 音频缓冲配置
    suffix_duration: float = 0.5  # 后缀音频时长（秒）
    buffer_duration: float = 3.0  # 缓冲区总时长（秒）
    
    # 推理配置
    num_threads: int = 2
    provider: str = "cpu"
    
    # 流重置配置（防止长时间运行内存累积）
    stream_reset_interval: int = 6000  # 每处理N个音频块后重建流（默认10分钟@100ms/块）
    
    @classmethod
    def from_json(cls, json_path: str) -> "KWSConfig":
        """从JSON文件加载配置"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, json_path: str) -> None:
        """保存配置到JSON文件"""
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def create_default(cls, model_dir: str) -> "KWSConfig":
        """
        创建默认配置
        
        Args:
            model_dir: 模型目录路径
        """
        model_dir = Path(model_dir)
        
        return cls(
            encoder_path=str(model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            decoder_path=str(model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            joiner_path=str(model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            tokens_path=str(model_dir / "tokens.txt"),
            keywords_file=str(model_dir / "keywords.txt"),
            mlp_model_path=str(model_dir.parent / "models" / "mlp_verifier.onnx"),
        )
