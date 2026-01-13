"""
多阶段关键词检测消融实验配置
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AblationConfig:
    """消融实验配置"""
    
    # 数据路径
    test_data_path: str = "/data/workspace/llm/audio-classification/dataset/kws_test_data_merged"
    positive_dir: str = "positive"
    negative_dir: str = "negative"
    
    # 样本数量
    positive_samples: int = 144
    negative_samples: int = 540
    
    # 模型路径
    model_dir: str = "/data/workspace/llm/keyword-spotting/exp/kws_finetune_v3"
    
    # 阶段1配置
    stage1_threshold: float = 0.3
    stage1_use_int8: bool = True
    
    # 阶段2验证方案列表
    verifiers: List[str] = field(default_factory=lambda: [
        "cnn", "asr", "dtw", "mlp"
    ])
    
    # ASR验证器配置
    asr_model_name: str = "openai/whisper-small"  # 基础Whisper模型，支持中文
    asr_target_text: str = "真真"
    asr_threshold: float = 0.5  # 文本匹配相似度阈值
    
    # DTW验证器配置
    dtw_template_dir: str = ""  # 模板音频目录
    dtw_threshold: float = 50.0  # DTW距离阈值
    dtw_n_mfcc: int = 13
    
    # CNN验证器配置
    cnn_model_path: str = ""
    cnn_threshold: float = 0.5
    
    # MLP验证器配置
    mlp_model_path: str = ""
    mlp_threshold: float = 0.5
    mlp_input_dim: int = 13 * 50  # MFCC特征维度
    
    # 后缀分割配置
    suffix_start_ratio: float = 0.4  # 从音频40%位置开始提取后缀
    suffix_min_duration_ms: int = 200  # 后缀最小时长
    suffix_max_duration_ms: int = 800  # 后缀最大时长
    
    # 音频配置
    sample_rate: int = 16000
    
    # 目标指标
    target_frr: float = 0.05  # 5%
    target_far: float = 0.10  # 10%
    
    # 输出配置
    results_dir: str = ""
    
    def __post_init__(self):
        if not self.results_dir:
            self.results_dir = str(
                Path(__file__).parent / "results"
            )
        if not self.dtw_template_dir:
            self.dtw_template_dir = str(
                Path(self.test_data_path) / self.positive_dir
            )


@dataclass
class VerifierResult:
    """验证器结果"""
    audio_path: str
    is_positive: bool  # 真实标签
    stage1_passed: bool  # 阶段1是否通过
    stage1_confidence: float = 0.0
    stage2_passed: bool = False  # 阶段2是否通过
    stage2_confidence: float = 0.0
    final_accepted: bool = False  # 最终是否接受
    verifier_name: str = ""
    process_time_ms: float = 0.0


@dataclass 
class AblationResult:
    """消融实验结果"""
    verifier_name: str
    
    # 样本统计
    total_positive: int = 0
    total_negative: int = 0
    
    # 阶段1统计
    stage1_passed_positive: int = 0
    stage1_passed_negative: int = 0
    
    # 阶段2统计
    stage2_passed_positive: int = 0
    stage2_passed_negative: int = 0
    
    # 最终结果
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0
    
    # 时间统计
    avg_process_time_ms: float = 0.0
    avg_audio_duration_ms: float = 0.0  # 平均音频时长(ms)
    
    @property
    def rtf(self) -> float:
        """Real-Time Factor: 处理时间/音频时长，小于1表示可实时处理"""
        if self.avg_audio_duration_ms > 0:
            return self.avg_process_time_ms / self.avg_audio_duration_ms
        return 0.0
    
    @property
    def frr(self) -> float:
        """False Rejection Rate"""
        total = self.true_positive + self.false_negative
        return self.false_negative / total if total > 0 else 0.0
    
    @property
    def far(self) -> float:
        """False Acceptance Rate"""
        total = self.false_positive + self.true_negative
        return self.false_positive / total if total > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """准确率"""
        total = (self.true_positive + self.false_negative + 
                 self.false_positive + self.true_negative)
        correct = self.true_positive + self.true_negative
        return correct / total if total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """精确率"""
        total = self.true_positive + self.false_positive
        return self.true_positive / total if total > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """召回率"""
        total = self.true_positive + self.false_negative
        return self.true_positive / total if total > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """F1分数"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def meets_target(self, config: AblationConfig) -> bool:
        """是否达到目标指标"""
        return self.frr <= config.target_frr and self.far <= config.target_far
    
    def to_dict(self) -> dict:
        return {
            "verifier_name": self.verifier_name,
            "total_positive": self.total_positive,
            "total_negative": self.total_negative,
            "stage1_passed_positive": self.stage1_passed_positive,
            "stage1_passed_negative": self.stage1_passed_negative,
            "stage2_passed_positive": self.stage2_passed_positive,
            "stage2_passed_negative": self.stage2_passed_negative,
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "frr": self.frr,
            "far": self.far,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "avg_process_time_ms": self.avg_process_time_ms,
            "avg_audio_duration_ms": self.avg_audio_duration_ms,
            "rtf": self.rtf,
            "meets_target": self.meets_target(AblationConfig()),
        }
