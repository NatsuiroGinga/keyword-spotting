#!/usr/bin/env python3
"""
流式 KWS 性能指标计算模块

包含：
- 检测准确率、精确率、召回率、F1
- 误报率 (FAR)、漏检率 (FRR)
- 延迟统计（平均、P50、P90、P99）
- 实时因子 (RTF)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path


@dataclass
class DetectionEvent:
    """检测事件"""
    audio_path: str
    label: int                    # 真实标签 (1=正样本, 0=负样本)
    detected: bool                # 是否检测到
    confidence: float = 0.0       # 检测置信度
    detection_time_ms: float = 0.0  # 检测时间戳（相对于音频开始）
    inference_time_ms: float = 0.0  # 推理耗时
    audio_duration_ms: float = 0.0  # 音频时长
    v3_triggered: bool = False    # V3 是否触发
    v3_score: float = 0.0         # V3 分数
    mlp_score: float = 0.0        # MLP 分数


@dataclass
class LatencyStats:
    """延迟统计"""
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    
    @classmethod
    def from_latencies(cls, latencies: List[float]) -> "LatencyStats":
        """从延迟列表计算统计"""
        if not latencies:
            return cls()
        
        arr = np.array(latencies)
        return cls(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99))
        )
    
    def to_dict(self) -> Dict:
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms
        }


@dataclass
class RTFStats:
    """实时因子统计"""
    overall_rtf: float = 0.0     # 总体 RTF
    mean_rtf: float = 0.0        # 平均 RTF
    max_rtf: float = 0.0         # 最大 RTF
    p99_rtf: float = 0.0         # P99 RTF
    is_realtime: bool = True     # 是否满足实时要求
    
    @classmethod
    def from_times(
        cls, 
        inference_times_ms: List[float], 
        audio_durations_ms: List[float]
    ) -> "RTFStats":
        """从推理时间和音频时长计算 RTF"""
        if not inference_times_ms or not audio_durations_ms:
            return cls()
        
        inference_arr = np.array(inference_times_ms)
        duration_arr = np.array(audio_durations_ms)
        
        # 每个样本的 RTF
        rtfs = inference_arr / np.maximum(duration_arr, 1e-6)
        
        # 总体 RTF
        overall_rtf = float(np.sum(inference_arr) / np.sum(duration_arr))
        
        return cls(
            overall_rtf=overall_rtf,
            mean_rtf=float(np.mean(rtfs)),
            max_rtf=float(np.max(rtfs)),
            p99_rtf=float(np.percentile(rtfs, 99)),
            is_realtime=overall_rtf < 1.0
        )
    
    def to_dict(self) -> Dict:
        return {
            "overall_rtf": self.overall_rtf,
            "mean_rtf": self.mean_rtf,
            "max_rtf": self.max_rtf,
            "p99_rtf": self.p99_rtf,
            "is_realtime": self.is_realtime
        }


@dataclass
class EvaluationMetrics:
    """评估指标"""
    # 样本统计
    total_samples: int = 0
    positive_samples: int = 0
    negative_samples: int = 0
    
    # 混淆矩阵
    tp: int = 0  # True Positive
    tn: int = 0  # True Negative
    fp: int = 0  # False Positive
    fn: int = 0  # False Negative
    
    # 检测指标
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # 误报/漏检率
    far: float = 0.0  # False Acceptance Rate
    frr: float = 0.0  # False Rejection Rate
    
    # 延迟统计
    latency_stats: LatencyStats = field(default_factory=LatencyStats)
    
    # RTF 统计
    rtf_stats: RTFStats = field(default_factory=RTFStats)
    
    # 目标达成
    far_target: float = 0.10   # FAR 目标 < 10%
    frr_target: float = 0.05   # FRR 目标 < 5%
    rtf_target: float = 1.0    # RTF 目标 < 1.0
    meets_far_target: bool = False
    meets_frr_target: bool = False
    meets_rtf_target: bool = False
    meets_all_targets: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "samples": {
                "total": self.total_samples,
                "positive": self.positive_samples,
                "negative": self.negative_samples
            },
            "confusion_matrix": {
                "tp": self.tp,
                "tn": self.tn,
                "fp": self.fp,
                "fn": self.fn
            },
            "detection": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score
            },
            "error_rates": {
                "far": self.far,
                "frr": self.frr
            },
            "latency": self.latency_stats.to_dict(),
            "rtf": self.rtf_stats.to_dict(),
            "targets": {
                "far_target": self.far_target,
                "frr_target": self.frr_target,
                "rtf_target": self.rtf_target,
                "meets_far": self.meets_far_target,
                "meets_frr": self.meets_frr_target,
                "meets_rtf": self.meets_rtf_target,
                "meets_all": self.meets_all_targets
            }
        }


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(
        self,
        far_target: float = 0.10,
        frr_target: float = 0.05,
        rtf_target: float = 1.0
    ):
        """
        初始化指标计算器
        
        Args:
            far_target: FAR 目标值
            frr_target: FRR 目标值
            rtf_target: RTF 目标值
        """
        self.far_target = far_target
        self.frr_target = frr_target
        self.rtf_target = rtf_target
    
    def calculate(self, events: List[DetectionEvent]) -> EvaluationMetrics:
        """
        计算评估指标
        
        Args:
            events: 检测事件列表
            
        Returns:
            评估指标
        """
        if not events:
            return EvaluationMetrics()
        
        # 提取数据
        labels = np.array([e.label for e in events])
        predictions = np.array([1 if e.detected else 0 for e in events])
        
        # 样本统计
        total_samples = len(events)
        positive_samples = int(np.sum(labels == 1))
        negative_samples = int(np.sum(labels == 0))
        
        # 混淆矩阵
        tp = int(np.sum((predictions == 1) & (labels == 1)))
        tn = int(np.sum((predictions == 0) & (labels == 0)))
        fp = int(np.sum((predictions == 1) & (labels == 0)))
        fn = int(np.sum((predictions == 0) & (labels == 1)))
        
        # 检测指标
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 误报/漏检率
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # 延迟统计（仅统计正样本的检测延迟）
        detection_latencies = [
            e.detection_time_ms for e in events 
            if e.label == 1 and e.detected
        ]
        latency_stats = LatencyStats.from_latencies(detection_latencies)
        
        # RTF 统计
        inference_times = [e.inference_time_ms for e in events]
        audio_durations = [e.audio_duration_ms for e in events]
        rtf_stats = RTFStats.from_times(inference_times, audio_durations)
        
        # 目标达成
        meets_far = far < self.far_target
        meets_frr = frr < self.frr_target
        meets_rtf = rtf_stats.overall_rtf < self.rtf_target
        
        return EvaluationMetrics(
            total_samples=total_samples,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            far=far,
            frr=frr,
            latency_stats=latency_stats,
            rtf_stats=rtf_stats,
            far_target=self.far_target,
            frr_target=self.frr_target,
            rtf_target=self.rtf_target,
            meets_far_target=meets_far,
            meets_frr_target=meets_frr,
            meets_rtf_target=meets_rtf,
            meets_all_targets=meets_far and meets_frr and meets_rtf
        )
    
    def find_optimal_threshold(
        self,
        events: List[DetectionEvent],
        scores: List[float],
        target_far: float = 0.10
    ) -> Tuple[float, EvaluationMetrics]:
        """
        搜索最优阈值
        
        Args:
            events: 检测事件列表
            scores: 对应的分数列表
            target_far: 目标 FAR
            
        Returns:
            (最优阈值, 对应指标)
        """
        labels = np.array([e.label for e in events])
        scores_arr = np.array(scores)
        
        best_threshold = 0.5
        best_metrics = None
        best_f1 = -1
        
        # 网格搜索
        for threshold in np.arange(0.1, 1.0, 0.05):
            # 临时修改 detected 状态
            temp_events = []
            for e, s in zip(events, scores):
                temp_e = DetectionEvent(
                    audio_path=e.audio_path,
                    label=e.label,
                    detected=s >= threshold,
                    confidence=s,
                    detection_time_ms=e.detection_time_ms,
                    inference_time_ms=e.inference_time_ms,
                    audio_duration_ms=e.audio_duration_ms
                )
                temp_events.append(temp_e)
            
            metrics = self.calculate(temp_events)
            
            # 在 FAR 约束下最大化 F1
            if metrics.far <= target_far:
                if metrics.f1_score > best_f1:
                    best_f1 = metrics.f1_score
                    best_threshold = threshold
                    best_metrics = metrics
        
        # 如果没有满足 FAR 约束的，返回 FAR 最低的
        if best_metrics is None:
            best_threshold = 0.9
            temp_events = []
            for e, s in zip(events, scores):
                temp_e = DetectionEvent(
                    audio_path=e.audio_path,
                    label=e.label,
                    detected=s >= best_threshold,
                    confidence=s,
                    detection_time_ms=e.detection_time_ms,
                    inference_time_ms=e.inference_time_ms,
                    audio_duration_ms=e.audio_duration_ms
                )
                temp_events.append(temp_e)
            best_metrics = self.calculate(temp_events)
        
        return best_threshold, best_metrics


def format_metrics_table(metrics: EvaluationMetrics) -> str:
    """
    格式化指标为表格字符串
    
    Args:
        metrics: 评估指标
        
    Returns:
        格式化的表格字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("流式 KWS 评估结果")
    lines.append("=" * 60)
    
    # 样本统计
    lines.append(f"\n【样本统计】")
    lines.append(f"  总样本数: {metrics.total_samples}")
    lines.append(f"  正样本数: {metrics.positive_samples}")
    lines.append(f"  负样本数: {metrics.negative_samples}")
    
    # 混淆矩阵
    lines.append(f"\n【混淆矩阵】")
    lines.append(f"  TP (正确检测): {metrics.tp}")
    lines.append(f"  TN (正确拒绝): {metrics.tn}")
    lines.append(f"  FP (误报): {metrics.fp}")
    lines.append(f"  FN (漏检): {metrics.fn}")
    
    # 检测指标
    lines.append(f"\n【检测指标】")
    lines.append(f"  准确率: {metrics.accuracy * 100:.2f}%")
    lines.append(f"  精确率: {metrics.precision * 100:.2f}%")
    lines.append(f"  召回率: {metrics.recall * 100:.2f}%")
    lines.append(f"  F1 分数: {metrics.f1_score * 100:.2f}%")
    
    # 误报/漏检率
    lines.append(f"\n【误报/漏检率】")
    far_status = "✓" if metrics.meets_far_target else "✗"
    frr_status = "✓" if metrics.meets_frr_target else "✗"
    lines.append(f"  FAR (误报率): {metrics.far * 100:.2f}% {far_status} (目标 < {metrics.far_target * 100:.0f}%)")
    lines.append(f"  FRR (漏检率): {metrics.frr * 100:.2f}% {frr_status} (目标 < {metrics.frr_target * 100:.0f}%)")
    
    # 延迟统计
    lines.append(f"\n【延迟统计】(正样本检测延迟)")
    lines.append(f"  平均: {metrics.latency_stats.mean_ms:.1f}ms")
    lines.append(f"  P50: {metrics.latency_stats.p50_ms:.1f}ms")
    lines.append(f"  P90: {metrics.latency_stats.p90_ms:.1f}ms")
    lines.append(f"  P99: {metrics.latency_stats.p99_ms:.1f}ms")
    
    # RTF 统计
    lines.append(f"\n【实时因子 (RTF)】")
    rtf_status = "✓" if metrics.meets_rtf_target else "✗"
    lines.append(f"  总体 RTF: {metrics.rtf_stats.overall_rtf:.4f} {rtf_status} (目标 < {metrics.rtf_target:.1f})")
    lines.append(f"  平均 RTF: {metrics.rtf_stats.mean_rtf:.4f}")
    lines.append(f"  P99 RTF: {metrics.rtf_stats.p99_rtf:.4f}")
    lines.append(f"  实时能力: {'是' if metrics.rtf_stats.is_realtime else '否'}")
    
    # 总体判定
    lines.append(f"\n【达标判定】")
    overall_status = "✓ 全部达标" if metrics.meets_all_targets else "✗ 未完全达标"
    lines.append(f"  {overall_status}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试指标计算
    calculator = MetricsCalculator()
    
    # 模拟测试数据
    events = [
        DetectionEvent("pos_1.wav", 1, True, 0.9, 500, 30, 1200),
        DetectionEvent("pos_2.wav", 1, True, 0.85, 480, 28, 1100),
        DetectionEvent("pos_3.wav", 1, False, 0.3, 0, 25, 1000),
        DetectionEvent("neg_1.wav", 0, False, 0.1, 0, 22, 800),
        DetectionEvent("neg_2.wav", 0, False, 0.2, 0, 24, 900),
        DetectionEvent("neg_3.wav", 0, True, 0.6, 300, 26, 850),
    ]
    
    metrics = calculator.calculate(events)
    print(format_metrics_table(metrics))
