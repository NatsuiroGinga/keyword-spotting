"""
评估指标计算工具
"""
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """指标计算结果"""
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    
    @property
    def frr(self) -> float:
        """False Rejection Rate = FN / (TP + FN)"""
        total = self.true_positive + self.false_negative
        return self.false_negative / total if total > 0 else 0.0
    
    @property
    def far(self) -> float:
        """False Acceptance Rate = FP / (FP + TN)"""
        total = self.false_positive + self.true_negative
        return self.false_positive / total if total > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """准确率 = (TP + TN) / Total"""
        total = (self.true_positive + self.false_negative + 
                 self.false_positive + self.true_negative)
        correct = self.true_positive + self.true_negative
        return correct / total if total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        """精确率 = TP / (TP + FP)"""
        total = self.true_positive + self.false_positive
        return self.true_positive / total if total > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """召回率 = TP / (TP + FN)"""
        total = self.true_positive + self.false_negative
        return self.true_positive / total if total > 0 else 0.0
    
    @property
    def f1(self) -> float:
        """F1分数 = 2 * P * R / (P + R)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def calculate_metrics(
    predictions: List[bool],
    labels: List[bool]
) -> MetricsResult:
    """
    计算评估指标
    
    Args:
        predictions: 预测结果列表
        labels: 真实标签列表
        
    Returns:
        MetricsResult对象
    """
    assert len(predictions) == len(labels), "预测和标签长度必须相同"
    
    tp = fn = fp = tn = 0
    
    for pred, label in zip(predictions, labels):
        if label:  # 正样本
            if pred:
                tp += 1
            else:
                fn += 1
        else:  # 负样本
            if pred:
                fp += 1
            else:
                tn += 1
    
    return MetricsResult(
        true_positive=tp,
        false_negative=fn,
        false_positive=fp,
        true_negative=tn
    )


def format_metrics_table(results: Dict[str, MetricsResult]) -> str:
    """
    格式化指标表格
    
    Args:
        results: {方案名称: MetricsResult}
        
    Returns:
        格式化的表格字符串
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'方案':<20} {'FRR':>8} {'FAR':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    lines.append("-" * 80)
    
    for name, metrics in results.items():
        lines.append(
            f"{name:<20} "
            f"{metrics.frr*100:>7.2f}% "
            f"{metrics.far*100:>7.2f}% "
            f"{metrics.accuracy*100:>7.2f}% "
            f"{metrics.precision*100:>7.2f}% "
            f"{metrics.recall*100:>7.2f}% "
            f"{metrics.f1*100:>7.2f}%"
        )
    
    lines.append("=" * 80)
    return "\n".join(lines)
