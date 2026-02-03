#!/usr/bin/env python3
"""
评估报告生成模块

生成流式 KWS 评估的可视化报告：
- 混淆矩阵
- ROC 曲线
- 延迟分布
- 指标对比表
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from metrics import EvaluationMetrics, DetectionEvent


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_confusion_matrix(
        self,
        events: List[DetectionEvent],
        title: str = "混淆矩阵",
        filename: str = "confusion_matrix.png"
    ) -> Path:
        """
        生成混淆矩阵图
        
        Args:
            events: 检测事件列表
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        labels = np.array([e.label for e in events])
        predictions = np.array([1 if e.detected else 0 for e in events])
        
        # 计算混淆矩阵
        tp = np.sum((predictions == 1) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # 绘制
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # 标签
        classes = ['负样本 (0)', '正样本 (1)']
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=['预测: 0', '预测: 1'],
            yticklabels=['真实: 0', '真实: 1'],
            title=title,
            ylabel='真实标签',
            xlabel='预测标签'
        )
        
        # 旋转标签
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # 添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                label_text = f"{cm[i, j]}"
                if i == 0 and j == 0:
                    label_text += "\n(TN)"
                elif i == 0 and j == 1:
                    label_text += "\n(FP)"
                elif i == 1 and j == 0:
                    label_text += "\n(FN)"
                else:
                    label_text += "\n(TP)"
                
                ax.text(j, i, label_text,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_latency_distribution(
        self,
        events: List[DetectionEvent],
        title: str = "检测延迟分布",
        filename: str = "latency_distribution.png"
    ) -> Path:
        """
        生成延迟分布图
        
        Args:
            events: 检测事件列表
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        # 提取正样本的检测延迟
        latencies = [e.detection_time_ms for e in events if e.label == 1 and e.detected]
        
        if not latencies:
            # 没有检测到的正样本，生成空图
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '无有效检测延迟数据', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1 = axes[0]
        ax1.hist(latencies, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(np.mean(latencies), color='red', linestyle='--', label=f'均值: {np.mean(latencies):.1f}ms')
        ax1.axvline(np.percentile(latencies, 50), color='orange', linestyle='--', label=f'P50: {np.percentile(latencies, 50):.1f}ms')
        ax1.axvline(np.percentile(latencies, 90), color='green', linestyle='--', label=f'P90: {np.percentile(latencies, 90):.1f}ms')
        ax1.set_xlabel('检测延迟 (ms)')
        ax1.set_ylabel('频次')
        ax1.set_title('检测延迟直方图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2 = axes[1]
        bp = ax2.boxplot(latencies, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.set_ylabel('检测延迟 (ms)')
        ax2.set_title('检测延迟箱线图')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f"样本数: {len(latencies)}\n"
        stats_text += f"最小值: {np.min(latencies):.1f}ms\n"
        stats_text += f"最大值: {np.max(latencies):.1f}ms\n"
        stats_text += f"均值: {np.mean(latencies):.1f}ms\n"
        stats_text += f"标准差: {np.std(latencies):.1f}ms"
        ax2.text(1.3, np.mean(latencies), stats_text, fontsize=10,
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_rtf_comparison(
        self,
        events: List[DetectionEvent],
        title: str = "RTF 分布",
        filename: str = "rtf_distribution.png"
    ) -> Path:
        """
        生成 RTF 分布图
        
        Args:
            events: 检测事件列表
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        # 计算每个样本的 RTF
        rtfs = []
        for e in events:
            if e.audio_duration_ms > 0:
                rtf = e.inference_time_ms / e.audio_duration_ms
                rtfs.append(rtf)
        
        if not rtfs:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '无有效 RTF 数据', ha='center', va='center', fontsize=14)
            ax.set_title(title)
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return output_path
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 直方图
        ax.hist(rtfs, bins=30, edgecolor='black', alpha=0.7, color='coral')
        
        # 添加参考线
        ax.axvline(1.0, color='red', linestyle='-', linewidth=2, label='实时阈值 (RTF=1.0)')
        ax.axvline(np.mean(rtfs), color='blue', linestyle='--', label=f'均值: {np.mean(rtfs):.4f}')
        ax.axvline(np.percentile(rtfs, 99), color='green', linestyle='--', label=f'P99: {np.percentile(rtfs, 99):.4f}')
        
        ax.set_xlabel('实时因子 (RTF)')
        ax.set_ylabel('频次')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置 x 轴范围
        ax.set_xlim(0, min(max(rtfs) * 1.1, 2.0))
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_score_distribution(
        self,
        events: List[DetectionEvent],
        title: str = "MLP 分数分布",
        filename: str = "score_distribution.png"
    ) -> Path:
        """
        生成 MLP 分数分布图
        
        Args:
            events: 检测事件列表
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        # 分离正负样本的 MLP 分数
        positive_scores = [e.mlp_score for e in events if e.label == 1 and e.v3_triggered]
        negative_scores = [e.mlp_score for e in events if e.label == 0 and e.v3_triggered]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.linspace(0, 1, 21)
        
        if positive_scores:
            ax.hist(positive_scores, bins=bins, alpha=0.6, label=f'正样本 (n={len(positive_scores)})', color='green')
        if negative_scores:
            ax.hist(negative_scores, bins=bins, alpha=0.6, label=f'负样本 (n={len(negative_scores)})', color='red')
        
        # 添加阈值线
        ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='阈值 (0.5)')
        
        ax.set_xlabel('MLP 置信度分数')
        ax.set_ylabel('频次')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_summary_dashboard(
        self,
        metrics: EvaluationMetrics,
        title: str = "评估摘要",
        filename: str = "summary_dashboard.png"
    ) -> Path:
        """
        生成摘要仪表板
        
        Args:
            metrics: 评估指标
            title: 图表标题
            filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 检测指标条形图
        ax1 = axes[0, 0]
        metric_names = ['准确率', '精确率', '召回率', 'F1']
        metric_values = [metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score]
        colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']
        
        bars = ax1.bar(metric_names, [v * 100 for v in metric_values], color=colors)
        ax1.set_ylabel('百分比 (%)')
        ax1.set_title('检测指标')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val*100:.1f}%', ha='center', fontsize=10)
        
        # 2. FAR/FRR 对比
        ax2 = axes[0, 1]
        rate_names = ['FAR\n(误报率)', 'FRR\n(漏检率)']
        rate_values = [metrics.far * 100, metrics.frr * 100]
        rate_targets = [metrics.far_target * 100, metrics.frr_target * 100]
        
        x = np.arange(len(rate_names))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, rate_values, width, label='实际值', color='tomato')
        bars2 = ax2.bar(x + width/2, rate_targets, width, label='目标值', color='lightgreen')
        
        ax2.set_ylabel('百分比 (%)')
        ax2.set_title('误报率 / 漏检率')
        ax2.set_xticks(x)
        ax2.set_xticklabels(rate_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加达标标记
        for i, (val, target) in enumerate(zip(rate_values, rate_targets)):
            status = "✓" if val < target else "✗"
            ax2.text(i - width/2, val + 1, f'{val:.1f}%\n{status}', ha='center', fontsize=9)
        
        # 3. RTF 指标
        ax3 = axes[1, 0]
        rtf_names = ['总体\nRTF', '平均\nRTF', 'P99\nRTF']
        rtf_values = [metrics.rtf_stats.overall_rtf, metrics.rtf_stats.mean_rtf, metrics.rtf_stats.p99_rtf]
        
        bars = ax3.bar(rtf_names, rtf_values, color='teal')
        ax3.axhline(1.0, color='red', linestyle='--', linewidth=2, label='实时阈值')
        ax3.set_ylabel('RTF')
        ax3.set_title('实时因子 (RTF < 1.0 为实时)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, rtf_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.4f}', ha='center', fontsize=10)
        
        # 4. 总体判定
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 创建判定表格
        checks = [
            ('FAR < 10%', metrics.meets_far_target, f'{metrics.far*100:.2f}%'),
            ('FRR < 5%', metrics.meets_frr_target, f'{metrics.frr*100:.2f}%'),
            ('RTF < 1.0', metrics.meets_rtf_target, f'{metrics.rtf_stats.overall_rtf:.4f}'),
        ]
        
        table_data = []
        for name, passed, value in checks:
            status = '✓ 达标' if passed else '✗ 未达标'
            color = 'green' if passed else 'red'
            table_data.append([name, value, status])
        
        table = ax4.table(
            cellText=table_data,
            colLabels=['指标', '实际值', '状态'],
            loc='center',
            cellLoc='center',
            colWidths=[0.3, 0.3, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # 设置表格样式
        for i in range(len(checks)):
            if checks[i][1]:
                table[(i+1, 2)].set_facecolor('lightgreen')
            else:
                table[(i+1, 2)].set_facecolor('lightcoral')
        
        # 总体判定
        overall_status = "✓ 全部达标！" if metrics.meets_all_targets else "✗ 部分指标未达标"
        overall_color = 'green' if metrics.meets_all_targets else 'red'
        ax4.text(0.5, 0.1, overall_status, ha='center', va='center', fontsize=16,
                 fontweight='bold', color=overall_color,
                 transform=ax4.transAxes)
        
        ax4.set_title('达标判定', fontsize=14, fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_full_report(
        self,
        events: List[DetectionEvent],
        metrics: EvaluationMetrics,
        config: Dict,
        prefix: str = ""
    ) -> Dict[str, Path]:
        """
        生成完整报告
        
        Args:
            events: 检测事件列表
            metrics: 评估指标
            config: 配置信息
            prefix: 文件名前缀
            
        Returns:
            生成的文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            prefix = f"{prefix}_"
        
        outputs = {}
        
        # 生成各类图表
        outputs["confusion_matrix"] = self.generate_confusion_matrix(
            events,
            title="流式 KWS 混淆矩阵",
            filename=f"{prefix}confusion_matrix_{timestamp}.png"
        )
        
        outputs["latency"] = self.generate_latency_distribution(
            events,
            title="检测延迟分布",
            filename=f"{prefix}latency_distribution_{timestamp}.png"
        )
        
        outputs["rtf"] = self.generate_rtf_comparison(
            events,
            title="RTF 分布",
            filename=f"{prefix}rtf_distribution_{timestamp}.png"
        )
        
        outputs["score"] = self.generate_score_distribution(
            events,
            title="MLP 分数分布",
            filename=f"{prefix}score_distribution_{timestamp}.png"
        )
        
        outputs["dashboard"] = self.generate_summary_dashboard(
            metrics,
            title="流式 KWS 评估摘要",
            filename=f"{prefix}summary_dashboard_{timestamp}.png"
        )
        
        print(f"\n已生成报告图表:")
        for name, path in outputs.items():
            print(f"  - {name}: {path}")
        
        return outputs


def generate_report_from_json(json_path: str, output_dir: str = None) -> Dict[str, Path]:
    """
    从 JSON 结果文件生成报告
    
    Args:
        json_path: JSON 结果文件路径
        output_dir: 输出目录（默认与 JSON 文件同目录）
        
    Returns:
        生成的文件路径字典
    """
    json_path = Path(json_path)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 重建事件列表
    events = []
    for e_data in data.get("events", []):
        events.append(DetectionEvent(
            audio_path=e_data["audio_path"],
            label=e_data["label"],
            detected=e_data["detected"],
            confidence=e_data.get("confidence", 0),
            detection_time_ms=e_data.get("detection_time_ms", 0),
            inference_time_ms=e_data.get("inference_time_ms", 0),
            audio_duration_ms=e_data.get("audio_duration_ms", 0),
            v3_triggered=e_data.get("v3_triggered", False),
            v3_score=e_data.get("v3_score", 0),
            mlp_score=e_data.get("mlp_score", 0)
        ))
    
    # 重建指标
    metrics_data = data.get("metrics", {})
    from metrics import LatencyStats, RTFStats
    
    metrics = EvaluationMetrics(
        total_samples=metrics_data.get("samples", {}).get("total", 0),
        positive_samples=metrics_data.get("samples", {}).get("positive", 0),
        negative_samples=metrics_data.get("samples", {}).get("negative", 0),
        tp=metrics_data.get("confusion_matrix", {}).get("tp", 0),
        tn=metrics_data.get("confusion_matrix", {}).get("tn", 0),
        fp=metrics_data.get("confusion_matrix", {}).get("fp", 0),
        fn=metrics_data.get("confusion_matrix", {}).get("fn", 0),
        accuracy=metrics_data.get("detection", {}).get("accuracy", 0),
        precision=metrics_data.get("detection", {}).get("precision", 0),
        recall=metrics_data.get("detection", {}).get("recall", 0),
        f1_score=metrics_data.get("detection", {}).get("f1_score", 0),
        far=metrics_data.get("error_rates", {}).get("far", 0),
        frr=metrics_data.get("error_rates", {}).get("frr", 0),
        latency_stats=LatencyStats(**metrics_data.get("latency", {})),
        rtf_stats=RTFStats(**metrics_data.get("rtf", {})),
        far_target=metrics_data.get("targets", {}).get("far_target", 0.1),
        frr_target=metrics_data.get("targets", {}).get("frr_target", 0.05),
        rtf_target=metrics_data.get("targets", {}).get("rtf_target", 1.0),
        meets_far_target=metrics_data.get("targets", {}).get("meets_far", False),
        meets_frr_target=metrics_data.get("targets", {}).get("meets_frr", False),
        meets_rtf_target=metrics_data.get("targets", {}).get("meets_rtf", False),
        meets_all_targets=metrics_data.get("targets", {}).get("meets_all", False),
    )
    
    # 生成报告
    output_dir = Path(output_dir) if output_dir else json_path.parent
    generator = ReportGenerator(output_dir)
    
    return generator.generate_full_report(
        events=events,
        metrics=metrics,
        config=data.get("config", {}),
        prefix="streaming"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成评估报告")
    parser.add_argument("--json", type=str, required=True, help="JSON 结果文件路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()
    
    outputs = generate_report_from_json(args.json, args.output_dir)
    print("\n报告生成完成！")
