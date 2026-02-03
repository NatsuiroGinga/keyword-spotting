#!/usr/bin/env python3
"""
消融实验可视化：生成 FAR-FRR 曲线、ROC 曲线等
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

def load_results(results_dir: Path) -> dict:
    """加载最新的实验结果"""
    json_files = sorted(results_dir.glob("ablation_results_*.json"))
    if not json_files:
        raise FileNotFoundError("No results found")
    
    with open(json_files[-1], "r") as f:
        return json.load(f)


def plot_far_frr_tradeoff(results: dict, output_path: Path):
    """绘制 FAR-FRR 权衡曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'V1_Baseline': 'blue', 'V2_Negative': 'green', 'V3_Finetuned': 'red'}
    markers = {'V1_Baseline': 'o', 'V2_Negative': 's', 'V3_Finetuned': '^'}
    
    for model_name, model_results in results.items():
        fars = [r['far'] * 100 for r in model_results]
        frrs = [r['frr'] * 100 for r in model_results]
        thresholds = [r['threshold'] for r in model_results]
        
        color = colors.get(model_name, 'gray')
        marker = markers.get(model_name, 'x')
        
        ax.plot(fars, frrs, f'-{marker}', color=color, label=model_name, 
                markersize=8, linewidth=2)
        
        # 标注关键阈值点
        for i, (far, frr, th) in enumerate(zip(fars, frrs, thresholds)):
            if th in [0.3, 0.5, 0.7]:
                ax.annotate(f'{th}', (far, frr), textcoords="offset points",
                           xytext=(5, 5), fontsize=8)
    
    # 绘制目标区域
    ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='FRR Target (5%)')
    ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5, label='FAR Target (10%)')
    
    # 高亮目标区域
    ax.fill_between([0, 10], [0, 0], [5, 5], color='green', alpha=0.1)
    ax.text(5, 2.5, 'Target Zone', ha='center', va='center', fontsize=10, color='green')
    
    ax.set_xlabel('FAR (False Accept Rate) %', fontsize=12)
    ax.set_ylabel('FRR (False Reject Rate) %', fontsize=12)
    ax.set_title('FAR-FRR Tradeoff Curve (Ablation Study)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path / 'far_frr_tradeoff.png', dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'far_frr_tradeoff.png'}")


def plot_f1_vs_threshold(results: dict, output_path: Path):
    """绘制 F1 vs 阈值曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'V1_Baseline': 'blue', 'V2_Negative': 'green', 'V3_Finetuned': 'red'}
    
    for model_name, model_results in results.items():
        thresholds = [r['threshold'] for r in model_results]
        f1_scores = [r['f1'] * 100 for r in model_results]
        
        color = colors.get(model_name, 'gray')
        ax.plot(thresholds, f1_scores, '-o', color=color, label=model_name,
                markersize=6, linewidth=2)
        
        # 找到最优 F1
        best_idx = np.argmax(f1_scores)
        ax.scatter([thresholds[best_idx]], [f1_scores[best_idx]], 
                  color=color, s=150, marker='*', zorder=5)
        ax.annotate(f'Best: {f1_scores[best_idx]:.1f}%',
                   (thresholds[best_idx], f1_scores[best_idx]),
                   textcoords="offset points", xytext=(10, 5), fontsize=9)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1 Score %', fontsize=12)
    ax.set_title('F1 Score vs Threshold (Ablation Study)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path / 'f1_vs_threshold.png', dpi=150)
    plt.close()
    print(f"Saved: {output_path / 'f1_vs_threshold.png'}")


def plot_bar_comparison(results: dict, output_path: Path):
    """绘制最优配置对比柱状图"""
    
    # 找到每个模型的最优 F1 配置
    optimal_configs = {}
    for model_name, model_results in results.items():
        best_result = max(model_results, key=lambda x: x['f1'])
        optimal_configs[model_name] = best_result
    
    models = list(optimal_configs.keys())
    metrics = ['far', 'frr', 'precision', 'recall', 'f1']
    metric_names = ['FAR', 'FRR', 'Precision', 'Recall', 'F1']
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    colors = {'V1_Baseline': 'blue', 'V2_Negative': 'green', 'V3_Finetuned': 'red'}
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        values = [optimal_configs[m][metric] * 100 for m in models]
        bars = ax.bar(models, values, color=[colors.get(m, 'gray') for m in models])
        
        ax.set_ylabel(f'{metric_name} %')
        ax.set_title(metric_name)
        ax.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle('Optimal Configuration Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'bar_comparison.png'}")


def plot_confusion_matrix_comparison(results: dict, output_path: Path):
    """绘制混淆矩阵对比"""
    
    # 找到 FAR 接近 10% 的配置
    target_far = 0.10
    optimal_configs = {}
    
    for model_name, model_results in results.items():
        # 找到 FAR 最接近 10% 的配置
        best = min(model_results, key=lambda x: abs(x['far'] - target_far))
        optimal_configs[model_name] = best
    
    n_models = len(optimal_configs)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, config) in enumerate(optimal_configs.items()):
        ax = axes[idx]
        
        cm = np.array([
            [config['tn'], config['fp']],
            [config['fn'], config['tp']]
        ])
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # 添加数值
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, str(cm[i, j]),
                              ha="center", va="center", color="black", fontsize=14)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_name}\n(th={config["threshold"]:.2f}, FAR={config["far"]*100:.1f}%)')
    
    plt.suptitle('Confusion Matrix @ FAR≈10%', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'confusion_matrix.png'}")


def generate_summary_table(results: dict) -> str:
    """生成 Markdown 格式的汇总表格"""
    
    lines = []
    lines.append("## 消融实验结果汇总")
    lines.append("")
    lines.append("### 各模型在不同 FAR 水平下的 FRR")
    lines.append("")
    lines.append("| 模型 | FAR≈5% | FAR≈10% | FAR≈15% | FAR≈20% |")
    lines.append("|------|--------|---------|---------|---------|")
    
    for model_name, model_results in results.items():
        row = [model_name]
        for target_far in [0.05, 0.10, 0.15, 0.20]:
            # 找到最接近目标 FAR 的配置
            closest = min(model_results, key=lambda x: abs(x['far'] - target_far))
            if abs(closest['far'] - target_far) < 0.03:
                row.append(f"{closest['frr']*100:.1f}%")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    lines.append("### 最优 F1 配置对比")
    lines.append("")
    lines.append("| 模型 | 最优阈值 | FAR | FRR | F1 | RTF |")
    lines.append("|------|----------|-----|-----|----|----|")
    
    for model_name, model_results in results.items():
        best = max(model_results, key=lambda x: x['f1'])
        lines.append(f"| {model_name} | {best['threshold']:.2f} | "
                    f"{best['far']*100:.2f}% | {best['frr']*100:.2f}% | "
                    f"{best['f1']*100:.2f}% | {best['rtf']:.4f} |")
    
    return "\n".join(lines)


def main():
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "experiments/streaming_cnn_kws/ablation_results"
    output_dir = results_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading results...")
    results = load_results(results_dir)
    
    print("Generating visualizations...")
    plot_far_frr_tradeoff(results, output_dir)
    plot_f1_vs_threshold(results, output_dir)
    plot_bar_comparison(results, output_dir)
    plot_confusion_matrix_comparison(results, output_dir)
    
    print("\nGenerating summary table...")
    summary = generate_summary_table(results)
    print(summary)
    
    # 保存汇总
    with open(output_dir / "summary.md", "w") as f:
        f.write(summary)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
