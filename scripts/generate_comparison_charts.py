#!/usr/bin/env python3
"""
Generate comparison charts for KWS inference mode evaluation.

Creates visualizations comparing Direct Inference vs Delayed Decision modes.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_comparison_results(json_path: str) -> Dict[str, Any]:
    """Load comparison results from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_detection_metrics_chart(
    direct_metrics: Dict[str, Any],
    delayed_metrics: Dict[str, Any],
    output_path: str,
):
    """Create bar chart comparing detection metrics."""
    metrics = ["FRR", "FAR", "Recall", "Specificity", "Accuracy", "Precision", "F1"]
    direct_values = [
        direct_metrics["frr_percent"],
        direct_metrics["far_percent"],
        direct_metrics["recall_percent"],
        direct_metrics["specificity_percent"],
        direct_metrics["accuracy_percent"],
        direct_metrics["precision_percent"],
        direct_metrics["f1_score"],
    ]
    delayed_values = [
        delayed_metrics["frr_percent"],
        delayed_metrics["far_percent"],
        delayed_metrics["recall_percent"],
        delayed_metrics["specificity_percent"],
        delayed_metrics["accuracy_percent"],
        delayed_metrics["precision_percent"],
        delayed_metrics["f1_score"],
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, direct_values, width, label='Direct Inference', color='#2196F3')
    bars2 = ax.bar(x + width/2, delayed_values, width, label='Delayed Decision', color='#4CAF50')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Detection Metrics Comparison: Direct vs Delayed Decision')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 110)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_rtf_comparison_chart(
    direct_rtf: Dict[str, Any],
    delayed_rtf: Dict[str, Any],
    output_path: str,
):
    """Create bar chart comparing RTF metrics."""
    metrics = ["Overall", "Mean", "Median", "P95", "P99", "Min", "Max"]
    direct_values = [
        direct_rtf["rtf_overall"],
        direct_rtf["rtf_mean"],
        direct_rtf["rtf_median"],
        direct_rtf["rtf_p95"],
        direct_rtf["rtf_p99"],
        direct_rtf["rtf_min"],
        direct_rtf["rtf_max"],
    ]
    delayed_values = [
        delayed_rtf["rtf_overall"],
        delayed_rtf["rtf_mean"],
        delayed_rtf["rtf_median"],
        delayed_rtf["rtf_p95"],
        delayed_rtf["rtf_p99"],
        delayed_rtf["rtf_min"],
        delayed_rtf["rtf_max"],
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, direct_values, width, label='Direct Inference', color='#2196F3')
    bars2 = ax.bar(x + width/2, delayed_values, width, label='Delayed Decision', color='#4CAF50')
    
    # Add real-time threshold line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Real-time threshold (RTF=1.0)')
    
    ax.set_xlabel('RTF Metrics')
    ax.set_ylabel('RTF Value')
    ax.set_title('RTF (Real-Time Factor) Comparison: Direct vs Delayed Decision')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_confusion_matrix_chart(
    direct_metrics: Dict[str, Any],
    delayed_metrics: Dict[str, Any],
    output_path: str,
):
    """Create confusion matrix comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Direct inference confusion matrix
    direct_cm = np.array([
        [direct_metrics["true_positive"], direct_metrics["false_negative"]],
        [direct_metrics["false_positive"], direct_metrics["true_negative"]]
    ])
    
    # Delayed decision confusion matrix
    delayed_cm = np.array([
        [delayed_metrics["true_positive"], delayed_metrics["false_negative"]],
        [delayed_metrics["false_positive"], delayed_metrics["true_negative"]]
    ])
    
    labels = ["Positive", "Negative"]
    
    for ax, cm, title in [
        (axes[0], direct_cm, "Direct Inference"),
        (axes[1], delayed_cm, "Delayed Decision")
    ]:
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Detected", "Not Detected"])
        ax.set_yticklabels(["Actual Positive", "Actual Negative"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, str(cm[i, j]),
                              ha="center", va="center", color="black", fontsize=14)
        
        fig.colorbar(im, ax=ax)
    
    plt.suptitle("Confusion Matrix Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_dashboard(
    results: Dict[str, Any],
    output_path: str,
):
    """Create a summary dashboard with all key metrics."""
    fig = plt.figure(figsize=(16, 10))
    
    direct = results["direct_inference"]
    delayed = results["delayed_decision"]
    comparison = results["comparison"]
    config = results["config"]
    
    # Title
    fig.suptitle(f"KWS Inference Mode Comparison Dashboard\n"
                 f"Model: {Path(config['model_dir']).name} | "
                 f"Samples: {config['positive_count']} pos + {config['negative_count']} neg",
                 fontsize=14, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. Detection metrics comparison (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ["FRR", "FAR", "Recall", "Precision", "F1"]
    direct_vals = [
        direct["detection_metrics"]["frr_percent"],
        direct["detection_metrics"]["far_percent"],
        direct["detection_metrics"]["recall_percent"],
        direct["detection_metrics"]["precision_percent"],
        direct["detection_metrics"]["f1_score"],
    ]
    delayed_vals = [
        delayed["detection_metrics"]["frr_percent"],
        delayed["detection_metrics"]["far_percent"],
        delayed["detection_metrics"]["recall_percent"],
        delayed["detection_metrics"]["precision_percent"],
        delayed["detection_metrics"]["f1_score"],
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width/2, direct_vals, width, label='Direct', color='#2196F3')
    ax1.bar(x + width/2, delayed_vals, width, label='Delayed', color='#4CAF50')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Detection Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 110)
    
    # 2. RTF comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    rtf_labels = ['Direct', 'Delayed']
    rtf_values = [
        direct["rtf_metrics"]["rtf_overall"],
        delayed["rtf_metrics"]["rtf_overall"]
    ]
    colors = ['#2196F3', '#4CAF50']
    bars = ax2.bar(rtf_labels, rtf_values, color=colors)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Real-time (RTF=1)')
    ax2.set_ylabel('RTF')
    ax2.set_title('Real-Time Factor')
    ax2.legend()
    for bar, val in zip(bars, rtf_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Detection counts (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    count_labels = ['TP', 'FN', 'TN', 'FP']
    direct_counts = [
        direct["detection_metrics"]["true_positive"],
        direct["detection_metrics"]["false_negative"],
        direct["detection_metrics"]["true_negative"],
        direct["detection_metrics"]["false_positive"],
    ]
    delayed_counts = [
        delayed["detection_metrics"]["true_positive"],
        delayed["detection_metrics"]["false_negative"],
        delayed["detection_metrics"]["true_negative"],
        delayed["detection_metrics"]["false_positive"],
    ]
    
    x = np.arange(len(count_labels))
    width = 0.35
    ax3.bar(x - width/2, direct_counts, width, label='Direct', color='#2196F3')
    ax3.bar(x + width/2, delayed_counts, width, label='Delayed', color='#4CAF50')
    ax3.set_ylabel('Count')
    ax3.set_title('Detection Counts')
    ax3.set_xticks(x)
    ax3.set_xticklabels(count_labels)
    ax3.legend()
    
    # 4. Processing time (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    time_labels = ['Audio Duration', 'Process Time']
    direct_times = [
        direct["rtf_metrics"]["total_audio_duration_sec"],
        direct["rtf_metrics"]["total_process_time_sec"],
    ]
    delayed_times = [
        delayed["rtf_metrics"]["total_audio_duration_sec"],
        delayed["rtf_metrics"]["total_process_time_sec"],
    ]
    
    x = np.arange(len(time_labels))
    width = 0.35
    ax4.bar(x - width/2, direct_times, width, label='Direct', color='#2196F3')
    ax4.bar(x + width/2, delayed_times, width, label='Delayed', color='#4CAF50')
    ax4.set_ylabel('Seconds')
    ax4.set_title('Processing Time')
    ax4.set_xticks(x)
    ax4.set_xticklabels(time_labels)
    ax4.legend()
    
    # 5. RTF distribution (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    rtf_metrics = ['Mean', 'Median', 'P95', 'P99']
    direct_rtf = [
        direct["rtf_metrics"]["rtf_mean"],
        direct["rtf_metrics"]["rtf_median"],
        direct["rtf_metrics"]["rtf_p95"],
        direct["rtf_metrics"]["rtf_p99"],
    ]
    delayed_rtf = [
        delayed["rtf_metrics"]["rtf_mean"],
        delayed["rtf_metrics"]["rtf_median"],
        delayed["rtf_metrics"]["rtf_p95"],
        delayed["rtf_metrics"]["rtf_p99"],
    ]
    
    x = np.arange(len(rtf_metrics))
    width = 0.35
    ax5.bar(x - width/2, direct_rtf, width, label='Direct', color='#2196F3')
    ax5.bar(x + width/2, delayed_rtf, width, label='Delayed', color='#4CAF50')
    ax5.set_ylabel('RTF')
    ax5.set_title('RTF Distribution')
    ax5.set_xticks(x)
    ax5.set_xticklabels(rtf_metrics)
    ax5.legend()
    
    # 6. Summary text (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                                    EVALUATION SUMMARY                                 ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║  Configuration:                                                                       ║
    ║    • Prefix timeout: {config['prefix_timeout_ms']}ms    • Chunk size: {config['chunk_size_ms']}ms                                        ║
    ║    • Test samples: {config['positive_count']} positive + {config['negative_count']} negative = {config['positive_count'] + config['negative_count']} total                          ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║  Comparison Results:                                                                  ║
    ║    • FRR difference: {comparison['frr_diff']:+.2f}%    • FAR difference: {comparison['far_diff']:+.2f}%                              ║
    ║    • RTF difference: {comparison['rtf_diff']:+.4f}                                                           ║
    ║    • FAR improved: {'✓ Yes' if comparison['far_improved'] else '✗ No'}    • FRR improved: {'✓ Yes' if comparison['frr_improved'] else '✗ No'}                                    ║
    ║    • Real-time capable: {'✓ Yes' if comparison['rtf_acceptable'] else '✗ No'}                                                            ║
    ╠══════════════════════════════════════════════════════════════════════════════════════╣
    ║  Recommendation:                                                                      ║
    ║    {'Both modes perform similarly. Choose based on latency requirements.' if comparison['frr_diff'] == 0 and comparison['far_diff'] == 0 else 'Use delayed decision mode for reduced false positives.' if comparison['far_improved'] else 'Use direct mode for lower latency.'}                  ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison charts for KWS evaluation"
    )
    parser.add_argument(
        "--results-json",
        type=str,
        required=True,
        help="Path to comparison results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for charts",
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_comparison_results(str(results_path))
    
    direct = results["direct_inference"]
    delayed = results["delayed_decision"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate charts
    print("Generating comparison charts...")
    
    # 1. Detection metrics chart
    create_detection_metrics_chart(
        direct["detection_metrics"],
        delayed["detection_metrics"],
        str(output_dir / f"detection_metrics_{timestamp}.png"),
    )
    
    # 2. RTF comparison chart
    create_rtf_comparison_chart(
        direct["rtf_metrics"],
        delayed["rtf_metrics"],
        str(output_dir / f"rtf_comparison_{timestamp}.png"),
    )
    
    # 3. Confusion matrix chart
    create_confusion_matrix_chart(
        direct["detection_metrics"],
        delayed["detection_metrics"],
        str(output_dir / f"confusion_matrix_{timestamp}.png"),
    )
    
    # 4. Summary dashboard
    create_summary_dashboard(
        results,
        str(output_dir / f"summary_dashboard_{timestamp}.png"),
    )
    
    print(f"\nAll charts saved to: {output_dir}")


if __name__ == "__main__":
    main()
