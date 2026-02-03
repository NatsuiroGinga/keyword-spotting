#!/usr/bin/env python3
"""
全面性能对比测试

对比多个模型在流式场景下的性能：
1. 预训练模型（如有ONNX）
2. V3微调模型（TTS数据训练）
3. V4微调模型（真实人声数据训练）
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent))

from streaming_evaluator import StreamingKWSEvaluator, print_metrics, EvaluationMetrics


def run_comprehensive_comparison():
    """运行全面对比测试"""
    base_dir = Path(__file__).parent.parent.parent
    
    # 测试数据目录
    test_dir = base_dir / "experiments/baseline_streaming/data_splits/test"
    
    # 模型目录
    models = {
        "v3": {
            "name": "V3模型 (TTS数据)",
            "dir": base_dir / "exp/kws_finetune_v3",
        },
        "v4": {
            "name": "V4模型 (真实人声)",
            "dir": base_dir / "experiments/baseline_streaming/exp_v4",
        },
    }
    
    # 输出目录
    output_dir = base_dir / "experiments/baseline_streaming/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("KWS 模型全面对比测试")
    print("=" * 70)
    print(f"测试数据: {test_dir}")
    print(f"测试样本数: {len(list(test_dir.glob('*.wav')))}")
    print()
    
    results = {}
    
    # 测试各模型
    for model_key, model_info in models.items():
        model_dir = model_info["dir"]
        model_name = model_info["name"]
        
        print("\n" + "=" * 70)
        print(f"测试: {model_name}")
        print("=" * 70)
        
        # 检查模型是否存在
        onnx_files = list(model_dir.glob("*.onnx")) if model_dir.exists() else []
        
        if not onnx_files:
            print(f"  [跳过] 模型目录不存在或没有ONNX文件: {model_dir}")
            continue
        
        # 测试不同阈值
        thresholds = [0.25, 0.3, 0.4, 0.5]
        best_metrics = None
        best_threshold = None
        best_f1 = 0
        
        for threshold in thresholds:
            try:
                evaluator = StreamingKWSEvaluator(
                    model_dir=str(model_dir),
                    keywords_threshold=threshold,
                    keywords_score=1.5,
                )
                
                metrics, details = evaluator.evaluate_dataset(str(test_dir), verbose=False)
                
                print(f"\n  阈值={threshold}:")
                print(f"    FAR={metrics.far*100:.2f}%, FRR={metrics.frr*100:.2f}%, "
                      f"F1={metrics.f1*100:.2f}%, RTF={metrics.avg_rtf:.4f}")
                
                # 综合评分：考虑FAR和FRR的平衡
                score = metrics.f1  # 使用F1作为主要指标
                
                if score > best_f1:
                    best_f1 = score
                    best_metrics = metrics
                    best_threshold = threshold
                
                results[f"{model_key}_t{threshold}"] = {
                    "name": f"{model_name} (t={threshold})",
                    "threshold": threshold,
                    "far": metrics.far,
                    "frr": metrics.frr,
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "avg_rtf": metrics.avg_rtf,
                    "max_rtf": metrics.max_rtf,
                    "avg_latency_ms": metrics.avg_latency_ms,
                }
                
            except Exception as e:
                print(f"    错误: {e}")
        
        if best_metrics:
            print(f"\n  最佳配置: 阈值={best_threshold}")
            print_metrics(best_metrics, f"{model_name} (最佳)")
            
            results[f"{model_key}_best"] = {
                "name": f"{model_name} (最佳)",
                "threshold": best_threshold,
                "far": best_metrics.far,
                "frr": best_metrics.frr,
                "accuracy": best_metrics.accuracy,
                "precision": best_metrics.precision,
                "recall": best_metrics.recall,
                "f1": best_metrics.f1,
                "avg_rtf": best_metrics.avg_rtf,
                "max_rtf": best_metrics.max_rtf,
                "avg_latency_ms": best_metrics.avg_latency_ms,
            }
    
    # 保存结果
    result_file = output_dir / f"comparison_results_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印对比表格
    print("\n" + "=" * 80)
    print("模型对比总结")
    print("=" * 80)
    print()
    print(f"{'模型':<35} {'FAR':>8} {'FRR':>8} {'F1':>8} {'RTF':>8} {'达标':>6}")
    print("-" * 80)
    
    for key, r in sorted(results.items()):
        passed = r['far'] < 0.1 and r['frr'] < 0.05 and r['avg_rtf'] < 1.0
        status = "✓" if passed else "✗"
        print(f"{r['name']:<35} {r['far']*100:>7.2f}% {r['frr']*100:>7.2f}% "
              f"{r['f1']*100:>7.2f}% {r['avg_rtf']:>7.4f} {status:>6}")
    
    print()
    print(f"结果已保存到: {result_file}")
    print()
    
    # 总结
    print("=" * 80)
    print("关键发现")
    print("=" * 80)
    
    # 找出最佳配置
    best_overall = None
    best_score = -1
    
    for key, r in results.items():
        # 综合评分：FAR和FRR越低越好，F1越高越好
        if r['far'] < 0.1 and r['frr'] < 0.05:
            score = r['f1']
            if score > best_score:
                best_score = score
                best_overall = r
    
    if best_overall:
        print(f"✓ 推荐配置: {best_overall['name']}")
        print(f"  FAR={best_overall['far']*100:.2f}%, FRR={best_overall['frr']*100:.2f}%, "
              f"F1={best_overall['f1']*100:.2f}%")
    else:
        print("✗ 没有找到满足所有指标要求的配置")
        print("  建议: 考虑增加二阶段验证（MLP）或重新训练模型")


if __name__ == "__main__":
    run_comprehensive_comparison()
