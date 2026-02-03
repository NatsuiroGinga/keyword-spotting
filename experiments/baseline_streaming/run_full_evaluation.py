#!/usr/bin/env python3
"""
全406样本综合评估
在训练、验证、测试三个子集上分别评估，并生成最终报告
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from streaming_evaluator import StreamingKWSEvaluator, print_metrics, EvaluationMetrics


def main():
    base_dir = Path(__file__).parent.parent.parent
    
    # 数据目录
    data_splits = {
        "train": base_dir / "experiments/baseline_streaming/data_splits/train",
        "val": base_dir / "experiments/baseline_streaming/data_splits/val",
        "test": base_dir / "experiments/baseline_streaming/data_splits/test",
        "all": base_dir / "data/all",
    }
    
    # 模型
    models = {
        "v3": {
            "name": "V3模型 (TTS训练)",
            "dir": base_dir / "exp/kws_finetune_v3",
        },
        "v4": {
            "name": "V4模型 (真实人声训练)",
            "dir": base_dir / "experiments/baseline_streaming/exp_v4",
        },
    }
    
    output_dir = base_dir / "experiments/baseline_streaming/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("KWS 全406样本综合评估报告")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # 对每个模型在每个数据集上测试
    for model_key, model_info in models.items():
        model_dir = model_info["dir"]
        model_name = model_info["name"]
        
        if not model_dir.exists() or not list(model_dir.glob("*.onnx")):
            print(f"跳过 {model_name}: 模型不存在")
            continue
        
        print(f"\n{'='*80}")
        print(f"模型: {model_name}")
        print(f"{'='*80}")
        
        results[model_key] = {}
        
        # 测试不同阈值
        thresholds = [0.25, 0.4, 0.5, 0.55]
        
        for threshold in thresholds:
            print(f"\n--- 阈值={threshold} ---")
            
            evaluator = StreamingKWSEvaluator(
                model_dir=str(model_dir),
                keywords_threshold=threshold,
                keywords_score=1.5,
            )
            
            for split_name, split_dir in data_splits.items():
                if not split_dir.exists():
                    continue
                
                wav_files = list(split_dir.glob("*.wav"))
                if not wav_files:
                    continue
                
                try:
                    metrics, details = evaluator.evaluate_dataset(str(split_dir), verbose=False)
                    
                    passed = metrics.far < 0.1 and metrics.frr < 0.05 and metrics.avg_rtf < 1.0
                    status = "✓" if passed else "✗"
                    
                    print(f"  {split_name:>5}: FAR={metrics.far*100:>6.2f}%, "
                          f"FRR={metrics.frr*100:>5.2f}%, "
                          f"F1={metrics.f1*100:>6.2f}%, "
                          f"RTF={metrics.avg_rtf:.4f} {status} "
                          f"(n={len(wav_files)})")
                    
                    key = f"{model_key}_t{threshold}_{split_name}"
                    results[model_key][key] = {
                        "model": model_name,
                        "threshold": threshold,
                        "split": split_name,
                        "samples": len(wav_files),
                        "far": metrics.far,
                        "frr": metrics.frr,
                        "accuracy": metrics.accuracy,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                        "avg_rtf": metrics.avg_rtf,
                        "max_rtf": metrics.max_rtf,
                        "tp": metrics.tp,
                        "tn": metrics.tn,
                        "fp": metrics.fp,
                        "fn": metrics.fn,
                        "passed": passed,
                    }
                    
                except Exception as e:
                    print(f"  {split_name}: 错误 - {e}")
    
    # 生成汇总表格
    print("\n" + "=" * 100)
    print("性能汇总表格")
    print("=" * 100)
    
    # V3 vs V4 在全部406样本上的对比
    print("\n【全406样本对比 (data/all)】")
    print("-" * 100)
    print(f"{'模型':<25} {'阈值':>6} {'FAR':>8} {'FRR':>8} {'准确率':>8} {'F1':>8} {'RTF':>8} {'达标':>6}")
    print("-" * 100)
    
    for model_key in ['v3', 'v4']:
        if model_key not in results:
            continue
        for key, r in sorted(results[model_key].items()):
            if "_all" in key:
                passed = "✓" if r['passed'] else "✗"
                model_name = r['model'][:20] + "..." if len(r['model']) > 20 else r['model']
                print(f"{model_name:<25} {r['threshold']:>6.2f} "
                      f"{r['far']*100:>7.2f}% {r['frr']*100:>7.2f}% "
                      f"{r['accuracy']*100:>7.2f}% {r['f1']*100:>7.2f}% "
                      f"{r['avg_rtf']:>7.4f} {passed:>6}")
    
    # 保存详细结果
    result_file = output_dir / f"full_evaluation_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {result_file}")
    
    # 最终结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    # 找出最佳配置
    best_config = None
    best_f1 = 0
    
    for model_key in results:
        for key, r in results[model_key].items():
            if "_all" in key and r['far'] < 0.1 and r['frr'] < 0.05:
                if r['f1'] > best_f1:
                    best_f1 = r['f1']
                    best_config = r
    
    if best_config:
        print(f"✓ 找到达标配置:")
        print(f"  模型: {best_config['model']}")
        print(f"  阈值: {best_config['threshold']}")
        print(f"  FAR: {best_config['far']*100:.2f}%")
        print(f"  FRR: {best_config['frr']*100:.2f}%")
        print(f"  F1: {best_config['f1']*100:.2f}%")
    else:
        print("✗ 没有找到完全达标的配置（FAR<10% 且 FRR<5%）")
        
        # 找最接近达标的配置
        closest = None
        min_gap = float('inf')
        
        for model_key in results:
            for key, r in results[model_key].items():
                if "_all" in key:
                    gap = max(0, r['far'] - 0.1) + max(0, r['frr'] - 0.05)
                    if gap < min_gap:
                        min_gap = gap
                        closest = r
        
        if closest:
            print(f"\n最接近达标的配置:")
            print(f"  模型: {closest['model']}")
            print(f"  阈值: {closest['threshold']}")
            print(f"  FAR: {closest['far']*100:.2f}% (目标<10%)")
            print(f"  FRR: {closest['frr']*100:.2f}% (目标<5%)")
            print(f"  F1: {closest['f1']*100:.2f}%")


if __name__ == "__main__":
    main()
