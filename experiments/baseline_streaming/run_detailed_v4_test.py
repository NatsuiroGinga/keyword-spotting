#!/usr/bin/env python3
"""
V4模型详细阈值测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from streaming_evaluator import StreamingKWSEvaluator, print_metrics


def main():
    base_dir = Path(__file__).parent.parent.parent
    
    test_dir = base_dir / "experiments/baseline_streaming/data_splits/test"
    model_dir = base_dir / "experiments/baseline_streaming/exp_v4"
    
    print("=" * 70)
    print("V4模型详细阈值测试")
    print("=" * 70)
    print(f"测试数据: {test_dir}")
    print(f"样本数: {len(list(test_dir.glob('*.wav')))}")
    print()
    
    # 测试更细粒度的阈值
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    results = []
    
    for threshold in thresholds:
        try:
            evaluator = StreamingKWSEvaluator(
                model_dir=str(model_dir),
                keywords_threshold=threshold,
                keywords_score=1.5,
            )
            
            metrics, details = evaluator.evaluate_dataset(str(test_dir), verbose=False)
            
            passed = metrics.far < 0.1 and metrics.frr < 0.05 and metrics.avg_rtf < 1.0
            status = "✓" if passed else "✗"
            
            results.append({
                "threshold": threshold,
                "far": metrics.far,
                "frr": metrics.frr,
                "f1": metrics.f1,
                "rtf": metrics.avg_rtf,
                "passed": passed,
            })
            
            print(f"阈值={threshold:.2f}: FAR={metrics.far*100:>6.2f}%, "
                  f"FRR={metrics.frr*100:>5.2f}%, F1={metrics.f1*100:>6.2f}%, "
                  f"RTF={metrics.avg_rtf:.4f} {status}")
            
            if passed:
                print(f"\n*** 找到达标配置: threshold={threshold} ***")
                print_metrics(metrics, f"V4模型 (threshold={threshold})")
            
        except Exception as e:
            print(f"阈值={threshold}: 错误 - {e}")
    
    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    
    # 分析最佳配置
    passed_configs = [r for r in results if r['passed']]
    if passed_configs:
        print(f"达标配置数: {len(passed_configs)}")
        best = max(passed_configs, key=lambda x: x['f1'])
        print(f"推荐配置: threshold={best['threshold']}")
        print(f"  FAR={best['far']*100:.2f}%, FRR={best['frr']*100:.2f}%, F1={best['f1']*100:.2f}%")
    else:
        print("没有找到完全达标的配置")
        # 找最接近达标的
        best = min(results, key=lambda x: abs(x['far'] - 0.1) + x['frr'])
        print(f"最接近达标: threshold={best['threshold']}")
        print(f"  FAR={best['far']*100:.2f}% (目标<10%), FRR={best['frr']*100:.2f}%")


if __name__ == "__main__":
    main()
