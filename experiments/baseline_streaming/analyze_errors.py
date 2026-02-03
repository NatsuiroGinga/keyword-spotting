#!/usr/bin/env python3
"""
V4模型详细错误分析
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from streaming_evaluator import StreamingKWSEvaluator


def main():
    base_dir = Path(__file__).parent.parent.parent
    
    test_dir = base_dir / "experiments/baseline_streaming/data_splits/test"
    model_dir = base_dir / "experiments/baseline_streaming/exp_v4"
    
    print("=" * 70)
    print("V4模型错误分析")
    print("=" * 70)
    
    evaluator = StreamingKWSEvaluator(
        model_dir=str(model_dir),
        keywords_threshold=0.50,  # 使用最佳阈值
        keywords_score=1.5,
    )
    
    metrics, results = evaluator.evaluate_dataset(str(test_dir), verbose=False)
    
    # 分析误报（FP）
    fp_samples = [r for r in results if r.label == 0 and r.detected]
    fn_samples = [r for r in results if r.label == 1 and not r.detected]
    tp_samples = [r for r in results if r.label == 1 and r.detected]
    tn_samples = [r for r in results if r.label == 0 and not r.detected]
    
    print(f"\n阈值=0.50时:")
    print(f"  TP: {len(tp_samples)}, TN: {len(tn_samples)}")
    print(f"  FP: {len(fp_samples)}, FN: {len(fn_samples)}")
    print(f"  FAR: {len(fp_samples)/(len(fp_samples)+len(tn_samples))*100:.2f}%")
    print(f"  FRR: {len(fn_samples)/(len(fn_samples)+len(tp_samples))*100:.2f}%")
    
    print("\n【误报样本 (FP) - 本应拒绝但误触发】")
    print("-" * 70)
    for r in fp_samples:
        print(f"  {r.text:<30} ({r.file})")
    
    print("\n【漏检样本 (FN) - 本应检测但未触发】")
    print("-" * 70)
    for r in fn_samples:
        print(f"  {r.text:<30} ({r.file})")
    
    print("\n【正确唤醒样本 (TP)】")
    print("-" * 70)
    for r in tp_samples[:5]:
        print(f"  {r.text:<30} ({r.file})")
    if len(tp_samples) > 5:
        print(f"  ... 还有 {len(tp_samples)-5} 个")
    
    # 使用阈值0.55再测试一次
    print("\n" + "=" * 70)
    print("阈值=0.55时的错误分析")
    print("=" * 70)
    
    evaluator2 = StreamingKWSEvaluator(
        model_dir=str(model_dir),
        keywords_threshold=0.55,
        keywords_score=1.5,
    )
    
    metrics2, results2 = evaluator2.evaluate_dataset(str(test_dir), verbose=False)
    
    fp_samples2 = [r for r in results2 if r.label == 0 and r.detected]
    fn_samples2 = [r for r in results2 if r.label == 1 and not r.detected]
    
    print(f"\n  FAR: {metrics2.far*100:.2f}%, FRR: {metrics2.frr*100:.2f}%")
    
    print("\n【误报样本 (FP)】")
    for r in fp_samples2:
        print(f"  {r.text:<30} ({r.file})")
    
    print("\n【漏检样本 (FN)】")
    for r in fn_samples2:
        print(f"  {r.text:<30} ({r.file})")


if __name__ == "__main__":
    main()
