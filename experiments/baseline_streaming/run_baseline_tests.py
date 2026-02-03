#!/usr/bin/env python3
"""
基线测试脚本

测试预训练模型和现有V3模型在测试集上的流式性能
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from streaming_evaluator import StreamingKWSEvaluator, print_metrics


def setup_pretrained_model_keywords(model_dir: Path) -> Path:
    """为预训练模型设置keywords.txt"""
    keywords_path = model_dir / "keywords.txt"
    
    # 创建关键词文件
    keyword_line = "n ǐ h ǎo zh ēn zh ēn @你好真真\n"
    with open(keywords_path, "w", encoding="utf-8") as f:
        f.write(keyword_line)
    
    return keywords_path


def run_baseline_tests():
    """运行基线测试"""
    base_dir = Path(__file__).parent.parent.parent
    
    # 数据目录
    test_dir = base_dir / "experiments/baseline_streaming/data_splits/test"
    
    # 模型目录
    pretrained_dir = base_dir / "icefall-kws-zipformer-wenetspeech-20240219/exp/kws"
    v3_dir = base_dir / "exp/kws_finetune_v3"
    
    # 输出目录
    output_dir = base_dir / "experiments/baseline_streaming/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("KWS 基线测试")
    print("=" * 70)
    print(f"测试数据: {test_dir}")
    print(f"测试样本数: {len(list(test_dir.glob('*.wav')))}")
    print()
    
    results = {}
    
    # 测试1: 预训练模型
    print("\n" + "=" * 70)
    print("测试 1: 预训练模型 (从未见过\"你好真真\")")
    print("=" * 70)
    
    # 检查是否有ONNX模型
    pretrained_onnx_files = list(pretrained_dir.glob("*.onnx")) if pretrained_dir.exists() else []
    
    if pretrained_onnx_files:
        # 确保keywords.txt存在
        setup_pretrained_model_keywords(pretrained_dir)
        
        evaluator = StreamingKWSEvaluator(
            model_dir=str(pretrained_dir),
            keywords_threshold=0.25,
            keywords_score=1.5,
        )
        
        metrics, details = evaluator.evaluate_dataset(str(test_dir), verbose=False)
        print_metrics(metrics, "预训练模型")
        
        results["pretrained"] = {
            "name": "预训练模型",
            "far": metrics.far,
            "frr": metrics.frr,
            "accuracy": metrics.accuracy,
            "f1": metrics.f1,
            "avg_rtf": metrics.avg_rtf,
        }
    else:
        print(f"  [跳过] 预训练模型没有ONNX文件，需要先导出")
        print(f"  提示: 可以运行 scripts/export/export_pretrained_onnx.sh 导出")
    
    # 测试2: V3微调模型
    print("\n" + "=" * 70)
    print("测试 2: V3微调模型 (在TTS数据上微调)")
    print("=" * 70)
    
    if v3_dir.exists():
        evaluator = StreamingKWSEvaluator(
            model_dir=str(v3_dir),
            keywords_threshold=0.25,
            keywords_score=1.5,
        )
        
        metrics, details = evaluator.evaluate_dataset(str(test_dir), verbose=False)
        print_metrics(metrics, "V3微调模型")
        
        results["v3"] = {
            "name": "V3微调模型",
            "far": metrics.far,
            "frr": metrics.frr,
            "accuracy": metrics.accuracy,
            "f1": metrics.f1,
            "avg_rtf": metrics.avg_rtf,
        }
    else:
        print(f"  [跳过] V3模型目录不存在: {v3_dir}")
    
    # 测试3: V3模型 + 不同阈值
    print("\n" + "=" * 70)
    print("测试 3: V3模型阈值调优")
    print("=" * 70)
    
    if v3_dir.exists():
        for threshold in [0.3, 0.35, 0.4, 0.5]:
            evaluator = StreamingKWSEvaluator(
                model_dir=str(v3_dir),
                keywords_threshold=threshold,
                keywords_score=1.5,
            )
            
            metrics, _ = evaluator.evaluate_dataset(str(test_dir), verbose=False)
            print(f"\n  阈值={threshold}:")
            print(f"    FAR={metrics.far*100:.2f}%, FRR={metrics.frr*100:.2f}%, "
                  f"Acc={metrics.accuracy*100:.2f}%, RTF={metrics.avg_rtf:.4f}")
            
            results[f"v3_threshold_{threshold}"] = {
                "name": f"V3 (threshold={threshold})",
                "threshold": threshold,
                "far": metrics.far,
                "frr": metrics.frr,
                "accuracy": metrics.accuracy,
                "f1": metrics.f1,
                "avg_rtf": metrics.avg_rtf,
            }
    
    # 保存结果
    result_file = output_dir / f"baseline_results_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印对比表格
    print("\n" + "=" * 70)
    print("基线测试结果对比")
    print("=" * 70)
    print()
    print(f"{'模型':<25} {'FAR':>8} {'FRR':>8} {'准确率':>8} {'F1':>8} {'RTF':>8}")
    print("-" * 70)
    for key, r in results.items():
        print(f"{r['name']:<25} {r['far']*100:>7.2f}% {r['frr']*100:>7.2f}% "
              f"{r['accuracy']*100:>7.2f}% {r['f1']*100:>7.2f}% {r['avg_rtf']:>7.4f}")
    print()
    print(f"结果已保存到: {result_file}")


if __name__ == "__main__":
    run_baseline_tests()
