#!/usr/bin/env python3
"""分析406样本数据集的类别分布"""

import os
from pathlib import Path
from collections import defaultdict
import re
import json

def analyze_dataset(data_dir: str = "data/all"):
    """分析数据集类别分布
    
    正样本定义：与"你好真真"拼音和声调相同的词语
    - 你好真真 (ni3 hao3 zhen1 zhen1)
    - 你好珍珍 (同音)
    - 你好甄甄 (同音)
    - 你好臻臻 (同音)
    """
    data_path = Path(data_dir)
    files = list(data_path.glob("*.wav"))
    
    print(f"=== 406样本数据集分析 ===")
    print(f"数据目录: {data_path.absolute()}")
    print(f"总文件数: {len(files)}")
    print()
    
    # 正样本关键词（与"你好真真"同音的词）
    positive_keywords = [
        "你好真真",
        "你好珍珍",
        "你好甄甄", 
        "你好臻臻",
        "你好桢桢",
    ]
    
    # 统计各类别
    categories = defaultdict(list)
    all_samples = []
    
    for f in files:
        name = f.stem
        # 去除时间戳前缀
        if "_" in name:
            text = name.split("_", 1)[1]
        else:
            # 处理没有下划线的旧格式（如20251212161558你好真真）
            text = re.sub(r"^\d+", "", name)
        
        sample = {
            "file": f.name,
            "text": text,
            "path": str(f.absolute()),
        }
        
        # 判断是否为正样本
        is_positive = False
        for kw in positive_keywords:
            if kw in text:
                is_positive = True
                sample["label"] = 1
                sample["category"] = f"positive_{kw}"
                categories[f"positive_{kw}"].append(f.name)
                break
        
        if not is_positive:
            sample["label"] = 0
            # 负样本细分
            if "你好" in text:
                sample["category"] = "negative_nihao_prefix"
                categories["negative_nihao_prefix"].append(f.name)
            elif "真真" in text or "珍珍" in text:
                sample["category"] = "negative_zhenzhen_suffix"
                categories["negative_zhenzhen_suffix"].append(f.name)
            else:
                sample["category"] = "negative_other"
                categories["negative_other"].append(f.name)
        
        all_samples.append(sample)
    
    # 打印统计
    print("=== 类别分布 ===")
    print()
    print("【正样本】(与\"你好真真\"同音):")
    positive_total = 0
    positive_samples = []
    for key in sorted(categories.keys()):
        if key.startswith("positive_"):
            count = len(categories[key])
            print(f"  {key.replace('positive_', '')}: {count}")
            positive_total += count
            positive_samples.extend(categories[key])
    print(f"  小计: {positive_total}")
    print()
    
    print("【负样本】:")
    negative_total = 0
    for key in sorted(categories.keys()):
        if key.startswith("negative_"):
            label = key.replace("negative_", "")
            count = len(categories[key])
            print(f"  {label}: {count}")
            negative_total += count
    print(f"  小计: {negative_total}")
    print()
    
    ratio = negative_total / max(positive_total, 1)
    print(f"正负样本比例: {positive_total}:{negative_total} = 1:{ratio:.2f}")
    print()
    
    # 打印正样本文件列表
    print("=== 正样本文件列表 ===")
    for i, f in enumerate(sorted(positive_samples), 1):
        print(f"  {i}. {f}")
    print()
    
    # 打印包含"你好"前缀的负样本
    print("=== 包含\"你好\"前缀的负样本（关键干扰样本）===")
    nihao_negatives = categories["negative_nihao_prefix"]
    for i, f in enumerate(sorted(nihao_negatives)[:20], 1):
        print(f"  {i}. {f}")
    if len(nihao_negatives) > 20:
        print(f"  ... 共 {len(nihao_negatives)} 个")
    
    # 保存分析结果
    result = {
        "total": len(files),
        "positive_count": positive_total,
        "negative_count": negative_total,
        "ratio": f"1:{ratio:.2f}",
        "categories": {k: len(v) for k, v in categories.items()},
        "samples": all_samples,
    }
    
    output_path = Path(__file__).parent / "dataset_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n分析结果已保存到: {output_path}")
    
    return result


if __name__ == "__main__":
    analyze_dataset()
