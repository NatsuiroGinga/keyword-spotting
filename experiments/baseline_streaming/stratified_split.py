#!/usr/bin/env python3
"""
分层数据集划分模块

按类别分层划分数据集，确保：
1. 训练集、验证集、测试集的正负样本比例一致
2. 各细分类别（你好前缀、其他负样本等）在各子集中分布均匀
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


class StratifiedDatasetSplitter:
    """分层数据集划分器
    
    确保训练集、验证集、测试集的类别分布保持一致
    """
    
    # 正样本关键词（与"你好真真"同音的词）
    POSITIVE_KEYWORDS = [
        "你好真真",
        "你好珍珍",
        "你好甄甄",
        "你好臻臻",
        "你好桢桢",
    ]
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        """
        Args:
            data_dir: 原始数据目录
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "比例之和必须为1"
        
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def _extract_text_from_filename(self, filename: str) -> str:
        """从文件名提取文本内容"""
        import re
        stem = Path(filename).stem
        if "_" in stem:
            return stem.split("_", 1)[1]
        else:
            return re.sub(r"^\d+", "", stem)
    
    def _classify_sample(self, filename: str) -> Tuple[int, str]:
        """
        分类样本
        
        Returns:
            (label, category): label为0/1，category为细分类别
        """
        text = self._extract_text_from_filename(filename)
        
        # 检查是否为正样本
        for kw in self.POSITIVE_KEYWORDS:
            if kw in text:
                return (1, "positive")
        
        # 负样本细分
        if "你好" in text:
            return (0, "negative_nihao")
        elif "真真" in text or "珍珍" in text:
            return (0, "negative_zhenzhen")
        else:
            return (0, "negative_other")
    
    def load_and_classify(self) -> Dict[str, List[str]]:
        """加载数据并分类
        
        Returns:
            按类别分组的文件列表
        """
        files = list(self.data_dir.glob("*.wav"))
        
        categories = defaultdict(list)
        for f in files:
            label, category = self._classify_sample(f.name)
            categories[category].append(f.name)
        
        return dict(categories)
    
    def stratified_split(
        self,
        categories: Dict[str, List[str]],
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        分层划分数据集
        
        确保每个类别在训练集、验证集、测试集中的比例一致
        """
        train_files = []
        val_files = []
        test_files = []
        
        for category, files in categories.items():
            random.shuffle(files)
            n = len(files)
            
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            # 剩余的都给测试集
            
            train_files.extend(files[:n_train])
            val_files.extend(files[n_train:n_train + n_val])
            test_files.extend(files[n_train + n_val:])
        
        # 打乱顺序
        random.shuffle(train_files)
        random.shuffle(val_files)
        random.shuffle(test_files)
        
        return train_files, val_files, test_files
    
    def validate_split(
        self,
        categories: Dict[str, List[str]],
        train_files: List[str],
        val_files: List[str],
        test_files: List[str],
    ) -> Dict:
        """验证划分结果的平衡性"""
        def count_by_category(files: List[str]) -> Dict[str, int]:
            counts = defaultdict(int)
            for f in files:
                _, category = self._classify_sample(f)
                counts[category] += 1
            return dict(counts)
        
        train_counts = count_by_category(train_files)
        val_counts = count_by_category(val_files)
        test_counts = count_by_category(test_files)
        
        # 计算各子集中正负样本比例
        def get_ratio(counts: Dict[str, int]) -> float:
            positive = counts.get("positive", 0)
            negative = sum(v for k, v in counts.items() if k != "positive")
            return positive / negative if negative > 0 else 0
        
        train_ratio = get_ratio(train_counts)
        val_ratio = get_ratio(val_counts)
        test_ratio = get_ratio(test_counts)
        
        return {
            "train": {
                "total": len(train_files),
                "counts": train_counts,
                "pos_neg_ratio": train_ratio,
            },
            "val": {
                "total": len(val_files),
                "counts": val_counts,
                "pos_neg_ratio": val_ratio,
            },
            "test": {
                "total": len(test_files),
                "counts": test_counts,
                "pos_neg_ratio": test_ratio,
            },
        }
    
    def copy_files(
        self,
        train_files: List[str],
        val_files: List[str],
        test_files: List[str],
    ):
        """将文件复制到对应子目录"""
        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }
        
        for split_name, files in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for f in files:
                src = self.data_dir / f
                dst = split_dir / f
                if src.exists():
                    shutil.copy2(src, dst)
    
    def create_manifests(
        self,
        train_files: List[str],
        val_files: List[str],
        test_files: List[str],
    ):
        """创建各子集的manifest文件"""
        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }
        
        for split_name, files in splits.items():
            manifest = []
            for f in files:
                label, category = self._classify_sample(f)
                text = self._extract_text_from_filename(f)
                manifest.append({
                    "file": f,
                    "text": text,
                    "label": label,
                    "category": category,
                })
            
            manifest_path = self.output_dir / f"{split_name}_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as fp:
                json.dump(manifest, fp, ensure_ascii=False, indent=2)
    
    def run(self) -> Dict:
        """执行完整的分层划分流程"""
        print("=" * 60)
        print("分层数据集划分")
        print("=" * 60)
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"划分比例: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        print(f"随机种子: {self.random_seed}")
        print()
        
        # 1. 加载并分类
        print("1. 加载数据并分类...")
        categories = self.load_and_classify()
        print(f"   类别分布:")
        for cat, files in sorted(categories.items()):
            print(f"     {cat}: {len(files)}")
        print()
        
        # 2. 分层划分
        print("2. 执行分层划分...")
        train_files, val_files, test_files = self.stratified_split(categories)
        print(f"   训练集: {len(train_files)}")
        print(f"   验证集: {len(val_files)}")
        print(f"   测试集: {len(test_files)}")
        print()
        
        # 3. 验证平衡性
        print("3. 验证划分平衡性...")
        validation = self.validate_split(categories, train_files, val_files, test_files)
        for split_name, info in validation.items():
            print(f"   {split_name}:")
            print(f"     总数: {info['total']}")
            print(f"     类别: {info['counts']}")
            print(f"     正负比: {info['pos_neg_ratio']:.4f}")
        print()
        
        # 4. 复制文件
        print("4. 复制文件到子目录...")
        self.copy_files(train_files, val_files, test_files)
        print(f"   完成")
        print()
        
        # 5. 创建manifest
        print("5. 创建manifest文件...")
        self.create_manifests(train_files, val_files, test_files)
        print(f"   完成")
        print()
        
        # 保存划分配置
        config = {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "split_sizes": {
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files),
            },
            "validation": validation,
        }
        config_path = self.output_dir / "split_config.json"
        with open(config_path, "w", encoding="utf-8") as fp:
            json.dump(config, fp, ensure_ascii=False, indent=2)
        print(f"配置已保存到: {config_path}")
        
        print()
        print("=" * 60)
        print("划分完成！")
        print("=" * 60)
        
        return config


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="分层数据集划分")
    parser.add_argument("--data-dir", type=str, default="data/all",
                        help="原始数据目录")
    parser.add_argument("--output-dir", type=str, 
                        default="experiments/baseline_streaming/data_splits",
                        help="输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="测试集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / args.data_dir
    output_dir = base_dir / args.output_dir
    
    splitter = StratifiedDatasetSplitter(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )
    
    splitter.run()


if __name__ == "__main__":
    main()
