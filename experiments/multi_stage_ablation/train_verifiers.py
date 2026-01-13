#!/usr/bin/env python3
"""
训练阶段2验证器（CNN和MLP）

使用正样本的后缀作为正类，负样本的后缀作为负类
"""
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf

from config import AblationConfig
from stage2 import CNNVerifier, MLPVerifier
from utils import load_audio, extract_suffix


def prepare_training_data(
    config: AblationConfig,
    max_positive: int = 144,
    max_negative: int = 300
) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    """
    准备训练数据
    
    从正样本提取"真真"后缀作为正类
    从负样本提取后缀作为负类
    
    Returns:
        (positive_samples, negative_samples)
        每个元素是 (audio_samples, sample_rate) 的元组
    """
    test_path = Path(config.test_data_path)
    positive_dir = test_path / config.positive_dir
    negative_dir = test_path / config.negative_dir
    
    positive_files = sorted(positive_dir.glob("*.wav"))[:max_positive]
    negative_files = sorted(negative_dir.glob("*.wav"))[:max_negative]
    
    positive_samples = []
    negative_samples = []
    
    print(f"加载正样本 ({len(positive_files)} 个)...")
    for audio_path in positive_files:
        try:
            samples, sr = load_audio(str(audio_path))
            # 提取后缀（"真真"部分）
            suffix = extract_suffix(
                samples, sr,
                start_ratio=config.suffix_start_ratio,
                min_duration_ms=config.suffix_min_duration_ms,
                max_duration_ms=config.suffix_max_duration_ms
            )
            positive_samples.append((suffix, sr))
        except Exception as e:
            print(f"  跳过 {audio_path.name}: {e}")
    
    print(f"加载负样本 ({len(negative_files)} 个)...")
    for audio_path in negative_files:
        try:
            samples, sr = load_audio(str(audio_path))
            # 提取后缀
            suffix = extract_suffix(
                samples, sr,
                start_ratio=config.suffix_start_ratio,
                min_duration_ms=config.suffix_min_duration_ms,
                max_duration_ms=config.suffix_max_duration_ms
            )
            negative_samples.append((suffix, sr))
        except Exception as e:
            print(f"  跳过 {audio_path.name}: {e}")
    
    print(f"\n训练数据: 正样本 {len(positive_samples)}, 负样本 {len(negative_samples)}")
    
    return positive_samples, negative_samples


def train_cnn(
    positive_samples: List[Tuple[np.ndarray, int]],
    negative_samples: List[Tuple[np.ndarray, int]],
    save_path: str,
    epochs: int = 100
) -> dict:
    """训练CNN验证器"""
    print("\n" + "=" * 50)
    print("训练CNN验证器")
    print("=" * 50)
    
    verifier = CNNVerifier(threshold=0.5)
    history = verifier.train(
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        epochs=epochs,
        batch_size=32,
        lr=0.001
    )
    
    verifier.save_model(save_path)
    
    return history


def train_mlp(
    positive_samples: List[Tuple[np.ndarray, int]],
    negative_samples: List[Tuple[np.ndarray, int]],
    save_path: str,
    epochs: int = 150
) -> dict:
    """训练MLP验证器"""
    print("\n" + "=" * 50)
    print("训练MLP验证器")
    print("=" * 50)
    
    verifier = MLPVerifier(threshold=0.5)
    history = verifier.train(
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        epochs=epochs,
        batch_size=32,
        lr=0.001
    )
    
    verifier.save_model(save_path)
    
    return history


def main():
    parser = argparse.ArgumentParser(description="训练阶段2验证器")
    parser.add_argument("--test-data", type=str,
                        default="/data/workspace/llm/audio-classification/dataset/kws_test_data_merged",
                        help="测试数据目录")
    parser.add_argument("--output-dir", type=str,
                        default="./models",
                        help="模型输出目录")
    parser.add_argument("--cnn-epochs", type=int, default=100,
                        help="CNN训练轮数")
    parser.add_argument("--mlp-epochs", type=int, default=150,
                        help="MLP训练轮数")
    parser.add_argument("--max-negative", type=int, default=300,
                        help="最大负样本数量")
    
    args = parser.parse_args()
    
    # 配置
    config = AblationConfig(test_data_path=args.test_data)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备训练数据
    positive_samples, negative_samples = prepare_training_data(
        config,
        max_negative=args.max_negative
    )
    
    # 训练CNN
    cnn_path = output_dir / "cnn_verifier.pt"
    cnn_history = train_cnn(
        positive_samples, negative_samples,
        str(cnn_path),
        epochs=args.cnn_epochs
    )
    
    # 训练MLP
    mlp_path = output_dir / "mlp_verifier.pt"
    mlp_history = train_mlp(
        positive_samples, negative_samples,
        str(mlp_path),
        epochs=args.mlp_epochs
    )
    
    print("\n" + "=" * 50)
    print("训练完成")
    print("=" * 50)
    print(f"CNN模型: {cnn_path}")
    print(f"  最终Loss: {cnn_history['loss'][-1]:.4f}")
    print(f"  最终Acc: {cnn_history['accuracy'][-1]:.4f}")
    print(f"MLP模型: {mlp_path}")
    print(f"  最终Loss: {mlp_history['loss'][-1]:.4f}")
    print(f"  最终Acc: {mlp_history['accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
