#!/usr/bin/env python3
"""
测试ASR验证器
"""
import time
from pathlib import Path
import numpy as np

from config import AblationConfig, AblationResult
from stage1.prefix_detector import PrefixDetector
from stage2 import ASRVerifier
from utils import load_audio, extract_suffix


def main():
    config = AblationConfig()
    
    print("=" * 60)
    print("测试 ASR 验证器")
    print("=" * 60)
    
    # 初始化阶段1
    print("\n初始化阶段1检测器...")
    stage1 = PrefixDetector(
        model_dir=config.model_dir,
        threshold=config.stage1_threshold
    )
    stage1.load_model()
    
    # 初始化ASR验证器
    print("\n初始化ASR验证器...")
    asr = ASRVerifier(
        model_name="openai/whisper-small",
        target_text="真真",
        threshold=0.5
    )
    asr.load_model()
    
    # 获取测试文件
    test_path = Path(config.test_data_path)
    positive_files = sorted((test_path / "positive").glob("*.wav"))[:20]  # 只测试20个
    negative_files = sorted((test_path / "negative").glob("*.wav"))[:50]  # 只测试50个
    
    result = AblationResult(verifier_name="asr")
    result.total_positive = len(positive_files)
    result.total_negative = len(negative_files)
    
    # 测试正样本
    print(f"\n测试正样本 ({len(positive_files)} 个)...")
    for audio_path in positive_files:
        samples, sr = load_audio(str(audio_path))
        detected, _, _ = stage1.detect(samples, sr)
        
        final_accepted = False
        if detected:
            result.stage1_passed_positive += 1
            suffix = extract_suffix(samples, sr, start_ratio=0.4)
            accepted, conf, text = asr.verify_with_transcription(suffix, sr)
            print(f"  {audio_path.name}: 转录=\"{text}\", 接受={accepted}")
            if accepted:
                result.stage2_passed_positive += 1
                final_accepted = True
        
        if final_accepted:
            result.true_positive += 1
        else:
            result.false_negative += 1
    
    # 测试负样本
    print(f"\n测试负样本 ({len(negative_files)} 个)...")
    for audio_path in negative_files:
        samples, sr = load_audio(str(audio_path))
        detected, _, _ = stage1.detect(samples, sr)
        
        final_accepted = False
        if detected:
            result.stage1_passed_negative += 1
            suffix = extract_suffix(samples, sr, start_ratio=0.4)
            accepted, conf, text = asr.verify_with_transcription(suffix, sr)
            print(f"  {audio_path.name}: 转录=\"{text}\", 接受={accepted}")
            if accepted:
                result.stage2_passed_negative += 1
                final_accepted = True
        
        if final_accepted:
            result.false_positive += 1
        else:
            result.true_negative += 1
    
    # 打印结果
    print("\n" + "=" * 60)
    print("ASR验证器结果")
    print("=" * 60)
    print(f"阶段1通过: 正样本 {result.stage1_passed_positive}/{result.total_positive}, "
          f"负样本 {result.stage1_passed_negative}/{result.total_negative}")
    print(f"阶段2通过: 正样本 {result.stage2_passed_positive}/{result.stage1_passed_positive}, "
          f"负样本 {result.stage2_passed_negative}/{result.stage1_passed_negative}")
    print(f"FRR: {result.frr*100:.2f}%")
    print(f"FAR: {result.far*100:.2f}%")
    print(f"准确率: {result.accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
