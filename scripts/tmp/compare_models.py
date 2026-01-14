#!/usr/bin/env python3
"""对比PyTorch模型和ONNX模型的输出"""
import sys
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
import torch

# 添加路径
sys.path.insert(0, '/data/workspace/llm/keyword-spotting/experiments/multi_stage_ablation')
sys.path.insert(0, '/data/workspace/llm/keyword-spotting/hf_upload')

# 导入两个版本的验证器
from stage2.mlp_verifier import MLPVerifier as PyTorchMLP
from src.models.mlp_verifier import MLPVerifierONNX
from src.audio.feature import FeatureExtractor

# 后缀提取
def extract_suffix(samples, sample_rate, start_ratio=0.4, min_duration_ms=200, max_duration_ms=800):
    total_samples = len(samples)
    start_sample = int(total_samples * start_ratio)
    min_samples = int(min_duration_ms * sample_rate / 1000)
    max_samples = int(max_duration_ms * sample_rate / 1000)
    
    suffix = samples[start_sample:]
    
    if len(suffix) < min_samples:
        new_start = max(0, total_samples - min_samples)
        suffix = samples[new_start:]
    elif len(suffix) > max_samples:
        suffix = suffix[:max_samples]
    
    return suffix

# 加载PyTorch模型
pt_model_path = '/data/workspace/llm/keyword-spotting/experiments/multi_stage_ablation/models/mlp_verifier.pt'
pt_mlp = PyTorchMLP(threshold=0.5, model_path=pt_model_path)
pt_mlp.load_model()

# 加载ONNX模型
onnx_model_path = '/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx'
onnx_mlp = MLPVerifierONNX(model_path=onnx_model_path, threshold=0.5)
onnx_mlp.load()

# 特征提取器（用于ONNX）
feature_extractor = FeatureExtractor(sample_rate=16000, n_mfcc=13, target_frames=50)

# 测试文件
test_dir = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive')
test_files = sorted(test_dir.glob('*.wav'))[:10]

print("对比PyTorch和ONNX模型输出:\n")
print(f"{'文件名':<55} | {'PyTorch':<10} | {'ONNX':<10} | {'差异':<10}")
print("-" * 95)

for test_file in test_files:
    audio, sr = sf.read(test_file)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    # 提取后缀
    suffix = extract_suffix(audio, 16000)
    
    # PyTorch验证
    _, pt_conf = pt_mlp.verify(suffix, 16000)
    
    # ONNX验证（使用相同的特征提取）
    features = feature_extractor.extract_for_mlp(suffix)
    _, onnx_conf = onnx_mlp.verify(features)
    
    diff = abs(pt_conf - onnx_conf)
    match = "✓" if diff < 0.01 else "✗"
    print(f"{test_file.name:<55} | {pt_conf:<10.4f} | {onnx_conf:<10.4f} | {diff:<10.4f} {match}")
