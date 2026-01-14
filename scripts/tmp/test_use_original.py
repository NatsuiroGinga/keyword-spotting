#!/usr/bin/env python3
"""测试：阶段1流式检测，阶段2使用原始音频验证"""
import sys
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa

sys.path.insert(0, '/data/workspace/llm/keyword-spotting/hf_upload')
from src.utils.config import KWSConfig
from src.pipeline.kws_stream import StreamingKWSPipeline
from src.audio.feature import FeatureExtractor
from src.models.mlp_verifier import MLPVerifierONNX

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

# 配置
model_dir = Path('/data/workspace/llm/keyword-spotting/hf_upload/kws_finetune_v3')
test_dir = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged')
pos_files = sorted((test_dir / 'positive').glob('*.wav'))
neg_files = sorted((test_dir / 'negative').glob('*.wav'))

config = KWSConfig(
    encoder_path=str(model_dir / 'encoder.int8.onnx'),
    decoder_path=str(model_dir / 'decoder.int8.onnx'),
    joiner_path=str(model_dir / 'joiner.int8.onnx'),
    tokens_path=str(model_dir / 'tokens.txt'),
    keywords_file=str(model_dir / 'keywords.txt'),
    mlp_model_path='/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx',
    keywords=['你好真真'],
    mlp_enabled=False,
)

pipeline = StreamingKWSPipeline(config)
pipeline.load()

mlp = MLPVerifierONNX(
    model_path='/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx',
    threshold=0.5
)
mlp.load()
feature_extractor = FeatureExtractor(sample_rate=16000, n_mfcc=13, target_frames=50)

def test_file(audio_path):
    """阶段1流式检测，阶段2使用原始音频验证"""
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    # 阶段1：流式检测
    pipeline.reset()
    chunk_size = 1600
    detected = False
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        result = pipeline.process_chunk(chunk)
        if result:
            detected = True
            break
    
    if not detected:
        return False
    
    # 阶段2：使用原始音频验证（而非缓冲区）
    suffix = extract_suffix(audio, 16000)
    features = feature_extractor.extract_for_mlp(suffix)
    verified, _ = mlp.verify(features)
    return verified

print("测试：阶段1流式检测 + 阶段2使用原始音频验证\n")

tp = sum(1 for f in pos_files if test_file(f))
fp = sum(1 for f in neg_files if test_file(f))

frr = (len(pos_files) - tp) / len(pos_files)
far = fp / len(neg_files)

print(f"正样本: TP={tp}/{len(pos_files)} ({tp/len(pos_files)*100:.1f}%)")
print(f"负样本: FP={fp}/{len(neg_files)} ({fp/len(neg_files)*100:.1f}%)")
print(f"\nFRR: {frr*100:.2f}%")
print(f"FAR: {far*100:.2f}%")

frr_ok = "✓" if frr <= 0.05 else "✗"
far_ok = "✓" if far <= 0.10 else "✗"
print(f"\n目标达成: FRR≤5% {frr_ok}, FAR≤10% {far_ok}")
