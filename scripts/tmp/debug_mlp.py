#!/usr/bin/env python3
"""调试MLP验证器"""
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

# 离线后缀提取
def extract_suffix_offline(samples, sample_rate, start_ratio=0.4, min_duration_ms=200, max_duration_ms=800):
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
config = KWSConfig(
    encoder_path=str(model_dir / 'encoder.int8.onnx'),
    decoder_path=str(model_dir / 'decoder.int8.onnx'),
    joiner_path=str(model_dir / 'joiner.int8.onnx'),
    tokens_path=str(model_dir / 'tokens.txt'),
    keywords_file=str(model_dir / 'keywords.txt'),
    mlp_model_path='/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx',
    keywords=['你好真真'],
    mlp_enabled=True,
)

pipeline = StreamingKWSPipeline(config)
pipeline.load()

# 独立的MLP验证器
mlp = MLPVerifierONNX(
    model_path='/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx',
    threshold=0.5
)
mlp.load()
feature_extractor = FeatureExtractor(sample_rate=16000, n_mfcc=13, target_frames=50)

# 测试正样本
test_dir = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive')
test_files = sorted(test_dir.glob('*.wav'))

print("正样本MLP验证分析:\n")

# 统计
offline_pass = 0
stream_pass = 0
both_pass = 0
offline_only = 0
stream_only = 0
both_fail = 0

for test_file in test_files:
    audio, sr = sf.read(test_file)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    # 离线验证（使用完整音频）
    suffix_offline = extract_suffix_offline(audio, 16000)
    feat_offline = feature_extractor.extract_for_mlp(suffix_offline)
    verified_offline, conf_offline = mlp.verify(feat_offline)
    
    # 流式验证
    pipeline.reset()
    chunk_size = 1600
    verified_stream = False
    conf_stream = 0.0
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        result = pipeline.process_chunk(chunk)
        if result:
            verified_stream = True
            conf_stream = result.mlp_confidence if result.mlp_confidence else 0.0
            break
    
    if verified_offline:
        offline_pass += 1
    if verified_stream:
        stream_pass += 1
    
    if verified_offline and verified_stream:
        both_pass += 1
    elif verified_offline and not verified_stream:
        offline_only += 1
    elif not verified_offline and verified_stream:
        stream_only += 1
    else:
        both_fail += 1

print(f"总样本: {len(test_files)}")
print(f"离线通过: {offline_pass} ({offline_pass/len(test_files)*100:.1f}%)")
print(f"流式通过: {stream_pass} ({stream_pass/len(test_files)*100:.1f}%)")
print(f"\n详细分析:")
print(f"  两者都通过: {both_pass}")
print(f"  仅离线通过: {offline_only}")
print(f"  仅流式通过: {stream_only}")
print(f"  两者都失败: {both_fail}")
