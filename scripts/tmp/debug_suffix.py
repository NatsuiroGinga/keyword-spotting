#!/usr/bin/env python3
"""对比离线和流式后缀提取的差异"""
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

# 离线后缀提取（与训练时一致）
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
    mlp_enabled=False,
)

pipeline = StreamingKWSPipeline(config)
pipeline.load()

# 加载MLP和特征提取器
mlp = MLPVerifierONNX(
    model_path='/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx',
    threshold=0.5
)
mlp.load()
feature_extractor = FeatureExtractor(sample_rate=16000, n_mfcc=13, target_frames=50)

# 测试文件
test_dir = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive')
test_files = sorted(test_dir.glob('*.wav'))[:10]

print("对比离线 vs 流式后缀提取:\n")
print(f"{'文件名':<60} | {'离线':<10} | {'流式':<10} | {'差异'}")
print("-" * 100)

for test_file in test_files:
    audio, sr = sf.read(test_file)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    # 离线方式：使用完整音频
    suffix_offline = extract_suffix_offline(audio, 16000)
    feat_offline = feature_extractor.extract_for_mlp(suffix_offline)
    verified_offline, conf_offline = mlp.verify(feat_offline)
    
    # 流式方式：模拟流式处理
    pipeline.reset()
    chunk_size = 1600
    detected = False
    buffer_audio = None
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        result = pipeline.process_chunk(chunk)
        if result:
            detected = True
            # 获取缓冲区音频
            buffer_audio = pipeline._audio_buffer.get_last(config.buffer_duration).copy()
            break
    
    if detected and buffer_audio is not None:
        # 从缓冲区提取后缀
        suffix_stream = extract_suffix_offline(buffer_audio, 16000)
        feat_stream = feature_extractor.extract_for_mlp(suffix_stream)
        verified_stream, conf_stream = mlp.verify(feat_stream)
        
        diff = "✓ 一致" if verified_offline == verified_stream else "✗ 不一致"
        print(f"{test_file.name:<60} | {conf_offline:.3f}{'✓' if verified_offline else '✗':<5} | {conf_stream:.3f}{'✓' if verified_stream else '✗':<5} | {diff}")
    else:
        print(f"{test_file.name:<60} | {conf_offline:.3f}{'✓' if verified_offline else '✗':<5} | 未检测    | -")
