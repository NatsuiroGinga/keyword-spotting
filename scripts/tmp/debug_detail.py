#!/usr/bin/env python3
"""详细调试：对比离线和流式的每个步骤"""
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
    mlp_enabled=False,  # 禁用MLP，手动验证
)

pipeline = StreamingKWSPipeline(config)
pipeline.load()

mlp = MLPVerifierONNX(
    model_path='/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx',
    threshold=0.5
)
mlp.load()
feature_extractor = FeatureExtractor(sample_rate=16000, n_mfcc=13, target_frames=50)

# 找一个仅离线通过的样本
test_dir = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive')
test_files = sorted(test_dir.glob('*.wav'))

print("查找仅离线通过的样本并详细对比:\n")

for test_file in test_files[:30]:
    audio, sr = sf.read(test_file)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    # 离线
    suffix_offline = extract_suffix_offline(audio, 16000)
    feat_offline = feature_extractor.extract_for_mlp(suffix_offline)
    verified_offline, conf_offline = mlp.verify(feat_offline)
    
    # 流式
    pipeline.reset()
    chunk_size = 1600
    buffer_audio = None
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        result = pipeline.process_chunk(chunk)
        if result:
            buffer_audio = pipeline._audio_buffer.get_last(config.buffer_duration).copy()
            break
    
    if buffer_audio is not None:
        suffix_stream = extract_suffix_offline(buffer_audio, 16000)
        feat_stream = feature_extractor.extract_for_mlp(suffix_stream)
        verified_stream, conf_stream = mlp.verify(feat_stream)
        
        if verified_offline and not verified_stream:
            print(f"文件: {test_file.name}")
            print(f"  原始音频长度: {len(audio)} samples = {len(audio)/16000:.3f}s")
            print(f"  缓冲区音频长度: {len(buffer_audio)} samples = {len(buffer_audio)/16000:.3f}s")
            print(f"  离线后缀长度: {len(suffix_offline)} samples = {len(suffix_offline)/16000:.3f}s")
            print(f"  流式后缀长度: {len(suffix_stream)} samples = {len(suffix_stream)/16000:.3f}s")
            print(f"  离线置信度: {conf_offline:.4f} -> {'通过' if verified_offline else '拒绝'}")
            print(f"  流式置信度: {conf_stream:.4f} -> {'通过' if verified_stream else '拒绝'}")
            
            # 对比后缀内容
            min_len = min(len(suffix_offline), len(suffix_stream))
            if min_len > 0:
                diff = np.abs(suffix_offline[:min_len] - suffix_stream[:min_len]).mean()
                print(f"  后缀差异(平均绝对差): {diff:.6f}")
            print()
