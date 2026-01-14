#!/usr/bin/env python3
"""调试延迟验证失败的样本"""
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

test_dir = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive')
test_files = sorted(test_dir.glob('*.wav'))

print("分析延迟验证失败的样本:\n")

fail_count = 0
for test_file in test_files:
    audio, sr = sf.read(test_file)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    # 离线验证
    suffix_orig = extract_suffix(audio, 16000)
    feat_orig = feature_extractor.extract_for_mlp(suffix_orig)
    _, conf_orig = mlp.verify(feat_orig)
    
    # 流式处理
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
    
    if not detected:
        print(f"{test_file.name}: 阶段1未检测到")
        fail_count += 1
        continue
    
    # 缓冲区验证
    buffer = pipeline._audio_buffer.get_last(config.buffer_duration).copy()
    suffix_buffer = extract_suffix(buffer, 16000)
    feat_buffer = feature_extractor.extract_for_mlp(suffix_buffer)
    _, conf_buffer = mlp.verify(feat_buffer)
    
    if conf_buffer < 0.5:
        print(f"{test_file.name}:")
        print(f"  原始音频: {len(audio)} samples")
        print(f"  缓冲区: {len(buffer)} samples")
        print(f"  离线置信度: {conf_orig:.4f}")
        print(f"  缓冲区置信度: {conf_buffer:.4f}")
        print(f"  后缀长度差异: {len(suffix_buffer) - len(suffix_orig)}")
        fail_count += 1
        if fail_count >= 10:
            break

print(f"\n失败样本数: {fail_count}")
