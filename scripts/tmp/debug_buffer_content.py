#!/usr/bin/env python3
"""调试缓冲区内容"""
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

# 测试一个样本
test_file = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive/positive_0000_zhmCNmXiaoxiaoNeural_p20pct_m10Hz_snr10dB.wav')

audio, sr = sf.read(test_file)
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
audio = audio.astype(np.float32)

print(f"文件: {test_file.name}")
print(f"原始音频长度: {len(audio)} samples = {len(audio)/16000:.3f}s")

# 处理所有chunk
pipeline.reset()
chunk_size = 1600

for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    if len(chunk) < chunk_size:
        # 这里是问题所在：最后一个chunk被padding了
        print(f"\n最后一个chunk: 原始{len(chunk)} samples, padding到{chunk_size}")
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
    pipeline.process_chunk(chunk)

buffer = pipeline._audio_buffer.get_last(config.buffer_duration).copy()
print(f"缓冲区长度: {len(buffer)} samples = {len(buffer)/16000:.3f}s")

# 计算缓冲区与原始音频的差异
# 缓冲区可能比原始音频长（因为padding）
if len(buffer) >= len(audio):
    # 比较前len(audio)个样本
    diff = np.abs(buffer[:len(audio)] - audio).mean()
    print(f"缓冲区前{len(audio)}样本与原始音频差异: {diff:.6f}")
    # 检查padding部分
    padding_part = buffer[len(audio):]
    print(f"Padding部分: {len(padding_part)} samples, 均值: {np.abs(padding_part).mean():.6f}")

# 提取后缀并验证
suffix_orig = extract_suffix(audio, 16000)
suffix_buffer = extract_suffix(buffer, 16000)

print(f"\n离线后缀长度: {len(suffix_orig)} samples")
print(f"缓冲区后缀长度: {len(suffix_buffer)} samples")

# 验证
feat_orig = feature_extractor.extract_for_mlp(suffix_orig)
feat_buffer = feature_extractor.extract_for_mlp(suffix_buffer)

_, conf_orig = mlp.verify(feat_orig)
_, conf_buffer = mlp.verify(feat_buffer)

print(f"\n离线验证置信度: {conf_orig:.4f}")
print(f"缓冲区验证置信度: {conf_buffer:.4f}")
