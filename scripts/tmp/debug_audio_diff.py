#!/usr/bin/env python3
"""详细分析离线和流式音频的差异"""
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

# 测试一个失败的样本
test_file = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive/positive_0000_zhmCNmXiaoxiaoNeural_p20pct_m10Hz_snr10dB.wav')

audio, sr = sf.read(test_file)
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
audio = audio.astype(np.float32)

print(f"文件: {test_file.name}")
print(f"原始音频长度: {len(audio)} samples = {len(audio)/16000:.3f}s")

# 离线后缀
suffix_offline = extract_suffix(audio, 16000)
print(f"\n离线后缀:")
print(f"  长度: {len(suffix_offline)} samples = {len(suffix_offline)/16000:.3f}s")
print(f"  起始位置: {int(len(audio)*0.4)} (40%)")

# 流式处理
pipeline.reset()
chunk_size = 1600
trigger_pos = 0

for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
    result = pipeline.process_chunk(chunk)
    if result:
        trigger_pos = i + chunk_size
        break

buffer_audio = pipeline._audio_buffer.get_last(config.buffer_duration).copy()
suffix_stream = extract_suffix(buffer_audio, 16000)

print(f"\n流式处理:")
print(f"  检测触发位置: {trigger_pos} samples = {trigger_pos/16000:.3f}s ({trigger_pos/len(audio)*100:.1f}%)")
print(f"  缓冲区长度: {len(buffer_audio)} samples = {len(buffer_audio)/16000:.3f}s")
print(f"  流式后缀长度: {len(suffix_stream)} samples = {len(suffix_stream)/16000:.3f}s")

# 关键：检查缓冲区音频与原始音频的对应关系
print(f"\n音频对齐分析:")
# 缓冲区应该包含从0到trigger_pos的音频
expected_buffer = audio[:trigger_pos]
print(f"  预期缓冲区长度: {len(expected_buffer)} samples")
print(f"  实际缓冲区长度: {len(buffer_audio)} samples")

# 检查是否一致
if len(buffer_audio) <= len(expected_buffer):
    # 取最后len(buffer_audio)个样本比较
    expected_part = expected_buffer[-len(buffer_audio):]
    diff = np.abs(buffer_audio - expected_part).mean()
    print(f"  缓冲区与预期音频差异: {diff:.6f}")
    if diff < 0.001:
        print("  ✓ 缓冲区音频与原始音频一致")
    else:
        print("  ✗ 缓冲区音频与原始音频不一致!")

# 验证结果
feat_offline = feature_extractor.extract_for_mlp(suffix_offline)
feat_stream = feature_extractor.extract_for_mlp(suffix_stream)

_, conf_offline = mlp.verify(feat_offline)
_, conf_stream = mlp.verify(feat_stream)

print(f"\nMLP验证结果:")
print(f"  离线置信度: {conf_offline:.4f}")
print(f"  流式置信度: {conf_stream:.4f}")

# 分析后缀差异
print(f"\n后缀内容分析:")
print(f"  离线后缀起始: 原始音频的 {int(len(audio)*0.4)/16000:.3f}s 位置")
print(f"  流式后缀起始: 缓冲区的 {int(len(buffer_audio)*0.4)/16000:.3f}s 位置")
print(f"  对应原始音频: {(trigger_pos - len(buffer_audio) + int(len(buffer_audio)*0.4))/16000:.3f}s 位置")

# 检查两个后缀的实际内容差异
min_len = min(len(suffix_offline), len(suffix_stream))
content_diff = np.abs(suffix_offline[:min_len] - suffix_stream[:min_len]).mean()
print(f"  后缀内容差异: {content_diff:.6f}")
