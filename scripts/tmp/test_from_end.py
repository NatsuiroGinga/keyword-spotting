#!/usr/bin/env python3
"""测试从缓冲区末尾提取后缀"""
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

# 配置
model_dir = Path('/data/workspace/llm/keyword-spotting/hf_upload/kws_finetune_v3')

# 测试数据
test_dir = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged')
pos_files = sorted((test_dir / 'positive').glob('*.wav'))
neg_files = sorted((test_dir / 'negative').glob('*.wav'))

# 分析离线后缀的实际长度
print("分析离线后缀长度:\n")

def extract_suffix_offline(samples, sample_rate, start_ratio=0.4, max_duration_ms=800):
    total_samples = len(samples)
    start_sample = int(total_samples * start_ratio)
    max_samples = int(max_duration_ms * sample_rate / 1000)
    
    suffix = samples[start_sample:]
    if len(suffix) > max_samples:
        suffix = suffix[:max_samples]
    return suffix

suffix_lengths = []
for f in pos_files[:20]:
    audio, sr = sf.read(f)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    suffix = extract_suffix_offline(audio, 16000)
    suffix_lengths.append(len(suffix) / 16000)

print(f"离线后缀长度统计:")
print(f"  最小: {min(suffix_lengths):.3f}s")
print(f"  最大: {max(suffix_lengths):.3f}s")
print(f"  平均: {np.mean(suffix_lengths):.3f}s")

# 测试从末尾提取
def test_from_end(end_duration):
    """从缓冲区末尾提取end_duration秒作为后缀"""
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
        pipeline.reset()
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32)
        
        chunk_size = 1600
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            result = pipeline.process_chunk(chunk)
            if result:
                # 从缓冲区末尾提取
                suffix = pipeline._audio_buffer.get_last(end_duration).copy()
                if len(suffix) < 1600:
                    return True
                features = feature_extractor.extract_for_mlp(suffix)
                verified, _ = mlp.verify(features)
                return verified
        return False
    
    tp = sum(1 for f in pos_files if test_file(f))
    fp = sum(1 for f in neg_files if test_file(f))
    
    frr = (len(pos_files) - tp) / len(pos_files)
    far = fp / len(neg_files)
    
    return tp, fp, frr, far

print("\n测试从缓冲区末尾提取后缀:\n")
print(f"{'end_duration':<12} | {'TP':<5} | {'FP':<5} | {'FRR':<10} | {'FAR':<10}")
print("-" * 55)

for duration in [0.75, 0.78, 0.80, 0.82, 0.85]:
    tp, fp, frr, far = test_from_end(duration)
    frr_ok = "✓" if frr <= 0.05 else ""
    far_ok = "✓" if far <= 0.10 else ""
    print(f"{duration:<12.2f} | {tp:<5} | {fp:<5} | {frr*100:<9.2f}% {frr_ok} | {far*100:<9.2f}% {far_ok}")
