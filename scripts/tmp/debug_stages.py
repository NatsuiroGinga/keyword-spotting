#!/usr/bin/env python3
"""分阶段调试"""
import sys
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa

sys.path.insert(0, '/data/workspace/llm/keyword-spotting/hf_upload')
from src.utils.config import KWSConfig
from src.pipeline.kws_stream import StreamingKWSPipeline

# 配置
model_dir = Path('/data/workspace/llm/keyword-spotting/hf_upload/kws_finetune_v3')

# 测试阶段1（禁用MLP）
config_stage1 = KWSConfig(
    encoder_path=str(model_dir / 'encoder.int8.onnx'),
    decoder_path=str(model_dir / 'decoder.int8.onnx'),
    joiner_path=str(model_dir / 'joiner.int8.onnx'),
    tokens_path=str(model_dir / 'tokens.txt'),
    keywords_file=str(model_dir / 'keywords.txt'),
    mlp_model_path='/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx',
    keywords=['你好真真'],
    mlp_enabled=False,
)

pipeline = StreamingKWSPipeline(config_stage1)
pipeline.load()

test_dir = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged')
pos_files = sorted((test_dir / 'positive').glob('*.wav'))
neg_files = sorted((test_dir / 'negative').glob('*.wav'))

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
            return True
    return False

print("=" * 60)
print("阶段1测试（仅Sherpa-ONNX KWS，无MLP验证）")
print("=" * 60)

# 正样本
pos_detected = sum(1 for f in pos_files if test_file(f))
print(f"\n正样本: {pos_detected}/{len(pos_files)} 检测到 ({pos_detected/len(pos_files)*100:.1f}%)")

# 负样本
neg_detected = sum(1 for f in neg_files if test_file(f))
print(f"负样本: {neg_detected}/{len(neg_files)} 误检 ({neg_detected/len(neg_files)*100:.1f}%)")

print(f"\n阶段1 FRR: {(len(pos_files)-pos_detected)/len(pos_files)*100:.2f}%")
print(f"阶段1 FAR: {neg_detected/len(neg_files)*100:.2f}%")
