#!/usr/bin/env python3
"""测试：不使用padding的流式处理"""
import sys
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa

sys.path.insert(0, '/data/workspace/llm/keyword-spotting/hf_upload')
from src.utils.config import KWSConfig
from src.pipeline.kws_stream import StreamingKWSPipeline

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
    mlp_enabled=True,
)

pipeline = StreamingKWSPipeline(config)
pipeline.load()

def test_file(audio_path):
    """不使用padding的流式处理"""
    pipeline.reset()
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    chunk_size = 1600
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        # 不padding最后一个chunk，直接处理原始长度
        result = pipeline.process_chunk(chunk)
        if result:
            return True
    return False

print("测试：不使用padding的流式处理\n")

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
