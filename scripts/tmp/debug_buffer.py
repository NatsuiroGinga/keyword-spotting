#!/usr/bin/env python3
"""调试缓冲区状态"""
import sys
import numpy as np
from pathlib import Path
import soundfile as sf

sys.path.insert(0, '/data/workspace/llm/keyword-spotting/hf_upload')
from src.utils.config import KWSConfig
from src.pipeline.kws_stream import StreamingKWSPipeline

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
    mlp_enabled=False,  # 先禁用MLP看第一阶段效果
)

pipeline = StreamingKWSPipeline(config)
pipeline.load()

# 测试几个正样本
test_dir = Path('/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive')
test_files = sorted(test_dir.glob('*.wav'))[:10]

print("分析检测触发时的缓冲区状态:\n")

for test_file in test_files:
    audio, sr = sf.read(test_file)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    pipeline.reset()
    chunk_size = 1600  # 100ms
    detected = False
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        result = pipeline.process_chunk(chunk)
        if result:
            buffer_len = pipeline._audio_buffer._total_samples
            audio_len = len(audio)
            trigger_pos = i + chunk_size
            print(f'{test_file.name}:')
            print(f'  音频总长: {audio_len/16000:.2f}s')
            print(f'  触发位置: {trigger_pos/16000:.2f}s ({trigger_pos/audio_len*100:.0f}%)')
            print(f'  缓冲区长: {buffer_len/16000:.2f}s')
            detected = True
            break
    
    if not detected:
        print(f'{test_file.name}: 未检测到')
