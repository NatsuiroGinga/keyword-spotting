#!/usr/bin/env python3
"""测试仅使用阶段1，调整阈值"""
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

def test_with_kws_threshold(kws_threshold):
    config = KWSConfig(
        encoder_path=str(model_dir / 'encoder.int8.onnx'),
        decoder_path=str(model_dir / 'decoder.int8.onnx'),
        joiner_path=str(model_dir / 'joiner.int8.onnx'),
        tokens_path=str(model_dir / 'tokens.txt'),
        keywords_file=str(model_dir / 'keywords.txt'),
        mlp_model_path='/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx',
        keywords=['你好真真'],
        mlp_enabled=False,  # 禁用MLP
        keywords_threshold=kws_threshold,
    )
    
    pipeline = StreamingKWSPipeline(config)
    pipeline.load()
    
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
    
    tp = sum(1 for f in pos_files if test_file(f))
    fp = sum(1 for f in neg_files if test_file(f))
    
    frr = (len(pos_files) - tp) / len(pos_files)
    far = fp / len(neg_files)
    
    return tp, fp, frr, far

print("测试仅阶段1（调整KWS阈值）:\n")
print(f"{'kws_threshold':<15} | {'TP':<5} | {'FP':<5} | {'FRR':<10} | {'FAR':<10}")
print("-" * 60)

for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]:
    tp, fp, frr, far = test_with_kws_threshold(threshold)
    frr_ok = "✓" if frr <= 0.05 else ""
    far_ok = "✓" if far <= 0.10 else ""
    print(f"{threshold:<15.2f} | {tp:<5} | {fp:<5} | {frr*100:<9.2f}% {frr_ok} | {far*100:<9.2f}% {far_ok}")
