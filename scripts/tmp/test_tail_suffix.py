#!/usr/bin/env python3
"""测试使用缓冲区末尾固定时长作为后缀"""
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

def test_with_tail_duration(tail_duration):
    """测试使用缓冲区末尾tail_duration秒作为后缀"""
    config = KWSConfig(
        encoder_path=str(model_dir / 'encoder.int8.onnx'),
        decoder_path=str(model_dir / 'decoder.int8.onnx'),
        joiner_path=str(model_dir / 'joiner.int8.onnx'),
        tokens_path=str(model_dir / 'tokens.txt'),
        keywords_file=str(model_dir / 'keywords.txt'),
        mlp_model_path='/data/workspace/llm/keyword-spotting/hf_upload/models/mlp_verifier.onnx',
        keywords=['你好真真'],
        mlp_enabled=False,  # 禁用内置MLP，手动验证
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
                # 阶段1检测到，进行阶段2验证
                # 使用缓冲区末尾tail_duration秒
                suffix = pipeline._audio_buffer.get_last(tail_duration).copy()
                if len(suffix) < 1600:  # 至少100ms
                    return True  # 太短，跳过验证
                features = feature_extractor.extract_for_mlp(suffix)
                verified, _ = mlp.verify(features)
                return verified
        return False
    
    tp = sum(1 for f in pos_files if test_file(f))
    fp = sum(1 for f in neg_files if test_file(f))
    
    frr = (len(pos_files) - tp) / len(pos_files)
    far = fp / len(neg_files)
    
    return tp, fp, frr, far

print("测试使用缓冲区末尾固定时长作为后缀:\n")
print(f"{'tail_duration':<15} | {'TP':<5} | {'FP':<5} | {'FRR':<10} | {'FAR':<10}")
print("-" * 60)

for duration in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
    tp, fp, frr, far = test_with_tail_duration(duration)
    frr_ok = "✓" if frr <= 0.05 else ""
    far_ok = "✓" if far <= 0.10 else ""
    print(f"{duration:<15.1f} | {tp:<5} | {fp:<5} | {frr*100:<9.2f}% {frr_ok} | {far*100:<9.2f}% {far_ok}")
