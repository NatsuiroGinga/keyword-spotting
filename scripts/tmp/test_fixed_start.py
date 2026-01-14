#!/usr/bin/env python3
"""测试使用固定起始时间提取后缀"""
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

def test_with_fixed_start(start_time_sec):
    """
    测试使用固定起始时间提取后缀
    从缓冲区的start_time_sec位置开始提取到末尾
    """
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
                # 获取缓冲区
                buffer = pipeline._audio_buffer.get_last(config.buffer_duration).copy()
                
                # 从固定起始时间提取后缀
                start_sample = int(start_time_sec * 16000)
                if start_sample >= len(buffer):
                    return True  # 缓冲区太短，跳过验证
                
                suffix = buffer[start_sample:]
                
                # 限制最大长度800ms
                max_samples = int(0.8 * 16000)
                if len(suffix) > max_samples:
                    suffix = suffix[:max_samples]
                
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

print("测试使用固定起始时间提取后缀:\n")
print(f"{'start_time':<12} | {'TP':<5} | {'FP':<5} | {'FRR':<10} | {'FAR':<10}")
print("-" * 55)

for start_time in [0.4, 0.45, 0.5, 0.52, 0.55, 0.6, 0.65, 0.7]:
    tp, fp, frr, far = test_with_fixed_start(start_time)
    frr_ok = "✓" if frr <= 0.05 else ""
    far_ok = "✓" if far <= 0.10 else ""
    print(f"{start_time:<12.2f} | {tp:<5} | {fp:<5} | {frr*100:<9.2f}% {frr_ok} | {far*100:<9.2f}% {far_ok}")
