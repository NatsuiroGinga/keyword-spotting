#!/usr/bin/env python3
"""
测试HuggingFace版本代码的效果
使用完整测试数据集（144正样本 + 540负样本）
"""
import sys
import time
import numpy as np
from pathlib import Path
import soundfile as sf

# 使用hf_upload目录的代码
sys.path.insert(0, '/data/workspace/llm/keyword-spotting/hf_upload')

from src.utils.config import KWSConfig
from src.pipeline.kws_stream import StreamingKWSPipeline

# 测试数据路径
TEST_DATA_PATH = "/data/workspace/llm/audio-classification/dataset/kws_test_data_merged"

# 配置 - 使用hf_upload中的ONNX模型
model_dir = Path('/data/workspace/llm/keyword-spotting/hf_upload/kws_finetune_v3')
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

print('=' * 60)
print('HuggingFace版本代码测试')
print('=' * 60)
print(f'模型目录: {model_dir}')
print(f'MLP模型: {config.mlp_model_path}')
print(f'测试数据: {TEST_DATA_PATH}')

# 加载模型
print('\n加载模型...')
pipeline = StreamingKWSPipeline(config)
pipeline.load()

def test_file(audio_path: Path) -> tuple:
    """测试单个文件，返回(是否检测到, 处理时间ms)"""
    pipeline.reset()
    
    audio, sr = sf.read(audio_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)
    
    start_time = time.perf_counter()
    
    # 分块处理（模拟流式）
    chunk_size = 1600  # 100ms
    detected = False
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        result = pipeline.process_chunk(chunk)
        if result:
            detected = True
            break
    
    process_time = (time.perf_counter() - start_time) * 1000
    audio_duration = len(audio) / 16000 * 1000
    
    return detected, process_time, audio_duration

# 获取测试文件
positive_dir = Path(TEST_DATA_PATH) / 'positive'
negative_dir = Path(TEST_DATA_PATH) / 'negative'
positive_files = sorted(positive_dir.glob('*.wav'))
negative_files = sorted(negative_dir.glob('*.wav'))

print(f'\n正样本: {len(positive_files)} 个')
print(f'负样本: {len(negative_files)} 个')

# 测试正样本
print('\n' + '-' * 60)
print('测试正样本...')
print('-' * 60)

tp = 0  # True Positive
fn = 0  # False Negative
pos_times = []
pos_durations = []

for i, f in enumerate(positive_files):
    detected, proc_time, audio_dur = test_file(f)
    pos_times.append(proc_time)
    pos_durations.append(audio_dur)
    if detected:
        tp += 1
    else:
        fn += 1
    if (i + 1) % 50 == 0:
        print(f'  已测试 {i+1}/{len(positive_files)}')

print(f'正样本结果: TP={tp}, FN={fn}')

# 测试负样本
print('\n' + '-' * 60)
print('测试负样本...')
print('-' * 60)

fp = 0  # False Positive
tn = 0  # True Negative
neg_times = []
neg_durations = []

for i, f in enumerate(negative_files):
    detected, proc_time, audio_dur = test_file(f)
    neg_times.append(proc_time)
    neg_durations.append(audio_dur)
    if detected:
        fp += 1
    else:
        tn += 1
    if (i + 1) % 100 == 0:
        print(f'  已测试 {i+1}/{len(negative_files)}')

print(f'负样本结果: FP={fp}, TN={tn}')

# 计算指标
frr = fn / (tp + fn) if (tp + fn) > 0 else 0
far = fp / (fp + tn) if (fp + tn) > 0 else 0
accuracy = (tp + tn) / (tp + fn + fp + tn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

all_times = pos_times + neg_times
all_durations = pos_durations + neg_durations
avg_time = np.mean(all_times)
avg_duration = np.mean(all_durations)
rtf = avg_time / avg_duration

# 输出结果
print('\n' + '=' * 60)
print('测试结果 (HuggingFace ONNX版本)')
print('=' * 60)
print(f'\n样本统计:')
print(f'  正样本: {len(positive_files)}')
print(f'  负样本: {len(negative_files)}')
print(f'  总计: {len(positive_files) + len(negative_files)}')

print(f'\n混淆矩阵:')
print(f'  TP (正确唤醒): {tp}')
print(f'  FN (漏唤醒): {fn}')
print(f'  FP (误唤醒): {fp}')
print(f'  TN (正确拒绝): {tn}')

print(f'\n性能指标:')
print(f'  FRR (漏唤醒率): {frr*100:.2f}%')
print(f'  FAR (误唤醒率): {far*100:.2f}%')
print(f'  准确率: {accuracy*100:.2f}%')
print(f'  精确率: {precision*100:.2f}%')
print(f'  召回率: {recall*100:.2f}%')
print(f'  F1分数: {f1*100:.2f}%')

print(f'\n时间性能:')
print(f'  平均处理时间: {avg_time:.2f}ms')
print(f'  平均音频时长: {avg_duration:.2f}ms')
print(f'  RTF: {rtf:.4f}')

print(f'\n目标达成:')
target_frr = 0.05
target_far = 0.10
frr_ok = '✓' if frr <= target_frr else '✗'
far_ok = '✓' if far <= target_far else '✗'
print(f'  FRR ≤ 5%: {frr_ok} (实际: {frr*100:.2f}%)')
print(f'  FAR ≤ 10%: {far_ok} (实际: {far*100:.2f}%)')

if frr <= target_frr and far <= target_far:
    print('\n🎉 达到目标!')
else:
    print('\n❌ 未达目标')
