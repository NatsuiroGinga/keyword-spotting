#!/usr/bin/env python3
"""V4 Epoch-98 细粒度阈值搜索"""

import numpy as np
import soundfile as sf
import sherpa_onnx
from pathlib import Path

BASE_DIR = Path('/data/workspace/llm/keyword-spotting')
EXP_DIR = BASE_DIR / 'experiments/baseline_streaming/exp_v4'
data_dir = BASE_DIR / 'data/all'

POSITIVE_KEYWORDS = ['你好真真', '你好珍珍', '你好甄甄', '你好臻臻', '你好桢桢']

def is_positive(text):
    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            return True
    return False

def extract_text(filename):
    stem = Path(filename).stem
    if '_' in stem:
        return stem.split('_', 1)[1]
    return stem

def test_threshold(threshold):
    model = sherpa_onnx.KeywordSpotter(
        encoder=str(EXP_DIR / 'encoder-epoch-98-avg-1-chunk-16-left-128.onnx'),
        decoder=str(EXP_DIR / 'decoder-epoch-98-avg-1-chunk-16-left-128.onnx'),
        joiner=str(EXP_DIR / 'joiner-epoch-98-avg-1-chunk-16-left-128.onnx'),
        tokens=str(EXP_DIR / 'tokens.txt'),
        keywords_file=str(EXP_DIR / 'keywords.txt'),
        keywords_threshold=threshold,
        keywords_score=1.5,
        num_threads=2,
        provider='cpu',
    )
    
    tp = tn = fp = fn = 0
    
    for audio_path in data_dir.glob('*.wav'):
        text = extract_text(audio_path.name)
        label = 1 if is_positive(text) else 0
        
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        stream = model.create_stream()
        chunk_size = int(0.03 * sr)
        detected = False
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            stream.accept_waveform(sr, chunk.tolist())
            while model.is_ready(stream):
                model.decode_stream(stream)
            result = model.get_result(stream)
            if isinstance(result, str) and result.strip():
                detected = True
                break
            elif hasattr(result, 'keyword') and result.keyword:
                detected = True
                break
        
        if label == 1:
            if detected: tp += 1
            else: fn += 1
        else:
            if detected: fp += 1
            else: tn += 1
    
    positive = tp + fn
    negative = tn + fp
    far = fp / negative if negative > 0 else 0
    frr = fn / positive if positive > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    passed = far < 0.10 and frr < 0.05
    
    return far, frr, f1, acc, passed, tp, fp, fn, tn

def main():
    print('V4 Epoch-98 细粒度阈值搜索')
    print('='*70)
    print(f"{'阈值':>6} {'FAR':>8} {'FRR':>8} {'F1':>8} {'准确率':>8} {'达标':>6}")
    print('-'*70)

    # 在0.45-0.55之间细粒度搜索
    thresholds = [0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54]
    best = None
    results = []

    for t in thresholds:
        far, frr, f1, acc, passed, tp, fp, fn, tn = test_threshold(t)
        mark = '✓' if passed else '✗'
        print(f'{t:>6.2f} {far*100:>7.2f}% {frr*100:>7.2f}% {f1*100:>7.2f}% {acc*100:>7.2f}% {mark:>6}')
        results.append((t, far, frr, f1, acc, passed, tp, fp, fn, tn))
        if passed and (best is None or f1 > best[3]):
            best = (t, far, frr, f1, acc)

    print()
    if best:
        print(f'最佳配置: 阈值={best[0]:.2f}, FAR={best[1]*100:.2f}%, FRR={best[2]*100:.2f}%, F1={best[3]*100:.2f}%')
    else:
        print('没有配置同时满足 FAR<10% 和 FRR<5%')
        # 找最接近的
        min_gap = float('inf')
        closest = None
        for r in results:
            t, far, frr, f1, acc, passed, _, _, _, _ = r
            gap = max(0, far - 0.10) + max(0, frr - 0.05)
            if gap < min_gap:
                min_gap = gap
                closest = (t, far, frr, f1, acc)
        if closest:
            print(f'最接近的配置: 阈值={closest[0]:.2f}, FAR={closest[1]*100:.2f}%, FRR={closest[2]*100:.2f}%, F1={closest[3]*100:.2f}%')

if __name__ == "__main__":
    main()
