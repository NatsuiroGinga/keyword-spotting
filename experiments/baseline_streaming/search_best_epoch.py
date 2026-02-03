#!/usr/bin/env python3
"""
在epoch 30-100之间搜索最佳权重
策略：
1. 粗粒度搜索：每5个epoch测试一次
2. 细粒度搜索：在最优区间内逐个epoch测试
3. 对每个epoch进行阈值优化，找到最佳阈值下的性能
"""

import os
import sys
import subprocess
import numpy as np
import soundfile as sf
import sherpa_onnx
from pathlib import Path
import json
from datetime import datetime

BASE_DIR = Path('/data/workspace/llm/keyword-spotting')
EXP_DIR = BASE_DIR / 'experiments/baseline_streaming/exp_v4'
ICEFALL_DIR = BASE_DIR / 'icefall/egs/wenetspeech/KWS'
DATA_DIR = BASE_DIR / 'data/all'
TOKENS = BASE_DIR / 'data/lang_partial_tone/tokens.txt'

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

def export_onnx(epoch):
    """导出指定epoch的ONNX模型"""
    encoder_file = EXP_DIR / f'encoder-epoch-{epoch}-avg-1-chunk-16-left-128.onnx'
    if encoder_file.exists():
        return True
    
    print(f"  导出epoch-{epoch} ONNX...")
    
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{BASE_DIR}/icefall:{env.get('PYTHONPATH', '')}"
    
    cmd = [
        '/data/workspace/llm/anaconda3/envs/kws-train/bin/python',
        './zipformer/export-onnx-streaming.py',
        '--exp-dir', str(EXP_DIR),
        '--tokens', str(TOKENS),
        '--epoch', str(epoch),
        '--avg', '1',
        '--use-averaged-model', '0',
        '--chunk-size', '16',
        '--left-context-frames', '128',
        '--decoder-dim', '320',
        '--joiner-dim', '320',
        '--num-encoder-layers', '1,1,1,1,1,1',
        '--feedforward-dim', '192,192,192,192,192,192',
        '--encoder-dim', '128,128,128,128,128,128',
        '--encoder-unmasked-dim', '128,128,128,128,128,128',
        '--causal', '1',
    ]
    
    result = subprocess.run(cmd, cwd=str(ICEFALL_DIR), env=env, 
                           capture_output=True, text=True)
    
    return encoder_file.exists()

def test_threshold(epoch, threshold):
    """测试指定epoch和阈值的性能"""
    encoder = EXP_DIR / f'encoder-epoch-{epoch}-avg-1-chunk-16-left-128.onnx'
    decoder = EXP_DIR / f'decoder-epoch-{epoch}-avg-1-chunk-16-left-128.onnx'
    joiner = EXP_DIR / f'joiner-epoch-{epoch}-avg-1-chunk-16-left-128.onnx'
    
    if not all([encoder.exists(), decoder.exists(), joiner.exists()]):
        return None
    
    keywords_file = EXP_DIR / 'keywords.txt'
    if not keywords_file.exists():
        with open(keywords_file, 'w') as f:
            f.write('n ǐ h ǎo zh ēn zh ēn @你好真真\n')
    
    model = sherpa_onnx.KeywordSpotter(
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        tokens=str(TOKENS),
        keywords_file=str(keywords_file),
        keywords_threshold=threshold,
        keywords_score=1.5,
        num_threads=2,
        provider='cpu',
    )
    
    tp = tn = fp = fn = 0
    
    for audio_path in DATA_DIR.glob('*.wav'):
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
    
    return {
        'far': far, 'frr': frr, 'f1': f1, 'acc': acc, 'passed': passed,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

def find_best_threshold(epoch):
    """对指定epoch进行阈值搜索，找到最佳阈值"""
    # 粗粒度搜索
    coarse_thresholds = [0.3, 0.4, 0.5, 0.6]
    best_coarse = None
    best_coarse_f1 = 0
    
    for t in coarse_thresholds:
        result = test_threshold(epoch, t)
        if result and result['passed'] and result['f1'] > best_coarse_f1:
            best_coarse = t
            best_coarse_f1 = result['f1']
    
    if best_coarse is None:
        # 没有达标的，找最接近的
        for t in coarse_thresholds:
            result = test_threshold(epoch, t)
            if result and result['f1'] > best_coarse_f1:
                best_coarse = t
                best_coarse_f1 = result['f1']
        if best_coarse is None:
            best_coarse = 0.5
    
    # 细粒度搜索 (±0.1范围内)
    fine_thresholds = np.arange(max(0.1, best_coarse - 0.1), 
                                min(0.9, best_coarse + 0.1) + 0.01, 0.02)
    
    best_threshold = best_coarse
    best_result = test_threshold(epoch, best_coarse)
    best_f1 = best_result['f1'] if best_result else 0
    
    for t in fine_thresholds:
        result = test_threshold(epoch, t)
        if result:
            # 优先选择达标且F1最高的
            if result['passed']:
                if not best_result or not best_result['passed'] or result['f1'] > best_f1:
                    best_threshold = t
                    best_result = result
                    best_f1 = result['f1']
            elif not best_result or (not best_result['passed'] and result['f1'] > best_f1):
                best_threshold = t
                best_result = result
                best_f1 = result['f1']
    
    return best_threshold, best_result

def main():
    print("=" * 80)
    print("V4模型 Epoch 30-100 最佳权重搜索")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 第一阶段：粗粒度搜索 (每5个epoch)
    print("【阶段1】粗粒度搜索 (每5个epoch)")
    print("-" * 80)
    
    coarse_epochs = list(range(30, 101, 5))
    coarse_results = []
    
    for epoch in coarse_epochs:
        checkpoint = EXP_DIR / f'epoch-{epoch}.pt'
        if not checkpoint.exists():
            print(f"  epoch-{epoch}: 跳过 (checkpoint不存在)")
            continue
        
        print(f"  epoch-{epoch}:", end=" ", flush=True)
        
        if not export_onnx(epoch):
            print("导出失败")
            continue
        
        best_t, result = find_best_threshold(epoch)
        if result:
            mark = '✓' if result['passed'] else '✗'
            print(f"阈值={best_t:.2f}, FAR={result['far']*100:.2f}%, "
                  f"FRR={result['frr']*100:.2f}%, F1={result['f1']*100:.2f}% {mark}")
            coarse_results.append({
                'epoch': epoch, 'threshold': best_t, **result
            })
        else:
            print("测试失败")
    
    # 找到最优区间
    passed_results = [r for r in coarse_results if r['passed']]
    if passed_results:
        best_coarse = max(passed_results, key=lambda x: x['f1'])
    else:
        best_coarse = max(coarse_results, key=lambda x: x['f1']) if coarse_results else None
    
    if not best_coarse:
        print("没有找到有效结果")
        return
    
    print()
    print(f"粗粒度最佳: epoch-{best_coarse['epoch']}, F1={best_coarse['f1']*100:.2f}%")
    
    # 第二阶段：细粒度搜索 (最优epoch ±5范围内)
    print()
    print("【阶段2】细粒度搜索 (最优区间内逐个epoch)")
    print("-" * 80)
    
    best_epoch = best_coarse['epoch']
    fine_epochs = list(range(max(30, best_epoch - 5), min(100, best_epoch + 5) + 1))
    fine_epochs = [e for e in fine_epochs if e not in coarse_epochs]
    
    all_results = coarse_results.copy()
    
    for epoch in fine_epochs:
        checkpoint = EXP_DIR / f'epoch-{epoch}.pt'
        if not checkpoint.exists():
            continue
        
        print(f"  epoch-{epoch}:", end=" ", flush=True)
        
        if not export_onnx(epoch):
            print("导出失败")
            continue
        
        best_t, result = find_best_threshold(epoch)
        if result:
            mark = '✓' if result['passed'] else '✗'
            print(f"阈值={best_t:.2f}, FAR={result['far']*100:.2f}%, "
                  f"FRR={result['frr']*100:.2f}%, F1={result['f1']*100:.2f}% {mark}")
            all_results.append({
                'epoch': epoch, 'threshold': best_t, **result
            })
        else:
            print("测试失败")
    
    # 汇总结果
    print()
    print("=" * 80)
    print("最终结果汇总")
    print("=" * 80)
    print()
    print(f"{'Epoch':>6} {'阈值':>6} {'FAR':>8} {'FRR':>8} {'F1':>8} {'准确率':>8} {'达标':>6}")
    print("-" * 60)
    
    all_results.sort(key=lambda x: x['epoch'])
    for r in all_results:
        mark = '✓' if r['passed'] else '✗'
        print(f"{r['epoch']:>6} {r['threshold']:>6.2f} {r['far']*100:>7.2f}% "
              f"{r['frr']*100:>7.2f}% {r['f1']*100:>7.2f}% {r['acc']*100:>7.2f}% {mark:>6}")
    
    # 找最佳
    passed_all = [r for r in all_results if r['passed']]
    if passed_all:
        best = max(passed_all, key=lambda x: x['f1'])
        print()
        print(f"🏆 最佳配置: epoch-{best['epoch']}, 阈值={best['threshold']:.2f}")
        print(f"   FAR={best['far']*100:.2f}%, FRR={best['frr']*100:.2f}%, F1={best['f1']*100:.2f}%")
    else:
        best = max(all_results, key=lambda x: x['f1'])
        print()
        print(f"⚠ 没有完全达标的配置，最接近的: epoch-{best['epoch']}, 阈值={best['threshold']:.2f}")
        print(f"   FAR={best['far']*100:.2f}%, FRR={best['frr']*100:.2f}%, F1={best['f1']*100:.2f}%")
    
    # 保存结果
    output_file = EXP_DIR / 'best_epoch_search_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best': best,
            'all_results': all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
