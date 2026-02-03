#!/usr/bin/env python3
"""测试V4 epoch-98模型在全406样本上的性能"""

import sys
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import sherpa_onnx
from dataclasses import dataclass
from typing import List

BASE_DIR = Path("/data/workspace/llm/keyword-spotting")
EXP_DIR = BASE_DIR / "experiments/baseline_streaming/exp_v4"

# 正样本关键词
POSITIVE_KEYWORDS = ["你好真真", "你好珍珍", "你好甄甄", "你好臻臻", "你好桢桢"]

def extract_text(filename: str) -> str:
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem

def is_positive(text: str) -> bool:
    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            return True
    return False

def create_kws_model(threshold: float = 0.25) -> sherpa_onnx.KeywordSpotter:
    """创建使用epoch-98的KWS模型"""
    return sherpa_onnx.KeywordSpotter(
        encoder=str(EXP_DIR / "encoder-epoch-98-avg-1-chunk-16-left-128.onnx"),
        decoder=str(EXP_DIR / "decoder-epoch-98-avg-1-chunk-16-left-128.onnx"),
        joiner=str(EXP_DIR / "joiner-epoch-98-avg-1-chunk-16-left-128.onnx"),
        tokens=str(EXP_DIR / "tokens.txt"),
        keywords_file=str(EXP_DIR / "keywords.txt"),
        keywords_threshold=threshold,
        keywords_score=1.5,
        num_threads=2,
        provider="cpu",
    )

def evaluate_file(model: sherpa_onnx.KeywordSpotter, audio_path: Path):
    """评估单个音频文件"""
    import time
    
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    stream = model.create_stream()
    
    # 流式处理
    chunk_size = int(0.03 * sr)  # 30ms
    detected = False
    
    start_time = time.perf_counter()
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        stream.accept_waveform(sr, chunk.tolist())
        
        while model.is_ready(stream):
            model.decode_stream(stream)
        
        result = model.get_result(stream)
        # result 可能是字符串或对象
        if isinstance(result, str):
            if result and result.strip():
                detected = True
                break
        elif hasattr(result, 'keyword') and result.keyword:
            detected = True
            break
    
    inference_time = (time.perf_counter() - start_time) * 1000
    audio_duration = len(audio) / sr * 1000
    
    return {
        "detected": detected,
        "inference_time_ms": inference_time,
        "audio_duration_ms": audio_duration,
        "rtf": inference_time / audio_duration,
    }

def evaluate_dataset(model: sherpa_onnx.KeywordSpotter, data_dir: Path):
    """评估数据集"""
    tp = tn = fp = fn = 0
    rtf_list = []
    
    audio_files = list(data_dir.glob("*.wav"))
    
    for audio_path in audio_files:
        text = extract_text(audio_path.name)
        label = 1 if is_positive(text) else 0
        
        result = evaluate_file(model, audio_path)
        detected = result["detected"]
        rtf_list.append(result["rtf"])
        
        if label == 1:
            if detected:
                tp += 1
            else:
                fn += 1
        else:
            if detected:
                fp += 1
            else:
                tn += 1
    
    total = tp + tn + fp + fn
    positive = tp + fn
    negative = tn + fp
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far = fp / negative if negative > 0 else 0  # False Accept Rate
    frr = fn / positive if positive > 0 else 0  # False Reject Rate
    avg_rtf = np.mean(rtf_list) if rtf_list else 0
    
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "frr": frr,
        "avg_rtf": avg_rtf,
    }

def main():
    print("=" * 80)
    print("V4 Epoch-98 模型评估 (全406样本)")
    print("=" * 80)
    print()
    
    data_dir = BASE_DIR / "data/all"
    
    # 确保keywords.txt存在
    keywords_file = EXP_DIR / "keywords.txt"
    if not keywords_file.exists():
        with open(keywords_file, "w") as f:
            f.write("n ǐ h ǎo zh ēn zh ēn @你好真真\n")
    
    # 确保tokens.txt存在
    tokens_file = EXP_DIR / "tokens.txt"
    if not tokens_file.exists():
        import shutil
        shutil.copy(BASE_DIR / "data/lang_partial_tone/tokens.txt", tokens_file)
    
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    results = []
    
    for threshold in thresholds:
        print(f"测试阈值: {threshold:.2f}")
        model = create_kws_model(threshold)
        metrics = evaluate_dataset(model, data_dir)
        
        passed = metrics["far"] < 0.10 and metrics["frr"] < 0.05
        
        print(f"  FAR={metrics['far']*100:.2f}%, FRR={metrics['frr']*100:.2f}%, "
              f"F1={metrics['f1']*100:.2f}%, RTF={metrics['avg_rtf']:.4f} "
              f"{'✓' if passed else '✗'}")
        
        results.append({
            "threshold": threshold,
            **metrics,
            "passed": passed,
        })
    
    print()
    print("=" * 80)
    print("结果汇总表格")
    print("=" * 80)
    print()
    print(f"{'阈值':>6} {'FAR':>8} {'FRR':>8} {'准确率':>8} {'F1':>8} {'RTF':>8} {'达标':>6}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['threshold']:>6.2f} {r['far']*100:>7.2f}% {r['frr']*100:>7.2f}% "
              f"{r['accuracy']*100:>7.2f}% {r['f1']*100:>7.2f}% {r['avg_rtf']:>8.4f} "
              f"{'✓' if r['passed'] else '✗':>6}")
    
    # 找最佳配置
    best = None
    for r in results:
        if r["passed"]:
            if best is None or r["f1"] > best["f1"]:
                best = r
    
    print()
    if best:
        print(f"最佳配置: 阈值={best['threshold']:.2f}, FAR={best['far']*100:.2f}%, "
              f"FRR={best['frr']*100:.2f}%, F1={best['f1']*100:.2f}%")
    else:
        print("警告: 没有配置同时满足 FAR<10% 和 FRR<5%")

if __name__ == "__main__":
    main()
