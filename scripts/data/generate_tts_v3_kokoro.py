#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Data Generation Script V3 - Kokoro Version
Generates positive samples using Kokoro-82M with multiple Chinese voices.
Based on plan_v2.md recommendations for high-quality synthetic data.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro


# Configuration
MODEL_PATH = "/data/workspace/llm/keyword-spotting/models/kokoro/kokoro-v1.0.onnx"
VOICES_PATH = "/data/workspace/llm/keyword-spotting/models/kokoro/voices-v1.0.bin"
OUTPUT_DIR = Path("/data/workspace/llm/keyword-spotting/data/raw_tts_v3/positive")

# Chinese voices in Kokoro
CHINESE_VOICES = [
    "zf_xiaobei",   # Female
    "zf_xiaoni",    # Female
    "zf_xiaoxiao",  # Female
    "zf_xiaoyi",    # Female
    "zm_yunjian",   # Male
    "zm_yunxi",     # Male
    "zm_yunxia",    # Male
    "zm_yunyang",   # Male
]

# Target keyword variations (based on plan_v2.md tone analysis)
# Standard: n ǐ h ǎo zh ēn zh ēn
# 上上变调: ni2 hao3 -> 实际发音第一个字变二声
# 轻声变体: zhen1 zhen0/zhen5 -> 第二个真可能轻声
TARGET_TEXTS = [
    "你好真真",      # Standard
]

# Speed variations (0.8x - 1.2x)
SPEED_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]

# Target: 1000+ samples = 8 voices × 5 speeds × ~25 variations each
# We'll generate multiple takes per combination to increase diversity


def create_sample_id(voice: str, text_idx: int, speed: float, take: int) -> str:
    """Create unique sample identifier."""
    speed_str = f"s{int(speed*100)}"
    return f"kokoro_{voice}_t{text_idx}_{speed_str}_take{take:02d}"


def generate_samples(kokoro: Kokoro, output_dir: Path, num_takes: int = 5) -> List[Dict]:
    """Generate TTS samples with various combinations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    total_generated = 0
    failed = 0

    total_combinations = len(CHINESE_VOICES) * len(TARGET_TEXTS) * len(SPEED_VALUES) * num_takes
    print(f"\nGenerating {total_combinations} samples...")
    print(f"Voices: {len(CHINESE_VOICES)}")
    print(f"Texts: {len(TARGET_TEXTS)}")
    print(f"Speeds: {len(SPEED_VALUES)}")
    print(f"Takes per combination: {num_takes}")

    start_time = time.time()

    for voice in CHINESE_VOICES:
        for text_idx, text in enumerate(TARGET_TEXTS):
            for speed in SPEED_VALUES:
                for take in range(num_takes):
                    sample_id = create_sample_id(voice, text_idx, speed, take)
                    output_path = output_dir / f"{sample_id}.wav"

                    if output_path.exists():
                        # Load existing metadata
                        try:
                            info = sf.info(str(output_path))
                            metadata.append({
                                "id": sample_id,
                                "file": output_path.name,
                                "voice": voice,
                                "text": text,
                                "speed": speed,
                                "take": take,
                                "duration": info.duration,
                                "sample_rate": info.samplerate,
                                "status": "skipped"
                            })
                            total_generated += 1
                        except Exception:
                            pass
                        continue

                    try:
                        # Generate audio
                        audio, sr = kokoro.create(
                            text=text,
                            voice=voice,
                            speed=speed,
                            lang="cmn"
                        )

                        # Note: Keep original sample rate (24kHz)
                        # Lhotse will handle resampling during training

                        # Save audio
                        sf.write(str(output_path), audio, sr)

                        duration = len(audio) / sr
                        metadata.append({
                            "id": sample_id,
                            "file": output_path.name,
                            "voice": voice,
                            "text": text,
                            "speed": speed,
                            "take": take,
                            "duration": round(duration, 3),
                            "sample_rate": sr,
                            "status": "success"
                        })
                        total_generated += 1

                        if total_generated % 50 == 0:
                            elapsed = time.time() - start_time
                            rate = total_generated / elapsed
                            remaining = (total_combinations - total_generated) / rate if rate > 0 else 0
                            print(f"  Generated: {total_generated}/{total_combinations} "
                                  f"({total_generated/total_combinations*100:.1f}%) "
                                  f"ETA: {remaining:.0f}s")

                    except Exception as e:
                        failed += 1
                        print(f"  Failed {sample_id}: {e}")
                        metadata.append({
                            "id": sample_id,
                            "voice": voice,
                            "text": text,
                            "speed": speed,
                            "take": take,
                            "status": "failed",
                            "error": str(e)
                        })

    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.1f}s")
    print(f"  Success: {total_generated}")
    print(f"  Failed: {failed}")

    return metadata


def save_metadata(metadata: List[Dict], output_dir: Path) -> None:
    """Save metadata to JSON file."""
    metadata_path = output_dir / "metadata.json"

    # Statistics
    success_count = sum(1 for m in metadata if m.get("status") == "success")
    skipped_count = sum(1 for m in metadata if m.get("status") == "skipped")
    failed_count = sum(1 for m in metadata if m.get("status") == "failed")

    total_duration = sum(m.get("duration", 0) for m in metadata if m.get("status") in ["success", "skipped"])

    output = {
        "generated_at": datetime.now().isoformat(),
        "generator": "kokoro-82m",
        "model_path": MODEL_PATH,
        "voices": CHINESE_VOICES,
        "target_texts": TARGET_TEXTS,
        "speed_values": SPEED_VALUES,
        "statistics": {
            "total": len(metadata),
            "success": success_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "total_duration_sec": round(total_duration, 2),
            "avg_duration_sec": round(total_duration / max(success_count + skipped_count, 1), 3)
        },
        "samples": metadata
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"  Total samples: {success_count + skipped_count}")
    print(f"  Total duration: {total_duration/60:.1f} minutes")


def main():
    print("=" * 60)
    print("KWS TTS Data Generation V3 - Kokoro-82M")
    print("=" * 60)

    # Initialize Kokoro
    print(f"\nLoading Kokoro model from: {MODEL_PATH}")
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    print(f"Model loaded successfully!")
    print(f"Available Chinese voices: {CHINESE_VOICES}")

    # Generate samples
    # Target: 1000+ samples
    # 8 voices × 1 text × 5 speeds × 25 takes = 1000 samples
    num_takes = 25  # Increase takes to get 1000 samples

    metadata = generate_samples(kokoro, OUTPUT_DIR, num_takes=num_takes)
    save_metadata(metadata, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
