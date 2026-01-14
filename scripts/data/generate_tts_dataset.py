#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Data Generation Script for KWS Fine-tuning
Generates positive samples using Edge-TTS with multiple voices and prosody variations.
"""

import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
import edge_tts

# Configuration
OUTPUT_DIR = Path("/data/workspace/llm/keyword-spotting/data/raw_tts/positive")
TARGET_TEXTS = ["你好真真"]
RATES = ["-30%", "-20%", "-10%", "+0%", "+10%", "+20%", "+30%"]
PITCHES = ["-15Hz", "-10Hz", "-5Hz", "+0Hz", "+5Hz", "+10Hz", "+15Hz"]


async def get_chinese_voices():
    """Get all Chinese (zh-CN, zh-TW) voices"""
    voices = await edge_tts.list_voices()
    zh_voices = [
        v['ShortName'] for v in voices
        if "zh-CN" in v.get('Locale', '') or "zh-TW" in v.get('Locale', '')
    ]
    return zh_voices


async def generate_samples():
    """Generate TTS samples with various voice and prosody combinations"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voices = await get_chinese_voices()
    print(f"Found {len(voices)} Chinese voices: {voices}")

    metadata = []
    generated_count = 0
    failed_count = 0

    sem = asyncio.Semaphore(5)  # Limit concurrency to avoid rate limiting

    async def generate_one(voice: str, text: str, rate: str, pitch: str) -> dict:
        nonlocal generated_count, failed_count
        async with sem:
            # Create filename
            short_voice = voice.split('-')[-1].replace("Neural", "")
            safe_rate = rate.replace("%", "pct").replace("+", "p").replace("-", "n")
            safe_pitch = pitch.replace("Hz", "hz").replace("+", "p").replace("-", "n")

            # Map text to ID
            text_id_map = {
                "你好真真": "nihaozhenzhen",
                "真真你好": "zhenzhen_nihao",
                "真真": "zhenzhen"
            }
            text_id = text_id_map.get(text, "unknown")

            filename = f"{short_voice}_{text_id}_{safe_rate}_{safe_pitch}.wav"
            filepath = OUTPUT_DIR / filename

            if filepath.exists():
                return {"status": "skipped", "file": filename}

            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            try:
                await communicate.save(str(filepath))
                generated_count += 1
                print(f"Generated [{generated_count}]: {filename}")
                return {
                    "status": "success",
                    "file": filename,
                    "voice": voice,
                    "text": text,
                    "rate": rate,
                    "pitch": pitch,
                    "filepath": str(filepath)
                }
            except Exception as e:
                failed_count += 1
                print(f"Failed {filename}: {e}")
                return {"status": "failed", "file": filename, "error": str(e)}

    tasks = []
    for voice in voices:
        for text in TARGET_TEXTS:
            for rate in RATES:
                for pitch in PITCHES:
                    tasks.append(generate_one(voice, text, rate, pitch))

    print(f"Starting generation of {len(tasks)} samples...")
    results = await asyncio.gather(*tasks)

    # Collect successful metadata
    for result in results:
        if result.get("status") == "success":
            metadata.append(result)

    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_samples": len(metadata),
            "target_texts": TARGET_TEXTS,
            "rates": RATES,
            "pitches": PITCHES,
            "samples": metadata
        }, f, ensure_ascii=False, indent=2)

    print(f"\nGeneration complete!")
    print(f"  Generated: {generated_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    asyncio.run(generate_samples())
