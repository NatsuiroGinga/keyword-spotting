#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS Data Generation Script V3 - Hard Negative Samples
Generates hard negative samples using Edge-TTS with multiple Chinese voices.
Based on plan_v2.md recommendations for adversarial training.

Hard Negatives Categories:
1. Prefix words: 你好, 您好 (most critical)
2. Extended prefix: 你好啊, 你好吗, 你好呀
3. Homophones: 泥豪, 李浩, 倪好
4. Suffix only: 真真
5. Long sentences containing prefix: "你好，今天天气不错"
"""

import asyncio
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import edge_tts


# Configuration
OUTPUT_DIR = Path("/data/workspace/llm/keyword-spotting/data/raw_tts_v3/negative")

# Hard negative categories with priorities
HARD_NEGATIVES = {
    # Category 1: Pure prefix (HIGHEST priority) - Target: 2000 samples
    "prefix": {
        "texts": ["你好", "您好"],
        "priority": "critical",
        "target_count": 2000,
    },
    # Category 2: Extended prefix (HIGH priority) - Target: 1500 samples
    "extended_prefix": {
        "texts": ["你好啊", "你好吗", "你好呀", "您好啊", "你好嘛"],
        "priority": "high",
        "target_count": 1500,
    },
    # Category 3: Homophones (MEDIUM-HIGH priority) - Target: 1000 samples
    "homophones": {
        "texts": ["泥豪", "李浩", "倪好", "拟好", "你濠", "尼好"],
        "priority": "medium_high",
        "target_count": 1000,
    },
    # Category 4: Suffix only (MEDIUM priority) - Target: 500 samples
    "suffix": {
        "texts": ["真真", "真真你好", "珍珍", "甄真"],
        "priority": "medium",
        "target_count": 500,
    },
    # Category 5: Long sentences with prefix (MEDIUM priority) - Target: 1500 samples
    "long_sentences": {
        "texts": [
            "你好，今天天气不错",
            "你好，请问现在几点了",
            "你好，我想问一下",
            "你好吗，最近怎么样",
            "你好啊，好久不见",
            "您好，请问有什么可以帮您的",
            "你好，我是新来的",
            "你好呀，我们聊聊天吧",
        ],
        "priority": "medium",
        "target_count": 1500,
    },
    # Category 6: General negatives (LOW priority) - Target: 500 samples
    "general": {
        "texts": [
            "早上好", "下午好", "晚上好",
            "谢谢", "不客气", "再见",
            "好的", "没问题", "可以",
            "今天天气真好", "明天见",
        ],
        "priority": "low",
        "target_count": 500,
    },
}

# Prosody variations for Edge-TTS
RATES = ["-20%", "-10%", "+0%", "+10%", "+20%"]
PITCHES = ["-10Hz", "+0Hz", "+10Hz"]


async def get_chinese_voices() -> List[str]:
    """Get Chinese voices from Edge-TTS."""
    voices = await edge_tts.list_voices()
    zh_voices = [
        v['ShortName'] for v in voices
        if "zh-CN" in v.get('Locale', '') or "zh-TW" in v.get('Locale', '')
    ]
    return zh_voices


async def generate_category_samples(
    category_name: str,
    category_config: Dict,
    voices: List[str],
    output_dir: Path,
    sem: asyncio.Semaphore,
) -> List[Dict]:
    """Generate samples for a specific category."""
    texts = category_config["texts"]
    target_count = category_config["target_count"]
    priority = category_config["priority"]

    # Calculate samples per text
    samples_per_text = target_count // len(texts)
    # Calculate how many voice/rate/pitch combinations we need
    combinations_per_text = samples_per_text // (len(RATES) * len(PITCHES))
    voices_to_use = voices[:max(1, combinations_per_text)]

    category_dir = output_dir / category_name
    category_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    generated = 0
    failed = 0

    tasks = []

    async def generate_one(text: str, voice: str, rate: str, pitch: str) -> Dict:
        nonlocal generated, failed

        async with sem:
            # Create filename
            text_id = text[:10].replace(" ", "_").replace("，", "_")
            voice_short = voice.split("-")[-1].replace("Neural", "")
            rate_safe = rate.replace("%", "pct").replace("+", "p").replace("-", "n")
            pitch_safe = pitch.replace("Hz", "hz").replace("+", "p").replace("-", "n")

            filename = f"{category_name}_{text_id}_{voice_short}_{rate_safe}_{pitch_safe}.wav"
            filepath = category_dir / filename

            if filepath.exists():
                return {
                    "category": category_name,
                    "priority": priority,
                    "text": text,
                    "voice": voice,
                    "rate": rate,
                    "pitch": pitch,
                    "file": filename,
                    "status": "skipped"
                }

            try:
                communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                await communicate.save(str(filepath))
                generated += 1

                return {
                    "category": category_name,
                    "priority": priority,
                    "text": text,
                    "voice": voice,
                    "rate": rate,
                    "pitch": pitch,
                    "file": filename,
                    "filepath": str(filepath),
                    "status": "success"
                }
            except Exception as e:
                failed += 1
                return {
                    "category": category_name,
                    "text": text,
                    "voice": voice,
                    "status": "failed",
                    "error": str(e)
                }

    # Build task list
    for text in texts:
        for voice in voices_to_use:
            for rate in RATES:
                for pitch in PITCHES:
                    tasks.append(generate_one(text, voice, rate, pitch))

    print(f"  Category '{category_name}': {len(tasks)} samples planned (target: {target_count})")

    # Execute tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, dict):
            metadata.append(result)
        else:
            print(f"  Error: {result}")

    success = sum(1 for m in metadata if m.get("status") == "success")
    skipped = sum(1 for m in metadata if m.get("status") == "skipped")
    print(f"    Generated: {success}, Skipped: {skipped}, Failed: {failed}")

    return metadata


async def main():
    print("=" * 70)
    print("KWS Hard Negative Sample Generation V3 - Edge-TTS")
    print("=" * 70)

    # Get Chinese voices
    voices = await get_chinese_voices()
    print(f"\nFound {len(voices)} Chinese voices")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Semaphore to limit concurrency
    sem = asyncio.Semaphore(10)

    all_metadata = []
    start_time = time.time()

    print("\nGenerating samples by category:")
    for category_name, category_config in HARD_NEGATIVES.items():
        category_metadata = await generate_category_samples(
            category_name,
            category_config,
            voices,
            OUTPUT_DIR,
            sem,
        )
        all_metadata.extend(category_metadata)

    elapsed = time.time() - start_time

    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"

    # Statistics by category
    category_stats = {}
    for category_name in HARD_NEGATIVES.keys():
        cat_samples = [m for m in all_metadata if m.get("category") == category_name]
        success = sum(1 for m in cat_samples if m.get("status") == "success")
        skipped = sum(1 for m in cat_samples if m.get("status") == "skipped")
        category_stats[category_name] = {
            "target": HARD_NEGATIVES[category_name]["target_count"],
            "actual": success + skipped,
            "priority": HARD_NEGATIVES[category_name]["priority"]
        }

    total_success = sum(1 for m in all_metadata if m.get("status") == "success")
    total_skipped = sum(1 for m in all_metadata if m.get("status") == "skipped")
    total_failed = sum(1 for m in all_metadata if m.get("status") == "failed")

    output = {
        "generated_at": datetime.now().isoformat(),
        "generator": "edge-tts",
        "elapsed_seconds": round(elapsed, 2),
        "categories": HARD_NEGATIVES,
        "category_statistics": category_stats,
        "statistics": {
            "total": len(all_metadata),
            "success": total_success,
            "skipped": total_skipped,
            "failed": total_failed,
        },
        "samples": all_metadata
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print("Generation Summary")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total samples: {total_success + total_skipped}")
    print("\nBy category:")
    for cat_name, stats in category_stats.items():
        print(f"  {cat_name}: {stats['actual']}/{stats['target']} ({stats['priority']})")
    print(f"\nMetadata saved to: {metadata_path}")


if __name__ == "__main__":
    asyncio.run(main())
