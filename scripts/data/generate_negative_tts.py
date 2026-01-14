#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Negative Sample TTS Data Generation Script for KWS Fine-tuning

Generates negative samples (non-wake-word audio) using Edge-TTS.
These include:
1. Hard negatives: Similar-sounding phrases like "你好", "您好", etc.
2. General negatives: Random Chinese sentences
"""

import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
import edge_tts

# Configuration
OUTPUT_DIR = Path("/data/workspace/llm/keyword-spotting/data/raw_tts/negative")

# Hard negative samples - similar-sounding phrases that should NOT trigger wake word
HARD_NEGATIVE_TEXTS = [
    # Direct "你好" variants
    "你好",
    "您好",
    "你好啊",
    "你好吗",
    "你好呀",
    # Homophones and similar sounds
    "泥豪",
    "李浩",
    "倪好",
    "尼豪",
    "你昊",
    # Partial matches
    "真真",
    "真真好",
    "好真真",
    # Extended phrases containing "你好"
    "你好朋友",
    "你好世界",
]

# General negative samples - random phrases
GENERAL_NEGATIVE_TEXTS = [
    # Greetings (different from wake word)
    "早上好",
    "晚上好",
    "下午好",
    "大家好",
    # Numbers
    "一二三四五",
    "六七八九十",
    # Common commands
    "打开灯",
    "关闭音乐",
    "播放歌曲",
    "调高音量",
    "设置闹钟",
    # Random sentences
    "今天天气很好",
    "明天去上班",
    "我想吃饭",
    "请稍等一下",
    "谢谢你",
    "不客气",
    "再见",
    "好的",
    "没问题",
    "知道了",
]

# Use fewer variations for negative samples to avoid too much data imbalance
RATES = ["-20%", "+0%", "+20%"]
PITCHES = ["-10Hz", "+0Hz", "+10Hz"]


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

    # Use a subset of voices to keep data size manageable
    # Select 5 diverse voices
    selected_voices = []
    for v in voices:
        if any(name in v for name in ['Yunjian', 'Xiaoyi', 'Xiaoxiao', 'Yunxi', 'Xiaobei']):
            selected_voices.append(v)

    if len(selected_voices) < 5:
        selected_voices = voices[:5]

    print(f"Using {len(selected_voices)} voices: {selected_voices}")

    metadata = []
    generated_count = 0
    failed_count = 0

    sem = asyncio.Semaphore(5)

    async def generate_one(voice: str, text: str, rate: str, pitch: str, category: str) -> dict:
        nonlocal generated_count, failed_count
        async with sem:
            short_voice = voice.split('-')[-1].replace("Neural", "")
            safe_rate = rate.replace("%", "pct").replace("+", "p").replace("-", "n")
            safe_pitch = pitch.replace("Hz", "hz").replace("+", "p").replace("-", "n")

            # Create a safe text ID
            text_id = text.replace(" ", "_")
            # Convert Chinese to pinyin-like representation for filename
            text_hash = hash(text) % 10000

            filename = f"{category}_{short_voice}_{text_hash}_{safe_rate}_{safe_pitch}.wav"
            filepath = OUTPUT_DIR / filename

            if filepath.exists():
                return {"status": "skipped", "file": filename}

            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            try:
                await communicate.save(str(filepath))
                generated_count += 1
                if generated_count % 20 == 0:
                    print(f"Generated [{generated_count}]: {filename}")
                return {
                    "status": "success",
                    "file": filename,
                    "voice": voice,
                    "text": text,
                    "category": category,
                    "rate": rate,
                    "pitch": pitch,
                    "filepath": str(filepath)
                }
            except Exception as e:
                failed_count += 1
                print(f"Failed {filename}: {e}")
                return {"status": "failed", "file": filename, "error": str(e)}

    tasks = []

    # Generate hard negatives with all variations
    for voice in selected_voices:
        for text in HARD_NEGATIVE_TEXTS:
            for rate in RATES:
                for pitch in PITCHES:
                    tasks.append(generate_one(voice, text, rate, pitch, "hard_neg"))

    # Generate general negatives with fewer variations
    for voice in selected_voices[:3]:  # Use only 3 voices for general negatives
        for text in GENERAL_NEGATIVE_TEXTS:
            for rate in ["+0%"]:  # Only neutral rate
                for pitch in ["+0Hz"]:  # Only neutral pitch
                    tasks.append(generate_one(voice, text, rate, pitch, "gen_neg"))

    print(f"Starting generation of {len(tasks)} negative samples...")
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
            "hard_negative_texts": HARD_NEGATIVE_TEXTS,
            "general_negative_texts": GENERAL_NEGATIVE_TEXTS,
            "rates": RATES,
            "pitches": PITCHES,
            "samples": metadata
        }, f, ensure_ascii=False, indent=2)

    print(f"\nGeneration complete!")
    print(f"  Generated: {generated_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Metadata saved to: {metadata_path}")

    # Summary by category
    hard_neg_count = sum(1 for m in metadata if m.get("category") == "hard_neg")
    gen_neg_count = sum(1 for m in metadata if m.get("category") == "gen_neg")
    print(f"\nBy category:")
    print(f"  Hard negatives: {hard_neg_count}")
    print(f"  General negatives: {gen_neg_count}")


if __name__ == "__main__":
    asyncio.run(generate_samples())
