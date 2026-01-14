#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lhotse Manifest Preparation Script V3
Prepares Lhotse manifests for KWS training with:
- Positive samples (target keyword)
- Hard negative samples (multiple categories)
- Sample weighting for balanced training
- Proper pinyin conversion for all samples

Based on plan_v2.md recommendations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
import re

from lhotse import CutSet, Recording, SupervisionSegment, RecordingSet, SupervisionSet
from lhotse.audio import info as audio_info
import soundfile as sf

# Import pypinyin for Chinese to pinyin conversion
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print("Warning: pypinyin not installed. Using predefined transcriptions only.")


# Configuration
BASE_DIR = Path("/data/workspace/llm/keyword-spotting")
DATA_DIR = BASE_DIR / "data/raw_tts_v3"
OUTPUT_DIR = BASE_DIR / "data/manifests_v3"

POSITIVE_DIR = DATA_DIR / "positive"
NEGATIVE_DIR = DATA_DIR / "negative"

# Target keyword transcription
TARGET_KEYWORD = "你好真真"
TARGET_PINYIN = "n ǐ h ǎo zh ēn zh ēn"

# Tone mark mapping for pypinyin output
TONE_MARKS = {
    'ā': 'ā', 'á': 'á', 'ǎ': 'ǎ', 'à': 'à',
    'ē': 'ē', 'é': 'é', 'ě': 'ě', 'è': 'è',
    'ī': 'ī', 'í': 'í', 'ǐ': 'ǐ', 'ì': 'ì',
    'ō': 'ō', 'ó': 'ó', 'ǒ': 'ǒ', 'ò': 'ò',
    'ū': 'ū', 'ú': 'ú', 'ǔ': 'ǔ', 'ù': 'ù',
    'ǖ': 'ǖ', 'ǘ': 'ǘ', 'ǚ': 'ǚ', 'ǜ': 'ǜ',
}

# Initials (声母) that need to be separated
INITIALS = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 
            'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'r', 'y', 'w']

# Predefined transcriptions for common phrases
PREDEFINED_TRANSCRIPTIONS = {
    # Target keyword
    "你好真真": "n ǐ h ǎo zh ēn zh ēn",
    
    # Prefix negatives
    "你好": "n ǐ h ǎo",
    "您好": "n ín h ǎo",
    
    # Extended prefix
    "你好啊": "n ǐ h ǎo ā",
    "你好吗": "n ǐ h ǎo m a",
    "你好呀": "n ǐ h ǎo y a",
    "您好啊": "n ín h ǎo ā",
    "你好嘛": "n ǐ h ǎo m a",
    
    # Homophones
    "泥豪": "n í h áo",
    "李浩": "l ǐ h ào",
    "倪好": "n í h ǎo",
    "拟好": "n ǐ h ǎo",
    "你濠": "n ǐ h áo",
    "尼好": "n í h ǎo",
    
    # Suffix
    "真真": "zh ēn zh ēn",
    "真真你好": "zh ēn zh ēn n ǐ h ǎo",
    "珍珍": "zh ēn zh ēn",
    "甄真": "zh ēn zh ēn",
}


def split_pinyin_syllable(syllable: str) -> str:
    """Split a pinyin syllable into initial + final with tone mark.
    
    Example: 'nǐ' -> 'n ǐ', 'hǎo' -> 'h ǎo', 'zhēn' -> 'zh ēn'
    """
    syllable = syllable.lower().strip()
    if not syllable:
        return ""
    
    # Try to match initials (longest first)
    for initial in sorted(INITIALS, key=len, reverse=True):
        if syllable.startswith(initial):
            final = syllable[len(initial):]
            if final:
                return f"{initial} {final}"
            else:
                return initial
    
    # No initial found, return as-is (pure vowel syllable like 'a', 'o', 'e')
    return syllable


def chinese_to_pinyin(text: str) -> str:
    """Convert Chinese text to space-separated pinyin with tone marks.
    
    Uses pypinyin library for conversion, then splits into initial+final format.
    """
    if not text:
        return ""
    
    # Check predefined transcriptions first
    if text in PREDEFINED_TRANSCRIPTIONS:
        return PREDEFINED_TRANSCRIPTIONS[text]
    
    if not HAS_PYPINYIN:
        return ""
    
    # Get pinyin with tone marks
    py_list = pinyin(text, style=Style.TONE, heteronym=False)
    
    # Process each syllable
    result_parts = []
    for py in py_list:
        syllable = py[0] if py else ""
        if syllable:
            split_syllable = split_pinyin_syllable(syllable)
            if split_syllable:
                result_parts.append(split_syllable)
    
    return " ".join(result_parts)


# Sample weights for training (higher = more frequently sampled)
SAMPLE_WEIGHTS = {
    "positive": 3.0,           # Highest weight for target keyword
    "prefix": 2.5,             # Critical - most common false positive
    "extended_prefix": 2.0,    # High priority
    "homophones": 1.5,         # Medium-high
    "suffix": 1.0,             # Medium
    "long_sentences": 1.0,     # Medium
    "general": 0.5,            # Low priority
}


def get_audio_info(audio_path: Path) -> Tuple[float, int, int]:
    """Get audio duration, sample rate, and channels."""
    try:
        info = sf.info(str(audio_path))
        return info.duration, info.samplerate, info.channels
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return 0.0, 0, 0


def create_recording(audio_path: Path, recording_id: str) -> Optional[Recording]:
    """Create a Lhotse Recording from an audio file."""
    duration, sample_rate, channels = get_audio_info(audio_path)
    if duration == 0:
        return None

    from lhotse.audio import AudioSource
    
    return Recording(
        id=recording_id,
        sources=[
            AudioSource(
                type="file",
                channels=list(range(channels)),
                source=str(audio_path),
            )
        ],
        sampling_rate=sample_rate,
        num_samples=int(duration * sample_rate),
        duration=duration,
    )


def create_supervision(
    recording_id: str,
    duration: float,
    text: str,
    pinyin: str,
    speaker: str,
    category: str,
    weight: float,
) -> SupervisionSegment:
    """Create a Lhotse SupervisionSegment."""
    return SupervisionSegment(
        id=f"{recording_id}_sup",
        recording_id=recording_id,
        start=0.0,
        duration=duration,
        channel=0,
        text=pinyin,  # Use pinyin for training (model uses pinyin tokenization)
        language="zh",
        speaker=speaker,
        custom={
            "text_zh": text,
            "category": category,
            "weight": weight,
            "is_positive": category == "positive",
        },
    )


def process_positive_samples(positive_dir: Path) -> Tuple[List[Recording], List[SupervisionSegment]]:
    """Process positive samples."""
    recordings = []
    supervisions = []

    metadata_path = positive_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        samples = metadata.get("samples", [])
    else:
        # Fallback: scan directory
        samples = [{"file": f.name} for f in positive_dir.glob("*.wav")]

    print(f"Processing {len(samples)} positive samples...")

    for sample in samples:
        if sample.get("status") == "failed":
            continue

        filename = sample.get("file")
        if not filename:
            continue

        audio_path = positive_dir / filename
        if not audio_path.exists():
            continue

        recording_id = f"pos_{audio_path.stem}"
        recording = create_recording(audio_path, recording_id)
        if recording is None:
            continue

        voice = sample.get("voice", "unknown")
        supervision = create_supervision(
            recording_id=recording_id,
            duration=recording.duration,
            text=TARGET_KEYWORD,
            pinyin=TARGET_PINYIN,
            speaker=f"tts_positive_{voice}",
            category="positive",
            weight=SAMPLE_WEIGHTS["positive"],
        )

        recordings.append(recording)
        supervisions.append(supervision)

    print(f"  Processed: {len(recordings)} recordings")
    return recordings, supervisions


def process_negative_samples_from_metadata(negative_dir: Path) -> Tuple[List[Recording], List[SupervisionSegment]]:
    """Process negative samples using metadata.json."""
    recordings = []
    supervisions = []
    
    metadata_path = negative_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Warning: No metadata.json found in {negative_dir}")
        return recordings, supervisions
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    samples = metadata.get("samples", [])
    print(f"Processing {len(samples)} negative samples from metadata...")
    
    skipped_no_file = 0
    skipped_no_pinyin = 0
    
    for sample in samples:
        if sample.get("status") == "failed":
            continue
        
        # Don't skip based on status - check if file actually exists instead
        # (status="skipped" in metadata may be outdated)
        
        category = sample.get("category", "general")
        text = sample.get("text", "")
        filename = sample.get("file", "")
        
        if not filename:
            continue
        
        # Find the audio file
        audio_path = negative_dir / category / filename
        if not audio_path.exists():
            # Try without category subdirectory
            audio_path = negative_dir / filename
            if not audio_path.exists():
                skipped_no_file += 1
                continue
        
        recording_id = f"neg_{category}_{audio_path.stem}"
        recording = create_recording(audio_path, recording_id)
        if recording is None:
            continue
        
        # Get pinyin transcription
        pinyin_text = chinese_to_pinyin(text)
        if not pinyin_text:
            skipped_no_pinyin += 1
            continue
        
        voice = sample.get("voice", "unknown")
        weight = SAMPLE_WEIGHTS.get(category, 1.0)
        
        supervision = create_supervision(
            recording_id=recording_id,
            duration=recording.duration,
            text=text,
            pinyin=pinyin_text,
            speaker=f"tts_negative_{voice}",
            category=category,
            weight=weight,
        )
        
        recordings.append(recording)
        supervisions.append(supervision)
    
    print(f"  Processed: {len(recordings)} recordings")
    print(f"  Skipped (no file): {skipped_no_file}")
    print(f"  Skipped (no pinyin): {skipped_no_pinyin}")
    
    return recordings, supervisions


def create_manifests() -> Tuple[RecordingSet, SupervisionSet]:
    """Create combined Lhotse manifests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process positive samples
    pos_recordings, pos_supervisions = process_positive_samples(POSITIVE_DIR)

    # Process negative samples from metadata
    neg_recordings, neg_supervisions = process_negative_samples_from_metadata(NEGATIVE_DIR)

    # Combine
    all_recordings = pos_recordings + neg_recordings
    all_supervisions = pos_supervisions + neg_supervisions

    print(f"\nTotal samples: {len(all_recordings)}")
    print(f"  Positive: {len(pos_recordings)}")
    print(f"  Negative: {len(neg_recordings)}")

    # Create Lhotse sets
    recording_set = RecordingSet.from_recordings(all_recordings)
    supervision_set = SupervisionSet.from_segments(all_supervisions)

    return recording_set, supervision_set


def save_manifests(recording_set: RecordingSet, supervision_set: SupervisionSet):
    """Save manifests to files."""
    # Save as JSONL.gz (compressed)
    recordings_path = OUTPUT_DIR / "kws_recordings_train_v3.jsonl.gz"
    supervisions_path = OUTPUT_DIR / "kws_supervisions_train_v3.jsonl.gz"

    recording_set.to_file(str(recordings_path))
    supervision_set.to_file(str(supervisions_path))

    print(f"\nManifests saved to:")
    print(f"  Recordings: {recordings_path}")
    print(f"  Supervisions: {supervisions_path}")
    
    # Create CutSet from recordings and supervisions
    cuts = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set,
    )
    cuts_path = OUTPUT_DIR / "kws_cuts.jsonl.gz"
    cuts.to_file(str(cuts_path))
    print(f"  Cuts: {cuts_path}")

    # Also save statistics
    stats = {
        "total_recordings": len(recording_set),
        "total_supervisions": len(supervision_set),
        "categories": {},
    }

    for sup in supervision_set:
        category = sup.custom.get("category", "unknown")
        if category not in stats["categories"]:
            stats["categories"][category] = {
                "count": 0,
                "total_duration": 0.0,
                "weight": SAMPLE_WEIGHTS.get(category, 1.0),
            }
        stats["categories"][category]["count"] += 1
        stats["categories"][category]["total_duration"] += sup.duration

    stats_path = OUTPUT_DIR / "manifest_stats_v3.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"  Statistics: {stats_path}")

    # Print summary
    print("\nCategory summary:")
    total_duration = 0
    for cat, cat_stats in sorted(stats["categories"].items()):
        total_duration += cat_stats["total_duration"]
        print(f"  {cat}: {cat_stats['count']} samples, "
              f"{cat_stats['total_duration']/60:.1f} min, "
              f"weight={cat_stats['weight']}")
    print(f"  Total duration: {total_duration/60:.1f} minutes")


def main():
    print("=" * 60)
    print("Lhotse Manifest Preparation V3")
    print("=" * 60)

    recording_set, supervision_set = create_manifests()
    save_manifests(recording_set, supervision_set)

    print("\n" + "=" * 60)
    print("Manifest Preparation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
