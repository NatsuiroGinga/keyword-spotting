#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Lhotse Manifests for KWS Fine-tuning
Converts generated TTS WAV files into Lhotse-compatible JSONL manifests.
Supports both positive (wake word) and negative (non-wake word) samples.
"""

import logging
from pathlib import Path
import json
import soundfile as sf
from lhotse import RecordingSet, SupervisionSet, SupervisionSegment, Recording, CutSet
from lhotse.audio import AudioSource
from pypinyin import pinyin, Style

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
POSITIVE_DIR = Path("/data/workspace/llm/keyword-spotting/data/raw_tts/positive")
NEGATIVE_DIR = Path("/data/workspace/llm/keyword-spotting/data/raw_tts/negative")
OUTPUT_DIR = Path("/data/workspace/llm/keyword-spotting/data/manifests")
TOKENS_FILE = Path("/data/workspace/llm/keyword-spotting/data/lang_partial_tone/tokens.txt")

# Legacy alias for backwards compatibility
CORPUS_DIR = POSITIVE_DIR


def text_to_pinyin_tokens(text: str) -> str:
    """
    Convert Chinese text to pinyin tokens (partial with tone).
    Example: "你好真真" -> "n ǐ h ǎo zh ēn zh ēn"
    """
    # Get pinyin with tone marks
    py_list = pinyin(text, style=Style.TONE3, heteronym=False)

    # Manual mapping for common initials/finals
    # This is a simplified version - for production, use the official tokens.txt mapping
    result = []
    for py in py_list:
        p = py[0]
        # Handle initials and finals
        if p.startswith(('zh', 'ch', 'sh')):
            initial = p[:2]
            final = p[2:]
        elif p[0] in 'bpmfdtnlgkhjqxrzcsyw':
            initial = p[0]
            final = p[1:]
        else:
            initial = ''
            final = p

        if initial:
            result.append(initial)
        if final:
            # Convert tone number to tone mark
            final = convert_tone_number_to_mark(final)
            result.append(final)

    return ' '.join(result)


def convert_tone_number_to_mark(syllable: str) -> str:
    """Convert tone number notation to tone mark notation."""
    tone_marks = {
        'a': ['ā', 'á', 'ǎ', 'à', 'a'],
        'e': ['ē', 'é', 'ě', 'è', 'e'],
        'i': ['ī', 'í', 'ǐ', 'ì', 'i'],
        'o': ['ō', 'ó', 'ǒ', 'ò', 'o'],
        'u': ['ū', 'ú', 'ǔ', 'ù', 'u'],
        'ü': ['ǖ', 'ǘ', 'ǚ', 'ǜ', 'ü'],
    }

    # Extract tone number (last character if digit)
    if syllable and syllable[-1].isdigit():
        tone = int(syllable[-1])
        syllable = syllable[:-1]
    else:
        tone = 5  # neutral tone

    if tone < 1 or tone > 5:
        tone = 5

    # Find the vowel to add tone mark
    # Priority: a, e, ou (mark o), else mark the last vowel
    result = syllable
    for vowel, marks in tone_marks.items():
        if vowel in syllable.lower():
            if vowel == 'i' and 'u' in syllable:
                # iu -> mark the second vowel
                continue
            if vowel == 'u' and 'i' in syllable:
                # ui -> mark the second vowel (i)
                continue
            result = syllable.replace(vowel, marks[tone - 1], 1)
            break

    return result


def prepare_manifests():
    """Prepare Lhotse manifests from TTS-generated WAV files (positive and negative samples)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recordings = []
    supervisions = []

    # Process positive samples
    logging.info("Processing positive samples...")
    pos_metadata_path = POSITIVE_DIR / "metadata.json"
    pos_text_map = {}
    if pos_metadata_path.exists():
        with open(pos_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            for sample in metadata.get('samples', []):
                pos_text_map[sample['file']] = sample['text']

    pos_wav_files = list(POSITIVE_DIR.glob("*.wav"))
    logging.info(f"Found {len(pos_wav_files)} positive WAV files in {POSITIVE_DIR}")

    for i, wav in enumerate(pos_wav_files):
        try:
            info = sf.info(str(wav))
            recording_id = f"pos_{wav.stem}"

            # Get text from metadata or infer from filename
            if wav.name in pos_text_map:
                text = pos_text_map[wav.name]
            elif "nihaozhenzhen" in wav.stem:
                text = "你好真真"
            elif "zhenzhen_nihao" in wav.stem:
                text = "真真你好"
            elif "zhenzhen" in wav.stem:
                text = "真真"
            else:
                text = "你好真真"  # default

            recording = Recording(
                id=recording_id,
                sources=[AudioSource(type="file", channels=[0], source=str(wav))],
                sampling_rate=int(info.samplerate),
                num_samples=int(info.frames),
                duration=float(info.duration)
            )
            recordings.append(recording)

            supervision = SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=0.0,
                duration=float(info.duration),
                channel=0,
                text=text,
                language="Chinese",
                speaker="tts_positive"
            )
            supervisions.append(supervision)

            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1}/{len(pos_wav_files)} positive files")

        except Exception as e:
            logging.warning(f"Error processing {wav}: {e}")

    # Process negative samples
    logging.info("Processing negative samples...")
    neg_metadata_path = NEGATIVE_DIR / "metadata.json"
    neg_text_map = {}
    if neg_metadata_path.exists():
        with open(neg_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            for sample in metadata.get('samples', []):
                neg_text_map[sample['file']] = sample['text']

    neg_wav_files = list(NEGATIVE_DIR.glob("*.wav")) if NEGATIVE_DIR.exists() else []
    logging.info(f"Found {len(neg_wav_files)} negative WAV files in {NEGATIVE_DIR}")

    for i, wav in enumerate(neg_wav_files):
        try:
            info = sf.info(str(wav))
            recording_id = f"neg_{wav.stem}"

            # Get text from metadata (required for negative samples)
            text = neg_text_map.get(wav.name, "")
            if not text:
                logging.warning(f"No text found for negative sample {wav.name}, skipping")
                continue

            recording = Recording(
                id=recording_id,
                sources=[AudioSource(type="file", channels=[0], source=str(wav))],
                sampling_rate=int(info.samplerate),
                num_samples=int(info.frames),
                duration=float(info.duration)
            )
            recordings.append(recording)

            supervision = SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=0.0,
                duration=float(info.duration),
                channel=0,
                text=text,
                language="Chinese",
                speaker="tts_negative"
            )
            supervisions.append(supervision)

            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1}/{len(neg_wav_files)} negative files")

        except Exception as e:
            logging.warning(f"Error processing {wav}: {e}")

    # Save as JSONL.GZ
    rec_set = RecordingSet.from_recordings(recordings)
    sup_set = SupervisionSet.from_segments(supervisions)

    rec_path = OUTPUT_DIR / "kws_recordings.jsonl.gz"
    sup_path = OUTPUT_DIR / "kws_supervisions.jsonl.gz"

    rec_set.to_file(rec_path)
    sup_set.to_file(sup_path)

    logging.info(f"Saved {len(recordings)} recordings to {rec_path}")
    logging.info(f"Saved {len(supervisions)} supervisions to {sup_path}")

    # Summary
    pos_count = len(pos_wav_files)
    neg_count = len(neg_wav_files)
    logging.info(f"\nSummary:")
    logging.info(f"  Positive samples: {pos_count}")
    logging.info(f"  Negative samples: {neg_count}")
    logging.info(f"  Total: {pos_count + neg_count}")

    # Create CutSet
    cuts = CutSet.from_manifests(recordings=rec_set, supervisions=sup_set)
    cuts_path = OUTPUT_DIR / "kws_cuts.jsonl.gz"
    cuts.to_file(cuts_path)
    logging.info(f"Saved {len(cuts)} cuts to {cuts_path}")

    return rec_set, sup_set, cuts


if __name__ == "__main__":
    prepare_manifests()
