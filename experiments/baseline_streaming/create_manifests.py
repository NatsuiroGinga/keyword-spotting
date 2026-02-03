#!/usr/bin/env python3
"""
从真实人声数据创建Lhotse Manifests

用于从预训练模型微调V3 KWS模型
"""

import json
import gzip
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional
import librosa
import soundfile as sf


# 正样本关键词（与"你好真真"同音的词）
POSITIVE_KEYWORDS = [
    "你好真真",
    "你好珍珍",
    "你好甄甄",
    "你好臻臻",
    "你好桢桢",
]

# 关键词的token序列（带声调的拼音）
KEYWORD_TOKENS = "n ǐ h ǎo zh ēn zh ēn"
KEYWORD_DISPLAY = "你好真真"


def extract_text_from_filename(filename: str) -> str:
    """从文件名提取文本内容"""
    import re
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_", 1)[1]
    else:
        return re.sub(r"^\d+", "", stem)


def is_positive_sample(text: str) -> bool:
    """判断是否为正样本"""
    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            return True
    return False


def get_text_for_transcript(text: str, is_positive: bool) -> str:
    """获取用于转录的文本
    
    对于正样本，使用标准关键词token序列
    对于负样本，使用空字符串（让模型学习不响应）
    """
    if is_positive:
        return KEYWORD_TOKENS
    else:
        return ""


def create_recording_entry(
    audio_path: Path,
    recording_id: str,
) -> Dict:
    """创建Recording条目"""
    # 获取音频信息
    info = sf.info(str(audio_path))
    duration = info.duration
    sample_rate = info.samplerate
    num_samples = int(duration * sample_rate)
    
    return {
        "id": recording_id,
        "sources": [
            {
                "type": "file",
                "channels": [0],
                "source": str(audio_path.absolute()),
            }
        ],
        "sampling_rate": sample_rate,
        "num_samples": num_samples,
        "duration": duration,
    }


def create_supervision_entry(
    recording_id: str,
    supervision_id: str,
    text: str,
    duration: float,
    is_positive: bool,
) -> Dict:
    """创建Supervision条目"""
    return {
        "id": supervision_id,
        "recording_id": recording_id,
        "start": 0.0,
        "duration": duration,
        "channel": 0,
        "text": text,
        "language": "Chinese",
        "custom": {
            "is_keyword": is_positive,
            "keyword": KEYWORD_DISPLAY if is_positive else "",
        },
    }


def create_manifests_for_split(
    split_dir: Path,
    output_dir: Path,
    split_name: str,
) -> Dict:
    """为单个数据集划分创建manifests"""
    audio_files = list(split_dir.glob("*.wav"))
    
    recordings = []
    supervisions = []
    
    stats = {
        "total": 0,
        "positive": 0,
        "negative": 0,
    }
    
    for audio_path in audio_files:
        text = extract_text_from_filename(audio_path.name)
        is_positive = is_positive_sample(text)
        
        # 生成唯一ID
        recording_id = f"kws_{split_name}_{audio_path.stem}"
        supervision_id = f"sup_{recording_id}"
        
        # 获取音频信息
        info = sf.info(str(audio_path))
        duration = info.duration
        
        # 获取转录文本
        transcript = get_text_for_transcript(text, is_positive)
        
        # 创建条目
        recording = create_recording_entry(audio_path, recording_id)
        supervision = create_supervision_entry(
            recording_id, supervision_id, transcript, duration, is_positive
        )
        
        recordings.append(recording)
        supervisions.append(supervision)
        
        stats["total"] += 1
        if is_positive:
            stats["positive"] += 1
        else:
            stats["negative"] += 1
    
    # 保存manifests
    output_dir.mkdir(parents=True, exist_ok=True)
    
    recordings_path = output_dir / f"kws_recordings_{split_name}.jsonl.gz"
    supervisions_path = output_dir / f"kws_supervisions_{split_name}.jsonl.gz"
    
    with gzip.open(recordings_path, "wt", encoding="utf-8") as f:
        for r in recordings:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    with gzip.open(supervisions_path, "wt", encoding="utf-8") as f:
        for s in supervisions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"  {split_name}:")
    print(f"    总数: {stats['total']}")
    print(f"    正样本: {stats['positive']}")
    print(f"    负样本: {stats['negative']}")
    print(f"    Recordings: {recordings_path}")
    print(f"    Supervisions: {supervisions_path}")
    
    return stats


def create_all_manifests(
    data_splits_dir: str = "experiments/baseline_streaming/data_splits",
    output_dir: str = "experiments/baseline_streaming/manifests",
):
    """创建所有数据集划分的manifests"""
    base_dir = Path(__file__).parent.parent.parent
    splits_dir = base_dir / data_splits_dir
    out_dir = base_dir / output_dir
    
    print("=" * 60)
    print("创建Lhotse Manifests")
    print("=" * 60)
    print(f"数据目录: {splits_dir}")
    print(f"输出目录: {out_dir}")
    print()
    
    all_stats = {}
    
    for split_name in ["train", "val", "test"]:
        split_dir = splits_dir / split_name
        if split_dir.exists():
            stats = create_manifests_for_split(split_dir, out_dir, split_name)
            all_stats[split_name] = stats
    
    # 保存统计信息
    stats_path = out_dir / "manifest_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print()
    print(f"统计信息已保存到: {stats_path}")
    print("=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    create_all_manifests()
