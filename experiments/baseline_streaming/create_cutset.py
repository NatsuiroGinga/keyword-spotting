#!/usr/bin/env python3
"""
从真实人声数据创建Lhotse CutSet

为负样本生成拼音文本，确保训练时不会出现空文本问题
"""

import json
import gzip
import os
import re
from pathlib import Path
from typing import Dict, List
import soundfile as sf
from lhotse import Recording, SupervisionSegment, CutSet, MonoCut
from lhotse.audio import AudioSource
from pypinyin import pinyin, Style


# 正样本关键词（与"你好真真"同音的词）
POSITIVE_KEYWORDS = [
    "你好真真",
    "你好珍珍",
    "你好甄甄",
    "你好臻臻",
    "你好桢桢",
]

# 关键词的token序列
KEYWORD_TOKENS = "n ǐ h ǎo zh ēn zh ēn"

# 声母表（按长度降序排列，确保先匹配长声母）
INITIALS = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 
            'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'r', 'y', 'w']


def extract_text_from_filename(filename: str) -> str:
    """从文件名提取文本内容"""
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


def split_pinyin_syllable(syllable: str) -> str:
    """分割单个拼音音节为声母+韵母格式
    
    使用 pypinyin 的 Style.TONE（直接带声调），声调已经在正确位置
    例如: 'nǐ' -> 'n ǐ', 'hǎo' -> 'h ǎo', 'zhēn' -> 'zh ēn'
    """
    syllable = syllable.lower().strip()
    if not syllable:
        return ""
    
    # 找声母
    for initial in INITIALS:
        if syllable.startswith(initial):
            final = syllable[len(initial):]
            if final:
                return f"{initial} {final}"
            else:
                return initial
    
    # 无声母（纯韵母音节如 'a', 'o', 'e', 'ai' 等）
    return syllable


def chinese_to_pinyin_tokens(text: str) -> str:
    """将中文转换为带声调的拼音token序列
    
    使用 Style.TONE 直接获取带声调的拼音，避免手动转换错误
    例如: "你好" -> "n ǐ h ǎo"
    """
    if not text:
        return "unk"
    
    # 使用 Style.TONE 获取带声调的拼音（声调符号在正确位置）
    py_list = pinyin(text, style=Style.TONE, heteronym=False)
    
    result_parts = []
    for py in py_list:
        syllable = py[0] if py else ""
        if syllable:
            split_result = split_pinyin_syllable(syllable)
            if split_result:
                result_parts.append(split_result)
    
    return " ".join(result_parts) if result_parts else "unk"


def get_transcript(text: str, is_positive: bool) -> str:
    """获取用于转录的文本
    
    正样本使用标准关键词token，负样本使用实际文本的拼音
    """
    if is_positive:
        return KEYWORD_TOKENS
    else:
        # 负样本：转换实际文本为拼音
        pinyin_text = chinese_to_pinyin_tokens(text)
        return pinyin_text if pinyin_text else "unk"


def create_cutset_from_audio_dir(
    audio_dir: Path,
    split_name: str,
) -> CutSet:
    """从音频目录创建CutSet"""
    audio_files = sorted(audio_dir.glob("*.wav"))
    cuts = []
    
    for audio_path in audio_files:
        # 获取音频信息
        info = sf.info(str(audio_path))
        duration = info.duration
        sample_rate = info.samplerate
        num_samples = int(duration * sample_rate)
        
        # 提取文本和标签
        text = extract_text_from_filename(audio_path.name)
        is_positive = is_positive_sample(text)
        transcript = get_transcript(text, is_positive)
        
        # 创建Recording
        recording_id = f"kws_{split_name}_{audio_path.stem}"
        recording = Recording(
            id=recording_id,
            sources=[
                AudioSource(
                    type="file",
                    channels=[0],
                    source=str(audio_path.absolute()),
                )
            ],
            sampling_rate=sample_rate,
            num_samples=num_samples,
            duration=duration,
        )
        
        # 创建Supervision
        supervision = SupervisionSegment(
            id=f"sup_{recording_id}",
            recording_id=recording_id,
            start=0.0,
            duration=duration,
            channel=0,
            text=transcript,
            language="Chinese",
            custom={
                "is_keyword": is_positive,
                "original_text": text,
            },
        )
        
        # 创建Cut
        cut = MonoCut(
            id=recording_id,
            start=0.0,
            duration=duration,
            channel=0,
            recording=recording,
            supervisions=[supervision],
        )
        cuts.append(cut)
    
    return CutSet.from_cuts(cuts)


def create_all_manifests(
    data_splits_dir: str = "experiments/baseline_streaming/data_splits",
    output_dir: str = "experiments/baseline_streaming/manifests",
):
    """创建所有数据集划分的manifests"""
    base_dir = Path(__file__).parent.parent.parent
    splits_dir = base_dir / data_splits_dir
    out_dir = base_dir / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("创建Lhotse CutSet Manifests (带拼音转录)")
    print("=" * 60)
    print(f"数据目录: {splits_dir}")
    print(f"输出目录: {out_dir}")
    print()
    
    all_stats = {}
    
    # 创建训练集CutSet
    train_dir = splits_dir / "train"
    if train_dir.exists():
        print("处理训练集...")
        train_cuts = create_cutset_from_audio_dir(train_dir, "train")
        
        # 统计
        n_positive = sum(1 for c in train_cuts if c.supervisions[0].custom.get("is_keyword", False))
        n_negative = len(train_cuts) - n_positive
        print(f"  总数: {len(train_cuts)}")
        print(f"  正样本: {n_positive}")
        print(f"  负样本: {n_negative}")
        
        # 打印几个样本
        print("  样本预览:")
        for i, cut in enumerate(list(train_cuts)[:3]):
            text = cut.supervisions[0].text[:40]
            is_kw = cut.supervisions[0].custom.get("is_keyword", False)
            print(f"    [{i+1}] {'正' if is_kw else '负'}: {text}")
        
        train_cuts.to_file(out_dir / "kws_cuts.jsonl.gz")
        train_cuts.to_file(out_dir / "kws_cuts_train.jsonl.gz")
        print(f"  保存到: {out_dir / 'kws_cuts.jsonl.gz'}")
        
        all_stats["train"] = {"total": len(train_cuts), "positive": n_positive, "negative": n_negative}
    
    # 创建验证集CutSet
    val_dir = splits_dir / "val"
    if val_dir.exists():
        print("\n处理验证集...")
        val_cuts = create_cutset_from_audio_dir(val_dir, "val")
        
        n_positive = sum(1 for c in val_cuts if c.supervisions[0].custom.get("is_keyword", False))
        n_negative = len(val_cuts) - n_positive
        print(f"  总数: {len(val_cuts)}")
        print(f"  正样本: {n_positive}")
        print(f"  负样本: {n_negative}")
        
        val_cuts.to_file(out_dir / "kws_cuts_val.jsonl.gz")
        print(f"  保存到: {out_dir / 'kws_cuts_val.jsonl.gz'}")
        
        all_stats["val"] = {"total": len(val_cuts), "positive": n_positive, "negative": n_negative}
    
    # 创建测试集CutSet
    test_dir = splits_dir / "test"
    if test_dir.exists():
        print("\n处理测试集...")
        test_cuts = create_cutset_from_audio_dir(test_dir, "test")
        
        n_positive = sum(1 for c in test_cuts if c.supervisions[0].custom.get("is_keyword", False))
        n_negative = len(test_cuts) - n_positive
        print(f"  总数: {len(test_cuts)}")
        print(f"  正样本: {n_positive}")
        print(f"  负样本: {n_negative}")
        
        test_cuts.to_file(out_dir / "kws_cuts_test.jsonl.gz")
        print(f"  保存到: {out_dir / 'kws_cuts_test.jsonl.gz'}")
        
        all_stats["test"] = {"total": len(test_cuts), "positive": n_positive, "negative": n_negative}
    
    # 保存统计信息
    stats_path = out_dir / "cutset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print()
    print(f"统计信息已保存到: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    create_all_manifests()
