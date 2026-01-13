#!/usr/bin/env python3
"""Debug sherpa-onnx keyword detection result format."""

import sys
from pathlib import Path

import sherpa_onnx
import soundfile as sf


def create_decoy_keywords_file(output_path: str) -> None:
    """Create keywords file with target and decoy keywords."""
    keywords = [
        {"pinyin": "n ǐ h ǎo zh ēn zh ēn", "text": "你好真真", "is_target": True},
        {"pinyin": "n ǐ h ǎo", "text": "你好", "is_target": False},
        {"pinyin": "n ǐ h ǎo a", "text": "你好啊", "is_target": False},
        {"pinyin": "n ín h ǎo", "text": "您好", "is_target": False},
        {"pinyin": "n ǐ h ǎo m a", "text": "你好吗", "is_target": False},
    ]

    lines = []
    for kw in keywords:
        boost = 2.0 if kw["is_target"] else 1.0
        threshold = 0.25
        line = f"{kw['pinyin']} :{boost:.2f} #{threshold:.2f} @{kw['text']}"
        lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Created keywords file: {output_path}")
    print("Contents:")
    for line in lines:
        print(f"  {line}")


def test_audio_detection(model_dir: Path, audio_path: str, keywords_path: str):
    """Test audio detection and print result format."""
    # Find model files
    suffix = ".int8.onnx"
    encoder_files = list(model_dir.glob(f"encoder-*{suffix}"))
    decoder_files = list(model_dir.glob(f"decoder-*{suffix}"))
    joiner_files = list(model_dir.glob(f"joiner-*{suffix}"))

    if not encoder_files or not decoder_files or not joiner_files:
        print(f"Error: Could not find model files in {model_dir}")
        return

    encoder_path = str(encoder_files[0])
    decoder_path = str(decoder_files[0])
    joiner_path = str(joiner_files[0])
    tokens_path = str(model_dir / "tokens.txt")

    print(f"\nModel files:")
    print(f"  Encoder: {encoder_path}")
    print(f"  Decoder: {decoder_path}")
    print(f"  Joiner: {joiner_path}")
    print(f"  Tokens: {tokens_path}")

    # Create spotter
    spotter = sherpa_onnx.KeywordSpotter(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        keywords_file=keywords_path,
        num_threads=4,
        provider="cpu",
    )

    # Load audio
    samples, sample_rate = sf.read(audio_path, dtype="float32")

    if sample_rate != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    if len(samples.shape) > 1:
        samples = samples[:, 0]

    print(f"\nAudio: {audio_path}")
    print(f"  Sample rate: {sample_rate}")
    print(f"  Duration: {len(samples) / sample_rate:.2f}s")

    # Process audio
    stream = spotter.create_stream()
    stream.accept_waveform(sample_rate, samples)

    tail_paddings = [0.0] * int(0.3 * sample_rate)
    stream.accept_waveform(sample_rate, tail_paddings)
    stream.input_finished()

    # Get results
    print("\nDetection results:")
    detection_count = 0
    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
        result = spotter.get_result(stream)
        if result:
            detection_count += 1
            print(f"\n  Detection #{detection_count}:")
            print(f"    Raw result: [{result}]")
            print(f"    Result type: {type(result)}")
            print(f"    Result repr: {repr(result)}")

            # Try to parse the result
            result_stripped = result.strip()
            print(f"    Stripped: [{result_stripped}]")

            # Check various matching patterns
            print(f"    Contains '你好真真': {'你好真真' in result}")
            print(f"    Contains '你好': {'你好' in result}")
            print(f"    Exact match '你好真真': {result_stripped == '你好真真'}")

            # Check if result starts/ends with certain strings
            print(f"    Starts with '你好真真': {result_stripped.startswith('你好真真')}")
            print(f"    Ends with '你好真真': {result_stripped.endswith('你好真真')}")

    if detection_count == 0:
        print("  No keyword detected")


def main():
    model_dir = Path("/data/workspace/llm/keyword-spotting/exp/kws_finetune_v2")

    # Create temporary keywords file
    keywords_path = "/tmp/debug_keywords.txt"
    create_decoy_keywords_file(keywords_path)

    # Test with positive samples
    positive_dir = Path("/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive")
    positive_files = sorted(positive_dir.glob("*.wav"))[:3]

    print("\n" + "=" * 60)
    print("Testing POSITIVE samples")
    print("=" * 60)

    for audio_path in positive_files:
        test_audio_detection(model_dir, str(audio_path), keywords_path)
        print("-" * 60)

    # Test with negative samples
    negative_dir = Path("/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/negative")
    negative_files = sorted(negative_dir.glob("*.wav"))[:3]

    print("\n" + "=" * 60)
    print("Testing NEGATIVE samples")
    print("=" * 60)

    for audio_path in negative_files:
        test_audio_detection(model_dir, str(audio_path), keywords_path)
        print("-" * 60)


if __name__ == "__main__":
    main()
