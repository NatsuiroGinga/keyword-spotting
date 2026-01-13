#!/usr/bin/env python3
"""
Analyze false positive samples to understand why they are being detected.
"""

import sys
from pathlib import Path

import sherpa_onnx
import soundfile as sf


def create_keyword_spotter(
    encoder_path: str,
    decoder_path: str,
    joiner_path: str,
    tokens_path: str,
    keywords_path: str,
    keywords_threshold: float = 0.0,
) -> sherpa_onnx.KeywordSpotter:
    return sherpa_onnx.KeywordSpotter(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        keywords_file=keywords_path,
        num_threads=4,
        keywords_threshold=keywords_threshold,
        provider="cpu",
    )


def test_audio_file(spotter, audio_path):
    samples, sample_rate = sf.read(audio_path, dtype="float32")
    if len(samples.shape) > 1:
        samples = samples[:, 0]

    stream = spotter.create_stream()
    stream.accept_waveform(sample_rate, samples)
    tail_paddings = [0.0] * int(0.3 * sample_rate)
    stream.accept_waveform(sample_rate, tail_paddings)
    stream.input_finished()

    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
        result = spotter.get_result(stream)
        if result:
            return True, result
    return False, ""


def main():
    model_dir = Path("/data/workspace/llm/keyword-spotting/exp/kws_finetune")
    negative_dir = Path("/data/workspace/llm/audio-classification/dataset/kws_test_data/negative")

    encoder_files = list(model_dir.glob("encoder-*.int8.onnx"))
    decoder_files = list(model_dir.glob("decoder-*.int8.onnx"))
    joiner_files = list(model_dir.glob("joiner-*.int8.onnx"))

    spotter = create_keyword_spotter(
        encoder_path=str(encoder_files[0]),
        decoder_path=str(decoder_files[0]),
        joiner_path=str(joiner_files[0]),
        tokens_path=str(model_dir / "tokens.txt"),
        keywords_path=str(model_dir / "keywords.txt"),
        keywords_threshold=0.0,
    )

    negative_files = sorted(negative_dir.glob("*.wav"))

    print("Analyzing false positives...")
    print("=" * 80)

    fp_by_prefix = {}
    for f in negative_files:
        detected, result = test_audio_file(spotter, str(f))
        if detected:
            # Extract prefix (negative_XXXX)
            prefix = "_".join(f.stem.split("_")[:2])
            if prefix not in fp_by_prefix:
                fp_by_prefix[prefix] = []
            fp_by_prefix[prefix].append(f.name)

    print(f"\nFalse positive groups (by original sample):")
    print("-" * 80)
    for prefix, files in sorted(fp_by_prefix.items()):
        print(f"\n{prefix}: {len(files)} files")
        for f in files[:3]:
            print(f"  - {f}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")

    print("\n" + "=" * 80)
    print(f"Total false positive groups: {len(fp_by_prefix)}")
    print(f"Total false positive files: {sum(len(files) for files in fp_by_prefix.values())}")


if __name__ == "__main__":
    main()
