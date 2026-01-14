#!/usr/bin/env python3
"""
Evaluate fine-tuned KWS model using sherpa-onnx.
Calculates FRR (False Rejection Rate) and FAR (False Accept Rate).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import sherpa_onnx
import soundfile as sf


def create_keyword_spotter(
    encoder_path: str,
    decoder_path: str,
    joiner_path: str,
    tokens_path: str,
    keywords_path: str,
    num_threads: int = 4,
    keywords_threshold: float = 0.0,
) -> sherpa_onnx.KeywordSpotter:
    """Create a sherpa-onnx KeywordSpotter."""
    return sherpa_onnx.KeywordSpotter(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        keywords_file=keywords_path,
        num_threads=num_threads,
        keywords_threshold=keywords_threshold,
        provider="cpu",
    )


def test_audio_file(
    spotter: sherpa_onnx.KeywordSpotter,
    audio_path: str,
) -> Tuple[bool, str]:
    """
    Test a single audio file for keyword detection.

    Returns:
        Tuple of (detected, keyword_text)
    """
    samples, sample_rate = sf.read(audio_path, dtype="float32")

    if sample_rate != 16000:
        import librosa
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    if len(samples.shape) > 1:
        samples = samples[:, 0]

    stream = spotter.create_stream()
    stream.accept_waveform(sample_rate, samples)

    tail_paddings = [0.0] * int(0.3 * sample_rate)
    stream.accept_waveform(sample_rate, tail_paddings)
    stream.input_finished()

    detected = False
    keyword_text = ""

    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
        result = spotter.get_result(stream)
        if result:
            detected = True
            keyword_text = result
            break

    return detected, keyword_text


def evaluate_dataset(
    spotter: sherpa_onnx.KeywordSpotter,
    audio_files: List[str],
    expected_positive: bool,
) -> Tuple[int, int, List[str]]:
    """
    Evaluate a dataset of audio files.

    Args:
        spotter: KeywordSpotter instance
        audio_files: List of audio file paths
        expected_positive: True if these are positive samples (should be detected)

    Returns:
        Tuple of (correct_count, total_count, error_files)
    """
    correct = 0
    total = len(audio_files)
    errors = []

    for audio_path in audio_files:
        try:
            detected, keyword = test_audio_file(spotter, audio_path)

            if expected_positive:
                if detected:
                    correct += 1
                else:
                    errors.append(audio_path)
            else:
                if not detected:
                    correct += 1
                else:
                    errors.append(f"{audio_path} (false positive: {keyword})")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            errors.append(f"{audio_path} (error: {e})")

    return correct, total, errors


def main():
    parser = argparse.ArgumentParser(description="Evaluate KWS model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/workspace/llm/keyword-spotting/exp/kws_finetune",
        help="Directory containing ONNX model files",
    )
    parser.add_argument(
        "--use-int8",
        action="store_true",
        default=True,
        help="Use INT8 quantized models",
    )
    parser.add_argument(
        "--positive-dir",
        type=str,
        default="/data/workspace/llm/audio-classification/dataset/kws_test_data/positive",
        help="Directory containing positive test samples",
    )
    parser.add_argument(
        "--negative-dir",
        type=str,
        default="/data/workspace/llm/audio-classification/dataset/kws_test_data/negative",
        help="Directory containing negative test samples",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keywords detection threshold",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error information",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    suffix = ".int8.onnx" if args.use_int8 else ".onnx"

    encoder_files = list(model_dir.glob(f"encoder-*{suffix}"))
    decoder_files = list(model_dir.glob(f"decoder-*{suffix}"))
    joiner_files = list(model_dir.glob(f"joiner-*{suffix}"))

    if not encoder_files or not decoder_files or not joiner_files:
        print(f"Error: Could not find model files with suffix {suffix} in {model_dir}")
        sys.exit(1)

    encoder_path = str(encoder_files[0])
    decoder_path = str(decoder_files[0])
    joiner_path = str(joiner_files[0])
    tokens_path = str(model_dir / "tokens.txt")
    keywords_path = str(model_dir / "keywords.txt")

    print("=" * 60)
    print("KWS Model Evaluation")
    print("=" * 60)
    print(f"Encoder: {encoder_path}")
    print(f"Decoder: {decoder_path}")
    print(f"Joiner: {joiner_path}")
    print(f"Tokens: {tokens_path}")
    print(f"Keywords: {keywords_path}")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)

    spotter = create_keyword_spotter(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        joiner_path=joiner_path,
        tokens_path=tokens_path,
        keywords_path=keywords_path,
        keywords_threshold=args.threshold,
    )

    positive_files = sorted(Path(args.positive_dir).glob("*.wav"))
    negative_files = sorted(Path(args.negative_dir).glob("*.wav"))

    print(f"\nPositive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print()

    print("Evaluating positive samples...")
    pos_correct, pos_total, pos_errors = evaluate_dataset(
        spotter, [str(f) for f in positive_files], expected_positive=True
    )

    print("Evaluating negative samples...")
    neg_correct, neg_total, neg_errors = evaluate_dataset(
        spotter, [str(f) for f in negative_files], expected_positive=False
    )

    frr = 1.0 - (pos_correct / pos_total) if pos_total > 0 else 0.0
    far = 1.0 - (neg_correct / neg_total) if neg_total > 0 else 0.0

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Positive samples: {pos_correct}/{pos_total} detected correctly")
    print(f"Negative samples: {neg_correct}/{neg_total} rejected correctly")
    print()
    print(f"FRR (False Rejection Rate): {frr:.2%}")
    print(f"FAR (False Accept Rate): {far:.2%}")
    print(f"Recall (True Positive Rate): {(1-frr):.2%}")
    print(f"Specificity (True Negative Rate): {(1-far):.2%}")
    print("=" * 60)

    if args.verbose and (pos_errors or neg_errors):
        print("\nErrors:")
        if pos_errors:
            print(f"\nFalse Rejections ({len(pos_errors)}):")
            for err in pos_errors[:10]:
                print(f"  - {err}")
            if len(pos_errors) > 10:
                print(f"  ... and {len(pos_errors) - 10} more")

        if neg_errors:
            print(f"\nFalse Accepts ({len(neg_errors)}):")
            for err in neg_errors[:10]:
                print(f"  - {err}")
            if len(neg_errors) > 10:
                print(f"  ... and {len(neg_errors) - 10} more")


if __name__ == "__main__":
    main()
