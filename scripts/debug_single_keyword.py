#!/usr/bin/env python3
"""Debug sherpa-onnx keyword detection with single keyword only."""

import sys
from pathlib import Path

import sherpa_onnx
import soundfile as sf


def create_single_keywords_file(output_path: str, boost: float = 1.0, threshold: float = 0.5) -> None:
    """Create keywords file with only target keyword."""
    line = f"n ǐ h ǎo zh ēn zh ēn :{boost:.2f} #{threshold:.2f} @你好真真"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"Created keywords file: {output_path}")
    print(f"  {line}")


def test_audio_detection(model_dir: Path, audio_path: str, keywords_path: str):
    """Test audio detection and print result format."""
    suffix = ".int8.onnx"
    encoder_files = list(model_dir.glob(f"encoder-*{suffix}"))
    decoder_files = list(model_dir.glob(f"decoder-*{suffix}"))
    joiner_files = list(model_dir.glob(f"joiner-*{suffix}"))

    if not encoder_files or not decoder_files or not joiner_files:
        print(f"Error: Could not find model files in {model_dir}")
        return None

    encoder_path = str(encoder_files[0])
    decoder_path = str(decoder_files[0])
    joiner_path = str(joiner_files[0])
    tokens_path = str(model_dir / "tokens.txt")

    spotter = sherpa_onnx.KeywordSpotter(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        keywords_file=keywords_path,
        num_threads=4,
        provider="cpu",
    )

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
    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
        result = spotter.get_result(stream)
        if result:
            detected = True
            break

    return detected


def main():
    model_dir = Path("/data/workspace/llm/keyword-spotting/exp/kws_finetune_v2")

    # Test different parameter combinations
    test_configs = [
        (0.3, 0.6),  # Low boost, high threshold
        (0.5, 0.5),
        (1.0, 0.5),
        (1.0, 0.6),
        (1.0, 0.65),  # Best config from previous run
        (1.5, 0.5),
        (2.0, 0.5),
    ]

    positive_dir = Path("/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive")
    negative_dir = Path("/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/negative")

    positive_files = sorted(positive_dir.glob("*.wav"))
    negative_files = sorted(negative_dir.glob("*.wav"))

    print(f"Total positive samples: {len(positive_files)}")
    print(f"Total negative samples: {len(negative_files)}")
    print()

    for boost, threshold in test_configs:
        keywords_path = "/tmp/single_keyword.txt"
        create_single_keywords_file(keywords_path, boost, threshold)

        # Test positive samples
        tp = 0
        for audio_path in positive_files:
            if test_audio_detection(model_dir, str(audio_path), keywords_path):
                tp += 1

        # Test negative samples
        fp = 0
        for audio_path in negative_files:
            if test_audio_detection(model_dir, str(audio_path), keywords_path):
                fp += 1

        frr = (len(positive_files) - tp) / len(positive_files) * 100
        far = fp / len(negative_files) * 100

        print(f"\nBoost={boost}, Threshold={threshold}:")
        print(f"  TP: {tp}/{len(positive_files)}, FP: {fp}/{len(negative_files)}")
        print(f"  FRR: {frr:.2f}%, FAR: {far:.2f}%")
        print()


if __name__ == "__main__":
    main()
