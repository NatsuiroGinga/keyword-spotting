#!/usr/bin/env python3
"""
Find optimal threshold for KWS model by testing multiple thresholds.
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
            return True
    return False


def evaluate_at_threshold(threshold, positive_files, negative_files, model_paths):
    spotter = create_keyword_spotter(
        encoder_path=model_paths["encoder"],
        decoder_path=model_paths["decoder"],
        joiner_path=model_paths["joiner"],
        tokens_path=model_paths["tokens"],
        keywords_path=model_paths["keywords"],
        keywords_threshold=threshold,
    )

    pos_detected = sum(1 for f in positive_files if test_audio_file(spotter, str(f)))
    neg_detected = sum(1 for f in negative_files if test_audio_file(spotter, str(f)))

    frr = 1.0 - (pos_detected / len(positive_files)) if positive_files else 0.0
    far = neg_detected / len(negative_files) if negative_files else 0.0

    return pos_detected, len(positive_files), neg_detected, len(negative_files), frr, far


def main():
    model_dir = Path("/data/workspace/llm/keyword-spotting/exp/kws_finetune")
    positive_dir = Path("/data/workspace/llm/audio-classification/dataset/kws_test_data/positive")
    negative_dir = Path("/data/workspace/llm/audio-classification/dataset/kws_test_data/negative")

    encoder_files = list(model_dir.glob("encoder-*.int8.onnx"))
    decoder_files = list(model_dir.glob("decoder-*.int8.onnx"))
    joiner_files = list(model_dir.glob("joiner-*.int8.onnx"))

    model_paths = {
        "encoder": str(encoder_files[0]),
        "decoder": str(decoder_files[0]),
        "joiner": str(joiner_files[0]),
        "tokens": str(model_dir / "tokens.txt"),
        "keywords": str(model_dir / "keywords.txt"),
    }

    positive_files = sorted(positive_dir.glob("*.wav"))
    negative_files = sorted(negative_dir.glob("*.wav"))

    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print()

    thresholds = [-1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("=" * 80)
    print(f"{'Threshold':>10} | {'Pos Det':>10} | {'Neg Det':>10} | {'FRR':>10} | {'FAR':>10} | {'F1':>10}")
    print("=" * 80)

    best_f1 = 0
    best_threshold = 0
    results = []

    for threshold in thresholds:
        pos_det, pos_total, neg_det, neg_total, frr, far = evaluate_at_threshold(
            threshold, positive_files, negative_files, model_paths
        )

        precision = pos_det / (pos_det + neg_det) if (pos_det + neg_det) > 0 else 0
        recall = 1 - frr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{threshold:>10.1f} | {pos_det:>7}/{pos_total:<3} | {neg_det:>7}/{neg_total:<3} | {frr:>9.2%} | {far:>9.2%} | {f1:>9.2%}")

        results.append({
            "threshold": threshold,
            "frr": frr,
            "far": far,
            "f1": f1,
            "pos_det": pos_det,
            "neg_det": neg_det,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("=" * 80)
    print(f"\nBest threshold: {best_threshold} (F1: {best_f1:.2%})")

    # Find threshold with FAR < 5% and lowest FRR
    low_far_results = [r for r in results if r["far"] <= 0.05]
    if low_far_results:
        best_low_far = min(low_far_results, key=lambda x: x["frr"])
        print(f"Best threshold with FAR <= 5%: {best_low_far['threshold']} (FRR: {best_low_far['frr']:.2%}, FAR: {best_low_far['far']:.2%})")

    # Find threshold with FAR < 10% and lowest FRR
    low_far_results = [r for r in results if r["far"] <= 0.10]
    if low_far_results:
        best_low_far = min(low_far_results, key=lambda x: x["frr"])
        print(f"Best threshold with FAR <= 10%: {best_low_far['threshold']} (FRR: {best_low_far['frr']:.2%}, FAR: {best_low_far['far']:.2%})")


if __name__ == "__main__":
    main()
