#!/usr/bin/env python3
"""
Parameter optimization with Decoy Approach for KWS model.
Uses multiple keywords (target + decoys) to filter false positives.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import sherpa_onnx
import soundfile as sf


# Decoy configuration - similar-sounding phrases to block
DECOY_KEYWORDS = [
    # Target keyword
    {"pinyin": "n ǐ h ǎo zh ēn zh ēn", "text": "你好真真", "is_target": True},
    # Decoys - similar-sounding variants
    {"pinyin": "n ǐ h ǎo", "text": "你好", "is_target": False},  # Most common false positive
    {"pinyin": "n ǐ h ǎo a", "text": "你好啊", "is_target": False},
    {"pinyin": "n ín h ǎo", "text": "您好", "is_target": False},
    {"pinyin": "n ǐ h ǎo m a", "text": "你好吗", "is_target": False},
]


@dataclass
class DecoyOptimizationResult:
    """Decoy optimization result."""
    target_boost: float
    decoy_boost: float
    threshold: float
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    decoy_blocked: int = 0  # Negatives correctly blocked by decoy
    total_positive: int = 0
    total_negative: int = 0
    process_time: float = 0.0

    @property
    def frr(self) -> float:
        if self.total_positive == 0:
            return 0.0
        return (self.false_negative / self.total_positive) * 100

    @property
    def far(self) -> float:
        if self.total_negative == 0:
            return 0.0
        return (self.false_positive / self.total_negative) * 100

    @property
    def recall(self) -> float:
        return 100 - self.frr

    @property
    def specificity(self) -> float:
        return 100 - self.far

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_boost": self.target_boost,
            "decoy_boost": self.decoy_boost,
            "threshold": self.threshold,
            "total_positive": self.total_positive,
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "frr_percent": round(self.frr, 2),
            "recall_percent": round(self.recall, 2),
            "total_negative": self.total_negative,
            "false_positive": self.false_positive,
            "decoy_blocked": self.decoy_blocked,
            "far_percent": round(self.far, 2),
            "specificity_percent": round(self.specificity, 2),
            "process_time_sec": round(self.process_time, 3),
        }

    def is_target_met(self, frr_target: float = 5.0, far_target: float = 20.0) -> bool:
        return self.frr <= frr_target and self.far <= far_target


def create_decoy_keywords_file(
    output_path: str,
    target_boost: float,
    decoy_boost: float,
    threshold: float,
    keywords: List[Dict] = None,
) -> None:
    """Create keywords file with target and decoy keywords.

    Args:
        output_path: Output file path
        target_boost: Boost for target keyword
        decoy_boost: Boost for decoy keywords
        threshold: Detection threshold
        keywords: List of keyword dicts with pinyin, text, is_target
    """
    if keywords is None:
        keywords = DECOY_KEYWORDS

    lines = []
    for kw in keywords:
        boost = target_boost if kw["is_target"] else decoy_boost
        line = f"{kw['pinyin']} :{boost:.2f} #{threshold:.2f} @{kw['text']}"
        lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def create_keyword_spotter(
    encoder_path: str,
    decoder_path: str,
    joiner_path: str,
    tokens_path: str,
    keywords_path: str,
    num_threads: int = 4,
    provider: str = "cpu",
) -> sherpa_onnx.KeywordSpotter:
    """Create sherpa-onnx KeywordSpotter."""
    return sherpa_onnx.KeywordSpotter(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        keywords_file=keywords_path,
        num_threads=num_threads,
        provider=provider,
    )


def test_audio_file_with_decoy(
    spotter: sherpa_onnx.KeywordSpotter,
    audio_path: str,
    target_text: str = "你好真真",
) -> Tuple[bool, Optional[str]]:
    """Test audio file with decoy filtering.

    Args:
        spotter: KeywordSpotter instance
        audio_path: Path to audio file
        target_text: Target keyword text

    Returns:
        (is_target_detected, detected_keyword_text)
        - is_target_detected: True only if target keyword detected (not decoy)
        - detected_keyword_text: The actual keyword detected (for analysis)
    """
    try:
        samples, sample_rate = sf.read(audio_path, dtype="float32")

        if sample_rate != 16000:
            try:
                import librosa
                samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except ImportError:
                return False, None

        if len(samples.shape) > 1:
            samples = samples[:, 0]

        stream = spotter.create_stream()
        stream.accept_waveform(sample_rate, samples)

        tail_paddings = [0.0] * int(0.3 * sample_rate)
        stream.accept_waveform(sample_rate, tail_paddings)
        stream.input_finished()

        detected_keyword = None
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
            result = spotter.get_result(stream)
            if result:
                detected_keyword = result.strip()
                break

        if detected_keyword:
            # Check if it's the target keyword (not a decoy)
            # Parse the keyword text from result
            # Result format varies - try to extract the text
            is_target = target_text in detected_keyword
            return is_target, detected_keyword

        return False, None
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False, None


def evaluate_decoy_config(
    spotter: sherpa_onnx.KeywordSpotter,
    positive_files: List[Path],
    negative_files: List[Path],
    target_text: str = "你好真真",
) -> Tuple[int, int, int, int, int, int, float]:
    """Evaluate decoy configuration.

    Returns:
        (tp, fn, fp, decoy_blocked, total_pos, total_neg, process_time)
    """
    start_time = time.time()

    tp = 0
    fn = 0
    for audio_path in positive_files:
        is_target, detected = test_audio_file_with_decoy(spotter, str(audio_path), target_text)
        if is_target:
            tp += 1
        else:
            fn += 1

    fp = 0
    decoy_blocked = 0
    for audio_path in negative_files:
        is_target, detected = test_audio_file_with_decoy(spotter, str(audio_path), target_text)
        if is_target:
            fp += 1
        elif detected:
            # Decoy was detected instead of target - this is good!
            decoy_blocked += 1

    process_time = time.time() - start_time
    return tp, fn, fp, decoy_blocked, len(positive_files), len(negative_files), process_time


def run_decoy_grid_search(
    model_dir: Path,
    positive_files: List[Path],
    negative_files: List[Path],
    target_boost_values: List[float],
    decoy_boost_values: List[float],
    threshold_values: List[float],
    num_threads: int = 4,
    provider: str = "cpu",
    output_dir: Path = None,
) -> Tuple[List[DecoyOptimizationResult], DecoyOptimizationResult]:
    """Run decoy parameter grid search."""
    if output_dir is None:
        output_dir = model_dir / "decoy_optimization"

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_keywords_file = output_dir / "keywords_temp.txt"

    # Find model files
    suffix = ".int8.onnx"
    encoder_files = list(model_dir.glob(f"encoder-*{suffix}"))
    decoder_files = list(model_dir.glob(f"decoder-*{suffix}"))
    joiner_files = list(model_dir.glob(f"joiner-*{suffix}"))

    if not encoder_files or not decoder_files or not joiner_files:
        raise RuntimeError(f"Could not find model files with suffix {suffix} in {model_dir}")

    encoder_path = str(encoder_files[0])
    decoder_path = str(decoder_files[0])
    joiner_path = str(joiner_files[0])
    tokens_path = str(model_dir / "tokens.txt")

    all_results: List[DecoyOptimizationResult] = []
    total_configs = len(target_boost_values) * len(decoy_boost_values) * len(threshold_values)

    print(f"\nStarting decoy parameter grid search: {total_configs} configurations")
    print(f"Target boost values: {target_boost_values}")
    print(f"Decoy boost values: {decoy_boost_values}")
    print(f"Threshold values: {threshold_values}")
    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print("=" * 60)

    config_idx = 0
    for target_boost in target_boost_values:
        for decoy_boost in decoy_boost_values:
            for threshold in threshold_values:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] Testing target_boost={target_boost:.2f}, "
                      f"decoy_boost={decoy_boost:.2f}, threshold={threshold:.2f}")

                # Create keywords file with decoys
                create_decoy_keywords_file(
                    str(temp_keywords_file),
                    target_boost=target_boost,
                    decoy_boost=decoy_boost,
                    threshold=threshold,
                )

                # Create spotter
                spotter = create_keyword_spotter(
                    encoder_path=encoder_path,
                    decoder_path=decoder_path,
                    joiner_path=joiner_path,
                    tokens_path=tokens_path,
                    keywords_path=str(temp_keywords_file),
                    num_threads=num_threads,
                    provider=provider,
                )

                # Evaluate
                tp, fn, fp, decoy_blocked, total_pos, total_neg, proc_time = evaluate_decoy_config(
                    spotter, positive_files, negative_files
                )

                result = DecoyOptimizationResult(
                    target_boost=target_boost,
                    decoy_boost=decoy_boost,
                    threshold=threshold,
                    true_positive=tp,
                    false_negative=fn,
                    false_positive=fp,
                    decoy_blocked=decoy_blocked,
                    total_positive=total_pos,
                    total_negative=total_neg,
                    process_time=proc_time,
                )
                all_results.append(result)

                print(f"  TP: {tp}/{total_pos}, FN: {fn}, FP: {fp}/{total_neg}, Blocked: {decoy_blocked}")
                print(f"  FRR: {result.frr:.2f}%, FAR: {result.far:.2f}%, Recall: {result.recall:.2f}%")

                del spotter

    # Find best result
    best_result = min(
        all_results,
        key=lambda r: (r.frr ** 2 + r.far, r.far if r.frr <= 5.0 else 100)
        if r.frr <= 10.0 else (r.frr, r.far),
    )

    if temp_keywords_file.exists():
        temp_keywords_file.unlink()

    return all_results, best_result


def generate_decoy_report(
    all_results: List[DecoyOptimizationResult],
    best_result: DecoyOptimizationResult,
    output_path: Path,
    target_frr: float = 5.0,
    target_far: float = 20.0,
) -> None:
    """Generate decoy optimization report."""
    valid_configs = [r for r in all_results if r.is_target_met(target_frr, target_far)]

    report_lines = [
        "=" * 70,
        "KWS Decoy Parameter Optimization Report",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Configuration Summary:",
        f"  Total configurations tested: {len(all_results)}",
        f"  Configurations meeting targets (FRR≤{target_frr}%, FAR≤{target_far}%): {len(valid_configs)}",
        "",
        "Best Configuration:",
        f"  Target Boost: {best_result.target_boost}",
        f"  Decoy Boost: {best_result.decoy_boost}",
        f"  Threshold: {best_result.threshold}",
        f"  FRR: {best_result.frr:.2f}%",
        f"  FAR: {best_result.far:.2f}%",
        f"  Recall: {best_result.recall:.2f}%",
        f"  Decoys Blocked: {best_result.decoy_blocked}",
        "",
        "=" * 70,
        "All Results (Sorted by FRR, then FAR):",
        "=" * 70,
    ]

    report_lines.append(
        f"{'TgtBoost':>8} | {'DecBoost':>8} | {'Thresh':>6} | {'TP':>3} | {'FN':>3} | "
        f"{'FP':>3} | {'Blocked':>7} | {'FRR%':>6} | {'FAR%':>6}"
    )
    report_lines.append("-" * 80)

    sorted_results = sorted(all_results, key=lambda r: (r.frr, r.far))

    for result in sorted_results:
        mark = " [BEST]" if result == best_result else ""
        mark += " [OK]" if result.is_target_met(target_frr, target_far) else ""
        report_lines.append(
            f"{result.target_boost:>8.2f} | {result.decoy_boost:>8.2f} | "
            f"{result.threshold:>6.2f} | {result.true_positive:>3} | "
            f"{result.false_negative:>3} | {result.false_positive:>3} | "
            f"{result.decoy_blocked:>7} | {result.frr:>6.2f} | "
            f"{result.far:>6.2f}{mark}"
        )

    if valid_configs:
        report_lines.extend([
            "",
            "=" * 70,
            "Valid Configurations (meeting targets):",
            "=" * 70,
        ])
        for result in sorted(valid_configs, key=lambda r: (r.frr + r.far)):
            report_lines.append(
                f"  target_boost={result.target_boost}, decoy_boost={result.decoy_boost}, "
                f"threshold={result.threshold}: FRR={result.frr:.2f}%, FAR={result.far:.2f}%"
            )

    report_lines.append("=" * 70)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimize KWS model with Decoy Approach")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/workspace/llm/keyword-spotting/exp/kws_finetune_v2",
        help="Directory containing ONNX model files",
    )
    parser.add_argument(
        "--positive-dir",
        type=str,
        default="/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive",
        help="Directory containing positive test samples",
    )
    parser.add_argument(
        "--negative-dir",
        type=str,
        default="/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/negative",
        help="Directory containing negative test samples",
    )
    parser.add_argument(
        "--target-boost-values",
        type=str,
        default="1.5,2.0,2.5,3.0",
        help="Comma-separated list of target boost values",
    )
    parser.add_argument(
        "--decoy-boost-values",
        type=str,
        default="0.5,1.0,1.5",
        help="Comma-separated list of decoy boost values",
    )
    parser.add_argument(
        "--threshold-values",
        type=str,
        default="0.15,0.20,0.25,0.30",
        help="Comma-separated list of threshold values",
    )
    parser.add_argument(
        "--target-frr",
        type=float,
        default=5.0,
        help="Target FRR (percentage)",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=20.0,
        help="Target FAR (percentage)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for inference",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="ONNX Runtime provider",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    target_boost_values = [float(x.strip()) for x in args.target_boost_values.split(",")]
    decoy_boost_values = [float(x.strip()) for x in args.decoy_boost_values.split(",")]
    threshold_values = [float(x.strip()) for x in args.threshold_values.split(",")]

    positive_files = sorted(Path(args.positive_dir).glob("*.wav"))
    negative_files = sorted(Path(args.negative_dir).glob("*.wav"))

    print("=" * 70)
    print("KWS Decoy Parameter Optimization")
    print("=" * 70)
    print(f"Model directory: {model_dir}")
    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print(f"Target boost values: {target_boost_values}")
    print(f"Decoy boost values: {decoy_boost_values}")
    print(f"Threshold values: {threshold_values}")
    print(f"Targets: FRR < {args.target_frr}%, FAR < {args.target_far}%")
    print("=" * 70)

    output_dir = model_dir / "decoy_optimization"
    all_results, best_result = run_decoy_grid_search(
        model_dir=model_dir,
        positive_files=positive_files,
        negative_files=negative_files,
        target_boost_values=target_boost_values,
        decoy_boost_values=decoy_boost_values,
        threshold_values=threshold_values,
        num_threads=args.num_threads,
        provider=args.provider,
        output_dir=output_dir,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"decoy_optimization_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_configurations": len(all_results),
            "best_configuration": best_result.to_dict(),
            "all_results": [r.to_dict() for r in all_results],
        }, f, ensure_ascii=False, indent=2)
    print(f"JSON results saved to: {json_path}")

    # Generate report
    report_path = output_dir / f"decoy_optimization_{timestamp}.txt"
    generate_decoy_report(
        all_results,
        best_result,
        report_path,
        target_frr=args.target_frr,
        target_far=args.target_far,
    )

    # Create recommended keywords file
    best_keywords_path = model_dir / "keywords_decoy.txt"
    create_decoy_keywords_file(
        str(best_keywords_path),
        target_boost=best_result.target_boost,
        decoy_boost=best_result.decoy_boost,
        threshold=best_result.threshold,
    )

    print("\n" + "=" * 70)
    print("Decoy Optimization Complete")
    print("=" * 70)
    print(f"Best configuration:")
    print(f"  Target Boost: {best_result.target_boost}")
    print(f"  Decoy Boost: {best_result.decoy_boost}")
    print(f"  Threshold: {best_result.threshold}")
    print(f"  FRR: {best_result.frr:.2f}%")
    print(f"  FAR: {best_result.far:.2f}%")
    print(f"  Decoys Blocked: {best_result.decoy_blocked}")
    print(f"\nRecommended keywords file: {best_keywords_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
