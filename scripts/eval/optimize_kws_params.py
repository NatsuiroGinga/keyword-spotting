#!/usr/bin/env python3
"""
Parameter optimization script for fine-tuned KWS model.
Performs grid search over boost and threshold parameters to find optimal configuration.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import sherpa_onnx
import soundfile as sf


@dataclass
class OptimizationResult:
    """Parameter optimization result."""
    boost: float
    threshold: float
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    total_positive: int = 0
    total_negative: int = 0
    process_time: float = 0.0

    @property
    def frr(self) -> float:
        """False Rejection Rate."""
        if self.total_positive == 0:
            return 0.0
        return (self.false_negative / self.total_positive) * 100

    @property
    def far(self) -> float:
        """False Accept Rate."""
        if self.total_negative == 0:
            return 0.0
        return (self.false_positive / self.total_negative) * 100

    @property
    def recall(self) -> float:
        """True Positive Rate."""
        return 100 - self.frr

    @property
    def specificity(self) -> float:
        """True Negative Rate."""
        return 100 - self.far

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "boost": self.boost,
            "threshold": self.threshold,
            "total_positive": self.total_positive,
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "frr_percent": round(self.frr, 2),
            "recall_percent": round(self.recall, 2),
            "total_negative": self.total_negative,
            "false_positive": self.false_positive,
            "far_percent": round(self.far, 2),
            "specificity_percent": round(self.specificity, 2),
            "process_time_sec": round(self.process_time, 3),
        }

    def is_target_met(self, frr_target: float = 5.0, far_target: float = 20.0) -> bool:
        """Check if targets are met."""
        return self.frr <= frr_target and self.far <= far_target


def create_keywords_file(
    output_path: str,
    boost: float,
    threshold: float,
    keyword_pinyin: str = "n ǐ h ǎo zh ēn zh ēn",
    keyword_text: str = "你好真真",
) -> None:
    """Create a parameterized keywords.txt file.

    Args:
        output_path: Output file path
        boost: Boost score parameter
        threshold: Trigger threshold parameter
        keyword_pinyin: Pinyin sequence
        keyword_text: Chinese text
    """
    # Format: pinyin_sequence :boost #threshold @text
    line = f"{keyword_pinyin} :{boost:.1f} #{threshold:.1f} @{keyword_text}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(line + "\n")


def create_keyword_spotter(
    encoder_path: str,
    decoder_path: str,
    joiner_path: str,
    tokens_path: str,
    keywords_path: str,
    num_threads: int = 4,
    provider: str = "cpu",
) -> sherpa_onnx.KeywordSpotter:
    """Create a sherpa-onnx KeywordSpotter.

    Args:
        encoder_path: Path to encoder ONNX model
        decoder_path: Path to decoder ONNX model
        joiner_path: Path to joiner ONNX model
        tokens_path: Path to tokens.txt file
        keywords_path: Path to keywords.txt file
        num_threads: Number of threads for CPU inference
        provider: ONNX Runtime provider ("cpu" or "cuda")
    """
    provider_kwargs = {}
    if provider == "cuda":
        provider_kwargs = {"provider": "cuda"}

    return sherpa_onnx.KeywordSpotter(
        tokens=tokens_path,
        encoder=encoder_path,
        decoder=decoder_path,
        joiner=joiner_path,
        keywords_file=keywords_path,
        num_threads=num_threads,
        provider=provider,
    )


def test_audio_file(
    spotter: sherpa_onnx.KeywordSpotter,
    audio_path: str,
) -> bool:
    """Test a single audio file for keyword detection.

    Args:
        spotter: KeywordSpotter instance
        audio_path: Path to audio file

    Returns:
        True if keyword detected
    """
    try:
        samples, sample_rate = sf.read(audio_path, dtype="float32")

        if sample_rate != 16000:
            try:
                import librosa
                samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except ImportError:
                print(f"Warning: librosa not available, skipping {audio_path}")
                return False

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
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False


def evaluate_config(
    spotter: sherpa_onnx.KeywordSpotter,
    positive_files: List[Path],
    negative_files: List[Path],
) -> Tuple[int, int, int, int, float]:
    """Evaluate a single parameter configuration.

    Args:
        spotter: KeywordSpotter instance
        positive_files: List of positive sample files
        negative_files: List of negative sample files

    Returns:
        (tp, fn, fp, total_pos, total_neg, process_time)
    """
    start_time = time.time()

    # Evaluate positive samples
    tp = 0
    fn = 0
    for audio_path in positive_files:
        detected = test_audio_file(spotter, str(audio_path))
        if detected:
            tp += 1
        else:
            fn += 1

    # Evaluate negative samples
    fp = 0
    for audio_path in negative_files:
        detected = test_audio_file(spotter, str(audio_path))
        if detected:
            fp += 1

    process_time = time.time() - start_time
    return tp, fn, fp, len(positive_files), len(negative_files), process_time


def run_parameter_grid_search(
    model_dir: Path,
    keywords_template_pinyin: str,
    keywords_template_text: str,
    positive_files: List[Path],
    negative_files: List[Path],
    boost_values: List[float],
    threshold_values: List[float],
    num_threads: int = 4,
    provider: str = "cpu",
    output_dir: Path = None,
) -> Tuple[List[OptimizationResult], OptimizationResult]:
    """Run parameter grid search.

    Args:
        model_dir: Model directory
        keywords_template_pinyin: Pinyin sequence template
        keywords_template_text: Chinese text template
        positive_files: Positive sample files
        negative_files: Negative sample files
        boost_values: List of boost values to test
        threshold_values: List of threshold values to test
        num_threads: Number of threads for inference
        provider: ONNX Runtime provider ("cpu" or "cuda")
        output_dir: Output directory for intermediate files

    Returns:
        (all_results, best_result)
    """
    if output_dir is None:
        output_dir = model_dir / "param_optimization"

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

    all_results: List[OptimizationResult] = []
    total_configs = len(boost_values) * len(threshold_values)

    print(f"\nStarting parameter grid search: {total_configs} configurations")
    print(f"Boost values: {boost_values}")
    print(f"Threshold values: {threshold_values}")
    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print("=" * 60)

    config_idx = 0
    for boost in boost_values:
        for threshold in threshold_values:
            config_idx += 1
            print(f"\n[{config_idx}/{total_configs}] Testing boost={boost:.1f}, threshold={threshold:.1f}")

            # Create temporary keywords file
            create_keywords_file(
                str(temp_keywords_file),
                boost=boost,
                threshold=threshold,
                keyword_pinyin=keywords_template_pinyin,
                keyword_text=keywords_template_text,
            )

            # Create spotter with current config
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
            tp, fn, fp, total_pos, total_neg, proc_time = evaluate_config(
                spotter, positive_files, negative_files
            )

            result = OptimizationResult(
                boost=boost,
                threshold=threshold,
                true_positive=tp,
                false_negative=fn,
                false_positive=fp,
                total_positive=total_pos,
                total_negative=total_neg,
                process_time=proc_time,
            )
            all_results.append(result)

            frr = result.frr
            far = result.far
            recall = result.recall

            print(f"  TP: {tp}/{total_pos}, FN: {fn}, FP: {fp}/{total_neg}")
            print(f"  FRR: {frr:.2f}%, FAR: {far:.2f}%, Recall: {recall:.2f}%")

            # Clean up spotter
            del spotter

    # Find best result (minimize FRR while keeping FAR reasonable)
    best_result = min(
        all_results,
        key=lambda r: (r.frr ** 2 + r.far, r.far if r.frr <= 5.0 else 100)
        if r.frr <= 10.0 else (r.frr, r.far),
    )

    # Clean up temp file
    if temp_keywords_file.exists():
        temp_keywords_file.unlink()

    return all_results, best_result


def generate_report(
    all_results: List[OptimizationResult],
    best_result: OptimizationResult,
    output_path: Path,
    target_frr: float = 5.0,
    target_far: float = 20.0,
) -> None:
    """Generate optimization report.

    Args:
        all_results: All optimization results
        best_result: Best result selected
        output_path: Output file path
        target_frr: Target FRR
        target_far: Target FAR
    """
    # Find configurations that meet targets
    valid_configs = [r for r in all_results if r.is_target_met(target_frr, target_far)]

    # Group by boost value for analysis
    boost_analysis: Dict[float, Dict[str, Any]] = {}
    for result in all_results:
        if result.boost not in boost_analysis:
            boost_analysis[result.boost] = []
        boost_analysis[result.boost].append(result)

    report_lines = [
        "=" * 70,
        "KWS Parameter Optimization Report",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Configuration Summary:",
        f"  Total configurations tested: {len(all_results)}",
        f"  Configurations meeting targets (FRR≤{target_frr}%, FAR≤{target_far}%): {len(valid_configs)}",
        "",
        "Best Configuration:",
        f"  Boost: {best_result.boost}",
        f"  Threshold: {best_result.threshold}",
        f"  FRR: {best_result.frr:.2f}%",
        f"  FAR: {best_result.far:.2f}%",
        f"  Recall: {best_result.recall:.2f}%",
        f"  Specificity: {best_result.specificity:.2f}%",
        "",
        "=" * 70,
        "All Results (Sorted by FRR, then FAR):",
        "=" * 70,
    ]

    # Header
    report_lines.append(f"{'Boost':>6} | {'Thresh':>6} | {'TP':>3} | {'FN':>3} | {'FP':>3} | {'FRR%':>6} | {'FAR%':>6}")
    report_lines.append("-" * 60)

    # Sort by FRR, then FAR
    sorted_results = sorted(all_results, key=lambda r: (r.frr, r.far))

    for result in sorted_results:
        mark = " [BEST]" if result == best_result else ""
        mark += " [OK]" if result.is_target_met(target_frr, target_far) else ""
        report_lines.append(
            f"{result.boost:>6.1f} | {result.threshold:>6.1f} | "
            f"{result.true_positive:>3} | {result.false_negative:>3} | "
            f"{result.false_positive:>3} | {result.frr:>6.2f} | "
            f"{result.far:>6.2f}{mark}"
        )

    # Add analysis section
    report_lines.extend([
        "",
        "=" * 70,
        "Boost Value Analysis (average performance by boost):",
        "=" * 70,
    ])

    for boost in sorted(boost_analysis.keys()):
        results = boost_analysis[boost]
        avg_frr = sum(r.frr for r in results) / len(results)
        avg_far = sum(r.far for r in results) / len(results)
        # Find best threshold for this boost
        best_for_boost = min(results, key=lambda r: (r.frr, r.far))
        report_lines.append(
            f"  Boost={boost:.1f}: Avg FRR={avg_frr:.2f}%, "
            f"Avg FAR={avg_far:.2f}%, "
            f"Best at thresh={best_for_boost.threshold:.1f} "
            f"(FRR={best_for_boost.frr:.2f}%)"
        )

    # Add valid configurations section
    if valid_configs:
        report_lines.extend([
            "",
            "=" * 70,
            "Valid Configurations (meeting targets):",
            "=" * 70,
        ])
        for result in sorted(valid_configs, key=lambda r: (r.frr + r.far)):
            report_lines.append(
                f"  boost={result.boost}, threshold={result.threshold}: "
                f"FRR={result.frr:.2f}%, FAR={result.far:.2f}%"
            )

    report_lines.append("=" * 70)

    # Write report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {output_path}")


def save_json_results(
    all_results: List[OptimizationResult],
    best_result: OptimizationResult,
    output_path: Path,
    metadata: Dict[str, Any] = None,
) -> None:
    """Save results as JSON.

    Args:
        all_results: All optimization results
        best_result: Best result selected
        output_path: Output file path
        metadata: Additional metadata
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_configurations": len(all_results),
        "metadata": metadata or {},
        "best_configuration": best_result.to_dict(),
        "all_results": [r.to_dict() for r in all_results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"JSON results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimize KWS model parameters")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/workspace/llm/keyword-spotting/exp/kws_finetune",
        help="Directory containing ONNX model files",
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
        "--keyword-pinyin",
        type=str,
        default="n ǐ h ǎo zh ēn zh ēn",
        help="Pinyin sequence for the keyword",
    )
    parser.add_argument(
        "--keyword-text",
        type=str,
        default="你好真真",
        help="Chinese text for the keyword",
    )
    parser.add_argument(
        "--boost-values",
        type=str,
        default="0.3,0.5,0.7,0.8,1.0,1.2,1.5",
        help="Comma-separated list of boost values to test",
    )
    parser.add_argument(
        "--threshold-values",
        type=str,
        default="0.4,0.45,0.5,0.55,0.6,0.65,0.7",
        help="Comma-separated list of threshold values to test",
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
        help="Number of threads for inference (CPU mode)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="ONNX Runtime provider: 'cpu' or 'cuda' (GPU acceleration)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: model_dir/param_optimization)",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Parse parameter grids
    boost_values = [float(x.strip()) for x in args.boost_values.split(",")]
    threshold_values = [float(x.strip()) for x in args.threshold_values.split(",")]

    # Get test files
    positive_files = sorted(Path(args.positive_dir).glob("*.wav"))
    negative_files = sorted(Path(args.negative_dir).glob("*.wav"))

    if not positive_files:
        print(f"Warning: No positive samples found in {args.positive_dir}")
    if not negative_files:
        print(f"Warning: No negative samples found in {args.negative_dir}")

    print("=" * 70)
    print("KWS Parameter Optimization")
    print("=" * 70)
    print(f"Model directory: {model_dir}")
    print(f"Keyword: {args.keyword_text}")
    print(f"Pinyin: {args.keyword_pinyin}")
    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print(f"Boost values: {boost_values}")
    print(f"Threshold values: {threshold_values}")
    print(f"Targets: FRR < {args.target_frr}%, FAR < {args.target_far}%")
    print("=" * 70)

    # Run grid search
    output_dir = model_dir / "param_optimization" if args.output_dir is None else Path(args.output_dir)
    all_results, best_result = run_parameter_grid_search(
        model_dir=model_dir,
        keywords_template_pinyin=args.keyword_pinyin,
        keywords_template_text=args.keyword_text,
        positive_files=positive_files,
        negative_files=negative_files,
        boost_values=boost_values,
        threshold_values=threshold_values,
        num_threads=args.num_threads,
        provider=args.provider,
        output_dir=output_dir,
    )

    # Generate output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"param_optimization_{timestamp}.json"
    metadata = {
        "model_dir": str(model_dir),
        "keyword_text": args.keyword_text,
        "keyword_pinyin": args.keyword_pinyin,
        "positive_samples_count": len(positive_files),
        "negative_samples_count": len(negative_files),
        "boost_values": boost_values,
        "threshold_values": threshold_values,
        "target_frr": args.target_frr,
        "target_far": args.target_far,
    }
    save_json_results(all_results, best_result, json_path, metadata)

    # Generate text report
    report_path = output_dir / f"param_optimization_{timestamp}.txt"
    generate_report(
        all_results,
        best_result,
        report_path,
        target_frr=args.target_frr,
        target_far=args.target_far,
    )

    # Create recommended keywords.txt
    best_keywords_path = model_dir / "keywords_recommended.txt"
    create_keywords_file(
        str(best_keywords_path),
        boost=best_result.boost,
        threshold=best_result.threshold,
        keyword_pinyin=args.keyword_pinyin,
        keyword_text=args.keyword_text,
    )

    print("\n" + "=" * 70)
    print("Optimization Complete")
    print("=" * 70)
    print(f"Best configuration:")
    print(f"  Boost: {best_result.boost}")
    print(f"  Threshold: {best_result.threshold}")
    print(f"  FRR: {best_result.frr:.2f}%")
    print(f"  FAR: {best_result.far:.2f}%")
    print(f"\nRecommended keywords file: {best_keywords_path}")
    print("=" * 70)

    # Create summary for commit message
    with open(model_dir / "optimization_summary.txt", "w") as f:
        f.write(f"# Parameter Optimization Summary\n\n")
        f.write(f"Best Configuration:\n")
        f.write(f"  boost={best_result.boost}\n")
        f.write(f"  threshold={best_result.threshold}\n\n")
        f.write(f"Results:\n")
        f.write(f"  FRR: {best_result.frr:.2f}%\n")
        f.write(f"  FAR: {best_result.far:.2f}%\n")
        f.write(f"  Recall: {best_result.recall:.2f}%\n")
        f.write(f"  Specificity: {best_result.specificity:.2f}%\n\n")
        f.write(f"Targets: FRR < {args.target_frr}%, FAR < {args.target_far}%\n")
        f.write(f"Pass: {best_result.is_target_met(args.target_frr, args.target_far)}\n")


if __name__ == "__main__":
    main()
