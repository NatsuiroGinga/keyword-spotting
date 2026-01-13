#!/usr/bin/env python3
"""
Compare Direct Inference vs Delayed Decision Inference for KWS.

This script runs both inference modes on the same test dataset and generates
a comprehensive comparison report including:
- FRR (False Rejection Rate)
- FAR (False Accept Rate)
- RTF (Real-Time Factor)
- Other metrics (Recall, Precision, F1, etc.)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_kws_with_rtf import DirectInferenceEvaluator, EvaluationStats as DirectStats
from evaluate_with_delay import (
    DelayedDecisionEvaluator, 
    DelayedDecisionConfig,
    EvaluationStats as DelayedStats,
)
from rtf_utils import compare_rtf_stats


def find_model_files(model_dir: Path, use_int8: bool = True) -> Dict[str, str]:
    """Find model files in directory."""
    suffix = ".int8.onnx" if use_int8 else ".onnx"
    
    encoder_files = list(model_dir.glob(f"encoder-*{suffix}"))
    decoder_files = list(model_dir.glob(f"decoder-*{suffix}"))
    joiner_files = list(model_dir.glob(f"joiner-*{suffix}"))
    
    if not encoder_files or not decoder_files or not joiner_files:
        raise RuntimeError(f"Could not find model files with suffix {suffix} in {model_dir}")
    
    return {
        "encoder": str(encoder_files[0]),
        "decoder": str(decoder_files[0]),
        "joiner": str(joiner_files[0]),
        "tokens": str(model_dir / "tokens.txt"),
        "keywords": str(model_dir / "keywords.txt"),
    }


def generate_comparison_report(
    direct_stats: DirectStats,
    delayed_stats: DelayedStats,
    config: Dict[str, Any],
) -> str:
    """Generate comparison report."""
    lines = [
        "=" * 70,
        "KWS Inference Mode Comparison Report",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Configuration:",
        f"  Model: {config['model_dir']}",
        f"  Positive samples: {config['positive_count']}",
        f"  Negative samples: {config['negative_count']}",
        f"  Prefix timeout: {config['prefix_timeout_ms']}ms",
        "",
        "=" * 70,
        "Detection Metrics Comparison",
        "=" * 70,
        "",
        f"{'Metric':<25} | {'Direct':>12} | {'Delayed':>12} | {'Diff':>12}",
        "-" * 70,
    ]
    
    # Detection metrics comparison
    metrics = [
        ("FRR (%)", direct_stats.frr, delayed_stats.frr),
        ("FAR (%)", direct_stats.far, delayed_stats.far),
        ("Recall (%)", direct_stats.recall, delayed_stats.recall),
        ("Specificity (%)", direct_stats.specificity, delayed_stats.specificity),
        ("Accuracy (%)", direct_stats.accuracy, delayed_stats.accuracy),
        ("Precision (%)", direct_stats.precision, delayed_stats.precision),
        ("F1 Score", direct_stats.f1_score, delayed_stats.f1_score),
    ]
    
    for name, val_direct, val_delayed in metrics:
        diff = val_delayed - val_direct
        diff_str = f"{diff:+.2f}"
        
        # Add indicator for improvement/degradation
        if "FRR" in name or "FAR" in name:
            # Lower is better
            indicator = "✓" if diff < 0 else ("✗" if diff > 0 else "=")
        else:
            # Higher is better
            indicator = "✓" if diff > 0 else ("✗" if diff < 0 else "=")
        
        lines.append(f"{name:<25} | {val_direct:>12.2f} | {val_delayed:>12.2f} | {diff_str:>10} {indicator}")
    
    lines.extend([
        "",
        "=" * 70,
        "Detection Counts",
        "=" * 70,
        "",
        f"{'Metric':<25} | {'Direct':>12} | {'Delayed':>12}",
        "-" * 70,
        f"{'True Positives':<25} | {direct_stats.true_positive:>12} | {delayed_stats.true_positive:>12}",
        f"{'False Negatives':<25} | {direct_stats.false_negative:>12} | {delayed_stats.false_negative:>12}",
        f"{'True Negatives':<25} | {direct_stats.true_negative:>12} | {delayed_stats.true_negative:>12}",
        f"{'False Positives':<25} | {direct_stats.false_positive:>12} | {delayed_stats.false_positive:>12}",
    ])
    
    # RTF comparison
    lines.extend([
        "",
        "=" * 70,
        "RTF (Real-Time Factor) Comparison",
        "=" * 70,
        "",
        f"{'Metric':<25} | {'Direct':>12} | {'Delayed':>12} | {'Diff':>12}",
        "-" * 70,
    ])
    
    rtf_metrics = [
        ("Overall RTF", direct_stats.rtf_stats.rtf, delayed_stats.rtf_stats.rtf),
        ("Mean RTF", direct_stats.rtf_stats.rtf_mean, delayed_stats.rtf_stats.rtf_mean),
        ("Median RTF", direct_stats.rtf_stats.rtf_median, delayed_stats.rtf_stats.rtf_median),
        ("P95 RTF", direct_stats.rtf_stats.rtf_p95, delayed_stats.rtf_stats.rtf_p95),
        ("P99 RTF", direct_stats.rtf_stats.rtf_p99, delayed_stats.rtf_stats.rtf_p99),
        ("Min RTF", direct_stats.rtf_stats.rtf_min, delayed_stats.rtf_stats.rtf_min),
        ("Max RTF", direct_stats.rtf_stats.rtf_max, delayed_stats.rtf_stats.rtf_max),
    ]
    
    for name, val_direct, val_delayed in rtf_metrics:
        diff = val_delayed - val_direct
        diff_pct = (diff / val_direct * 100) if val_direct > 0 else 0
        diff_str = f"{diff:+.4f} ({diff_pct:+.1f}%)"
        lines.append(f"{name:<25} | {val_direct:>12.4f} | {val_delayed:>12.4f} | {diff_str}")
    
    # Real-time capability
    direct_rt = "Yes" if direct_stats.rtf_stats.is_realtime else "No"
    delayed_rt = "Yes" if delayed_stats.rtf_stats.is_realtime else "No"
    lines.extend([
        "-" * 70,
        f"{'Real-time capable':<25} | {direct_rt:>12} | {delayed_rt:>12}",
    ])
    
    # Processing time
    lines.extend([
        "",
        "=" * 70,
        "Processing Time",
        "=" * 70,
        "",
        f"{'Metric':<25} | {'Direct':>12} | {'Delayed':>12}",
        "-" * 70,
        f"{'Total audio (sec)':<25} | {direct_stats.rtf_stats.total_audio_duration_sec:>12.2f} | {delayed_stats.rtf_stats.total_audio_duration_sec:>12.2f}",
        f"{'Total process (sec)':<25} | {direct_stats.rtf_stats.total_process_time_sec:>12.2f} | {delayed_stats.rtf_stats.total_process_time_sec:>12.2f}",
    ])
    
    # Delayed decision specific stats
    if hasattr(delayed_stats, 'prefix_detections'):
        lines.extend([
            "",
            "=" * 70,
            "Delayed Decision State Machine Statistics",
            "=" * 70,
            "",
            f"  Prefix detections: {delayed_stats.prefix_detections}",
            f"  Suffix confirmations: {delayed_stats.suffix_confirmations}",
            f"  Timeouts (rejected): {delayed_stats.timeouts}",
            f"  Direct triggers: {delayed_stats.direct_triggers}",
        ])
    
    # Summary and recommendation
    lines.extend([
        "",
        "=" * 70,
        "Summary",
        "=" * 70,
        "",
    ])
    
    # Analyze results
    frr_improved = delayed_stats.frr < direct_stats.frr
    far_improved = delayed_stats.far < direct_stats.far
    rtf_acceptable = delayed_stats.rtf_stats.rtf < 1.0
    
    if far_improved and not frr_improved:
        lines.append("  Delayed decision mode REDUCES FAR (fewer false positives)")
        lines.append(f"  FAR improvement: {direct_stats.far:.2f}% -> {delayed_stats.far:.2f}%")
        if delayed_stats.frr > direct_stats.frr:
            lines.append(f"  Trade-off: FRR increased from {direct_stats.frr:.2f}% to {delayed_stats.frr:.2f}%")
    elif frr_improved and not far_improved:
        lines.append("  Delayed decision mode REDUCES FRR (fewer false negatives)")
        lines.append(f"  FRR improvement: {direct_stats.frr:.2f}% -> {delayed_stats.frr:.2f}%")
    elif frr_improved and far_improved:
        lines.append("  Delayed decision mode IMPROVES BOTH FRR and FAR")
    else:
        lines.append("  No significant improvement with delayed decision mode")
    
    lines.append("")
    if rtf_acceptable:
        lines.append(f"  RTF is acceptable for real-time use: {delayed_stats.rtf_stats.rtf:.4f}")
    else:
        lines.append(f"  WARNING: RTF exceeds real-time threshold: {delayed_stats.rtf_stats.rtf:.4f}")
    
    # Recommendation
    lines.extend([
        "",
        "Recommendation:",
    ])
    
    if far_improved and rtf_acceptable:
        lines.append("  ✓ Use DELAYED DECISION mode for production")
        lines.append("    - Reduces false positives while maintaining real-time performance")
    elif not rtf_acceptable:
        lines.append("  ✗ Delayed decision mode is TOO SLOW for real-time use")
        lines.append("    - Consider optimizing chunk size or using direct mode")
    else:
        lines.append("  → Consider trade-offs based on your specific requirements")
        lines.append("    - Direct mode: Lower latency, potentially more false positives")
        lines.append("    - Delayed mode: Higher latency, potentially fewer false positives")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def run_comparison(
    model_dir: Path,
    positive_files: List[str],
    negative_files: List[str],
    use_int8: bool = True,
    prefix_timeout_ms: int = 600,
    chunk_size_ms: int = 100,
    threshold: float = 0.0,
    num_threads: int = 4,
    verbose: bool = False,
) -> tuple:
    """Run comparison between direct and delayed inference modes."""
    
    # Find model files
    model_files = find_model_files(model_dir, use_int8)
    
    print("\n" + "=" * 60)
    print("Running Direct Inference Evaluation")
    print("=" * 60)
    
    # Direct inference evaluation
    direct_evaluator = DirectInferenceEvaluator(
        encoder_path=model_files["encoder"],
        decoder_path=model_files["decoder"],
        joiner_path=model_files["joiner"],
        tokens_path=model_files["tokens"],
        keywords_path=model_files["keywords"],
        keywords_threshold=threshold,
        num_threads=num_threads,
    )
    
    direct_stats = direct_evaluator.evaluate_dataset(
        positive_files=positive_files,
        negative_files=negative_files,
        verbose=verbose,
    )
    
    print(f"\nDirect mode - FRR: {direct_stats.frr:.2f}%, FAR: {direct_stats.far:.2f}%, RTF: {direct_stats.rtf_stats.rtf:.4f}")
    
    print("\n" + "=" * 60)
    print("Running Delayed Decision Evaluation")
    print("=" * 60)
    
    # Delayed decision evaluation
    delay_config = DelayedDecisionConfig(
        prefix_timeout_ms=prefix_timeout_ms,
        chunk_size_ms=chunk_size_ms,
    )
    
    delayed_evaluator = DelayedDecisionEvaluator(
        encoder_path=model_files["encoder"],
        decoder_path=model_files["decoder"],
        joiner_path=model_files["joiner"],
        tokens_path=model_files["tokens"],
        keywords_path=model_files["keywords"],
        config=delay_config,
        num_threads=num_threads,
    )
    
    delayed_stats = delayed_evaluator.evaluate_dataset(
        positive_files=positive_files,
        negative_files=negative_files,
        verbose=verbose,
    )
    
    print(f"\nDelayed mode - FRR: {delayed_stats.frr:.2f}%, FAR: {delayed_stats.far:.2f}%, RTF: {delayed_stats.rtf_stats.rtf:.4f}")
    
    return direct_stats, delayed_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compare Direct vs Delayed Decision inference modes"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/workspace/llm/keyword-spotting/exp/kws_finetune_v3",
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
        "--prefix-timeout",
        type=int,
        default=600,
        help="Prefix timeout in milliseconds for delayed decision",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size in milliseconds for streaming simulation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Keywords detection threshold",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for inference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get test files
    positive_files = sorted(Path(args.positive_dir).glob("*.wav"))
    negative_files = sorted(Path(args.negative_dir).glob("*.wav"))
    
    if not positive_files:
        print(f"Error: No positive samples found in {args.positive_dir}")
        return
    if not negative_files:
        print(f"Error: No negative samples found in {args.negative_dir}")
        return
    
    print("=" * 70)
    print("KWS Inference Mode Comparison")
    print("=" * 70)
    print(f"Model directory: {model_dir}")
    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print(f"Prefix timeout: {args.prefix_timeout}ms")
    print(f"Chunk size: {args.chunk_size}ms")
    print("=" * 70)
    
    # Run comparison
    try:
        direct_stats, delayed_stats = run_comparison(
            model_dir=model_dir,
            positive_files=[str(f) for f in positive_files],
            negative_files=[str(f) for f in negative_files],
            use_int8=args.use_int8,
            prefix_timeout_ms=args.prefix_timeout,
            chunk_size_ms=args.chunk_size,
            threshold=args.threshold,
            num_threads=args.num_threads,
            verbose=args.verbose,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    # Generate comparison report
    config = {
        "model_dir": str(model_dir),
        "positive_count": len(positive_files),
        "negative_count": len(negative_files),
        "prefix_timeout_ms": args.prefix_timeout,
        "chunk_size_ms": args.chunk_size,
        "threshold": args.threshold,
    }
    
    report = generate_comparison_report(direct_stats, delayed_stats, config)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save text report
    report_path = output_dir / f"comparison_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save JSON results
    json_path = output_dir / f"comparison_results_{timestamp}.json"
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "direct_inference": direct_stats.to_dict(),
        "delayed_decision": delayed_stats.to_dict(),
        "comparison": {
            "frr_diff": delayed_stats.frr - direct_stats.frr,
            "far_diff": delayed_stats.far - direct_stats.far,
            "rtf_diff": delayed_stats.rtf_stats.rtf - direct_stats.rtf_stats.rtf,
            "far_improved": delayed_stats.far < direct_stats.far,
            "frr_improved": delayed_stats.frr < direct_stats.frr,
            "rtf_acceptable": delayed_stats.rtf_stats.rtf < 1.0,
        },
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
