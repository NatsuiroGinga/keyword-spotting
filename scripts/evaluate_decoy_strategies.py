#!/usr/bin/env python3
"""
Evaluate and compare Decoy strategies for reducing FAR.

Strategy A: High-threshold Decoy filtering
  - Use decoys with higher boost to absorb false positives
  - Only trigger when full keyword "你好真真" is detected

Strategy B: Delayed Decision + Decoy combination
  - Combine decoy detection with delayed decision logic
  - Wait for timeout to confirm full keyword
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import sherpa_onnx
import soundfile as sf


# Decoy configurations
DECOY_KEYWORDS = [
    {"pinyin": "n ǐ h ǎo zh ēn zh ēn", "text": "你好真真", "is_target": True},
    {"pinyin": "n ǐ h ǎo", "text": "你好", "is_target": False},
    {"pinyin": "n ǐ h ǎo a", "text": "你好啊", "is_target": False},
    {"pinyin": "n ín h ǎo", "text": "您好", "is_target": False},
    {"pinyin": "n ǐ h ǎo m a", "text": "你好吗", "is_target": False},
]


@dataclass
class EvaluationResult:
    """Evaluation result for a strategy."""
    strategy_name: str
    config: Dict[str, Any]
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0
    decoy_blocked: int = 0
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

    @property
    def accuracy(self) -> float:
        total = self.total_positive + self.total_negative
        if total == 0:
            return 0.0
        correct = self.true_positive + self.true_negative
        return (correct / total) * 100

    @property
    def f1_score(self) -> float:
        precision = self.true_positive / (self.true_positive + self.false_positive) if (self.true_positive + self.false_positive) > 0 else 0
        recall = self.true_positive / (self.true_positive + self.false_negative) if (self.true_positive + self.false_negative) > 0 else 0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "config": self.config,
            "total_positive": self.total_positive,
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "total_negative": self.total_negative,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "decoy_blocked": self.decoy_blocked,
            "frr_percent": round(self.frr, 2),
            "far_percent": round(self.far, 2),
            "recall_percent": round(self.recall, 2),
            "specificity_percent": round(self.specificity, 2),
            "accuracy_percent": round(self.accuracy, 2),
            "f1_score": round(self.f1_score, 2),
            "process_time_sec": round(self.process_time, 3),
        }


def create_keywords_file(
    output_path: str,
    target_boost: float,
    target_threshold: float,
    decoy_boost: float,
    decoy_threshold: float,
    keywords: List[Dict] = None,
) -> None:
    """Create keywords file with target and decoy keywords."""
    if keywords is None:
        keywords = DECOY_KEYWORDS

    lines = []
    for kw in keywords:
        if kw["is_target"]:
            boost = target_boost
            threshold = target_threshold
        else:
            boost = decoy_boost
            threshold = decoy_threshold
        line = f"{kw['pinyin']} :{boost:.2f} #{threshold:.2f} @{kw['text']}"
        lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def create_target_only_keywords_file(
    output_path: str,
    boost: float,
    threshold: float,
) -> None:
    """Create keywords file with only target keyword (baseline)."""
    line = f"n ǐ h ǎo zh ēn zh ēn :{boost:.2f} #{threshold:.2f} @你好真真"
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


def test_audio_file(
    spotter: sherpa_onnx.KeywordSpotter,
    audio_path: str,
) -> Tuple[bool, Optional[str]]:
    """Test audio file and return (detected, keyword_text)."""
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

        return detected_keyword is not None, detected_keyword
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False, None


def test_audio_file_chunked(
    spotter: sherpa_onnx.KeywordSpotter,
    audio_path: str,
    chunk_size_ms: int = 100,
    prefix_timeout_ms: int = 600,
    target_text: str = "你好真真",
) -> Tuple[bool, Optional[str], str]:
    """
    Test audio file with chunked processing and delayed decision.
    
    Returns:
        (is_target_detected, detected_keyword, decision_type)
        decision_type: "direct", "delayed_confirm", "delayed_reject", "timeout", "none"
    """
    try:
        samples, sample_rate = sf.read(audio_path, dtype="float32")

        if sample_rate != 16000:
            try:
                import librosa
                samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except ImportError:
                return False, None, "error"

        if len(samples.shape) > 1:
            samples = samples[:, 0]

        chunk_size = int(chunk_size_ms * sample_rate / 1000)
        timeout_chunks = int(prefix_timeout_ms / chunk_size_ms)

        stream = spotter.create_stream()
        
        first_detection = None
        first_detection_chunk = -1
        final_detection = None
        
        num_chunks = (len(samples) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(samples))
            chunk = samples[start:end].tolist()
            
            stream.accept_waveform(sample_rate, chunk)
            
            while spotter.is_ready(stream):
                spotter.decode_stream(stream)
                result = spotter.get_result(stream)
                if result:
                    keyword = result.strip()
                    if first_detection is None:
                        first_detection = keyword
                        first_detection_chunk = i
                    final_detection = keyword

        # Add tail padding
        tail_paddings = [0.0] * int(0.3 * sample_rate)
        stream.accept_waveform(sample_rate, tail_paddings)
        stream.input_finished()

        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
            result = spotter.get_result(stream)
            if result:
                keyword = result.strip()
                if first_detection is None:
                    first_detection = keyword
                    first_detection_chunk = num_chunks
                final_detection = keyword

        # Decision logic
        if final_detection is None:
            return False, None, "none"

        is_target = target_text in final_detection

        if first_detection and target_text in first_detection:
            return True, final_detection, "direct"
        elif first_detection and target_text not in first_detection:
            # First detected a decoy/prefix
            if is_target:
                return True, final_detection, "delayed_confirm"
            else:
                return False, final_detection, "delayed_reject"
        else:
            return is_target, final_detection, "direct"

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False, None, "error"


def evaluate_baseline(
    model_dir: Path,
    positive_files: List[Path],
    negative_files: List[Path],
    boost: float = 1.5,
    threshold: float = 0.3,
    num_threads: int = 4,
    provider: str = "cpu",
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate baseline (no decoy)."""
    print("\n" + "=" * 60)
    print("Evaluating BASELINE (no decoy)")
    print("=" * 60)

    # Find model files
    suffix = ".int8.onnx"
    encoder_files = list(model_dir.glob(f"encoder-*{suffix}"))
    decoder_files = list(model_dir.glob(f"decoder-*{suffix}"))
    joiner_files = list(model_dir.glob(f"joiner-*{suffix}"))

    if not encoder_files or not decoder_files or not joiner_files:
        raise RuntimeError(f"Could not find model files in {model_dir}")

    encoder_path = str(encoder_files[0])
    decoder_path = str(decoder_files[0])
    joiner_path = str(joiner_files[0])
    tokens_path = str(model_dir / "tokens.txt")

    # Create temp keywords file
    temp_keywords = model_dir / "keywords_baseline_temp.txt"
    create_target_only_keywords_file(str(temp_keywords), boost, threshold)

    spotter = create_keyword_spotter(
        encoder_path, decoder_path, joiner_path, tokens_path,
        str(temp_keywords), num_threads, provider
    )

    result = EvaluationResult(
        strategy_name="Baseline (no decoy)",
        config={"boost": boost, "threshold": threshold},
        total_positive=len(positive_files),
        total_negative=len(negative_files),
    )

    start_time = time.time()

    # Test positive samples
    print(f"\nTesting {len(positive_files)} positive samples...")
    for audio_path in positive_files:
        detected, keyword = test_audio_file(spotter, str(audio_path))
        if detected and "你好真真" in (keyword or ""):
            result.true_positive += 1
        else:
            result.false_negative += 1
            if verbose:
                print(f"  FN: {audio_path.name} -> {keyword}")

    # Test negative samples
    print(f"Testing {len(negative_files)} negative samples...")
    for audio_path in negative_files:
        detected, keyword = test_audio_file(spotter, str(audio_path))
        if detected and "你好真真" in (keyword or ""):
            result.false_positive += 1
            if verbose:
                print(f"  FP: {audio_path.name} -> {keyword}")
        else:
            result.true_negative += 1

    result.process_time = time.time() - start_time

    # Cleanup
    del spotter
    if temp_keywords.exists():
        temp_keywords.unlink()

    print(f"\nBaseline Results:")
    print(f"  FRR: {result.frr:.2f}% ({result.false_negative}/{result.total_positive})")
    print(f"  FAR: {result.far:.2f}% ({result.false_positive}/{result.total_negative})")
    print(f"  Accuracy: {result.accuracy:.2f}%")

    return result


def evaluate_strategy_a(
    model_dir: Path,
    positive_files: List[Path],
    negative_files: List[Path],
    target_boost: float = 1.5,
    target_threshold: float = 0.3,
    decoy_boost: float = 3.0,
    decoy_threshold: float = 0.15,
    num_threads: int = 4,
    provider: str = "cpu",
    verbose: bool = False,
) -> EvaluationResult:
    """
    Strategy A: High-threshold Decoy filtering.
    
    Decoys have higher boost to absorb false positives.
    Only count as positive if "你好真真" is detected (not decoy).
    """
    print("\n" + "=" * 60)
    print("Evaluating STRATEGY A: High-threshold Decoy Filtering")
    print("=" * 60)
    print(f"  Target: boost={target_boost}, threshold={target_threshold}")
    print(f"  Decoy:  boost={decoy_boost}, threshold={decoy_threshold}")

    # Find model files
    suffix = ".int8.onnx"
    encoder_files = list(model_dir.glob(f"encoder-*{suffix}"))
    decoder_files = list(model_dir.glob(f"decoder-*{suffix}"))
    joiner_files = list(model_dir.glob(f"joiner-*{suffix}"))

    if not encoder_files or not decoder_files or not joiner_files:
        raise RuntimeError(f"Could not find model files in {model_dir}")

    encoder_path = str(encoder_files[0])
    decoder_path = str(decoder_files[0])
    joiner_path = str(joiner_files[0])
    tokens_path = str(model_dir / "tokens.txt")

    # Create temp keywords file with decoys
    temp_keywords = model_dir / "keywords_strategy_a_temp.txt"
    create_keywords_file(
        str(temp_keywords),
        target_boost, target_threshold,
        decoy_boost, decoy_threshold
    )

    spotter = create_keyword_spotter(
        encoder_path, decoder_path, joiner_path, tokens_path,
        str(temp_keywords), num_threads, provider
    )

    result = EvaluationResult(
        strategy_name="Strategy A: High-threshold Decoy",
        config={
            "target_boost": target_boost,
            "target_threshold": target_threshold,
            "decoy_boost": decoy_boost,
            "decoy_threshold": decoy_threshold,
        },
        total_positive=len(positive_files),
        total_negative=len(negative_files),
    )

    start_time = time.time()

    # Test positive samples
    print(f"\nTesting {len(positive_files)} positive samples...")
    for audio_path in positive_files:
        detected, keyword = test_audio_file(spotter, str(audio_path))
        if detected and keyword:
            if "你好真真" in keyword:
                result.true_positive += 1
            else:
                # Detected decoy instead of target - this is FN
                result.false_negative += 1
                if verbose:
                    print(f"  FN (decoy): {audio_path.name} -> {keyword}")
        else:
            result.false_negative += 1
            if verbose:
                print(f"  FN (none): {audio_path.name}")

    # Test negative samples
    print(f"Testing {len(negative_files)} negative samples...")
    for audio_path in negative_files:
        detected, keyword = test_audio_file(spotter, str(audio_path))
        if detected and keyword:
            if "你好真真" in keyword:
                result.false_positive += 1
                if verbose:
                    print(f"  FP: {audio_path.name} -> {keyword}")
            else:
                # Detected decoy - this is good (blocked by decoy)
                result.decoy_blocked += 1
                result.true_negative += 1
        else:
            result.true_negative += 1

    result.process_time = time.time() - start_time

    # Cleanup
    del spotter
    if temp_keywords.exists():
        temp_keywords.unlink()

    print(f"\nStrategy A Results:")
    print(f"  FRR: {result.frr:.2f}% ({result.false_negative}/{result.total_positive})")
    print(f"  FAR: {result.far:.2f}% ({result.false_positive}/{result.total_negative})")
    print(f"  Decoy Blocked: {result.decoy_blocked}")
    print(f"  Accuracy: {result.accuracy:.2f}%")

    return result


def evaluate_strategy_b(
    model_dir: Path,
    positive_files: List[Path],
    negative_files: List[Path],
    target_boost: float = 1.5,
    target_threshold: float = 0.3,
    decoy_boost: float = 3.0,
    decoy_threshold: float = 0.15,
    chunk_size_ms: int = 100,
    prefix_timeout_ms: int = 600,
    num_threads: int = 4,
    provider: str = "cpu",
    verbose: bool = False,
) -> EvaluationResult:
    """
    Strategy B: Delayed Decision + Decoy combination.
    
    Process audio in chunks, wait for timeout to confirm full keyword.
    """
    print("\n" + "=" * 60)
    print("Evaluating STRATEGY B: Delayed Decision + Decoy")
    print("=" * 60)
    print(f"  Target: boost={target_boost}, threshold={target_threshold}")
    print(f"  Decoy:  boost={decoy_boost}, threshold={decoy_threshold}")
    print(f"  Chunk size: {chunk_size_ms}ms, Timeout: {prefix_timeout_ms}ms")

    # Find model files
    suffix = ".int8.onnx"
    encoder_files = list(model_dir.glob(f"encoder-*{suffix}"))
    decoder_files = list(model_dir.glob(f"decoder-*{suffix}"))
    joiner_files = list(model_dir.glob(f"joiner-*{suffix}"))

    if not encoder_files or not decoder_files or not joiner_files:
        raise RuntimeError(f"Could not find model files in {model_dir}")

    encoder_path = str(encoder_files[0])
    decoder_path = str(decoder_files[0])
    joiner_path = str(joiner_files[0])
    tokens_path = str(model_dir / "tokens.txt")

    # Create temp keywords file with decoys
    temp_keywords = model_dir / "keywords_strategy_b_temp.txt"
    create_keywords_file(
        str(temp_keywords),
        target_boost, target_threshold,
        decoy_boost, decoy_threshold
    )

    spotter = create_keyword_spotter(
        encoder_path, decoder_path, joiner_path, tokens_path,
        str(temp_keywords), num_threads, provider
    )

    result = EvaluationResult(
        strategy_name="Strategy B: Delayed Decision + Decoy",
        config={
            "target_boost": target_boost,
            "target_threshold": target_threshold,
            "decoy_boost": decoy_boost,
            "decoy_threshold": decoy_threshold,
            "chunk_size_ms": chunk_size_ms,
            "prefix_timeout_ms": prefix_timeout_ms,
        },
        total_positive=len(positive_files),
        total_negative=len(negative_files),
    )

    decision_stats = {"direct": 0, "delayed_confirm": 0, "delayed_reject": 0, "none": 0}

    start_time = time.time()

    # Test positive samples
    print(f"\nTesting {len(positive_files)} positive samples...")
    for audio_path in positive_files:
        is_target, keyword, decision_type = test_audio_file_chunked(
            spotter, str(audio_path), chunk_size_ms, prefix_timeout_ms
        )
        decision_stats[decision_type] = decision_stats.get(decision_type, 0) + 1
        
        if is_target:
            result.true_positive += 1
        else:
            result.false_negative += 1
            if verbose:
                print(f"  FN ({decision_type}): {audio_path.name} -> {keyword}")

    # Test negative samples
    print(f"Testing {len(negative_files)} negative samples...")
    for audio_path in negative_files:
        is_target, keyword, decision_type = test_audio_file_chunked(
            spotter, str(audio_path), chunk_size_ms, prefix_timeout_ms
        )
        
        if is_target:
            result.false_positive += 1
            if verbose:
                print(f"  FP ({decision_type}): {audio_path.name} -> {keyword}")
        else:
            result.true_negative += 1
            if keyword and "你好真真" not in (keyword or ""):
                result.decoy_blocked += 1

    result.process_time = time.time() - start_time

    # Cleanup
    del spotter
    if temp_keywords.exists():
        temp_keywords.unlink()

    print(f"\nStrategy B Results:")
    print(f"  FRR: {result.frr:.2f}% ({result.false_negative}/{result.total_positive})")
    print(f"  FAR: {result.far:.2f}% ({result.false_positive}/{result.total_negative})")
    print(f"  Decoy Blocked: {result.decoy_blocked}")
    print(f"  Accuracy: {result.accuracy:.2f}%")
    print(f"  Decision Stats: {decision_stats}")

    return result


def generate_comparison_report(
    results: List[EvaluationResult],
    output_dir: Path,
) -> None:
    """Generate comparison report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Text report
    report_lines = [
        "=" * 70,
        "Decoy Strategy Comparison Report",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Summary table
    report_lines.extend([
        "Summary:",
        "-" * 70,
        f"{'Strategy':<40} | {'FRR%':>6} | {'FAR%':>6} | {'Acc%':>6} | {'Blocked':>7}",
        "-" * 70,
    ])

    for r in results:
        report_lines.append(
            f"{r.strategy_name:<40} | {r.frr:>6.2f} | {r.far:>6.2f} | "
            f"{r.accuracy:>6.2f} | {r.decoy_blocked:>7}"
        )

    report_lines.extend([
        "-" * 70,
        "",
        "Detailed Results:",
        "=" * 70,
    ])

    for r in results:
        report_lines.extend([
            "",
            f"Strategy: {r.strategy_name}",
            f"Config: {r.config}",
            f"  Positive samples: {r.total_positive}",
            f"    True Positive: {r.true_positive}",
            f"    False Negative: {r.false_negative}",
            f"    FRR: {r.frr:.2f}%",
            f"  Negative samples: {r.total_negative}",
            f"    True Negative: {r.true_negative}",
            f"    False Positive: {r.false_positive}",
            f"    Decoy Blocked: {r.decoy_blocked}",
            f"    FAR: {r.far:.2f}%",
            f"  Overall:",
            f"    Accuracy: {r.accuracy:.2f}%",
            f"    F1 Score: {r.f1_score:.2f}",
            f"    Process Time: {r.process_time:.2f}s",
        ])

    report_lines.extend([
        "",
        "=" * 70,
        "Analysis:",
        "=" * 70,
    ])

    # Find best strategy
    baseline = results[0]
    best_far = min(results, key=lambda r: r.far)
    best_acc = max(results, key=lambda r: r.accuracy)

    report_lines.extend([
        f"Baseline FAR: {baseline.far:.2f}%",
        f"Best FAR: {best_far.strategy_name} with {best_far.far:.2f}% (improvement: {baseline.far - best_far.far:.2f}%)",
        f"Best Accuracy: {best_acc.strategy_name} with {best_acc.accuracy:.2f}%",
    ])

    # Check if any strategy meets targets
    target_frr = 1.39
    target_far = 7.46
    meeting_targets = [r for r in results if r.frr <= target_frr and r.far <= target_far]
    
    if meeting_targets:
        report_lines.append(f"\nStrategies meeting targets (FRR≤{target_frr}%, FAR≤{target_far}%):")
        for r in meeting_targets:
            report_lines.append(f"  - {r.strategy_name}: FRR={r.frr:.2f}%, FAR={r.far:.2f}%")
    else:
        report_lines.append(f"\nNo strategy meets targets (FRR≤{target_frr}%, FAR≤{target_far}%)")

    report_lines.append("=" * 70)

    # Save text report
    report_path = output_dir / f"decoy_strategy_comparison_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to: {report_path}")

    # Save JSON
    json_path = output_dir / f"decoy_strategy_comparison_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
        }, f, ensure_ascii=False, indent=2)
    print(f"JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Decoy strategies for KWS")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/workspace/llm/keyword-spotting/exp/kws_finetune_v3",
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
        "--target-boost",
        type=float,
        default=1.5,
        help="Boost for target keyword",
    )
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=0.3,
        help="Threshold for target keyword",
    )
    parser.add_argument(
        "--decoy-boost",
        type=float,
        default=3.0,
        help="Boost for decoy keywords",
    )
    parser.add_argument(
        "--decoy-threshold",
        type=float,
        default=0.15,
        help="Threshold for decoy keywords",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size in ms for Strategy B",
    )
    parser.add_argument(
        "--prefix-timeout",
        type=int,
        default=600,
        help="Prefix timeout in ms for Strategy B",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = model_dir / "decoy_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_files = sorted(Path(args.positive_dir).glob("*.wav"))
    negative_files = sorted(Path(args.negative_dir).glob("*.wav"))

    print("=" * 70)
    print("Decoy Strategy Comparison")
    print("=" * 70)
    print(f"Model directory: {model_dir}")
    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    results = []

    # 1. Baseline (no decoy)
    baseline_result = evaluate_baseline(
        model_dir, positive_files, negative_files,
        boost=args.target_boost, threshold=args.target_threshold,
        num_threads=args.num_threads, provider=args.provider,
        verbose=args.verbose,
    )
    results.append(baseline_result)

    # 2. Strategy A: High-threshold Decoy
    strategy_a_result = evaluate_strategy_a(
        model_dir, positive_files, negative_files,
        target_boost=args.target_boost, target_threshold=args.target_threshold,
        decoy_boost=args.decoy_boost, decoy_threshold=args.decoy_threshold,
        num_threads=args.num_threads, provider=args.provider,
        verbose=args.verbose,
    )
    results.append(strategy_a_result)

    # 3. Strategy B: Delayed Decision + Decoy
    strategy_b_result = evaluate_strategy_b(
        model_dir, positive_files, negative_files,
        target_boost=args.target_boost, target_threshold=args.target_threshold,
        decoy_boost=args.decoy_boost, decoy_threshold=args.decoy_threshold,
        chunk_size_ms=args.chunk_size, prefix_timeout_ms=args.prefix_timeout,
        num_threads=args.num_threads, provider=args.provider,
        verbose=args.verbose,
    )
    results.append(strategy_b_result)

    # Generate comparison report
    generate_comparison_report(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<40} | {'FRR%':>6} | {'FAR%':>6} | {'Acc%':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r.strategy_name:<40} | {r.frr:>6.2f} | {r.far:>6.2f} | {r.accuracy:>6.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
