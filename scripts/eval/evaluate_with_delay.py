#!/usr/bin/env python3
"""
Evaluate KWS model with delayed decision inference logic.

This script integrates the delayed decision state machine with comprehensive
evaluation metrics including FRR, FAR, and RTF.

Delayed Decision Logic:
- State machine: IDLE -> PREFIX_DETECTED -> WAITING_SUFFIX -> TRIGGERED/REJECTED
- Two-phase detection: "你好" -> wait 600ms for "真真"
- Reduces false positives from prefix-only triggers
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import sherpa_onnx
import soundfile as sf

from rtf_utils import RTFStats, RTFMeasurement, RTFTimer, get_audio_duration


class KWSState(Enum):
    """KWS state machine states."""
    IDLE = "idle"
    PREFIX_DETECTED = "prefix_detected"
    WAITING_SUFFIX = "waiting_suffix"
    TRIGGERED = "triggered"
    REJECTED = "rejected"


@dataclass
class DelayedDecisionConfig:
    """Configuration for delayed decision inference."""
    prefix_timeout_ms: int = 600
    detection_window_ms: int = 800
    chunk_size_ms: int = 100
    sample_rate: int = 16000
    
    # Keywords
    full_keyword: str = "你好真真"
    prefix_keyword: str = "你好"
    suffix_keyword: str = "真真"


@dataclass
class EvaluationResult:
    """Evaluation result for a single audio file."""
    audio_path: str
    expected_positive: bool
    detected: bool
    keyword_text: str = ""
    process_time_sec: float = 0.0
    audio_duration_sec: float = 0.0
    state_transitions: List[str] = field(default_factory=list)
    
    @property
    def is_correct(self) -> bool:
        """Check if detection is correct."""
        if self.expected_positive:
            return self.detected
        else:
            return not self.detected
    
    @property
    def rtf(self) -> float:
        """Calculate RTF."""
        if self.audio_duration_sec <= 0:
            return float('inf')
        return self.process_time_sec / self.audio_duration_sec


@dataclass
class EvaluationStats:
    """Aggregated evaluation statistics."""
    # Detection metrics
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0
    total_positive: int = 0
    total_negative: int = 0
    
    # RTF metrics
    rtf_stats: RTFStats = field(default_factory=RTFStats)
    
    # State transition stats
    prefix_detections: int = 0
    suffix_confirmations: int = 0
    timeouts: int = 0
    direct_triggers: int = 0
    
    # Results
    results: List[EvaluationResult] = field(default_factory=list)
    
    @property
    def frr(self) -> float:
        """False Rejection Rate (%)."""
        if self.total_positive == 0:
            return 0.0
        return (self.false_negative / self.total_positive) * 100
    
    @property
    def far(self) -> float:
        """False Accept Rate (%)."""
        if self.total_negative == 0:
            return 0.0
        return (self.false_positive / self.total_negative) * 100
    
    @property
    def recall(self) -> float:
        """Recall / True Positive Rate (%)."""
        return 100 - self.frr
    
    @property
    def specificity(self) -> float:
        """Specificity / True Negative Rate (%)."""
        return 100 - self.far
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy (%)."""
        total = self.total_positive + self.total_negative
        if total == 0:
            return 0.0
        correct = self.true_positive + self.true_negative
        return (correct / total) * 100
    
    @property
    def precision(self) -> float:
        """Precision (%)."""
        detected = self.true_positive + self.false_positive
        if detected == 0:
            return 0.0
        return (self.true_positive / detected) * 100
    
    @property
    def f1_score(self) -> float:
        """F1 Score."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    def add_result(self, result: EvaluationResult):
        """Add an evaluation result."""
        self.results.append(result)
        
        if result.expected_positive:
            self.total_positive += 1
            if result.detected:
                self.true_positive += 1
            else:
                self.false_negative += 1
        else:
            self.total_negative += 1
            if result.detected:
                self.false_positive += 1
            else:
                self.true_negative += 1
        
        # Add RTF measurement
        if result.audio_duration_sec > 0:
            measurement = RTFMeasurement(
                audio_path=result.audio_path,
                audio_duration_sec=result.audio_duration_sec,
                process_time_sec=result.process_time_sec,
            )
            self.rtf_stats.add_measurement(measurement)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detection_metrics": {
                "total_positive": self.total_positive,
                "total_negative": self.total_negative,
                "true_positive": self.true_positive,
                "false_negative": self.false_negative,
                "false_positive": self.false_positive,
                "true_negative": self.true_negative,
                "frr_percent": round(self.frr, 2),
                "far_percent": round(self.far, 2),
                "recall_percent": round(self.recall, 2),
                "specificity_percent": round(self.specificity, 2),
                "accuracy_percent": round(self.accuracy, 2),
                "precision_percent": round(self.precision, 2),
                "f1_score": round(self.f1_score, 2),
            },
            "state_machine_stats": {
                "prefix_detections": self.prefix_detections,
                "suffix_confirmations": self.suffix_confirmations,
                "timeouts": self.timeouts,
                "direct_triggers": self.direct_triggers,
            },
            "rtf_metrics": self.rtf_stats.to_dict(),
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "Evaluation Results (Delayed Decision Mode)",
            "=" * 60,
            "",
            "Detection Metrics:",
            f"  Positive samples: {self.true_positive}/{self.total_positive} detected",
            f"  Negative samples: {self.true_negative}/{self.total_negative} rejected",
            f"  FRR: {self.frr:.2f}%",
            f"  FAR: {self.far:.2f}%",
            f"  Recall: {self.recall:.2f}%",
            f"  Specificity: {self.specificity:.2f}%",
            f"  Accuracy: {self.accuracy:.2f}%",
            f"  Precision: {self.precision:.2f}%",
            f"  F1 Score: {self.f1_score:.2f}",
            "",
            "State Machine Statistics:",
            f"  Prefix detections: {self.prefix_detections}",
            f"  Suffix confirmations: {self.suffix_confirmations}",
            f"  Timeouts (rejected): {self.timeouts}",
            f"  Direct triggers: {self.direct_triggers}",
            "",
            self.rtf_stats.summary(),
            "=" * 60,
        ]
        return "\n".join(lines)


class DelayedDecisionEvaluator:
    """Evaluator with delayed decision inference logic."""
    
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        tokens_path: str,
        keywords_path: str,
        config: DelayedDecisionConfig = None,
        num_threads: int = 4,
    ):
        self.config = config or DelayedDecisionConfig()
        
        # Create keyword spotter
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            keywords_file=keywords_path,
            num_threads=num_threads,
            provider="cpu",
        )
        
        # State machine
        self.state = KWSState.IDLE
        self.state_transitions: List[str] = []
        self.prefix_start_time: Optional[float] = None
        
        # Statistics
        self.stats = EvaluationStats()
    
    def _reset_state(self):
        """Reset state machine."""
        self.state = KWSState.IDLE
        self.state_transitions = []
        self.prefix_start_time = None
    
    def _transition_to(self, new_state: KWSState):
        """Transition to a new state."""
        self.state_transitions.append(f"{self.state.value} -> {new_state.value}")
        self.state = new_state
    
    def _check_prefix_timeout(self, current_time: float) -> bool:
        """Check if prefix has timed out."""
        if self.prefix_start_time is None:
            return False
        elapsed_ms = (current_time - self.prefix_start_time) * 1000
        return elapsed_ms > self.config.prefix_timeout_ms
    
    def _process_detection(self, keyword: str, current_time: float) -> bool:
        """
        Process a keyword detection through the state machine.
        
        Returns:
            True if final trigger, False otherwise
        """
        if self.state == KWSState.IDLE:
            if keyword == self.config.full_keyword:
                # Direct full keyword detection
                self._transition_to(KWSState.TRIGGERED)
                self.stats.direct_triggers += 1
                return True
            elif self.config.prefix_keyword in keyword:
                # Prefix detected
                self._transition_to(KWSState.PREFIX_DETECTED)
                self._transition_to(KWSState.WAITING_SUFFIX)
                self.prefix_start_time = current_time
                self.stats.prefix_detections += 1
                return False
        
        elif self.state == KWSState.WAITING_SUFFIX:
            if self._check_prefix_timeout(current_time):
                # Timeout - reject
                self._transition_to(KWSState.REJECTED)
                self.stats.timeouts += 1
                self._reset_state()
                return False
            
            if keyword == self.config.full_keyword:
                # Full keyword confirmed
                self._transition_to(KWSState.TRIGGERED)
                self.stats.suffix_confirmations += 1
                return True
            elif self.config.suffix_keyword in keyword:
                # Suffix detected
                self._transition_to(KWSState.TRIGGERED)
                self.stats.suffix_confirmations += 1
                return True
        
        return False
    
    def evaluate_audio_file(
        self,
        audio_path: str,
        expected_positive: bool,
    ) -> EvaluationResult:
        """
        Evaluate a single audio file with delayed decision logic.
        
        Args:
            audio_path: Path to audio file
            expected_positive: Whether this is a positive sample
        
        Returns:
            EvaluationResult
        """
        self._reset_state()
        
        # Read audio
        try:
            samples, sample_rate = sf.read(audio_path, dtype="float32")
        except Exception as e:
            return EvaluationResult(
                audio_path=audio_path,
                expected_positive=expected_positive,
                detected=False,
                keyword_text=f"Error: {e}",
            )
        
        # Resample if needed
        if sample_rate != self.config.sample_rate:
            try:
                import librosa
                samples = librosa.resample(
                    samples, 
                    orig_sr=sample_rate, 
                    target_sr=self.config.sample_rate
                )
                sample_rate = self.config.sample_rate
            except ImportError:
                pass
        
        # Convert to mono
        if len(samples.shape) > 1:
            samples = samples[:, 0]
        
        audio_duration = len(samples) / sample_rate
        
        # Process in chunks (simulating real-time)
        chunk_samples = int(self.config.chunk_size_ms * sample_rate / 1000)
        detected = False
        keyword_text = ""
        
        start_time = time.perf_counter()
        stream = self.spotter.create_stream()
        
        for i in range(0, len(samples), chunk_samples):
            chunk = samples[i:i + chunk_samples]
            current_time = time.perf_counter()
            
            # Check for timeout in waiting state
            if self.state == KWSState.WAITING_SUFFIX:
                if self._check_prefix_timeout(current_time):
                    self._transition_to(KWSState.REJECTED)
                    self.stats.timeouts += 1
                    self._reset_state()
            
            # Feed audio to stream
            stream.accept_waveform(sample_rate, chunk)
            
            # Decode
            while self.spotter.is_ready(stream):
                self.spotter.decode_stream(stream)
                result = self.spotter.get_result(stream)
                if result:
                    # Process through state machine
                    triggered = self._process_detection(result, current_time)
                    if triggered:
                        detected = True
                        keyword_text = result
                        break
            
            if detected:
                break
        
        # Final processing with tail padding
        if not detected:
            tail_paddings = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
            stream.accept_waveform(sample_rate, tail_paddings)
            stream.input_finished()
            
            while self.spotter.is_ready(stream):
                self.spotter.decode_stream(stream)
                result = self.spotter.get_result(stream)
                if result:
                    current_time = time.perf_counter()
                    triggered = self._process_detection(result, current_time)
                    if triggered:
                        detected = True
                        keyword_text = result
                        break
        
        process_time = time.perf_counter() - start_time
        
        result = EvaluationResult(
            audio_path=audio_path,
            expected_positive=expected_positive,
            detected=detected,
            keyword_text=keyword_text,
            process_time_sec=process_time,
            audio_duration_sec=audio_duration,
            state_transitions=self.state_transitions.copy(),
        )
        
        return result
    
    def evaluate_dataset(
        self,
        positive_files: List[str],
        negative_files: List[str],
        verbose: bool = False,
    ) -> EvaluationStats:
        """
        Evaluate a dataset.
        
        Args:
            positive_files: List of positive sample paths
            negative_files: List of negative sample paths
            verbose: Print progress
        
        Returns:
            EvaluationStats
        """
        self.stats = EvaluationStats()
        
        total = len(positive_files) + len(negative_files)
        processed = 0
        
        # Evaluate positive samples
        if verbose:
            print(f"Evaluating {len(positive_files)} positive samples...")
        
        for audio_path in positive_files:
            result = self.evaluate_audio_file(audio_path, expected_positive=True)
            self.stats.add_result(result)
            processed += 1
            if verbose and processed % 50 == 0:
                print(f"  Progress: {processed}/{total}")
        
        # Evaluate negative samples
        if verbose:
            print(f"Evaluating {len(negative_files)} negative samples...")
        
        for audio_path in negative_files:
            result = self.evaluate_audio_file(audio_path, expected_positive=False)
            self.stats.add_result(result)
            processed += 1
            if verbose and processed % 50 == 0:
                print(f"  Progress: {processed}/{total}")
        
        return self.stats


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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KWS model with delayed decision inference"
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
        help="Prefix timeout in milliseconds",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size in milliseconds for streaming simulation",
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
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "delayed_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model files
    try:
        model_files = find_model_files(model_dir, args.use_int8)
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    # Get test files
    positive_files = sorted(Path(args.positive_dir).glob("*.wav"))
    negative_files = sorted(Path(args.negative_dir).glob("*.wav"))
    
    print("=" * 60)
    print("KWS Evaluation with Delayed Decision")
    print("=" * 60)
    print(f"Model directory: {model_dir}")
    print(f"Encoder: {model_files['encoder']}")
    print(f"Positive samples: {len(positive_files)}")
    print(f"Negative samples: {len(negative_files)}")
    print(f"Prefix timeout: {args.prefix_timeout}ms")
    print(f"Chunk size: {args.chunk_size}ms")
    print("=" * 60)
    
    # Create config
    config = DelayedDecisionConfig(
        prefix_timeout_ms=args.prefix_timeout,
        chunk_size_ms=args.chunk_size,
    )
    
    # Create evaluator
    evaluator = DelayedDecisionEvaluator(
        encoder_path=model_files["encoder"],
        decoder_path=model_files["decoder"],
        joiner_path=model_files["joiner"],
        tokens_path=model_files["tokens"],
        keywords_path=model_files["keywords"],
        config=config,
        num_threads=args.num_threads,
    )
    
    # Run evaluation
    stats = evaluator.evaluate_dataset(
        positive_files=[str(f) for f in positive_files],
        negative_files=[str(f) for f in negative_files],
        verbose=args.verbose,
    )
    
    # Print results
    print("\n" + stats.summary())
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON results
    json_path = output_dir / f"delayed_eval_{timestamp}.json"
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_dir": str(model_dir),
            "prefix_timeout_ms": args.prefix_timeout,
            "chunk_size_ms": args.chunk_size,
            "positive_samples": len(positive_files),
            "negative_samples": len(negative_files),
        },
        "results": stats.to_dict(),
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    # Error analysis
    if args.verbose:
        print("\n" + "=" * 60)
        print("Error Analysis")
        print("=" * 60)
        
        false_negatives = [r for r in stats.results if r.expected_positive and not r.detected]
        false_positives = [r for r in stats.results if not r.expected_positive and r.detected]
        
        if false_negatives:
            print(f"\nFalse Negatives ({len(false_negatives)}):")
            for r in false_negatives[:10]:
                print(f"  - {Path(r.audio_path).name}")
                if r.state_transitions:
                    print(f"    Transitions: {' -> '.join(r.state_transitions)}")
        
        if false_positives:
            print(f"\nFalse Positives ({len(false_positives)}):")
            for r in false_positives[:10]:
                print(f"  - {Path(r.audio_path).name}: '{r.keyword_text}'")
                if r.state_transitions:
                    print(f"    Transitions: {' -> '.join(r.state_transitions)}")


if __name__ == "__main__":
    main()
