#!/usr/bin/env python3
"""
Evaluate sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 pretrained model
on real human voice test dataset.

Calculates FAR, FRR, and RTF metrics.
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import sherpa_onnx
import soundfile as sf


@dataclass
class EvaluationResult:
    """Evaluation result for a single audio file."""
    audio_path: str
    expected_positive: bool
    detected: bool
    keyword_text: str = ""
    process_time_sec: float = 0.0
    audio_duration_sec: float = 0.0
    
    @property
    def is_correct(self) -> bool:
        if self.expected_positive:
            return self.detected
        else:
            return not self.detected
    
    @property
    def rtf(self) -> float:
        if self.audio_duration_sec <= 0:
            return float('inf')
        return self.process_time_sec / self.audio_duration_sec


@dataclass
class EvaluationStats:
    """Aggregated evaluation statistics."""
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0
    total_positive: int = 0
    total_negative: int = 0
    
    total_process_time: float = 0.0
    total_audio_duration: float = 0.0
    
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
    
    @property
    def rtf(self) -> float:
        """Real-Time Factor."""
        if self.total_audio_duration <= 0:
            return float('inf')
        return self.total_process_time / self.total_audio_duration
    
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
        
        self.total_process_time += result.process_time_sec
        self.total_audio_duration += result.audio_duration_sec
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_counts": {
                "total_positive": self.total_positive,
                "total_negative": self.total_negative,
                "true_positive": self.true_positive,
                "false_negative": self.false_negative,
                "false_positive": self.false_positive,
                "true_negative": self.true_negative,
            },
            "metrics": {
                "frr_percent": round(self.frr, 2),
                "far_percent": round(self.far, 2),
                "recall_percent": round(self.recall, 2),
                "specificity_percent": round(self.specificity, 2),
                "accuracy_percent": round(self.accuracy, 2),
                "precision_percent": round(self.precision, 2),
                "f1_score": round(self.f1_score, 2),
                "rtf": round(self.rtf, 4),
            },
            "timing": {
                "total_process_time_sec": round(self.total_process_time, 2),
                "total_audio_duration_sec": round(self.total_audio_duration, 2),
            },
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 70,
            "Pretrained Model Evaluation Results",
            "=" * 70,
            "",
            "Sample Statistics:",
            f"  Positive samples: {self.true_positive}/{self.total_positive} detected (TP/Total)",
            f"  Negative samples: {self.true_negative}/{self.total_negative} rejected (TN/Total)",
            "",
            "Detection Metrics:",
            f"  FRR (False Rejection Rate): {self.frr:.2f}%",
            f"  FAR (False Accept Rate):    {self.far:.2f}%",
            f"  Recall:                     {self.recall:.2f}%",
            f"  Specificity:                {self.specificity:.2f}%",
            f"  Accuracy:                   {self.accuracy:.2f}%",
            f"  Precision:                  {self.precision:.2f}%",
            f"  F1 Score:                   {self.f1_score:.2f}",
            "",
            "Performance Metrics:",
            f"  RTF (Real-Time Factor):     {self.rtf:.4f}",
            f"  Total process time:         {self.total_process_time:.2f}s",
            f"  Total audio duration:       {self.total_audio_duration:.2f}s",
            "=" * 70,
        ]
        return "\n".join(lines)


class PretrainedModelEvaluator:
    """Evaluator for sherpa-onnx pretrained KWS model."""
    
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        tokens_path: str,
        keywords_path: str,
        keywords_threshold: float = 0.0,
        num_threads: int = 4,
    ):
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            keywords_file=keywords_path,
            num_threads=num_threads,
            keywords_threshold=keywords_threshold,
            provider="cpu",
        )
        self.sample_rate = 16000
        self.stats = EvaluationStats()
    
    def evaluate_audio_file(
        self,
        audio_path: str,
        expected_positive: bool,
    ) -> EvaluationResult:
        """Evaluate a single audio file."""
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
        if sample_rate != self.sample_rate:
            try:
                import librosa
                samples = librosa.resample(
                    samples, 
                    orig_sr=sample_rate, 
                    target_sr=self.sample_rate
                )
                sample_rate = self.sample_rate
            except ImportError:
                pass
        
        # Convert to mono
        if len(samples.shape) > 1:
            samples = samples[:, 0]
        
        audio_duration = len(samples) / sample_rate
        
        # Process audio
        start_time = time.perf_counter()
        
        stream = self.spotter.create_stream()
        stream.accept_waveform(sample_rate, samples)
        
        # Add tail padding
        tail_paddings = [0.0] * int(0.3 * sample_rate)
        stream.accept_waveform(sample_rate, tail_paddings)
        stream.input_finished()
        
        detected = False
        keyword_text = ""
        
        while self.spotter.is_ready(stream):
            self.spotter.decode_stream(stream)
            result = self.spotter.get_result(stream)
            if result:
                detected = True
                keyword_text = result
                break
        
        process_time = time.perf_counter() - start_time
        
        return EvaluationResult(
            audio_path=audio_path,
            expected_positive=expected_positive,
            detected=detected,
            keyword_text=keyword_text,
            process_time_sec=process_time,
            audio_duration_sec=audio_duration,
        )
    
    def evaluate_dataset(
        self,
        audio_files: List[str],
        keyword_pattern: str = "你好真真",
        verbose: bool = False,
    ) -> EvaluationStats:
        """Evaluate a dataset of audio files."""
        self.stats = EvaluationStats()
        
        total = len(audio_files)
        
        for i, audio_path in enumerate(audio_files):
            # Determine if positive sample by checking filename
            filename = Path(audio_path).name
            expected_positive = keyword_pattern in filename
            
            result = self.evaluate_audio_file(audio_path, expected_positive)
            self.stats.add_result(result)
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{total}")
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained KWS model on real human voice test data"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/workspace/llm/audio-classification/models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
        help="Directory containing pretrained ONNX model files",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="/data/workspace/llm/keyword-spotting/data/all",
        help="Directory containing test audio files",
    )
    parser.add_argument(
        "--keywords-file",
        type=str,
        default=None,
        help="Keywords file (default: model_dir/keywords_nihao_zhenzhen.txt)",
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
        default="/data/workspace/llm/keyword-spotting/log/evaluation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find INT8 model files
    encoder_path = str(model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx")
    decoder_path = str(model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx")
    joiner_path = str(model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx")
    tokens_path = str(model_dir / "tokens.txt")
    
    # Use keywords file for "你好真真"
    if args.keywords_file:
        keywords_path = args.keywords_file
    else:
        keywords_path = str(model_dir / "keywords_nihao_zhenzhen.txt")
    
    # Get test files
    audio_files = sorted(test_dir.glob("*.wav"))
    
    print("=" * 70)
    print("Pretrained KWS Model Evaluation")
    print("Model: sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 (INT8)")
    print("=" * 70)
    print(f"Model directory:  {model_dir}")
    print(f"Test directory:   {test_dir}")
    print(f"Keywords file:    {keywords_path}")
    print(f"Total test files: {len(audio_files)}")
    print(f"Threshold:        {args.threshold}")
    print("=" * 70)
    
    # Verify files exist
    for path in [encoder_path, decoder_path, joiner_path, tokens_path, keywords_path]:
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            return
    
    # Create evaluator
    evaluator = PretrainedModelEvaluator(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        joiner_path=joiner_path,
        tokens_path=tokens_path,
        keywords_path=keywords_path,
        keywords_threshold=args.threshold,
        num_threads=args.num_threads,
    )
    
    # Run evaluation
    print("\nEvaluating...")
    stats = evaluator.evaluate_dataset(
        audio_files=[str(f) for f in audio_files],
        keyword_pattern="你好真真",
        verbose=args.verbose,
    )
    
    # Print results
    print("\n" + stats.summary())
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"pretrained_eval_{timestamp}.json"
    
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "model": "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
        "model_type": "INT8",
        "config": {
            "model_dir": str(model_dir),
            "test_dir": str(test_dir),
            "keywords_file": keywords_path,
            "threshold": args.threshold,
            "total_files": len(audio_files),
        },
        "results": stats.to_dict(),
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    # Error analysis
    print("\n" + "=" * 70)
    print("Error Analysis")
    print("=" * 70)
    
    false_negatives = [r for r in stats.results if r.expected_positive and not r.detected]
    false_positives = [r for r in stats.results if not r.expected_positive and r.detected]
    
    if false_negatives:
        print(f"\nFalse Negatives (Missed detections): {len(false_negatives)}")
        for r in false_negatives[:10]:
            print(f"  - {Path(r.audio_path).name}")
        if len(false_negatives) > 10:
            print(f"  ... and {len(false_negatives) - 10} more")
    
    if false_positives:
        print(f"\nFalse Positives (Wrong detections): {len(false_positives)}")
        for r in false_positives[:10]:
            print(f"  - {Path(r.audio_path).name}: '{r.keyword_text}'")
        if len(false_positives) > 10:
            print(f"  ... and {len(false_positives) - 10} more")


if __name__ == "__main__":
    main()
