#!/usr/bin/env python3
"""
RTF (Real-Time Factor) Calculation Utilities for KWS Evaluation.

RTF = Processing Time / Audio Duration
RTF < 1.0 means real-time capable
RTF > 1.0 means slower than real-time
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import soundfile as sf


@dataclass
class RTFMeasurement:
    """Single RTF measurement for an audio file."""
    audio_path: str
    audio_duration_sec: float
    process_time_sec: float
    
    @property
    def rtf(self) -> float:
        """Calculate RTF."""
        if self.audio_duration_sec <= 0:
            return float('inf')
        return self.process_time_sec / self.audio_duration_sec


@dataclass
class RTFStats:
    """Aggregated RTF statistics."""
    total_audio_duration_sec: float = 0.0
    total_process_time_sec: float = 0.0
    measurements: List[RTFMeasurement] = field(default_factory=list)
    
    @property
    def rtf(self) -> float:
        """Overall RTF."""
        if self.total_audio_duration_sec <= 0:
            return float('inf')
        return self.total_process_time_sec / self.total_audio_duration_sec
    
    @property
    def rtf_mean(self) -> float:
        """Mean RTF across all files."""
        if not self.measurements:
            return 0.0
        return np.mean([m.rtf for m in self.measurements])
    
    @property
    def rtf_std(self) -> float:
        """Standard deviation of RTF."""
        if not self.measurements:
            return 0.0
        return np.std([m.rtf for m in self.measurements])
    
    @property
    def rtf_min(self) -> float:
        """Minimum RTF."""
        if not self.measurements:
            return 0.0
        return min(m.rtf for m in self.measurements)
    
    @property
    def rtf_max(self) -> float:
        """Maximum RTF."""
        if not self.measurements:
            return 0.0
        return max(m.rtf for m in self.measurements)
    
    @property
    def rtf_median(self) -> float:
        """Median RTF."""
        if not self.measurements:
            return 0.0
        return float(np.median([m.rtf for m in self.measurements]))
    
    @property
    def rtf_p95(self) -> float:
        """95th percentile RTF."""
        if not self.measurements:
            return 0.0
        return float(np.percentile([m.rtf for m in self.measurements], 95))
    
    @property
    def rtf_p99(self) -> float:
        """99th percentile RTF."""
        if not self.measurements:
            return 0.0
        return float(np.percentile([m.rtf for m in self.measurements], 99))
    
    @property
    def is_realtime(self) -> bool:
        """Check if overall RTF is real-time capable."""
        return self.rtf < 1.0
    
    @property
    def file_count(self) -> int:
        """Number of files processed."""
        return len(self.measurements)
    
    def add_measurement(self, measurement: RTFMeasurement):
        """Add a measurement."""
        self.measurements.append(measurement)
        self.total_audio_duration_sec += measurement.audio_duration_sec
        self.total_process_time_sec += measurement.process_time_sec
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_count": self.file_count,
            "total_audio_duration_sec": round(self.total_audio_duration_sec, 3),
            "total_process_time_sec": round(self.total_process_time_sec, 3),
            "rtf_overall": round(self.rtf, 4),
            "rtf_mean": round(self.rtf_mean, 4),
            "rtf_std": round(self.rtf_std, 4),
            "rtf_min": round(self.rtf_min, 4),
            "rtf_max": round(self.rtf_max, 4),
            "rtf_median": round(self.rtf_median, 4),
            "rtf_p95": round(self.rtf_p95, 4),
            "rtf_p99": round(self.rtf_p99, 4),
            "is_realtime": self.is_realtime,
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"RTF Statistics ({self.file_count} files):",
            f"  Total audio duration: {self.total_audio_duration_sec:.2f}s",
            f"  Total process time: {self.total_process_time_sec:.2f}s",
            f"  Overall RTF: {self.rtf:.4f}",
            f"  Mean RTF: {self.rtf_mean:.4f} ± {self.rtf_std:.4f}",
            f"  Median RTF: {self.rtf_median:.4f}",
            f"  Min/Max RTF: {self.rtf_min:.4f} / {self.rtf_max:.4f}",
            f"  P95/P99 RTF: {self.rtf_p95:.4f} / {self.rtf_p99:.4f}",
            f"  Real-time capable: {'Yes' if self.is_realtime else 'No'}",
        ]
        return "\n".join(lines)


class RTFTimer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


def get_audio_duration(audio_path: str, sample_rate: int = 16000) -> float:
    """
    Get audio duration in seconds.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Expected sample rate (for resampling calculation)
    
    Returns:
        Duration in seconds
    """
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        # Fallback: read the file
        try:
            samples, sr = sf.read(audio_path, dtype="float32")
            if len(samples.shape) > 1:
                samples = samples[:, 0]
            return len(samples) / sr
        except Exception:
            return 0.0


def measure_inference_rtf(
    inference_fn,
    audio_files: List[str],
    warmup_count: int = 3,
) -> RTFStats:
    """
    Measure RTF for an inference function.
    
    Args:
        inference_fn: Function that takes audio_path and returns result
        audio_files: List of audio file paths
        warmup_count: Number of warmup runs (not counted)
    
    Returns:
        RTFStats with all measurements
    """
    stats = RTFStats()
    
    # Warmup runs
    for i, audio_path in enumerate(audio_files[:warmup_count]):
        if Path(audio_path).exists():
            try:
                inference_fn(audio_path)
            except Exception:
                pass
    
    # Actual measurements
    for audio_path in audio_files:
        if not Path(audio_path).exists():
            continue
        
        audio_duration = get_audio_duration(audio_path)
        if audio_duration <= 0:
            continue
        
        try:
            with RTFTimer() as timer:
                inference_fn(audio_path)
            
            measurement = RTFMeasurement(
                audio_path=audio_path,
                audio_duration_sec=audio_duration,
                process_time_sec=timer.elapsed,
            )
            stats.add_measurement(measurement)
            
        except Exception as e:
            print(f"Error measuring {audio_path}: {e}")
    
    return stats


def compare_rtf_stats(
    stats_a: RTFStats,
    stats_b: RTFStats,
    label_a: str = "Mode A",
    label_b: str = "Mode B",
) -> str:
    """
    Compare two RTF statistics.
    
    Args:
        stats_a: First RTF stats
        stats_b: Second RTF stats
        label_a: Label for first mode
        label_b: Label for second mode
    
    Returns:
        Comparison summary string
    """
    lines = [
        "=" * 60,
        "RTF Comparison",
        "=" * 60,
        f"{'Metric':<25} | {label_a:>12} | {label_b:>12} | {'Diff':>10}",
        "-" * 60,
    ]
    
    metrics = [
        ("Overall RTF", stats_a.rtf, stats_b.rtf),
        ("Mean RTF", stats_a.rtf_mean, stats_b.rtf_mean),
        ("Median RTF", stats_a.rtf_median, stats_b.rtf_median),
        ("P95 RTF", stats_a.rtf_p95, stats_b.rtf_p95),
        ("P99 RTF", stats_a.rtf_p99, stats_b.rtf_p99),
        ("Min RTF", stats_a.rtf_min, stats_b.rtf_min),
        ("Max RTF", stats_a.rtf_max, stats_b.rtf_max),
    ]
    
    for name, val_a, val_b in metrics:
        diff = val_b - val_a
        diff_pct = (diff / val_a * 100) if val_a > 0 else 0
        diff_str = f"{diff:+.4f} ({diff_pct:+.1f}%)"
        lines.append(f"{name:<25} | {val_a:>12.4f} | {val_b:>12.4f} | {diff_str:>10}")
    
    lines.extend([
        "-" * 60,
        f"{'Real-time capable':<25} | {'Yes' if stats_a.is_realtime else 'No':>12} | {'Yes' if stats_b.is_realtime else 'No':>12} |",
        "=" * 60,
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the module
    print("RTF Utils Module Test")
    print("=" * 40)
    
    # Create mock measurements
    stats = RTFStats()
    for i in range(10):
        m = RTFMeasurement(
            audio_path=f"test_{i}.wav",
            audio_duration_sec=2.0,
            process_time_sec=0.1 + i * 0.01,
        )
        stats.add_measurement(m)
    
    print(stats.summary())
    print()
    print("Dict output:")
    for k, v in stats.to_dict().items():
        print(f"  {k}: {v}")
