#!/usr/bin/env python3
"""
Simple test script to verify sherpa-onnx CUDA acceleration support.

This script tests whether sherpa-onnx can use GPU (CUDA) for inference
and compares performance between CPU and CUDA providers.
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import sherpa_onnx
    import soundfile as sf
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required packages:")
    print("  pip install sherpa-onnx soundfile")
    sys.exit(1)


def test_provider(
    encoder_path: str,
    decoder_path: str,
    joiner_path: str,
    tokens_path: str,
    provider: str,
    audio_path: str,
    keyword_path: str,
    duration_sec: int = 10,
) -> bool:
    """
    Test if a specific provider works with sherpa-onnx.

    Args:
        encoder_path: Path to encoder ONNX model
        decoder_path: Path to decoder ONNX model
        joiner_path: Path to joiner ONNX model
        tokens_path: Path to tokens.txt
        provider: Provider name (cpu or cuda)
        audio_path: Path to test audio file
        keyword_path: Path to keywords.txt
        duration_sec: Test duration in seconds

    Returns:
        True if provider works, False otherwise
    """
    print(f"\nTesting provider: {provider.upper()}")
    print("-" * 50)

    try:
        # Create KeywordSpotter with specified provider
        spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            keywords_file=keyword_path,
            num_threads=4,
            keywords_threshold=0.0,
            provider=provider,
        )
        print(f"[OK] KeywordSpotter created with provider '{provider}'")

        # Test inference
        samples, sample_rate = sf.read(audio_path, dtype="float32")

        if sample_rate != 16000:
            print(f"[INFO] Resampling from {sample_rate}Hz to 16000Hz")
            try:
                import librosa
                samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except ImportError:
                print("[INFO] librosa not available, skipping test")
                return True

        if len(samples.shape) > 1:
            samples = samples[:, 0]

        # Warmup runs
        for _ in range(3):
            stream = spotter.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()
            while spotter.is_ready(stream):
                spotter.decode_stream(stream)

        # Performance test for specified duration
        print(f"[INFO] Running {duration_sec}s performance test...")
        times = []
        count = 0
        start_time = time.time()

        while time.time() - start_time < duration_sec:
            stream = spotter.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()

            iter_start = time.time()
            while spotter.is_ready(stream):
                spotter.decode_stream(stream)
            iter_end = time.time()

            times.append(iter_end - iter_start)
            count += 1

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)
        throughput = count / duration_sec

        print(f"[OK] Ran {count} inferences in {duration_sec:.1f}s")
        print(f"      Throughput: {throughput:.1f} inferences/sec")
        print(f"      Latency:    avg={avg_time*1000:.2f}ms, min={min_time*1000:.2f}ms, max={max_time*1000:.2f}ms")
        print(f"      RTF:        {total_time*1000/(duration_sec*1000):.3f}realtime")
        return True

    except Exception as e:
        print(f"[FAIL] Error with provider '{provider}': {e}")
        return False


def find_test_files(model_dir: Path) -> dict:
    """Find required model and test files."""
    result = {}

    # Find ONNX model files
    encoder_files = list(model_dir.glob("encoder-*.onnx"))
    decoder_files = list(model_dir.glob("decoder-*.onnx"))
    joiner_files = list(model_dir.glob("joiner-*.onnx"))

    if encoder_files:
        result["encoder"] = str(encoder_files[0])
    else:
        encoder_files_int8 = list(model_dir.glob("encoder-*.int8.onnx"))
        if encoder_files_int8:
            result["encoder"] = str(encoder_files_int8[0])

    if decoder_files:
        result["decoder"] = str(decoder_files[0])
    else:
        decoder_files_int8 = list(model_dir.glob("decoder-*.int8.onnx"))
        if decoder_files_int8:
            result["decoder"] = str(decoder_files_int8[0])

    if joiner_files:
        result["joiner"] = str(joiner_files[0])
    else:
        joiner_files_int8 = list(model_dir.glob("joiner-*.int8.onnx"))
        if joiner_files_int8:
            result["joiner"] = str(joiner_files_int8[0])

    # Find tokens.txt and keywords.txt
    tokens_file = model_dir / "tokens.txt"
    keywords_file = model_dir / "keywords.txt"

    if tokens_file.exists():
        result["tokens"] = str(tokens_file)

    if keywords_file.exists():
        result["keywords"] = str(keywords_file)

    # Find test audio file
    test_audio = model_dir / "test.wav"
    if not test_audio.exists():
        # Try data/raw_tts directory
        test_audio = Path("/data/workspace/llm/keyword-spotting/data/raw_tts")
        if test_audio.exists():
            wav_files = list(test_audio.glob("*.wav"))
            if wav_files:
                result["audio"] = str(wav_files[0])

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test sherpa-onnx CUDA acceleration support"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/data/workspace/llm/keyword-spotting/exp/kws_finetune",
        help="Directory containing ONNX model files",
    )
    parser.add_argument(
        "--test-audio",
        type=str,
        help="Path to test audio file (optional)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Only test CPU provider (skip CUDA)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration in seconds (default: 10)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    print("=" * 60)
    print("Sherpa-ONNX CUDA Acceleration Test")
    print("=" * 60)
    print(f"Test duration: {args.duration} seconds per provider")
    print("=" * 60)

    # Find required files
    files = find_test_files(model_dir)
    if args.test_audio:
        files["audio"] = args.test_audio

    required = ["encoder", "decoder", "joiner", "tokens", "keywords", "audio"]
    missing = [k for k in required if k not in files]

    if missing:
        print(f"Error: Missing required files: {', '.join(missing)}")
        print(f"Model directory: {model_dir}")
        sys.exit(1)

    print(f"Model directory: {model_dir}")
    print(f"Encoder: {Path(files['encoder']).name}")
    print(f"Decoder: {Path(files['decoder']).name}")
    print(f"Joiner:  {Path(files['joiner']).name}")
    print(f"Tokens:  {files['tokens']}")
    print(f"Keywords: {files['keywords']}")
    print(f"Test audio: {files['audio']}")
    print("=" * 60)

    results = {}

    # Test CPU first (always available)
    results["cpu"] = test_provider(
        files["encoder"],
        files["decoder"],
        files["joiner"],
        files["tokens"],
        "cpu",
        files["audio"],
        files["keywords"],
        args.duration,
    )

    # Test CUDA if not skipped
    if not args.cpu_only:
        print("\n")
        print("=" * 60)
        print("CUDA Test")
        print("=" * 60)
        results["cuda"] = test_provider(
            files["encoder"],
            files["decoder"],
            files["joiner"],
            files["tokens"],
            "cuda",
            files["audio"],
            files["keywords"],
            args.duration,
        )
    else:
        results["cuda"] = False

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"CPU:    {'[OK]' if results['cpu'] else '[FAIL]'}")
    print(f"CUDA:   {'[OK]' if results['cuda'] else '[FAIL]'}")
    print("=" * 60)

    if results["cpu"] and not results["cuda"]:
        print("\nNote: CPU works but CUDA failed.")
        print("This may indicate:")
        print("  1. No GPU available")
        print("  2. sherpa-onnx was built without CUDA support")
        print("  3. CUDA provider is not compatible with your GPU")
    elif results["cuda"]:
        print("\nCUDA acceleration is available!")
        print("You can use provider='cuda' in sherpa-onnx for faster inference.")

    sys.exit(0 if results["cpu"] else 1)


if __name__ == "__main__":
    main()
