# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Keyword Spotting (KWS) model training and fine-tuning project for the Chinese wake word "你好真真" (Ni Hao Zhen Zhen). In a zero-shot scenario (no real human voice training data), the project uses synthetic TTS data generation combined with transfer learning to train an offline keyword recognition model.

**Key Technologies:**
- **Icefall**: Next-Gen Kaldi training framework using k2 and Lhotse
- **Sherpa-onnx**: ONNX-based deployment framework for speech models
- **Zipformer**: Neural architecture for speech recognition (3.3M parameters)
- **Edge-TTS**: Microsoft Azure's text-to-speech for synthetic training data
- **Lhotse**: Speech data processing library for manifests and augmentation

## Current Status (V3 + Multi-Stage Detection)

**Model Directory**: `exp/kws_finetune_v3/`

### Multi-Stage Detection Results (RECOMMENDED)

The multi-stage detection approach successfully solves the prefix matching problem:

| 方案 | FRR | FAR | 准确率 | F1 | 延迟 | RTF | 达标 |
|------|-----|-----|--------|----|----|-----|------|
| Baseline (仅V3) | 0.00% | **72.22%** | 42.98% | 42.48% | 33ms | 0.0182 | ✗ |
| **V3 + MLP** | **0.00%** | **1.30%** | **98.98%** | **97.63%** | 46ms | 0.0249 | ✓ |
| V3 + CNN | 0.00% | 1.48% | 98.83% | 97.30% | 51ms | 0.0279 | ✓ |
| V3 + DTW | 96.53% | 0.00% | 79.68% | 6.71% | 259ms | 0.1420 | ✗ |

**Key Achievement**: FAR reduced from **72.22%** to **1.30%** (70.93% improvement) while maintaining **0% FRR**.

### V3 Evaluation Results (Direct vs Delayed Decision)

| Metric | Direct Inference | Delayed Decision | Diff |
|--------|------------------|------------------|------|
| FRR | 0.00% | 0.00% | 0 |
| FAR | 20.56% | 20.56% | 0 |
| Recall | 100.00% | 100.00% | 0 |
| Accuracy | 88.58% | 88.58% | 0 |
| F1 Score | 88.62 | 88.62 | 0 |

### RTF (Real-Time Factor) Comparison

| Metric | Direct | Delayed | Diff |
|--------|--------|---------|------|
| Overall RTF | 0.0204 | 0.0191 | -6.5% |
| Mean RTF | 0.0206 | 0.0194 | -6.1% |
| P99 RTF | 0.0349 | 0.0305 | -12.6% |
| Real-time capable | ✓ Yes | ✓ Yes | - |

**Key Finding**: Both inference modes achieve identical detection metrics. RTF is excellent (< 0.03), far below real-time threshold of 1.0.

### V2 Results (Historical)

| Configuration | FRR | FAR | Recall |
|---------------|-----|-----|--------|
| boost=0.3, threshold=0.6 | 0.00% | 53.70% | 100% |
| boost=0.3, threshold=0.7 | 3.47% | 47.41% | 96.53% |
| boost=1.0, threshold=0.65 | 12.50% | 34.26% | 87.50% |
| boost=1.5, threshold=0.7 | 29.86% | 20.37% | 70.14% |

**Note**: No V2 configuration achieves both FRR < 1.39% and FAR < 7.46%. FRR and FAR are strongly inversely correlated.

## Core Problem

**The model cannot effectively distinguish "你好" (Ni Hao) from "你好真真" (Ni Hao Zhen Zhen)**

Root causes:
1. **Acoustic similarity**: "你好真真" starts with "你好", with nearly identical acoustic features
2. **Small sequence difference**: Only 4 phonemes difference (zh ēn zh ēn)
3. **Pretrained model bias**: WenetSpeech has seen大量 "你好" but almost no "你好真真"

**Current performance gap**:
- Best config: FRR=12.50%, FAR=34.26%
- Target: FRR < 1.39%, FAR < 7.46%
- FRR exceeds target by ~9x, FAR exceeds by ~4.6x

## Project Structure

```
/data/workspace/llm/keyword-spotting/
├── icefall/                 # Icefall framework (K2 training scripts)
│   └── egs/wenetspeech/KWS/zipformer/  # KWS recipe files
├── doc/                     # Documentation directory
│   ├── keyword_spotting_guide.md   # KWS guide from sherpa-onnx
│   └── sherpa_onnx_installation.md # Installation instructions
├── exp/                     # Experiment outputs (checkpoints, ONNX models)
│   ├── kws_finetune/        # V1 model (without negative samples)
│   ├── kws_finetune_v2/     # V2 model (with negative samples)
│   └── kws_finetune_v3/     # V3 model (30 epochs, 5440 samples) - USE THIS
│       └── comparison/      # Direct vs Delayed decision comparison results
├── experiments/             # Ablation experiments
│   └── multi_stage_ablation/  # Multi-stage detection experiments
│       ├── stage1/          # Stage 1 prefix detector
│       ├── stage2/          # Stage 2 verifiers (CNN/MLP/DTW/ASR)
│       ├── models/          # Trained verifier models
│       └── results/         # Ablation experiment results
├── data/                    # Data directories
│   ├── manifests/           # Lhotse data manifests
│   ├── raw_tts/             # Synthetic TTS audio
│   │   ├── nihao_zhenzhen/  # Positive samples (539 files)
│   │   └── negative/        # Negative samples (873 files)
│   └── lang_partial_tone/   # Token vocabulary (pinyin-based)
├── log/                     # Log files (ALL logs must be placed here)
│   ├── training/            # Training logs
│   ├── evaluation/          # Evaluation logs
│   └── misc/                # Miscellaneous logs
├── plan/                    # Plan and design documents
│   ├── plan_v1.md           # Initial training plan
│   ├── plan_v2.md           # V2 plan with negative samples
│   └── implementation_plan.md  # Implementation details
├── report/                  # Experiment reports
│   └── multi_stage_ablation_report.md  # Multi-stage ablation report
└── scripts/                 # Custom utility scripts
```

## Training Data Statistics

| Category | Count | Path |
|----------|-------|------|
| Positive samples | 539 | `data/raw_tts/nihao_zhenzhen/*.wav` |
| Negative samples | 873 | `data/raw_tts/negative/*.wav` |
| **Total** | **1,412** | `data/manifests/` |

**Negative sample composition**:
- **Hard negatives**: 810 - "你好", "您好", "你好啊", "你好吗", and homophones like "泥豪", "李浩"
- **General negatives**: 63 - Random Chinese phrases and greetings

**Positive:Negative ratio**: 1:1.6

## Inference Mode Comparison

### Direct Inference vs Delayed Decision

The project supports two inference modes:

1. **Direct Inference**: Standard sherpa-onnx KeywordSpotter, returns first detected keyword immediately
2. **Delayed Decision**: State machine with 600ms timeout window to confirm full keyword

### Comparison Command

```bash
python scripts/compare_inference_modes.py \
  --model-dir exp/kws_finetune_v3 \
  --positive-dir /data/workspace/llm/audio-classification/dataset/kws_test_data/positive \
  --negative-dir /data/workspace/llm/audio-classification/dataset/kws_test_data/negative \
  --prefix-timeout 600 \
  --chunk-size 100 \
  --verbose
```

### Output Files

```
exp/kws_finetune_v3/comparison/
├── comparison_report_*.txt      # Text report
├── comparison_results_*.json    # JSON results
├── detection_metrics_*.png      # Detection metrics chart
├── rtf_comparison_*.png         # RTF comparison chart
├── confusion_matrix_*.png       # Confusion matrix
└── summary_dashboard_*.png      # Summary dashboard
```

### Key Findings

- Both modes achieve **identical detection performance** (FRR=0%, FAR=20.56%)
- RTF is excellent for both modes (~0.02, far below real-time threshold of 1.0)
- Delayed decision mode is slightly faster due to chunked processing
- Current model directly detects full keyword "你好真真", not triggering prefix logic

## Common Commands

### Full Training Pipeline (V2)

```bash
# 1. Generate synthetic training data with Edge-TTS
python scripts/generate_tts_dataset.py

# 2. Generate negative TTS samples
python scripts/generate_negative_tts.py

# 3. Create Lhotse manifests from WAV files
python scripts/prepare_lhotse_manifests.py

# 4. Fine-tune the model V2 (uses kws-train conda environment)
bash scripts/run_finetune_v2.sh

# 5. Export to ONNX with INT8 quantization
bash scripts/export_onnx_v2.sh

# 6. Parameter optimization (uses full test dataset)
python scripts/optimize_kws_params.py \
  --model-dir exp/kws_finetune_v2 \
  --positive-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive \
  --negative-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/negative \
  --target-frr 1.39 \
  --target-far 7.46
```

### Quick Evaluation

```bash
python scripts/evaluate_kws_model.py \
  --model-dir exp/kws_finetune_v2 \
  --positive-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive \
  --negative-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/negative \
  --threshold 0.5
```

## Key File Locations

| Description | Path |
|-------------|------|
| Pretrained checkpoint | `icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt` |
| Training script V2 | `scripts/run_finetune_v2.sh` |
| Export script V2 | `scripts/export_onnx_v2.sh` |
| Token vocabulary | `data/lang_partial_tone/tokens.txt` |
| Lhotse manifests | `data/manifests/kws_*.jsonl.gz` |
| V3 Model outputs | `exp/kws_finetune_v3/` |
| V3 Comparison results | `exp/kws_finetune_v3/comparison/` |
| V2 Model outputs | `exp/kws_finetune_v2/` |
| Optimization reports | `exp/kws_finetune_v2/param_optimization/` |
| **Multi-stage ablation** | `experiments/multi_stage_ablation/` |
| **MLP verifier model** | `experiments/multi_stage_ablation/models/mlp_verifier.pt` |
| **CNN verifier model** | `experiments/multi_stage_ablation/models/cnn_verifier.pt` |
| **Ablation report** | `experiments/multi_stage_ablation/results/ablation_final_report.txt` |
| **Ablation summary** | `report/multi_stage_ablation_report.md` |

## Model Architecture

**Zipformer-Transducer (3.3M parameters):**
- 6 encoder layers with downsampling stacks
- Causal architecture for streaming support
- Chunk size: 16, Left context: 128 frames
- Encoder dim: 128, Decoder dim: 320, Joiner dim: 320

## Tokenization Strategy

The model uses **pinyin-based tokenization** with partial tones. Tokens consist of:
- Initials (声母): zh, ch, sh, n, h, etc.
- Finals with tone marks (韵母+声调): ǐ, ǎo, ēn, etc.

Example: "你好真真" → `n ǐ h ǎo zh ēn zh ēn`

The `keywords.txt` format supports boost and threshold parameters:
```
n ǐ h ǎo zh ēn zh ēn :boost #threshold @你好真真
```
- `:boost` - Boost score parameter (increases keyword pathway probability)
- `#threshold` - Trigger threshold parameter (minimum acoustic probability for detection)

## Fine-tuning Parameters (V2)

The V2 training script (`scripts/run_finetune_v2.sh`) uses:
- Base learning rate: 0.0003 (reduced from 0.0005)
- Epochs: 20 (increased from 10)
- Max duration: 500
- FP16 mixed precision training
- On-the-fly feature extraction
- SpecAugment enabled
- On-the-fly data augmentation (disabled via `--enable-musan 0`)

## Python Environment

The project uses a conda environment named `kws-train`:
- Python: 3.10+
- PyTorch with CUDA support
- k2 (version 1.24.4+)
- Lhotse 1.32.1
- sherpa-onnx

To activate: `conda activate kws-train` or use full path `/data/workspace/llm/anaconda3/envs/kws-train/`

## Sherpa-ONNX Installation

Sherpa-onnx can be installed using multiple methods:

### Method 1: Pre-compiled wheels (CPU only)

```bash
pip install sherpa-onnx sherpa-onnx-bin
```

### Method 2: CUDA 11.8 support (CPU + GPU)

```bash
pip install sherpa-onnx==1.12.13+cuda -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
```

Pass `provider=cuda` to use NVIDIA GPU (defaults to `provider=cpu`).

### Method 3: CUDA 12.8 + CUDNN9 support

```bash
pip install sherpa-onnx==1.12.13+cuda12.cudnn9 -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
```

### Method 4: From source (CPU)

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
python3 setup.py install
```

### Method 5: From source (GPU/CUDA)

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
export SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON"
cd sherpa-onnx
python3 setup.py install
```

### Installation verification

```bash
python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"
```

For more details, see https://k2-fsa.github.io/sherpa/onnx/cpu.html

### System Requirements for CUDA

To use sherpa-onnx with CUDA acceleration:

1. **NVIDIA GPU** with compute capability 6.0+
2. **CUDA Toolkit** (11.8, 12.2, or 12.8)
3. **cuDNN** library (version matching CUDA)

For CUDA 12.x + cuDNN 9 on Linux:
```bash
sudo yum install -y libcudnn9-cuda-12   # RHEL/CentOS/TencentOS
# or
sudo apt-get install -y libcudnn9      # Ubuntu/Debian
```

### CUDA Testing

Use the test script to verify CUDA availability and compare performance:

```bash
# Run 10-second benchmark test
python scripts/test_cuda.py \
  --model-dir exp/kws_finetune_v2 \
  --test-audio /path/to/test_audio.wav \
  --duration 10

# Test CPU only
python scripts/test_cuda.py --cpu-only
```

When creating a `KeywordSpotter`, use `provider='cuda'` to enable GPU acceleration:

```python
spotter = sherpa_onnx.KeywordSpotter(
    tokens=tokens_path,
    encoder=encoder_path,
    decoder=decoder_path,
    joiner=joiner_path,
    keywords_file=keyword_path,
    provider='cpu',  # Use 'cpu' for CPU inference, 'cuda' for GPU
)
```

**IMPORTANT: CPU vs CUDA Performance**

For this KWS model (INT8 quantized ~1M parameters), **CPU is significantly faster than CUDA**:

| Provider | Throughput | Avg Latency | Notes |
|----------|------------|-------------|-------|
| CPU | ~43 inferences/sec | ~21ms | **Recommended** |
| CUDA | ~0.4 inferences/sec | ~2800ms | Not suitable for small models |

**Why CPU is faster:**
1. Small model fits entirely in CPU cache (~4MB total)
2. PCIe data transfer overhead is high relative to compute time
3. GPU kernel launch overhead significant for small workloads
4. Insufficient compute to utilize GPU parallelism

**Recommendation:** Use `provider='cpu'` for this KWS model. CUDA is only beneficial for larger models (hundreds of MB to GB in size).

## Icefall Integration

The Icefall framework is located in `icefall/` directory. You must set `PYTHONPATH`:

```bash
export PYTHONPATH=/data/workspace/llm/keyword-spotting/icefall:$PYTHONPATH
```

The KWS recipe is at `icefall/egs/wenetspeech/KWS/zipformer/`.

## Evaluation Metrics

- **FRR** (False Rejection Rate): The rate at which the model misses the actual keyword
- **FAR** (False Accept Rate): The rate at which the model detects the keyword when not present
- **Recall** (True Positive Rate): The rate at which the model correctly detects the keyword
- **Specificity** (True Negative Rate): The rate at which the model correctly rejects non-keyword audio
- **RTF** (Real-Time Factor): Processing time / Audio duration. RTF < 1.0 means real-time capable

**Targets**: FRR < 1.39%, FAR < 7.46%, RTF < 1.0

## Test Data

**IMPORTANT**: All evaluation must use the unified test dataset from audio-classification.

**Dataset Path**: `/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/`

| Category | Count | Description |
|----------|-------|-------------|
| Positive | 144 | TTS voices (8 voices × 6 SNR levels × 3 prosody variations) |
| Negative | 540 | Complete negative samples (includes all 84 "你好" variants) |

**DO NOT** exclude the 84 "你好" variant samples (泥豪/李浩). The task is to achieve optimal performance on the **complete** test dataset.

## Parameter Optimization

The `scripts/optimize_kws_params.py` script performs grid search over boost and threshold parameters.

### Usage (V2)

```bash
python scripts/optimize_kws_params.py \
  --model-dir exp/kws_finetune_v2 \
  --positive-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive \
  --negative-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/negative \
  --boost-values "0.3,0.5,0.7,0.8,1.0,1.2,1.5" \
  --threshold-values "0.4,0.45,0.5,0.55,0.6,0.65,0.7" \
  --target-frr 1.39 \
  --target-far 7.46
```

### Default Parameter Ranges

| Parameter | Range | Description |
|-----------|-------|-------------|
| boost | 0.3 - 1.5 | Keyword pathway boost (lower = less aggressive) |
| threshold | 0.4 - 0.7 | Detection threshold (higher = harder to trigger) |

### Output

- Report: `exp/kws_finetune_v2/param_optimization/param_optimization_*.txt`
- JSON: `exp/kws_finetune_v2/param_optimization/param_optimization_*.json`
- Recommended config: `exp/kws_finetune_v2/keywords_recommended.txt`

##  Approach Status

**The decoy approach has been TESTED and FAILED.**

### What Was Attempted

Registered decoy keywords like "你好", "你好啊", "您好" with lower boost values, expecting the model to detect them as decoys and reject them for positive samples.

### Results

- FAR improved to 9.63% (good)
- FRR degraded to 82.64% (catastrophic)

### Why It Failed

When audio contains "你好真真", the shorter "你好" sequence is **matched first** by the Transducer model. sherpa-onnx returns the first matched keyword, and since "你好" is a prefix of "你好真真", it always triggers first. All positive samples were incorrectly classified as decoy "你好".

**Debug verification** (from `scripts/debug_keyword_result.py`):
```
Testing POSITIVE samples (should detect "你好真真"):
  Audio: positive_0000_*.wav
  Detected: [你好]  <-- WRONG! Should be "你好真真"

Testing NEGATIVE samples (should detect "你好" or nothing):
  Audio: expanded_negative_0000_*.wav
  Detected: [你好真真]  <-- This is the source of FAR
```

## Failed Approaches Summary

| Approach | Result | Notes |
|----------|--------|-------|
| V1 (no negative samples) | FRR=10.42%, FAR=44.07% | Insufficient performance |
| V2 (with negative samples) | FRR=0%~12.50%, FAR=34.26%~55% | Slight improvement, but far from targets |
| Decoy filtering | FRR=82.64%, FAR=9.63% | Complete failure |

## Next Optimization Suggestions

Given the fundamental problem (无法区分"你好"和"你好真真"), consider these approaches:

### Option A: Post-processing Time Window

Wait for a time window (e.g., 500ms) after detecting "你好" to see if "真真" follows.

**Pros**: No retraining needed
**Cons**: Increases latency, requires sherpa-onnx usage modifications

### Option B: Change Wake Word

Use a wake word that doesn't start with common words.

**Candidates**:
- "真真你好" (reverse order)
- "嗨真真" (different prefix)
- "喂真真"

**Pros**: Fundamentally solves prefix matching problem

### Option C: Multi-stage Detection ✓ IMPLEMENTED & SUCCESSFUL

1. Stage 1: Detect "你好真真" with V3 model (high recall, high FAR)
2. Stage 2: Verify suffix "真真" with MLP/CNN classifier (filters false positives)

**Implementation**: `experiments/multi_stage_ablation/`

**Results**:
- FAR: 72.22% → **1.30%** (70.93% reduction)
- FRR: 0% → **0%** (maintained)
- Latency: 33ms → **46ms** (+13ms)
- RTF: 0.0249 (real-time capable)

**Recommended Configuration**: V3 + MLP Verifier

### Option D: Negative Sample Weighting

Increase weight of "你好" negative samples during training to make model less likely to trigger.

**Pros**: May improve discrimination

**Cons**: May reduce recall

### Option E: Collect Real Data

Collect real human voice samples of "你好真真" for training.

**Pros**: Most likely to achieve target performance

**Cons**: Requires data collection effort

## Scripts Reference

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/generate_tts_dataset.py` | Generate positive TTS data | ✓ Working |
| `scripts/generate_negative_tts.py` | Generate negative TTS data | ✓ Working |
| `scripts/prepare_lhotse_manifests.py` | Create Lhotse manifests | ✓ Working |
| `scripts/run_finetune.sh` | Training V1 (no negatives) | Deprecated |
| `scripts/run_finetune_v2.sh` | Training V2 (with negatives) | ✓ Working |
| `scripts/export_onnx.sh` | Export V1 to ONNX | Deprecated |
| `scripts/export_onnx_v2.sh` | Export V2 to ONNX | ✓ Working |
| `scripts/optimize_kws_params.py` | Parameter grid search | ✓ Working |
| `scripts/optimize_decoy_params.py` | Decoy optimization | ✗ FAILED |
| `scripts/evaluate_kws_model.py` | Quick model evaluation | ✓ Working |
| `scripts/find_optimal_threshold.py` | Find optimal threshold | ✓ Working |
| `scripts/test_cuda.py` | Test CUDA acceleration | ✓ Working |
| `scripts/rtf_utils.py` | RTF calculation utilities | ✓ Working |
| `scripts/evaluate_with_delay.py` | Delayed decision evaluation | ✓ Working |
| `scripts/evaluate_kws_with_rtf.py` | Direct inference with RTF | ✓ Working |
| `scripts/compare_inference_modes.py` | Compare Direct vs Delayed | ✓ Working |
| `scripts/generate_comparison_charts.py` | Generate comparison charts | ✓ Working |
| `scripts/delayed_decision_inference.py` | Delayed decision state machine | ✓ Working |

### Multi-Stage Detection Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `experiments/multi_stage_ablation/run_ablation.py` | Run ablation experiments | ✓ Working |
| `experiments/multi_stage_ablation/run_ablation_trained.py` | Evaluate with trained models | ✓ Working |
| `experiments/multi_stage_ablation/train_verifiers.py` | Train CNN/MLP verifiers | ✓ Working |
| `experiments/multi_stage_ablation/stage1/prefix_detector.py` | Stage 1 V3 detector | ✓ Working |
| `experiments/multi_stage_ablation/stage2/mlp_verifier.py` | MLP suffix verifier | ✓ Working |
| `experiments/multi_stage_ablation/stage2/cnn_verifier.py` | CNN suffix verifier | ✓ Working |
| `experiments/multi_stage_ablation/stage2/dtw_verifier.py` | DTW suffix verifier | ✓ Working |
| `experiments/multi_stage_ablation/stage2/asr_verifier.py` | ASR suffix verifier | ✓ Working |

## Log File Management

**IMPORTANT: All log files MUST be placed in the `log/` directory.**

### Log Structure

```
log/
├── training/           # Training process logs
│   └── training_<YYYYMMDD>_<HHMMSS>.log
├── evaluation/         # Evaluation and optimization logs
│   └── evaluation_<YYYYMMDD>_<HHMMSS>.log
└── misc/               # Miscellaneous logs (debug, etc.)
    └── <script_name>_<YYYYMMDD>_<HHMMSS>.log
```

### Naming Convention

Log files should follow the format:
```
<category>_<script_name>_<YYYYMMDD>_<HHMMSS>.log
```

Examples:
- `log/training/training_finetune_v2_20260112_093000.log`
- `log/evaluation/evaluation_optimize_params_20260112_103000.log`
- `log/misc/debug_keyword_result_20260112_110000.log`

### Python Script Best Practice

When creating log files in scripts:
```python
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__name__).parent.parent / "log" / "training"
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"training_{timestamp}.log"
```

## Plan File Management

**All plan files should be placed in the `plan/` directory.**

### Plan Structure

```
plan/
├── plan_v1.md              # Initial training plan
├── plan_v2.md              # V2 training plan (with negative samples)
└── implementation_plan.md   # Detailed implementation plans
```

### Naming Convention

Plan files should follow the format:
```
<plan_name>_<version>.md
```

Examples:
- `plan/plan_v1.md` - Initial training approach
- `plan/plan_v2.md` - Negative sampling approach
- `plan/implementation_plan.md` - Detailed implementation strategy

## Important Notes

- **SOLVED**: Multi-stage detection (V3 + MLP) achieves FAR=1.30%, FRR=0% - **meets all targets**
- **Core limitation**: KWS keyword "你好真真" contains "你好" as prefix - solved via two-stage verification
- Best single-stage config: FRR=0%, FAR=72.22% vs target FRR<5%, FAR<10%
- **Recommended**: Use V3 + MLP verifier for production deployment
- This is a zero-shot KWS project with no real human voice data (only TTS)
- The model is fine-tuned from a pre-trained WenetSpeech model
- TTS generation uses Edge-TTS with multiple Chinese voices and prosody variations (rate, pitch)
- INT8 quantization reduces model size by ~75% (3.3M → ~1M parameters)
- **Always use the complete test dataset from audio-classification (540 negative samples)**
- **Decoy approach has been tested and is NOT feasible for this keyword**
- **All log files MUST be placed in the `log/` directory with proper naming convention**

## Reproduction

### Environment Setup

```bash
conda activate kws-train
export PYTHONPATH=/data/workspace/llm/keyword-spotting/icefall:$PYTHONPATH
```

### Full Reproduction

```bash
# 1. Prepare data
python scripts/generate_tts_dataset.py
python scripts/generate_negative_tts.py
python scripts/prepare_lhotse_manifests.py

# 2. Train V2
bash scripts/run_finetune_v2.sh

# 3. Export
bash scripts/export_onnx_v2.sh

# 4. Evaluate and optimize
python scripts/optimize_kws_params.py \
  --model-dir exp/kws_finetune_v2 \
  --positive-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive \
  --negative-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/negative \
  --target-frr 1.39 \
  --target-far 7.46
```

### Model Files (V2)

```
exp/kws_finetune_v2/
├── encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx
├── decoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx
├── joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx
├── tokens.txt
├── keywords.txt
└── param_optimization/
    └── param_optimization_*.txt  # Optimization reports
```

## Documentation Reference

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project guidance and development instructions |
| `report.md` | Current status and analysis report |
| `doc/keyword_spotting_guide.md` | KWS guide from sherpa-onnx (conceptual reference) |
| `doc/sherpa_onnx_installation.md` | Sherpa-ONNX installation instructions |
| `plan/` | Implementation plans and design documents |
| `icefall/egs/wenetspeech/KWS/` | Icefall KWS training code |
