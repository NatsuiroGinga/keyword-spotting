# CODEBUDDY.md

This file provides guidance to CodeBuddy when working with code in this repository.

## Project Overview

This is a Keyword Spotting (KWS) model training and fine-tuning project for the Chinese wake word "你好真真" (Ni Hao Zhen Zhen). In a zero-shot scenario (no real human voice training data), the project uses synthetic TTS data generation combined with transfer learning to train an offline keyword recognition model.

**Key Technologies:**
- **Icefall**: Next-Gen Kaldi training framework using k2 and Lhotse
- **Sherpa-onnx**: ONNX-based deployment framework for speech models
- **Zipformer**: Neural architecture for speech recognition (3.3M parameters)
- **Edge-TTS**: Microsoft Azure's text-to-speech for synthetic training data
- **Lhotse**: Speech data processing library for manifests and augmentation

## Current Status (V4 Streaming Model - PRODUCTION READY)

**🏆 Best Model**: `experiments/baseline_streaming/exp_v4/epoch-33`
**🚀 HuggingFace**: https://huggingface.co/Heehobino/nihao-zhenzhen-kws

### V4 Streaming Model Performance

| 版本 | 大小 | 阈值 | F1 | FAR | FRR | RTF | 达标 |
|------|------|------|-----|-----|-----|-----|------|
| **FP32** | 13MB | 0.52 | **89.47%** | 5.64% | 2.30% | 0.0185 | ✓ |
| **INT8** | 5MB | 0.46 | 89.01% | 5.96% | 2.30% | 0.0140 | ✓ |

**Key Achievement**: 
- FAR < 10% ✓, FRR < 5% ✓, RTF < 1.0 ✓
- INT8量化：体积减小62%，速度提升24%

### Previous Models Comparison

| Model | FAR | FRR | F1 | Notes |
|-------|-----|-----|-----|-------|
| V3 + MLP | 1.30% | 0.00% | 97.63% | Multi-stage, higher latency |
| **V4 (epoch-33)** | **5.64%** | **2.30%** | **89.47%** | **Single-stage streaming, recommended** |

## Deployable Package

**独立可发布目录**: `nihao-zhenzhen-kws/`

```
nihao-zhenzhen-kws/
├── model/                    # FP32模型 (13MB)
│   ├── encoder.onnx
│   ├── decoder.onnx
│   ├── joiner.onnx
│   ├── tokens.txt
│   └── keywords.txt
├── model_int8/               # INT8模型 (5MB)
│   └── ...
├── examples/
│   └── realtime_detection.py # 麦克风实时检测
├── config.json               # 配置文件
├── inference.py              # Python推理接口
├── requirements.txt
└── README.md
```

**Quick Start**:
```python
from inference import load_model

# FP32 (默认)
detector = load_model()

# INT8 (更小更快)
detector = load_model(variant="int8")

# 检测音频
result = detector.detect("audio.wav")
```

## Test Data

**IMPORTANT**: All evaluation must use the real human voice test dataset.

**Primary Dataset (Real Human Voice)**: `data/all/` (from `data/kws-data-all.zip`)

| Category | Count | Description |
|----------|-------|-------------|
| Positive (你好真真) | 10 | Real human voice recordings of "你好真真" |
| Similar (你好珍珍/娟娟) | 3 | Similar-sounding keywords (should be rejected) |
| Negative | 36 | Other commands and phrases |
| **Total** | **49** | Real-world test samples |

**Priority**: Use `data/all/` (real human voice) as the primary test dataset for all experiments.

## Project Structure

```
/data/workspace/llm/keyword-spotting/
├── nihao-zhenzhen-kws/       # 🚀 Deployable package (HuggingFace)
├── icefall/                  # Icefall framework (K2 training scripts)
│   └── egs/wenetspeech/KWS/zipformer/  # KWS recipe files
├── experiments/
│   ├── baseline_streaming/   # V4 streaming model experiments
│   │   └── exp_v4/           # Best model: epoch-33
│   └── multi_stage_ablation/ # V3 + MLP experiments
├── exp/                      # Legacy experiment outputs
│   └── kws_finetune_v3/      # V3 model
├── data/
│   ├── all/                  # Real human voice test data (49 files)
│   ├── manifests/            # Lhotse data manifests
│   └── lang_partial_tone/    # Token vocabulary (pinyin-based)
├── log/                      # Log files (ALL logs must be placed here)
├── scripts/                  # Custom utility scripts
└── src/                      # Source code
```

## Key File Locations

| Description | Path |
|-------------|------|
| **V4 Best Model (FP32)** | `experiments/baseline_streaming/exp_v4/epoch-33.pt` |
| **V4 ONNX (FP32)** | `experiments/baseline_streaming/exp_v4/*epoch-33*.onnx` |
| **V4 ONNX (INT8)** | `experiments/baseline_streaming/exp_v4/*epoch-33*.int8.onnx` |
| **Deployable Package** | `nihao-zhenzhen-kws/` |
| Pretrained checkpoint | `icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt` |
| Token vocabulary | `data/lang_partial_tone/tokens.txt` |
| Real human voice test data | `data/all/` |

## Python Environment

The project uses a conda environment named `kws-train`:
- Python: 3.10+
- PyTorch with CUDA support
- k2 (version 1.24.4+)
- Lhotse 1.32.1
- sherpa-onnx

To activate: `conda activate kws-train` or use full path `/data/workspace/llm/anaconda3/envs/kws-train/`

## Common Commands

### Quick Test with Deployable Package

```bash
cd nihao-zhenzhen-kws/examples
python realtime_detection.py
```

### Export V4 Model to ONNX

```bash
cd icefall/egs/wenetspeech/KWS
python ./zipformer/export-onnx-streaming.py \
    --exp-dir /path/to/experiments/baseline_streaming/exp_v4 \
    --tokens /path/to/data/lang_partial_tone/tokens.txt \
    --epoch 33 --avg 1 \
    --chunk-size 16 --left-context-frames 128 \
    --causal 1
```

### Upload to HuggingFace

```bash
cd nihao-zhenzhen-kws
huggingface-cli upload Heehobino/nihao-zhenzhen-kws . --repo-type model
```

## Evaluation Metrics

- **FRR** (False Rejection Rate): The rate at which the model misses the actual keyword
- **FAR** (False Accept Rate): The rate at which the model detects the keyword when not present
- **F1**: Harmonic mean of precision and recall
- **RTF** (Real-Time Factor): Processing time / Audio duration. RTF < 1.0 means real-time capable

**Targets**: FRR < 5%, FAR < 10%, RTF < 1.0

## Important Notes

- **PRODUCTION READY**: V4 streaming model (epoch-33) meets all targets
- **HuggingFace**: Model published at `Heehobino/nihao-zhenzhen-kws`
- **Recommended**: Use FP32 for best accuracy, INT8 for edge/mobile deployment
- **Primary test dataset**: Use `data/all/` (real human voice) for all experiments
- This is a zero-shot KWS project with no real human voice data (only TTS) for training
- **All log files MUST be placed in the `log/` directory with proper naming convention**
