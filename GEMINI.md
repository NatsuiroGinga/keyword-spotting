# GEMINI.md

This file provides context for the Gemini CLI agent when working with this repository.

## Project Overview

This is a **Keyword Spotting (KWS)** model training and fine-tuning project for the Chinese wake word **"你好真真"** (Ni Hao Zhen Zhen). Operating in a zero-shot scenario (no real human voice training data), the project uses synthetic TTS data generation combined with transfer learning to train an offline keyword recognition model using the Icefall framework and Sherpa-onnx for deployment.

### Key Technologies

- **Icefall**: Next-Gen Kaldi training framework using k2 and Lhotse for speech recognition training
- **Sherpa-onnx**: ONNX-based deployment framework for speech models
- **Zipformer**: Neural architecture for speech recognition (3.3M parameters)
- **Edge-TTS**: Microsoft Azure's text-to-speech for synthetic training data
- **Lhotse**: Speech data processing library for manifests and augmentation
- **PyTorch**: Deep learning framework with FP16 mixed precision training

### Project Type

**Code Project** - This is a complete machine learning pipeline for training, fine-tuning, and deploying a keyword spotting model.

## Directory Structure

```
/data/workspace/llm/keyword-spotting/
├── icefall/                           # Icefall framework (K2 training scripts)
│   └── egs/wenetspeech/KWS/zipformer/ # KWS recipe files
│       ├── finetune.py                # Fine-tuning script
│       ├── export-onnx-streaming.py   # ONNX export script
│       ├── asr_datamodule.py          # Data loading and preprocessing
│       ├── model.py                   # Model definition
│       ├── zipformer.py               # Zipformer encoder
│       ├── decoder.py                 # Predictor/Decoder
│       └── joiner.py                  # Joiner network
├── exp/kws_finetune/                  # Experiment outputs
│   ├── best-train-loss.pt             # Best training checkpoint
│   ├── best-valid-loss.pt             # Best validation checkpoint
│   ├── epoch-*.pt                     # Epoch checkpoints
│   ├── encoder-*.onnx                 # Exported ONNX encoder
│   ├── decoder-*.onnx                 # Exported ONNX decoder
│   ├── joiner-*.onnx                  # Exported ONNX joiner
│   ├── keywords.txt                   # Keyword configuration
│   └── tokens.txt                     # Token vocabulary
├── data/                              # Data directories
│   ├── manifests/                     # Lhotse data manifests
│   │   ├── kws_recordings.jsonl.gz
│   │   ├── kws_supervisions.jsonl.gz
│   │   └── kws_cuts.jsonl.gz
│   ├── raw_tts/                       # Synthetic TTS audio
│   │   └── positive/                  # Positive samples (wake word)
│   └── lang_partial_tone/             # Token vocabulary (pinyin-based)
│       ├── tokens.txt
│       └── words.txt
├── scripts/                           # Custom utility scripts
│   ├── generate_tts_dataset.py        # Generate TTS training data
│   ├── prepare_lhotse_manifests.py    # Create Lhotse manifests
│   ├── run_finetune.sh                # Training shell script
│   ├── export_onnx.sh                 # Export and quantize models
│   ├── evaluate_kws_model.py          # Evaluate model performance
│   ├── find_optimal_threshold.py      # Find detection threshold
│   └── analyze_false_positives.py     # Analyze false positives
├── icefall-kws-zipformer-wenetspeech-20240219/ # Pretrained model
│   └── exp/pretrained.pt
└── doc.md, plan.md, CLAUDE.md         # Documentation files
```

## Building and Running

### Prerequisites

- Python 3.10+ with conda environment `kws-train`
- PyTorch with CUDA support
- k2 library (version 1.24.4+)
- Lhotse 1.32.1
- sherpa-onnx
- Edge-TTS

### Full Training Pipeline

```bash
# 1. Generate synthetic training data with Edge-TTS
python scripts/generate_tts_dataset.py

# 2. Create Lhotse manifests from WAV files
python scripts/prepare_lhotse_manifests.py

# 3. Fine-tune the model (uses kws-train conda environment)
bash scripts/run_finetune.sh

# 4. Export to ONNX with INT8 quantization
bash scripts/export_onnx.sh

# 5. Evaluate model performance
python scripts/evaluate_kws_model.py --threshold 0.0

# 6. Find optimal detection threshold
python scripts/find_optimal_threshold.py
```

### Individual Component Commands

#### Data Generation
```bash
python scripts/generate_tts_dataset.py
```
- Uses Edge-TTS with multiple Chinese voices (zh-CN, zh-TW)
- Generates prosody variations (rate: -30% to +30%, pitch: -15Hz to +15Hz)
- Outputs to `data/raw_tts/positive/`
- Creates `metadata.json` with generation details

#### Manifest Preparation
```bash
python scripts/prepare_lhotse_manifests.py
```
- Scans `data/raw_tts/positive/` directory
- Creates Lhotse RecordingSet and SupervisionSet
- Outputs compressed JSONL files to `data/manifests/`

#### Model Fine-tuning
```bash
bash scripts/run_finetune.sh
```
Key parameters from the script:
- Pretrained checkpoint: `icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt`
- Epochs: 10
- Base learning rate: 0.0005
- FP16 mixed precision: enabled
- SpecAugment: enabled
- MUSAN augmentation: disabled (`--enable-musan 0`)
- Chunk size: 16, Left context: 128 (streaming)

Manual fine-tuning command:
```bash
export PYTHONPATH=/data/workspace/llm/keyword-spotting/icefall:$PYTHONPATH
/data/workspace/llm/anaconda3/envs/kws-train/bin/python \
    ./icefall/egs/wenetspeech/KWS/zipformer/finetune.py \
    --world-size 1 \
    --num-epochs 10 \
    --start-epoch 1 \
    --exp-dir exp/kws_finetune \
    --lang-dir data/lang_partial_tone \
    --manifest-dir data/manifests \
    --pinyin-type partial_with_tone \
    --use-fp16 1 \
    --use-mux 0 \
    --use-custom-kws-data 1 \
    --on-the-fly-feats 1 \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --finetune-ckpt icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt
```

#### ONNX Export
```bash
bash scripts/export_onnx.sh
```
- Exports encoder, decoder, and joiner to ONNX format
- Quantizes models to INT8 using onnxruntime
- Creates `keywords.txt` with target wake word configuration
- Outputs to `exp/kws_finetune/`

#### Model Evaluation
```bash
python scripts/evaluate_kws_model.py \
    --model-dir exp/kws_finetune \
    --use-int8 \
    --positive-dir /path/to/positive/test/data \
    --negative-dir /path/to/negative/test/data \
    --threshold 0.0
```

## Development Conventions

### Model Architecture

**Zipformer-Transducer (3.3M parameters):**
- 6 encoder layers with downsampling stacks
- Causal architecture for streaming support
- Chunk size: 16 frames, Left context: 128 frames
- Encoder dim: 128, Decoder dim: 320, Joiner dim: 320
- Num encoder layers: 1,1,1,1,1,1
- Feedforward dim: 192,192,192,192,192,192

### Tokenization Strategy

The model uses **pinyin-based tokenization with partial tones**. Tokens consist of:
- Initials (声母): zh, ch, sh, n, h, etc.
- Finals with tone marks (韵母+声调): ǐ, ǎo, ēn, etc.

Example conversion: "你好真真" → `n ǐ h ǎo zh ēn zh ēn`

This is defined in `keywords.txt` as:
```
n ǐ h ǎo zh ēn zh ēn @你好真真
```

### Training Configuration

From `scripts/run_finetune.sh`:
- Base learning rate: 0.0005
- Epochs: 10
- FP16 mixed precision training
- On-the-fly feature extraction
- SpecAugment enabled
- Data augmentation (MUSAN) disabled by default
- No data mixing (use-mux: 0) - uses only custom KWS data

### Python Environment

The project uses a conda environment named `kws-train`:
- Full path: `/data/workspace/llm/anaconda3/envs/kws-train/`
- Python: 3.10+
- Key packages: k2, lhotse, sherpa-onnx, torch, edge-tts

Activate with:
```bash
conda activate kws-train
# or use full path:
export PATH=/data/workspace/llm/anaconda3/envs/kws-train/bin:$PATH
```

### PYTHONPATH Configuration

The Icefall framework must be in the Python path:
```bash
export PYTHONPATH=/data/workspace/llm/keyword-spotting/icefall:$PYTHONPATH
```

### Key File Locations

| Description | Path |
|-------------|------|
| Pretrained checkpoint | `icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt` |
| Training script | `icefall/egs/wenetspeech/KWS/zipformer/finetune.py` |
| Data module | `icefall/egs/wenetspeech/KWS/zipformer/asr_datamodule.py` |
| Token vocabulary | `data/lang_partial_tone/tokens.txt` |
| Lhotse manifests | `data/manifests/kws_*.jsonl.gz` |
| Model outputs | `exp/kws_finetune/` |
| ONNX export script | `icefall/egs/wenetspeech/KWS/zipformer/export-onnx-streaming.py` |

### Evaluation Metrics

- **FRR** (False Rejection Rate): The rate at which the model misses the actual keyword
- **FAR** (False Accept Rate): The rate at which the model detects the keyword when not present
- **Target**: FAR < 1/hour, FRR < 5%

### Important Notes

1. **Zero-shot KWS**: This project has no real human voice data; all training data is synthetic TTS
2. **Pretrained Model**: Fine-tuned from a pre-trained WenetSpeech model
3. **TTS Generation**: Uses Edge-TTS with multiple Chinese voices and prosody variations
4. **INT8 Quantization**: Reduces model size by ~75% (3.3M → ~1M parameters)
5. **Streaming Support**: Model is causal and supports streaming inference
6. **Pinyin Tokenization**: Model operates on pinyin tokens, not characters directly

### Code Style

- Python code follows standard conventions (similar to PEP 8)
- Shell scripts use Bash
- Function names and variables use snake_case
- Logging is used throughout for training progress monitoring

### Testing

No automated tests are configured in this project. Evaluation is performed manually using:
- `scripts/evaluate_kws_model.py` - Tests on positive/negative audio datasets
- `scripts/find_optimal_threshold.py` - Searches for optimal detection threshold
- Manual inference using sherpa-onnx Python API

### Data Flow

1. **TTS Generation**: Edge-TTS generates WAV files with various voice and prosody combinations
2. **Manifest Creation**: Lhotse scans WAV files and creates RecordingSet/SupervisionSet
3. **Feature Extraction**: On-the-fly Fbank feature extraction during training
4. **Training**: Zipformer-Transducer model fine-tuned on synthetic data
5. **Export**: Model exported to ONNX and quantized to INT8
6. **Deployment**: sherpa-onnx runtime uses ONNX models for keyword detection

### Dependencies

Core Python dependencies from `icefall/requirements.txt`:
- k2 (speech recognition framework)
- lhotse (speech data processing)
- kaldifst, kaldilm, kaldialign (Kaldi utilities)
- sentencepiece (tokenizer)
- pypinyin==0.50.0 (Chinese pinyin conversion)
- tensorboard (logging)
- onnx, onnxruntime (ONNX export and runtime)
- sherpa-onnx (deployment framework)
