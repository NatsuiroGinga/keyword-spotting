# Scripts Directory

This directory contains utility scripts for the Keyword Spotting project, organized by function.

## Directory Structure

### `data/` - Data Generation and Preparation
- `generate_tts_dataset.py` - Generate positive TTS training data
- `generate_tts_v3_kokoro.py` - Generate V3 positive data with Kokoro TTS
- `generate_negative_tts.py` - Generate negative TTS samples
- `generate_tts_v3_negative.py` - Generate V3 negative samples
- `prepare_lhotse_manifests.py` - Create Lhotse manifests
- `prepare_lhotse_manifests_v3.py` - Create V3 manifests

### `training/` - Model Training
- `run_finetune.sh` - Run fine-tuning (V1)
- `run_finetune_v2.sh` - Run fine-tuning (V2)
- `run_finetune_v3.sh` - Run fine-tuning (V3)
- `run_simple_train_v3.sh` - Simple training script V3
- `run_train_v3.sh` - Training script V3
- `simple_train_v3.py` - Simple Python training V3
- `train_kws_v3.py` - Python training script V3

### `export/` - Model Export
- `export_onnx.sh` - Export model to ONNX (V1)
- `export_onnx_v2.sh` - Export model to ONNX (V2)
- `export_onnx_v3.sh` - Export model to ONNX (V3)

### `eval/` - Evaluation and Optimization
- `analyze_false_positives.py` - Analyze false positive cases
- `compare_inference_modes.py` - Compare Direct vs Delayed inference
- `evaluate_decoy_strategies.py` - Evaluate decoy keyword strategies
- `evaluate_kws_model.py` - Quick model evaluation
- `evaluate_kws_with_rtf.py` - Evaluation with RTF metrics
- `evaluate_with_delay.py` - Delayed decision evaluation
- `find_optimal_threshold.py` - Find optimal detection threshold
- `generate_comparison_charts.py` - Generate comparison charts
- `optimize_decoy_params.py` - Optimize decoy parameters
- `optimize_kws_params.py` - Optimize boost and threshold parameters

### `inference/` - Inference and Testing
- `debug_keyword_result.py` - Debug keyword detection results
- `debug_single_keyword.py` - Debug single keyword detection
- `delayed_decision_inference.py` - Delayed decision state machine
- `test_cuda.py` - Test CUDA acceleration

### `utils/` - Utilities
- `rtf_utils.py` - RTF (Real-Time Factor) calculation utilities

### `tmp/` - Temporary Scripts
Temporary debug and test scripts created during development. See `tmp/README.md` for details.
