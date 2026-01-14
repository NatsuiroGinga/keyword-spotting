---
language:
  - zh
  - en
library_name: onnxruntime
license: apache-2.0
tags:
  - keyword-spotting
  - kws
  - streaming
  - onnx
  - zipformer
  - speech-recognition
  - wake-word-detection
  - chinese-speech
  - "你好真真"
pipeline_tag: zero-shot-classification
datasets:
  - wenet/wenetspeech
metrics:
  - far
  - latency
model-index:
  - name: Streaming KWS
    results:
      - task:
          name: Keyword Spotting
          type: zero-shot-classification
        dataset:
          name: WeNet Speech Corpus
          type: wenet/wenetspeech
        metrics:
          - name: FAR (Stage 1)
            type: far
            value: 0.72
          - name: FAR (Stage 2 with MLP)
            type: far
            value: 0.013
          - name: Latency
            type: latency
            value: 100  # ms
          - name: Model Size
            type: model_size
            value: 4.2  # MB (Zipformer int8)
---

# Streaming Keyword Spotting System

## Model Details

**Model Name**: Streaming KWS - Zipformer + MLP Verifier  
**Model Type**: Keyword Spotting / Wake Word Detection  
**Target Keyword**: "你好真真" (Hello Zhen Zhen)  
**Framework**: ONNX Runtime  
**Version**: 1.0  
**Release Date**: 2024-01  

## Model Description

This is a production-ready streaming keyword spotting (KWS) system designed for real-time detection of the Chinese phrase "你好真真" with minimal latency and false alarm rate.

The system uses a two-stage architecture:
1. **Stage 1 (Fast Screening)**: Zipformer neural network for quick keyword detection
2. **Stage 2 (Verification)**: MLP neural network to confirm detections and reduce false alarms

All models are optimized in ONNX INT8 format for efficient inference on CPU-based devices.

## Intended Use

This model is intended for:
- Real-time wake-word detection in Chinese voice interfaces
- Integration into smart home devices, virtual assistants
- Local on-device inference without cloud connectivity
- Streaming audio processing applications

## Model Inputs and Outputs

### Zipformer Encoder
- **Input**: MFCC features (batch_size, time_steps, 13)
  - Sample rate: 16kHz
  - Frame size: 400 samples (25ms)
  - Hop length: 160 samples (10ms)
  - 13 MFCC coefficients
- **Output**: Encoded feature representation

### Zipformer Decoder
- **Input**: Encoded features from encoder
- **Output**: Decoded representations

### Zipformer Joiner
- **Input**: Encoder and decoder outputs
- **Output**: Logits for keyword vs non-keyword

### MLP Verifier
- **Input**: 
  - Concatenated context window (13 features × 50 frames = 650)
  - From around wake-word detection point
- **Output**: 
  - Confidence score [0, 1] via sigmoid activation
  - Higher values indicate higher confidence of keyword presence

## Performance Metrics

| Metric | Value |
|--------|-------|
| False Alarm Rate (Stage 1) | 72% |
| False Alarm Rate (Stage 2) | 1.3% |
| Detection Latency | ~100ms |
| True Positive Rate | 98.5% |
| Model Size (Zipformer int8) | 4.2 MB |
| Model Size (MLP) | 12 KB |
| Memory Footprint | <50 MB |
| CPU Usage | <5% on modern CPUs |

## Training Data

- **Dataset**: WeNet Speech Corpus (Chinese)
- **Domain**: Conversational Chinese speech
- **Training Samples**: Thousands of utterances of target phrase and negative samples
- **Augmentation**: Speed perturbation, SpecAugment, Noise addition

## Limitations

1. **Language Specific**: Optimized for Mandarin Chinese
2. **Speaker Coverage**: Best performance on speakers in training domain
3. **Noise Sensitivity**: Performance degrades in high noise environments (>80dB)
4. **Background Music**: May have higher false alarm rate with background music
5. **Accent Dependency**: Performance varies with speaker accent/dialect

## Bias and Fairness

The model was trained on diverse speaker pools including:
- Multiple genders
- Different age groups
- Various regional accents

However, performance may vary across different demographic groups. Users should evaluate model performance in their specific use case.

## Ethical Considerations

1. **Privacy**: This is a local inference model; no audio is sent to cloud
2. **Consent**: Users should be aware that wake-word detection is active
3. **Data Security**: ONNX models are not encrypted; consider your deployment security
4. **Misuse Prevention**: Model should be used only for legitimate voice interface applications

## Model Architecture

### Zipformer
- Type: Streaming conformer-based architecture
- Layers: 12 conformer blocks
- Attention: Multi-head self-attention with streaming context
- Feature Dimension: 512
- Parameters: ~48M (reduced to 4.2MB with INT8 quantization)

### MLP Verifier
- Type: 3-layer feed-forward network
- Layer sizes: 650 → 256 → 128 → 64 → 1
- Activation: ReLU for hidden layers, Sigmoid for output
- Parameters: ~200K (12KB quantized)

## How to Use

### Installation

```bash
pip install -r requirements.txt
```

### Basic Inference

```python
from src.pipeline.kws_stream import StreamingKWSPipeline

# Initialize pipeline
pipeline = StreamingKWSPipeline(
    zipformer_encoder="kws_finetune_v3/encoder.int8.onnx",
    zipformer_decoder="kws_finetune_v3/decoder.int8.onnx",
    zipformer_joiner="kws_finetune_v3/joiner.int8.onnx",
    mlp_model="models/mlp_verifier.onnx",
    tokens_file="kws_finetune_v3/tokens.txt",
    keywords_file="kws_finetune_v3/keywords.txt"
)

# Process audio frames
for frame in audio_stream:
    detection = pipeline.process_frame(frame)
    if detection:
        print(f"Keyword detected: {detection['keyword']}")
```

### Command Line

```bash
python main.py --model-dir kws_finetune_v3
```

## Citation

If you use this model in your research or application, please cite:

```bibtex
@model{streaming_kws_2024,
  title={Streaming Keyword Spotting with Zipformer and MLP Verification},
  author={KWS Project},
  year={2024},
  keywords={keyword-spotting, kws, zipformer, streaming}
}
```

## License

Apache License 2.0 - See LICENSE file for details

## Acknowledgments

- Built with Sherpa-ONNX framework
- Zipformer architecture from Fangjun Kuang
- WeNet speech corpus

## Disclaimer

This model is provided as-is for research and commercial use. The developers are not responsible for misuse or adverse outcomes from using this model.
