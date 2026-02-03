---
language:
  - zh
license: mit
library_name: sherpa-onnx
tags:
  - keyword-spotting
  - wake-word
  - speech-recognition
  - onnx
  - zipformer
  - streaming
  - int8
  - quantization
pipeline_tag: audio-classification
metrics:
  - f1
  - accuracy
---

# 你好真真 - 中文关键词检测模型

中文唤醒词/关键词检测模型，用于检测 "你好真真" (Ni Hao Zhen Zhen) 关键词。

## 模型变体

提供FP32和INT8两个版本：

| 版本 | 大小 | 阈值 | F1 | FAR | FRR | RTF |
|------|------|------|-----|-----|-----|-----|
| **FP32** (默认) | 13MB | 0.52 | 89.47% | 5.64% | 2.30% | 0.0185 |
| **INT8** | 5MB | 0.46 | 89.01% | 5.96% | 2.30% | 0.0140 |

- **FP32**: 精度更高，推荐用于服务器端
- **INT8**: 体积小62%，速度快24%，推荐用于嵌入式/移动端

## 安装

```bash
pip install -r requirements.txt
```

或手动安装依赖：

```bash
pip install sherpa-onnx sounddevice soundfile numpy
```

## 快速开始

### 实时麦克风检测

```bash
cd examples
python realtime_detection.py
```

运行后对着麦克风说 "你好真真" 即可看到检测结果。

### Python API

```python
from inference import load_model

# 加载FP32模型（默认）
detector = load_model()

# 或加载INT8模型（更小更快）
detector = load_model(variant="int8")

# 检测音频文件
result = detector.detect("path/to/audio.wav")
print(f"检测到关键词: {result['detected']}")
print(f"关键词: {result['keyword']}")
```

### 流式检测

```python
import numpy as np
from inference import load_model

detector = load_model()
stream = detector.create_stream()

# 模拟流式音频输入
chunk_size = 1600  # 100ms @ 16kHz
audio_data = np.random.randn(chunk_size).astype(np.float32)

stream.accept_waveform(16000, audio_data.tolist())
while detector._kws.is_ready(stream):
    detector._kws.decode_stream(stream)

result = detector._kws.get_result(stream)
print(result)
```

## 目录结构

```
nihao-zhenzhen-kws/
├── model/                    # FP32模型 (13MB)
│   ├── encoder.onnx
│   ├── decoder.onnx
│   ├── joiner.onnx
│   ├── tokens.txt
│   └── keywords.txt
├── model_int8/               # INT8模型 (5MB)
│   ├── encoder.onnx
│   ├── decoder.onnx
│   ├── joiner.onnx
│   ├── tokens.txt
│   └── keywords.txt
├── examples/
│   └── realtime_detection.py
├── config.json
├── inference.py
├── requirements.txt
└── README.md
```

## 性能指标

在49个真实人声测试样本上的评估结果：

### FP32模型
- **F1分数**: 89.47%
- **准确率**: 95.07%
- **误报率 (FAR)**: 5.64%
- **漏检率 (FRR)**: 2.30%
- **RTF**: 0.0185

### INT8模型
- **F1分数**: 89.01%
- **准确率**: 94.33%
- **误报率 (FAR)**: 5.96%
- **漏检率 (FRR)**: 2.30%
- **RTF**: 0.0140

## License

MIT License

## 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - 语音识别推理框架
- [icefall](https://github.com/k2-fsa/icefall) - 模型训练框架
