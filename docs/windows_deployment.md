# Windows平台部署指南

本文档介绍如何在Windows系统上部署和运行"你好真真"流式关键词识别系统。

## 目录

1. [系统要求](#系统要求)
2. [环境准备](#环境准备)
3. [安装依赖](#安装依赖)
4. [模型准备](#模型准备)
5. [运行程序](#运行程序)
6. [配置说明](#配置说明)
7. [常见问题](#常见问题)

---

## 系统要求

### 硬件要求
- **CPU**: x86_64架构，支持SSE4.2指令集
- **内存**: 至少2GB可用内存
- **麦克风**: 支持16kHz采样率的音频输入设备

### 软件要求
- **操作系统**: Windows 10/11 (64位)
- **Python**: 3.8 - 3.11 (推荐3.10)
- **Visual C++ Redistributable**: 2019或更高版本

---

## 环境准备

### 1. 安装Python

1. 下载Python安装包：https://www.python.org/downloads/windows/
2. 运行安装程序，**勾选"Add Python to PATH"**
3. 验证安装：
   ```powershell
   python --version
   # 应显示: Python 3.10.x
   ```

### 2. 安装Visual C++ Redistributable

如果运行时出现DLL缺失错误，请安装：
- 下载地址：https://aka.ms/vs/17/release/vc_redist.x64.exe
- 运行安装程序并重启电脑

### 3. 创建虚拟环境（推荐）

```powershell
# 进入项目目录
cd keyword-spotting

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\activate

# 验证激活成功（命令行前缀应显示 (venv)）
```

---

## 安装依赖

### 方式一：使用requirements.txt（推荐）

```powershell
# 确保已激活虚拟环境
pip install -r requirements.txt
```

### 方式二：手动安装

```powershell
# 核心依赖
pip install numpy onnx onnxruntime

# 音频处理
pip install sounddevice librosa

# sherpa-onnx关键词识别引擎
pip install sherpa-onnx
```

### 验证安装

```powershell
python -c "import sherpa_onnx; import sounddevice; import librosa; print('依赖安装成功!')"
```

---

## 模型准备

### 模型文件结构

确保以下模型文件存在：

```
keyword-spotting/
├── exp/kws_finetune_v3/
│   ├── encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx  # ~2.3MB
│   ├── decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx  # ~0.5MB
│   ├── joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx   # ~1.5MB
│   ├── tokens.txt                                          # 词表
│   └── keywords.txt                                        # 关键词配置
└── models/
    └── mlp_verifier.onnx                                   # ~12KB
```

### 生成MLP验证器ONNX模型

如果`models/mlp_verifier.onnx`不存在，需要先导出：

```powershell
# 需要安装PyTorch
pip install torch

# 导出模型
python tools/export_mlp_onnx.py
```

### 配置关键词文件

编辑`exp/kws_finetune_v3/keywords.txt`，确保包含目标关键词：

```
你 好 真 真 :1.5 #0.25
```

格式说明：
- `你 好 真 真`: 关键词的token序列（空格分隔）
- `:1.5`: 加分权重（可选，默认1.0）
- `#0.25`: 触发阈值（可选，默认0.25）

---

## 运行程序

### 基本用法

```powershell
# 使用默认配置运行
python main.py --model-dir exp/kws_finetune_v3
```

### 查看可用音频设备

```powershell
python main.py --list-devices
```

输出示例：
```
可用音频设备:
   0 Microsoft Sound Mapper - Input, MME (2 in, 0 out)
>  1 麦克风 (Realtek Audio), MME (2 in, 0 out)
   2 Microsoft Sound Mapper - Output, MME (0 in, 2 out)
```

### 指定音频设备

```powershell
# 使用设备索引1
python main.py --model-dir exp/kws_finetune_v3 --device 1
```

### 调整检测灵敏度

```powershell
# 降低误报率（提高阈值）
python main.py --model-dir exp/kws_finetune_v3 --kws-threshold 0.35 --mlp-threshold 0.6

# 提高检测率（降低阈值）
python main.py --model-dir exp/kws_finetune_v3 --kws-threshold 0.2 --mlp-threshold 0.4
```

### 禁用MLP二阶段验证

```powershell
python main.py --model-dir exp/kws_finetune_v3 --no-mlp
```

### 完整参数列表

```powershell
python main.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir` | `exp/kws_finetune_v3` | 模型目录路径 |
| `--mlp-model` | `models/mlp_verifier.onnx` | MLP验证器路径 |
| `--keywords` | `你好真真` | 关键词列表 |
| `--kws-score` | `1.5` | 关键词加分权重 |
| `--kws-threshold` | `0.25` | KWS触发阈值 |
| `--mlp-threshold` | `0.5` | MLP验证阈值 |
| `--no-mlp` | `False` | 禁用MLP验证 |
| `--device` | `None` | 音频设备索引 |
| `--num-threads` | `2` | 推理线程数 |
| `--provider` | `cpu` | 计算提供者 |

---

## 配置说明

### 阈值调优指南

| 场景 | KWS阈值 | MLP阈值 | 说明 |
|------|---------|---------|------|
| 高灵敏度 | 0.15-0.20 | 0.3-0.4 | 容易触发，可能有误报 |
| 平衡模式 | 0.25-0.30 | 0.5-0.6 | 推荐设置 |
| 低误报 | 0.35-0.45 | 0.7-0.8 | 减少误报，可能漏检 |

### 性能优化

1. **减少延迟**：降低`chunk_duration_ms`（默认100ms）
2. **降低CPU占用**：减少`num_threads`（默认2）
3. **GPU加速**：安装`onnxruntime-gpu`并设置`--provider cuda`

---

## 常见问题

### Q1: 提示"No module named 'sherpa_onnx'"

**解决方案**：
```powershell
pip install sherpa-onnx
```

### Q2: 提示"No input device"或无法检测到麦克风

**解决方案**：
1. 检查麦克风是否正确连接
2. 在Windows设置中确认麦克风权限已开启
3. 使用`--list-devices`查看可用设备
4. 尝试指定设备索引：`--device 0`

### Q3: 运行时出现DLL缺失错误

**解决方案**：
1. 安装Visual C++ Redistributable 2019+
2. 重启电脑后重试

### Q4: 检测不到关键词

**解决方案**：
1. 确认麦克风正常工作（可用其他录音软件测试）
2. 降低检测阈值：`--kws-threshold 0.15`
3. 确认`keywords.txt`中的关键词配置正确
4. 尝试禁用MLP验证：`--no-mlp`

### Q5: 误报率太高

**解决方案**：
1. 提高KWS阈值：`--kws-threshold 0.35`
2. 提高MLP阈值：`--mlp-threshold 0.7`
3. 确保MLP验证器已启用（不使用`--no-mlp`）

### Q6: CPU占用过高

**解决方案**：
1. 减少推理线程：`--num-threads 1`
2. 使用int8量化模型（已默认使用）

---

## 开发者信息

### 项目结构

```
keyword-spotting/
├── main.py                 # 主程序入口
├── requirements.txt        # Python依赖
├── src/
│   ├── audio/
│   │   ├── capture.py      # 麦克风采集
│   │   └── feature.py      # 特征提取
│   ├── models/
│   │   └── mlp_verifier.py # MLP验证器
│   ├── pipeline/
│   │   └── kws_stream.py   # 流式KWS管道
│   └── utils/
│       └── config.py       # 配置管理
├── tools/
│   └── export_mlp_onnx.py  # ONNX导出工具
├── models/                 # 模型文件
└── docs/                   # 文档
```

### API使用示例

```python
from src.utils.config import KWSConfig
from src.pipeline.kws_stream import StreamingKWSPipeline

# 创建配置
config = KWSConfig(
    encoder_path="exp/kws_finetune_v3/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    decoder_path="exp/kws_finetune_v3/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    joiner_path="exp/kws_finetune_v3/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    tokens_path="exp/kws_finetune_v3/tokens.txt",
    keywords_file="exp/kws_finetune_v3/keywords.txt",
    mlp_model_path="models/mlp_verifier.onnx",
)

# 创建管道
pipeline = StreamingKWSPipeline(config)
pipeline.load()

# 设置检测回调
def on_detection(result):
    print(f"检测到: {result.keyword}")

pipeline.set_on_detection(on_detection)

# 处理音频块
import numpy as np
audio_chunk = np.random.randn(1600).astype(np.float32)  # 100ms @ 16kHz
result = pipeline.process_chunk(audio_chunk)
```

---

## 技术支持

如遇到问题，请检查：
1. Python版本是否在3.8-3.11之间
2. 所有依赖是否正确安装
3. 模型文件是否完整
4. 麦克风是否正常工作

如问题仍未解决，请提供以下信息：
- Windows版本
- Python版本
- 完整错误信息
- 使用的命令行参数
