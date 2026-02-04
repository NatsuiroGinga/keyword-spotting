# Keyword Spotting System

中文唤醒词 **「你好真真」** 实时识别系统，基于 Icefall 和 Sherpa-ONNX 的两阶段检测方案。

## 项目简介

这是一个零样本关键词检测（KWS）项目，在没有真实人声训练数据的情况下，通过合成 TTS 数据和迁移学习训练离线关键词识别模型。

### 关键指标

| 方案 | FRR | FAR | 准确率 | F1 | 延迟 | RTF |
|------|-----|-----|--------|----|----|----|
| Baseline (仅V3) | 0.00% | 72.22% | 42.98% | 42.48% | 33ms | 0.0182 |
| V3 + MLP | 0.00% | 1.30% | 98.98% | 97.63% | 46ms | 0.0249 |
| **V4 Streaming (FP32)** | **2.30%** | **5.64%** | **95.07%** | **89.47%** | - | 0.0185 |
| V4 Streaming (INT8) | 2.30% | 5.96% | 94.33% | 89.01% | - | 0.0140 |

**核心成果**:
- V3 + MLP: FAR 从 72.22% 降至 1.30%（降低 70.93%），同时保持 FRR 为 0%
- V4 Streaming: 单阶段流式检测，支持 FP32/INT8 量化，INT8 体积减小 62%，速度提升 24%

**推荐**: V4 Streaming (epoch-33) - 单阶段，实时性好，已部署到 HuggingFace

### 技术栈

- **Icefall**: 基于 k2 和 Lhotse 的下一代 Kaldi 训练框架
- **Sherpa-onnx**: 基于 ONNX 的语音模型部署框架
- **Zipformer**: 神经架构，参数量 3.3M
- **Edge-TTS**: 微软 Azure 的文本转语音（合成训练数据）
- **Lhotse**: 语音数据处理库

## 项目结构

```
keyword-spotting/
├── main.py                   # 主程序入口
├── requirements.txt          # Python 依赖
├── README.md                 # 本文件
├── CLAUDE.md                 # 项目开发指南（详细技术文档）
├── QUICK_START.md            # HuggingFace 快速部署指南
├── DEPLOYMENT_SUMMARY.md     # 部署总结
│
├── data/                     # 数据目录
│   ├── all/                  # 真人语音测试数据 (406 个文件)
│   ├── manifests/            # Lhotse 数据清单
│   ├── raw_tts/              # 合成 TTS 音频
│   └── lang_partial_tone/    # 词汇表（拼音+声调）
│
├── scripts/                  # 脚本目录
│   ├── data/                 # 数据生成脚本
│   ├── training/             # 模型训练脚本
│   ├── export/               # 模型导出脚本
│   ├── eval/                 # 评估和优化脚本
│   ├── inference/            # 推理和测试脚本
│   ├── utils/                # 工具函数
│   └── tmp/                  # 临时调试脚本
│
├── src/                      # 源代码
│   ├── audio/                # 音频采集和处理
│   ├── models/               # 模型定义
│   ├── pipeline/             # 流式 KWS 管道
│   └── utils/                # 配置和工具
│
├── doc/                      # 文档目录
│   ├── keyword_spotting_guide.md    # KWS 指南
│   ├── sherpa_onnx_installation.md  # 安装说明
│   ├── hf_upload_guide.md          # HF 上传指南
│   └── windows_deployment.md       # Windows 部署指南
│
├── exp/                      # 旧实验输出
│   └── kws_finetune_v3/      # V3 模型（30 epochs）
│
├── experiments/              # 实验目录
│   ├── baseline_streaming/   # V4 流式模型实验 (推荐)
│   │   └── exp_v4/          # 最佳模型: epoch-33
│   └── multi_stage_ablation/ # V3 + MLP 多阶段检测实验
│       ├── models/           # 训练好的验证器模型
│       └── results/          # 实验结果
│
├── nihao-zhenzhen-kws/       # 可部署包 (HuggingFace)
│   ├── model/                # FP32 模型 (13MB)
│   ├── model_int8/           # INT8 模型 (5MB)
│   ├── inference.py          # Python 推理接口
│   └── examples/             # 示例代码
├── icefall/                  # Icefall 框架
│   └── egs/wenetspeech/KWS/zipformer/
│
├── log/                      # 日志目录
├── plan/                     # 实现计划文档
├── report/                   # 实验报告
└── tools/                    # 部署工具
```

## 快速开始

### 使用可部署包（推荐）

```bash
cd nihao-zhenzhen-kws/examples
python realtime_detection.py
```

### Python API

```python
from inference import load_model

# 加载 FP32 模型（默认）
detector = load_model()

# 或加载 INT8 模型（更小更快）
detector = load_model(variant="int8")

# 检测音频
result = detector.detect("audio.wav")
```

### 训练环境设置

```bash
# 创建 conda 环境
conda create -n kws-train python=3.10 -y
conda activate kws-train

# 安装依赖
pip install -r requirements.txt

# 设置 PythonPath
export PYTHONPATH=/path/to/keyword-spotting/icefall:$PYTHONPATH
```

### 训练新模型

```bash
# 1. 生成合成训练数据
python scripts/data/generate_tts_v3_kokoro.py
python scripts/data/generate_tts_v3_negative.py

# 2. 生成 Lhotse 清单
python scripts/data/prepare_lhotse_manifests_v3.py

# 3. 训练模型
bash scripts/training/run_finetune_v3.sh

# 4. 导出 ONNX 模型
bash scripts/export/export_onnx_v3.sh
```

### 评估模型

```bash
# 快速评估
python scripts/eval/evaluate_kws_model.py \
  --model-dir exp/kws_finetune_v3 \
  --positive-dir /path/to/positive \
  --negative-dir /path/to/negative

# 参数优化
python scripts/eval/optimize_kws_params.py \
  --model-dir exp/kws_finetune_v3 \
  --target-frr 1.39 \
  --target-far 7.46
```

## 核心问题与解决方案

### 核心问题

模型无法有效区分 **「你好」** 和 **「你好真真」**，因为：

1. **声学相似**: "你好真真" 以 "你好" 开头，声学特征几乎相同
2. **序列差异小**: 仅相差 4 个音素 (zh ēn zh ēn)
3. **预训练偏倚**: WenetSpeech 训练数据中包含大量「你好」，但几乎没有「你好真真」

### 解决方案：多阶段检测

采用两阶段检测架构，成功解决前缀匹配问题：

| 阶段 | 方法 | 作用 |
|------|------|---|
| Stage 1 | Zipformer V3 | 快速筛选（高召回率，高 FAR 72%） |
| Stage 2 | MLP/CNN 验证器 | 后缀验证（将 FAR 降至 1.3%） |

**推荐配置**: V3 + MLP Verifier

## 文档

- **[CLAUDE.md](./CLAUDE.md)** - 项目开发指南（详细技术文档和常见命令）
- **[QUICK_START.md](./QUICK_START.md)** - HuggingFace 快速部署指南
- **[DEPLOYMENT_SUMMARY.md](./DEPLOYMENT_SUMMARY.md)** - 部署总结
- **[doc/hf_upload_guide.md](./doc/hf_upload_guide.md)** - HuggingFace 上传指南
- **[doc/windows_deployment.md](./doc/windows_deployment.md)** - Windows 部署指南
- **[doc/sherpa_onnx_installation.md](./doc/sherpa_onnx_installation.md)** - Sherpa-ONNX 安装说明

## 测试数据

### 主要测试集（真人语音）

**路径**: `data/all/` (来自 `data/kws-data-all.zip`)

| 类别 | 数量 | 描述 |
|------|------|------|
| 正样本 (你好真真) | 63 | 真人语音录制的 "你好真真" |
| 相似词 (你好珍珍/娟娟) | 28 | 发音相似的关键词（应被拒绝） |
| 负样本 | 315 | 其他命令和短语 |
| **总计** | **406** | 真实场景测试样本 |

### 训练数据统计

| 类别 | 数量 | 路径 |
|------|------|------|
| 正样本 | 539 | `data/raw_tts/nihao_zhenzhen/*.wav` |
| 负样本 | 873 | `data/raw_tts/negative/*.wav` |
| **总计** | **1,412** | `data/manifests/` |

**负样本构成**:
- 困难负样本: 810 个 - 「你好」、「您好」、「你好啊」、「您好啊」及谐音如「泥豪」、「李浩」
- 普通负样本: 63 个 - 随机中文短语和问候语

**正负样本比例**: 1:1.6

## 模型导出与部署

### 可部署包 (V4 Streaming)

已准备好完整部署包：`nihao-zhenzhen-kws/`

**HuggingFace**: https://huggingface.co/Heehobino/nihao-zhenzhen-kws

上传到 HuggingFace:
```bash
cd nihao-zhenzhen-kws
huggingface-cli upload Heehobino/nihao-zhenzhen-kws . --repo-type model
```

### 训练模型导出

```bash
# 导出 V4 模型为 ONNX
cd icefall/egs/wenetspeech/KWS
python ./zipformer/export-onnx-streaming.py \
    --exp-dir /path/to/experiments/baseline_streaming/exp_v4 \
    --tokens /path/to/data/lang_partial_tone/tokens.txt \
    --epoch 33 --avg 1 \
    --chunk-size 16 --left-context-frames 128 \
    --causal 1
```

## 许可证

Apache License 2.0

## 相关链接

- [Icefall Documentation](https://github.com/k2-fsa/icefall)
- [Sherpa-ONNX Documentation](https://k2-fsa.github.io/sherpa/onnx/)
- [HuggingFace Model](https://huggingface.co/Heehobino/nihao-zhenzhen-kws)
- [GitHub Repository](https://github.com/NatsuiroGinga/keyword-spotting)
