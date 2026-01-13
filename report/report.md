# KWS模型微调优化现状报告

**生成日期**: 2026-01-12
**项目目录**: `/data/workspace/llm/keyword-spotting`
**目标关键词**: "你好真真" (Ni Hao Zhen Zhen)

---

## 一、项目目标

开发一个关键词检测（KWS）模型，用于检测唤醒词"你好真真"。

### 性能目标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| FRR (False Rejection Rate) | < 1.39% | 漏检率，目标唤醒词被错误拒绝的比例 |
| FAR (False Accept Rate) | < 7.46% | 误检率，非唤醒词被错误接受的比例 |

### 技术约束

- **零样本场景**：没有真实人声训练数据，只能使用TTS合成数据
- **目标设备**：边缘设备，模型需要INT8量化
- **模型架构**：Zipformer-Transducer (3.3M参数)
- **推理框架**：sherpa-onnx

---

## 二、当前状态

### 2.1 已完成的工作

| 阶段 | 描述 | 状态 |
|------|------|------|
| V1模型训练 | 仅使用正样本训练 | 完成，效果差 |
| 负样本生成 | 生成873个负样本（TTS合成） | 完成 |
| V2模型训练 | 使用正负样本混合训练 | 完成 |
| 参数优化 | 网格搜索boost和threshold | 完成 |
| Decoy方案尝试 | 使用诱词过滤误检 | 失败 |

### 2.2 当前最佳模型性能

**模型目录**: `exp/kws_finetune_v2/`

| 配置 | FRR | FAR | 召回率 |
|------|-----|-----|--------|
| boost=0.3, threshold=0.6 | 0.00% | 53.70% | 100% |
| boost=0.3, threshold=0.7 | 3.47% | 47.41% | 96.53% |
| **boost=1.0, threshold=0.65** | **12.50%** | **34.26%** | 87.50% |
| boost=1.5, threshold=0.7 | 29.86% | 20.37% | 70.14% |

**关键发现**：
- **没有任何参数配置能同时满足FRR<1.39%和FAR<7.46%的目标**
- FRR和FAR呈现强负相关：降低FRR会显著提高FAR，反之亦然
- 最接近目标的配置FRR=12.50%、FAR=34.26%，两个指标都大幅超标

### 2.3 测试数据集

**路径**: `/data/workspace/llm/audio-classification/dataset/kws_test_data_merged/`

| 类别 | 数量 | 说明 |
|------|------|------|
| Positive | 144 | TTS生成的"你好真真"样本（8种声音 × 6种SNR × 3种韵律变化） |
| Negative | 540 | 负样本，包含84个"你好"变体（泥豪、李浩等） |

---

## 三、根本问题分析

### 3.1 核心问题

**模型无法有效区分"你好"和"你好真真"**

这是因为：

1. **声学高度相似**："你好真真"的前半部分就是"你好"，声学特征几乎一致
2. **序列长度差异小**：两者只差4个音素（zh ēn zh ēn）
3. **预训练模型偏差**：WenetSpeech预训练模型见过大量"你好"，但几乎没见过"你好真真"

### 3.2 负样本构成分析

540个负样本中：

| 类型 | 数量 | 说明 |
|------|------|------|
| "你好"及其变体 | 84 | 包括"泥豪"、"李浩"等谐音 |
| 其他负样本 | 456 | 随机语音、环境噪声等 |

大部分FAR来自对"你好"变体的误检。

### 3.3 Decoy方案失败原因

尝试注册"你好"、"你好啊"、"您好"等作为诱词（decoy），期望模型检测到短的"你好"时不触发。

**失败原因**：
- 当音频说"你好真真"时，"你好"这个更短的序列会**先被匹配**
- sherpa-onnx返回第一个匹配的关键词
- 由于"你好"是"你好真真"的前缀，它总是会先触发
- 导致所有正样本都被错误地识别为decoy "你好"

**调试验证**（来自`scripts/debug_keyword_result.py`输出）：
```
Testing POSITIVE samples (应该检测为"你好真真"):
  Audio: positive_0000_*.wav
  Detected: [你好]  <-- 错误！应该是"你好真真"

Testing NEGATIVE samples (应该检测为"你好"或不检测):
  Audio: expanded_negative_0000_*.wav
  Detected: [你好真真]  <-- 这才是FAR的来源
```

---

## 四、训练数据现状

### 4.1 训练数据统计

| 类别 | 数量 | 路径 |
|------|------|------|
| 正样本 | 539 | `data/raw_tts/nihao_zhenzhen/*.wav` |
| 负样本 | 873 | `data/raw_tts/negative/*.wav` |
| **总计** | **1412** | `data/manifests/` |

### 4.2 负样本构成

**硬负样本（Hard Negatives）**: 810个
- "你好"、"您好"、"你好啊"、"你好吗"
- 谐音词："泥豪"、"李浩"、"倪好"等
- 8种TTS声音 × 多种韵律变化

**通用负样本（General Negatives）**: 63个
- 随机中文短句
- 常见问候语（非"你好"相关）

### 4.3 Lhotse Manifests

```bash
# Manifest文件位置
data/manifests/kws_recordings_train_merged.jsonl.gz
data/manifests/kws_supervisions_train_merged.jsonl.gz

# 训练数据正负样本比例
正样本:负样本 ≈ 1:1.6
```

---

## 五、模型架构与训练配置

### 5.1 模型架构

```python
# Zipformer-Transducer (3.3M参数)
num_encoder_layers = "1,1,1,1,1,1"
encoder_dim = "128,128,128,128,128,128"
feedforward_dim = "192,192,192,192,192,192"
decoder_dim = 320
joiner_dim = 320
causal = True  # 流式支持
chunk_size = 16
left_context_frames = 128
```

### 5.2 训练配置（V2）

```bash
# scripts/run_finetune_v2.sh
--num-epochs 20
--base-lr 0.0003
--max-duration 500
--enable-spec-aug 1
--enable-musan 0
```

### 5.3 关键词格式（Tokenization）

```
# keywords.txt格式
n ǐ h ǎo zh ēn zh ēn :boost #threshold @你好真真

# 拼音分解
"你好真真" → n ǐ h ǎo zh ēn zh ēn
```

---

## 六、关键文件清单

### 6.1 核心脚本

| 文件 | 功能 |
|------|------|
| `scripts/generate_tts_dataset.py` | 生成正样本TTS数据 |
| `scripts/generate_negative_tts.py` | 生成负样本TTS数据 |
| `scripts/prepare_lhotse_manifests.py` | 创建Lhotse训练manifests |
| `scripts/run_finetune_v2.sh` | 训练脚本（V2版本） |
| `scripts/export_onnx_v2.sh` | 导出ONNX模型 |
| `scripts/optimize_kws_params.py` | 参数网格搜索优化 |
| `scripts/optimize_decoy_params.py` | Decoy方案优化（已证实不可行） |

### 6.2 模型文件

```
exp/kws_finetune_v2/
├── encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx
├── decoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx
├── joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx
├── tokens.txt
├── keywords.txt
└── param_optimization/
    └── param_optimization_*.txt  # 参数优化报告
```

### 6.3 参考文档

- `CLAUDE.md` - 项目说明文档
- `icefall/egs/wenetspeech/KWS/` - Icefall KWS训练代码

---

## 七、已尝试但失败的方案

### 7.1 方案：增加负样本训练

**假设**：通过训练模型识别"你好"等负样本，让模型学会区分

**结果**：轻微改善，但不足以满足目标
- V1 (无负样本): FRR=10.42%, FAR=44.07%
- V2 (有负样本): FRR=0%~12.50%, FAR=34.26%~55%

**分析**：Transducer是序列识别模型，不是二分类器。它能识别"你好"序列，但在"你好真真"中会先匹配到"你好"部分。

### 7.2 方案：Decoy诱词过滤

**假设**：注册多个关键词，当检测到短的"你好"时判定为非目标

**结果**：完全失败
- FAR降至9.63%（好）
- FRR飙升至82.64%（灾难性）

**原因**：正样本"你好真真"总是先被识别为"你好"decoy，导致几乎所有正样本被过滤。

---

## 八、下一步优化建议

### 8.1 建议方案A：后处理时间窗口

**思路**：检测到"你好"后，等待一个时间窗口（如500ms），看是否继续检测到"真真"

**优点**：
- 不需要重新训练模型
- 可以在推理阶段实现

**缺点**：
- 增加延迟
- 需要修改sherpa-onnx使用方式

### 8.2 建议方案B：更换关键词

**思路**：使用一个不包含常见词汇前缀的唤醒词

**候选词**：
- "真真你好" (颠倒顺序)
- "嗨真真" (换前缀)
- "喂真真"

**优点**：从根本上解决前缀匹配问题

### 8.3 建议方案C：多阶段检测

**思路**：
1. 第一阶段：检测"你好"系列
2. 第二阶段：如果检测到，继续分析是否包含"真真"

**实现**：可能需要训练两个模型或修改模型输出

### 8.4 建议方案D：负样本加权

**思路**：在训练时大幅增加"你好"负样本的权重，让模型更倾向于不触发

**优点**：可能提高区分能力

**缺点**：可能导致召回率下降

### 8.5 建议方案E：收集真实数据

**思路**：收集真实人声"你好真真"样本进行训练

**优点**：最有可能达到目标性能

**缺点**：需要数据收集成本

---

## 九、复现说明

### 9.1 环境配置

```bash
conda activate kws-train
export PYTHONPATH=/data/workspace/llm/keyword-spotting/icefall:$PYTHONPATH
```

### 9.2 运行参数优化

```bash
python scripts/optimize_kws_params.py \
  --model-dir exp/kws_finetune_v2 \
  --positive-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/positive \
  --negative-dir /data/workspace/llm/audio-classification/dataset/kws_test_data_merged/negative \
  --target-frr 1.39 \
  --target-far 7.46
```

### 9.3 重新训练模型

```bash
# 1. 准备数据
python scripts/generate_tts_dataset.py
python scripts/generate_negative_tts.py
python scripts/prepare_lhotse_manifests.py

# 2. 训练
bash scripts/run_finetune_v2.sh

# 3. 导出
bash scripts/export_onnx_v2.sh

# 4. 评估
python scripts/optimize_kws_params.py
```

---

## 十、总结

### 当前困境

**核心矛盾**：目标唤醒词"你好真真"包含极其常见的中文词汇"你好"作为前缀，而Transducer模型会在检测到"你好"序列时立即触发，无法等待后续的"真真"。

### 数据事实

- 当前最佳配置：FRR=12.50%, FAR=34.26%
- 目标：FRR<1.39%, FAR<7.46%
- 差距：FRR超标约9倍，FAR超标约4.6倍

### 根本建议

**强烈建议重新考虑唤醒词选择**。"你好真真"这个唤醒词在技术上存在固有的区分难度。更换为不包含"你好"前缀的唤醒词（如"真真你好"、"嗨真真"）将大幅降低技术难度。

---

*报告生成者: Claude AI*
*最后更新: 2026-01-12*
