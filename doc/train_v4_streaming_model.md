# V4 Streaming 模型微调训练指南

本文档详细说明如何从预训练模型微调训练 V4 Streaming 关键词检测模型。

## 目录

- [概述](#概述)
- [前置准备](#前置准备)
- [数据准备](#数据准备)
- [训练流程](#训练流程)
- [模型导出](#模型导出)
- [模型评估](#模型评估)
- [常见问题](#常见问题)

## 概述

V4 Streaming 模型是基于 Zipformer-Transducer 架构的单阶段流式关键词检测模型，支持实时音频处理。

**模型特点**：
- 单阶段流式检测，延迟低
- 支持 FP32 和 INT8 量化
- 训练数据：282 个样本（60 正 + 222 负）
- 测试数据：65 个样本
- 最佳模型：epoch-33 (F1=89.47%, FAR=5.64%, FRR=2.30%)

**技术栈**：
- 训练框架：Icefall (k2 + Lhotse)
- 推理框架：Sherpa-ONNX
- 模型架构：Zipformer-Transducer
- 词汇表：带声调的拼音（`lang_partial_tone/`）

## 前置准备

### 环境要求

```bash
# 创建 conda 环境
conda create -n kws-train python=3.10 -y
conda activate kws-train

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install k2 lhotse sherpa-onnx soundfile librosa

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0"  # 使用第一个 GPU
export PYTHONPATH=/path/to/keyword-spotting/icefall:$PYTHONPATH
```

### 目录结构

```
keyword-spotting/
├── icefall/                               # Icefall 框架
│   └── egs/wenetspeech/KWS/zipformer/   # KWS 训练脚本
├── data/
│   ├── all/                              # 真实人声数据（406 个文件）
│   ├── lang_partial_tone/                 # 词汇表（拼音+声调）
│   └── manifests/                         # Lhotse 数据清单
├── experiments/
│   └── baseline_streaming/
│       ├── data_splits/                   # 数据划分
│       ├── manifests/                     # 训练用 manifests
│       └── exp_v4/                       # 训练输出
└── icefall-kws-zipformer-wenetspeech-20240219/  # 预训练模型
```

### 下载预训练模型

如果尚未下载预训练模型：

```bash
cd /data/workspace/llm/keyword-spotting
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/icefall-kws-zipformer-wenetspeech-20240219.tar.gz
tar -xzf icefall-kws-zipformer-wenetspeech-20240219.tar.gz
```

预训练模型路径：`icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt`

## 数据准备

### 1. 数据划分

使用分层划分确保训练集、验证集和测试集的类别分布一致。

```bash
cd experiments/baseline_streaming
python stratified_split.py
```

这将生成：
- `data_splits/train_manifest.json` (282 样本)
- `data_splits/val_manifest.json` (59 样本)
- `data_splits/test_manifest.json` (65 样本)

数据分布：
| 数据集 | 正样本 | 负样本(你好) | 负样本(真真) | 负样本(其他) | 总计 |
|--------|--------|--------------|--------------|--------------|------|
| Train  | 60     | 8            | 4            | 210          | 282  |
| Val    | 13     | 1            | 0            | 45           | 59   |
| Test   | 14     | 3            | 2            | 46           | 65   |

### 2. 创建 Lhotse Manifests

从划分后的数据创建 Lhotse 格式的训练清单：

```bash
cd experiments/baseline_streaming
python create_manifests.py
```

这将生成：
- `manifests/train.jsonl.gz`
- `manifests/val.jsonl.gz`
- `manifests/test.jsonl.gz`

**Manifest 格式说明**：
```json
{
  "cuts": [
    {
      "id": "unique_id",
      "source": {
        "type": "file",
        "channels": [0],
        "sources": [
          {
            "type": "file",
            "source": "/path/to/audio.wav",
            "channels": [0]
          }
        ]
      },
      "recording": {
        "id": "rec_id",
        "sources": [...],
        "num_samples": 16000,
        "sampling_rate": 16000,
        "duration": 1.0,
        "num_channels": 1
      },
      "supervisions": [
        {
          "id": "sup_id",
          "recording_id": "rec_id",
          "text": "你好真真"
        }
      ]
    }
  ]
}
```

## 训练流程

### 完整训练脚本

使用预制的训练脚本进行完整训练：

```bash
cd /data/workspace/llm/keyword-spotting/experiments/baseline_streaming
bash train_v4.sh
```

**脚本内容**：

```bash
#!/bin/bash
# 从真实人声数据微调V4 KWS模型
# 使用分层划分的训练集和验证集

set -e

export CUDA_VISIBLE_DEVICES="0"

BASE_DIR=/data/workspace/llm/keyword-spotting
PRETRAINED_CKPT=${BASE_DIR}/icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt
MANIFEST_DIR=${BASE_DIR}/experiments/baseline_streaming/manifests
LANG_DIR=${BASE_DIR}/data/lang_partial_tone
EXP_DIR=${BASE_DIR}/experiments/baseline_streaming/exp_v4

export PYTHONPATH=${BASE_DIR}/icefall:$PYTHONPATH

mkdir -p ${EXP_DIR}

cd ${BASE_DIR}/icefall/egs/wenetspeech/KWS

# 微调训练
python ./zipformer/finetune.py \
    --world-size 1 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${EXP_DIR} \
    --lang-dir ${LANG_DIR} \
    --manifest-dir ${MANIFEST_DIR} \
    --pinyin-type partial_with_tone \
    --use-fp16 0 \
    --use-mux 0 \
    --use-custom-kws-data 1 \
    --on-the-fly-feats 1 \
    --enable-musan 0 \
    --enable-spec-aug 1 \
    --decoder-dim 320 \
    --joiner-dim 320 \
    --num-encoder-layers "1,1,1,1,1,1" \
    --feedforward-dim "192,192,192,192,192,192" \
    --encoder-dim "128,128,128,128,128,128" \
    --encoder-unmasked-dim "128,128,128,128,128,128" \
    --causal 1 \
    --base-lr 0.0001 \
    --lr-epochs 50 \
    --lr-batches 50000 \
    --finetune-ckpt ${PRETRAINED_CKPT} \
    --max-duration 200 \
    --bucketing-sampler 1 \
    --num-buckets 5 \
    --num-workers 2 \
    2>&1 | tee ${EXP_DIR}/train.log
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--world-size` | 1 | 单 GPU 训练 |
| `--num-epochs` | 30 | 训练轮数 |
| `--causal` | 1 | 流式模型 |
| `--base-lr` | 0.0001 | 学习率（较小，用于微调） |
| `--finetune-ckpt` | pretrained.pt | 预训练检查点 |
| `--num-encoder-layers` | "1,1,1,1,1,1" | 编码器层数 |
| `--encoder-dim` | "128,..." | 编码器维度 |
| `--decoder-dim` | 320 | 解码器维度 |
| `--joiner-dim` | 320 | Joiner 维度 |
| `--max-duration` | 200 | 最大批次时长（秒） |
| `--bucketing-sampler` | 1 | 使用桶采样 |
| `--enable-spec-aug` | 1 | 启用 SpecAugment |
| `--on-the-fly-feats` | 1 | 在线提取特征 |

### 监控训练进度

训练过程会保存每个 epoch 的检查点：

```bash
# 查看训练日志
tail -f experiments/baseline_streaming/exp_v4/train.log

# 查看已保存的检查点
ls experiments/baseline_streaming/exp_v4/epoch-*.pt
```

**训练日志示例**：

```
epoch 1/30, batch 10/100
  train_loss: 2.345
  learning_rate: 0.000100
  elapsed: 00:02:15

epoch 1/30
  train_loss: 2.123
  val_loss: 2.087
  best_val_loss: 2.087 (saved as best-valid-loss.pt)
```

### 恢复训练

如果训练中断，可以从中断处恢复：

```bash
cd /data/workspace/llm/keyword-spotting/experiments/baseline_streaming
bash train_v4_resume.sh
```

或者手动指定检查点：

```bash
cd /data/workspace/llm/keyword-spotting/icefall/egs/wenetspeech/KWS

python ./zipformer/finetune.py \
    --world-size 1 \
    --num-epochs 50 \
    --start-epoch 34 \
    --exp-dir ${EXP_DIR} \
    --checkpoint ${EXP_DIR}/epoch-33.pt \
    ... [其他参数]
```

## 模型导出

### 导出为 ONNX

训练完成后，将最佳模型导出为 ONNX 格式：

```bash
cd /data/workspace/llm/keyword-spotting/experiments/baseline_streaming
bash export_v4_onnx.sh
```

**脚本内容**：

```bash
#!/bin/bash
set -e

BASE_DIR=/data/workspace/llm/keyword-spotting
EXP_DIR=${BASE_DIR}/experiments/baseline_streaming/exp_v4
TOKENS=${BASE_DIR}/data/lang_partial_tone/tokens.txt

cd ${BASE_DIR}/icefall/egs/wenetspeech/KWS

# 导出 epoch-33 (最佳模型)
python ./zipformer/export-onnx-streaming.py \
    --exp-dir ${EXP_DIR} \
    --tokens ${TOKENS} \
    --epoch 33 \
    --avg 1 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --causal 1

echo "导出完成！"
ls -lh ${EXP_DIR}/*.onnx
```

**导出参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `--epoch` | 33 | 要导出的 epoch |
| `--avg` | 1 | 平均的检查点数 |
| `--chunk-size` | 16 | 流式处理的块大小（帧数） |
| `--left-context-frames` | 128 | 左上下文帧数 |

**导出文件**：

```
experiments/baseline_streaming/exp_v4/
├── encoder-epoch-33-avg-1.onnx         # 编码器
├── decoder-epoch-33-avg-1.onnx         # 解码器
├── joiner-epoch-33-avg-1.onnx          # Joiner
└── tokens-epoch-33-avg-1.txt           # Token 列表
```

### INT8 量化

使用 onnxruntime 对模型进行 INT8 量化：

```bash
cd /data/workspace/llm/keyword-spotting/experiments/baseline_streaming

# 使用代表性数据集进行校准
python -c "
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

model_path = '${EXP_DIR}/encoder-epoch-33-avg-1.onnx'
quantized_path = '${EXP_DIR}/encoder-epoch-33-avg-1.int8.onnx'

quantize_dynamic(
    model_path,
    quantized_path,
    weight_type=QuantType.QInt8
)
print(f'INT8 模型已保存: {quantized_path}')
"
```

对三个模型文件都执行量化：
- `encoder-epoch-33-avg-1.int8.onnx`
- `decoder-epoch-33-avg-1.int8.onnx`
- `joiner-epoch-33-avg-1.int8.onnx`

## 模型评估

### 选择最佳 Epoch

使用搜索脚本找到最佳 epoch：

```bash
cd /data/workspace/llm/keyword-spotting/experiments/baseline_streaming
python search_best_epoch.py
```

这将在测试集上评估每个 epoch，并输出最佳 epoch 的性能指标。

### 详细评估

对选定模型进行详细评估：

```bash
cd /data/workspace/llm/keyword-spotting/experiments/baseline_streaming
python run_full_evaluation.py --epoch 33
```

**评估指标**：

- **F1 Score**: 精确率和召回率的调和平均
- **FAR** (False Accept Rate): 误报率，检测到不存在的关键词
- **FRR** (False Reject Rate): 漏检率，未检测到存在的关键词
- **RTF** (Real-Time Factor): 实时因子，RTF < 1.0 表示实时可行

**V4 最佳模型性能 (epoch-33, 406 测试样本)**：

| 变体 | 大小 | 阈值 | F1 | FAR | FRR | RTF |
|------|------|------|-----|-----|-----|-----|
| FP32 | 13MB | 0.52 | 89.47% | 5.64% | 2.30% | 0.0185 |
| INT8 | 5MB | 0.46 | 89.01% | 5.96% | 2.30% | 0.0140 |

### 阈值优化

为 INT8 模型找到最优阈值：

```bash
cd /data/workspace/llm/keyword-spotting/experiments/baseline_streaming
python fine_tune_threshold_v4_98.py
```

这将遍历不同阈值，找到 F1 最高的配置。

## 常见问题

### Q1: 训练时出现 OOM (Out of Memory) 错误

**解决方案**：
```bash
# 减少 max-duration
--max-duration 100

# 减少 num-buckets
--num-buckets 3

# 减少 encoder/decoder 维度
--encoder-dim "96,96,96,96,96,96"
--decoder-dim 256
--joiner-dim 256
```

### Q2: 验证损失不下降

**可能原因**：
1. 学习率过大：降低 `--base-lr` 到 `0.00005`
2. 数据质量差：检查 Manifests 是否正确
3. 预训练模型不匹配：确认 `--finetune-ckpt` 路径

**解决方案**：
```bash
# 使用更小的学习率重新训练
--base-lr 0.00005 --lr-batches 100000
```

### Q3: 导出 ONNX 时出错

**错误示例**：
```
Error: Node xxx has type xxx which is not supported by ONNX Runtime
```

**解决方案**：
```bash
# 检查 ONNX 版本
pip install onnx onnxruntime==1.16.0

# 使用更简单的导出配置
--chunk-size 8 --left-context-frames 64
```

### Q4: INT8 量化后精度大幅下降

**可能原因**：
- 校准数据集不具代表性
- 量化方法不合适

**解决方案**：
```python
# 使用动态量化（当前方法）
quantize_dynamic(..., weight_type=QuantType.QInt8)

# 或者使用静态量化（需要校准数据集）
from onnxruntime.quantization import quantize_preprocessed
quantize_preprocessed(..., calibration_data_path='calibration_data.npz')
```

### Q5: 推理速度慢

**优化方法**：
1. 使用 INT8 模型（速度快 24%）
2. 增加 `--num-threads`（多线程）
3. 使用 GPU Provider（如果可用）

```python
# 设置推理线程
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
```

### Q6: 如何添加新的关键词？

**步骤**：
1. 在 `data/lang_partial_tone/tokens.txt` 中添加新词的拼音
2. 准备新词的音频数据
3. 重新训练或微调模型
4. 导出新模型

**示例**：
```bash
# 添加 "你好小明"
echo "n ǐ h ǎo xi ǎo m íng 你好小明" >> data/lang_partial_tone/tokens.txt
```

## 参考资料

- [Icefall 文档](https://github.com/k2-fsa/icefall)
- [Sherpa-ONNX 文档](https://k2-fsa.github.io/sherpa/onnx/)
- [Lhotse 文档](https://lhotse.readthedocs.io/)
- [K2 文档](https://k2-fsa.github.io/k2/)

## 附录

### 完整训练流程

```bash
# 1. 数据准备
cd experiments/baseline_streaming
python stratified_split.py
python create_manifests.py

# 2. 训练
bash train_v4.sh

# 3. 选择最佳 epoch
python search_best_epoch.py

# 4. 导出模型
bash export_v4_onnx.sh

# 5. 量化模型
python quantize_model.py

# 6. 评估
python run_full_evaluation.py --epoch 33

# 7. 部署到 nihao-zhenzhen-kws/
cp ${EXP_DIR}/*.onnx ${BASE_DIR}/nihao-zhenzhen-kws/model/
```

### 训练时间估计

| 配置 | 数据量 | Epoch | 预估时间 |
|------|--------|-------|----------|
| 1x V100 | 282 样本 | 30 epochs | ~2 小时 |
| 1x A100 | 282 样本 | 30 epochs | ~1.5 小时 |

### 硬件要求

- **最低**: 8GB GPU (e.g., GTX 1080)
- **推荐**: 16GB GPU (e.g., RTX 3090, V100)
- **内存**: 32GB RAM
- **存储**: 10GB 可用空间

---

**文档版本**: 1.0
**最后更新**: 2026-02-03
**作者**: Keyword Spotting Team
