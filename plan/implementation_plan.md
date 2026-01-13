# KWS模型优化实施计划

**基于**: `plan_v2.md` 深度技术分析报告
**目标**: FRR < 1.39%, FAR < 7.46%
**唤醒词**: "你好真真" (Ni Hao Zhen Zhen)

---

## 核心策略总结

根据 `plan_v2.md` 的分析，解决前缀触发问题需要**三位一体闭环**：

1. **数据仿真** - 使用多引擎TTS + Lhotse增强构建高保真数据集
2. **对抗训练** - 困难负样本挖掘 + 加权采样
3. **延迟决策** - 推理端状态机实现两阶段检测

---

## 阶段1: 高质量TTS数据生成

### 1.1 正样本生成 (目标: 1200条)

| 引擎 | 数量 | 变体类型 | 说明 |
|------|------|----------|------|
| Kokoro-82M | 1000 | 基础变体 | 8种音色 × 多种语速(0.9x-1.1x) |
| ChatTTS | 200 | 情感变体 | 带情感、停顿、语气词 |

**关键：声调变体**
```
标准: n ǐ h ǎo zh ēn zh ēn (你好真真)
变调1: n í h ǎo zh ēn zh ēn (上上变调)
变调2: n í h ǎo zh ēn zhen (轻声变体)
```

### 1.2 困难负样本生成 (目标: 6000条)

| 类型 | 文本 | 数量 | 优先级 |
|------|------|------|--------|
| 前缀词 | 你好、您好、泥豪、李浩 | 2000 | 最高 |
| 扩展前缀 | 你好啊、你好吗、你好呀 | 1500 | 高 |
| 后缀词 | 真真、真真你好 | 500 | 中 |
| 长句含前缀 | "你好，今天天气不错" | 1500 | 中 |
| 通用负样本 | 随机短句 | 500 | 低 |

### 1.3 实施脚本

```bash
# 创建新的数据生成脚本
scripts/generate_tts_v3_kokoro.py   # Kokoro正样本
scripts/generate_tts_v3_chattts.py  # ChatTTS正样本+困难负样本
```

---

## 阶段2: Lhotse增强流水线

### 2.1 增强策略

| 增强类型 | 参数 | 目的 |
|----------|------|------|
| 加性噪声 (Musan) | SNR 5-15dB | 模拟真实环境噪声 |
| 卷积混响 (RIR) | RT60 0.2-0.8s | 模拟远场识别 |
| 速度扰动 | 0.9x, 1.0x, 1.1x | 增加鲁棒性 |
| 音量扰动 | ±6dB | 模拟距离变化 |
| SpecAugment | 频域+时域 | 防止过拟合 |

### 2.2 数据配比

```
训练集构成:
- 正样本: 1200 × 3(速度变体) × 增强 ≈ 3600+
- 困难负样本: 6000 × 增强 ≈ 6000+
- 总计: ~10000+ 样本

每个Batch配比:
- 正样本: 30-40%
- 困难负样本: 40-50%
- 通用负样本: 10-20%
```

### 2.3 实施脚本

```bash
scripts/prepare_lhotse_manifests_v3.py  # 新版manifest准备
scripts/apply_lhotse_augmentation.py    # Lhotse增强流水线
```

---

## 阶段3: 模型微调 (Icefall)

### 3.1 训练策略

| 参数 | 值 | 说明 |
|------|-----|------|
| base_lr | 1e-4 | 极小学习率防止灾难性遗忘 |
| num_epochs | 20-30 | 使用早停 |
| max_duration | 300-500 | Batch duration |
| sampler | WeightedSimpleCutSampler | 加权采样 |

### 3.2 加权采样实现

```python
# 采样权重
weights = {
    "positive": 3.0,      # 正样本高权重
    "hard_negative": 2.0, # 困难负样本次高权重
    "general_negative": 0.5  # 通用负样本低权重
}
```

### 3.3 修改文件

```bash
icefall/egs/wenetspeech/KWS/zipformer/finetune.py  # 添加加权采样
scripts/run_finetune_v3.sh                          # 新训练脚本
```

---

## 阶段4: 延迟决策推理逻辑

### 4.1 状态机设计

```
状态: IDLE -> PREFIX_DETECTED -> WAITING_SUFFIX -> TRIGGERED/REJECTED

IDLE:
  - 持续监听
  - 检测到任何关键词 -> PREFIX_DETECTED

PREFIX_DETECTED:
  - 检查是否是完整"你好真真"
  - 如果是 -> TRIGGERED
  - 如果只是"你好" -> WAITING_SUFFIX, 启动600ms计时器

WAITING_SUFFIX:
  - 继续缓存音频
  - 检测到"真真" -> TRIGGERED
  - 600ms超时 -> REJECTED, 回到IDLE

TRIGGERED:
  - 触发唤醒回调
  - 重置状态 -> IDLE

REJECTED:
  - 丢弃缓冲
  - 重置状态 -> IDLE
```

### 4.2 实现方案

**方案A: 两阶段检测 (推荐)**
- 第一阶段: 检测"你好"作为触发器
- 第二阶段: 在600ms窗口内验证"真真"

**方案B: 单模型延迟决策**
- 只注册"你好真真"
- 检测到后延迟200ms确认

### 4.3 实施脚本

```bash
scripts/kws_inference_with_delay.py    # 延迟决策推理
scripts/evaluate_with_delay.py         # 带延迟的评估脚本
```

---

## 阶段5: 综合评估与调优

### 5.1 评估流程

```bash
# 1. 导出ONNX模型
bash scripts/export_onnx_v3.sh

# 2. 参数网格搜索 (boost + threshold)
python scripts/optimize_kws_params.py \
  --model-dir exp/kws_finetune_v3 \
  --target-frr 1.39 \
  --target-far 7.46

# 3. 带延迟决策的评估
python scripts/evaluate_with_delay.py \
  --delay-ms 600 \
  --model-dir exp/kws_finetune_v3
```

### 5.2 调优参数

| 参数 | 范围 | 说明 |
|------|------|------|
| boost | 1.5-3.0 | 关键词增强分数 |
| threshold | 0.3-0.6 | 触发阈值 |
| delay_ms | 400-800 | 延迟决策窗口 |

---

## 文件清单

### 新建文件

| 文件 | 功能 | 优先级 |
|------|------|--------|
| `scripts/generate_tts_v3_kokoro.py` | Kokoro TTS生成 | P0 |
| `scripts/generate_tts_v3_chattts.py` | ChatTTS生成 | P0 |
| `scripts/prepare_lhotse_manifests_v3.py` | V3 manifest准备 | P0 |
| `scripts/apply_lhotse_augmentation.py` | Lhotse增强 | P1 |
| `scripts/run_finetune_v3.sh` | V3训练脚本 | P1 |
| `scripts/kws_inference_with_delay.py` | 延迟决策推理 | P1 |
| `scripts/evaluate_with_delay.py` | 带延迟评估 | P2 |
| `scripts/export_onnx_v3.sh` | V3模型导出 | P2 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `icefall/.../finetune.py` | 添加WeightedSimpleCutSampler支持 |

---

## 执行顺序

```
Week 1: 阶段1 + 阶段2
├── Day 1-2: 安装Kokoro/ChatTTS，编写TTS生成脚本
├── Day 3-4: 生成正样本和困难负样本
└── Day 5-7: 构建Lhotse增强流水线

Week 2: 阶段3
├── Day 1-2: 修改finetune.py支持加权采样
├── Day 3-5: 训练V3模型
└── Day 6-7: 导出ONNX并初步评估

Week 3: 阶段4 + 阶段5
├── Day 1-3: 实现延迟决策推理逻辑
├── Day 4-5: 综合评估和参数调优
└── Day 6-7: 迭代优化
```

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| TTS数据过于完美 | 使用重度噪声增强和SpecAugment |
| 延迟决策增加响应时间 | 优化窗口大小(400-600ms) |
| 合成数据与真实数据分布差异 | 必须用真实测试集校准 |
| 模型过拟合合成数据 | 使用早停，监控验证集Loss |

---

## 成功标准

- [ ] FRR < 1.39% (漏检率)
- [ ] FAR < 7.46% (误检率)
- [ ] 响应延迟 < 800ms
- [ ] 在测试集上稳定复现

---

*计划制定日期: 2026-01-12*
*基于: plan_v2.md 技术分析报告*
