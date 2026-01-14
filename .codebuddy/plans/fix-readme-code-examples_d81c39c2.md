---
name: fix-readme-code-examples
overview: 修复hf_upload/README.md中的示例代码，使其与实际API一致，包括类名、初始化参数和方法调用
todos:
  - id: explore-actual-api
    content: 使用 [subagent:code-explorer] 查找AudioCapture、StreamingKWSPipeline、KWSConfig的实际API定义
    status: completed
  - id: fix-class-names
    content: 修复README中的类名，将MicrophoneCapture改为AudioCapture
    status: completed
    dependencies:
      - explore-actual-api
  - id: fix-init-params
    content: 修复StreamingKWSPipeline初始化示例，改为使用KWSConfig对象
    status: completed
    dependencies:
      - explore-actual-api
  - id: fix-method-calls
    content: 修复方法调用，将process_frame改为process_chunk
    status: completed
    dependencies:
      - explore-actual-api
  - id: update-api-reference
    content: 更新API Reference部分的参数说明，确保与实际代码一致
    status: completed
    dependencies:
      - explore-actual-api
---

## 产品概述

修复 `hf_upload/README.md` 文档中的示例代码，确保文档与实际代码API保持一致，提升开发者使用体验。

## 核心功能

- 修复类名错误：将 `MicrophoneCapture` 更正为 `AudioCapture`
- 修复初始化参数：将 `StreamingKWSPipeline` 的多路径参数改为接收 `KWSConfig` 对象
- 修复方法调用：将 `process_frame` 更正为 `process_chunk`
- 更新 API Reference：确保参数说明与实际代码实现一致

## 技术方案

### 问题分析

根据用户反馈，README.md 中存在以下API不一致问题：

| 文档中的错误 | 实际API |
| --- | --- |
| `MicrophoneCapture` 类 | `AudioCapture` 类 |
| `StreamingKWSPipeline(model_path, labels_path, ...)` | `StreamingKWSPipeline(config: KWSConfig)` |
| `process_frame()` 方法 | `process_chunk()` 方法 |


### 修复策略

#### 1. 类名修正

```python
# 错误示例
from audio import MicrophoneCapture

# 正确示例
from audio import AudioCapture
```

#### 2. 初始化参数修正

```python
# 错误示例
pipeline = StreamingKWSPipeline(
    model_path="model.onnx",
    labels_path="labels.txt",
    ...
)

# 正确示例
from kws import KWSConfig, StreamingKWSPipeline

config = KWSConfig(
    model_path="model.onnx",
    labels_path="labels.txt",
    ...
)
pipeline = StreamingKWSPipeline(config)
```

#### 3. 方法调用修正

```python
# 错误示例
result = pipeline.process_frame(audio_data)

# 正确示例
result = pipeline.process_chunk(audio_data)
```

### 验证方法

- 对照源代码文件确认实际API定义
- 确保修复后的示例代码可以正常运行

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 探索项目代码库，查找 `AudioCapture`、`StreamingKWSPipeline`、`KWSConfig` 等类的实际定义和方法签名
- Expected outcome: 获取准确的类名、初始化参数和方法名，确保README修复内容与源代码完全一致