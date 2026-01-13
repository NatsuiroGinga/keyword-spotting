---
name: update-gitignore
overview: 更新 .gitignore 文件以适应HuggingFace部署工作流，解决ONNX模型和临时上传目录的跟踪问题。
todos:
  - id: read-gitignore
    content: 读取当前 .gitignore 文件内容
    status: completed
  - id: update-gitignore
    content: 更新 .gitignore 文件修正 ONNX 和 HuggingFace 规则
    status: completed
    dependencies:
      - read-gitignore
  - id: verify-tracking
    content: 验证 models/mlp_verifier.onnx 可被正确跟踪
    status: completed
    dependencies:
      - update-gitignore
---

## 产品概述

更新 .gitignore 配置文件，优化 HuggingFace 部署工作流的文件跟踪规则。

## 核心功能

- 修正 ONNX 文件跟踪规则，允许 `models/mlp_verifier.onnx` 被版本控制
- 确保 models 目录可跟踪关键文件
- 忽略 HuggingFace 上传临时目录 `hf_upload/`
- 添加 HuggingFace 缓存文件忽略规则（如 `.cache/huggingface/`、`__pycache__/` 等）

## 技术栈

- 配置文件管理：Shell 脚本配置

## 技术架构

### 修改说明

这是一个简单的配置文件更新任务，不涉及复杂架构。主要工作是编辑 `.gitignore` 文件，调整忽略规则以满足项目需求。

### 关键修改点

1. 将 `*.onnx` 改为更具体的路径模式，保留 `!models/mlp_verifier.onnx` 例外规则
2. 确保 `models/` 目录未被全局忽略
3. 添加 `hf_upload/` 忽略规则
4. 添加 HuggingFace 相关忽略模式

### 目录结构（修改部分）

```
project-root/
├── .gitignore              # 需要更新的文件
├── models/
│   └── mlp_verifier.onnx   # 需要被跟踪的文件
└── hf_upload/              # 需要被忽略的临时目录
```

### 关键代码结构

**.gitignore 规则示例**：

```
# Python 编译文件
__pycache__/
*.py[cod]
*$py.class

# ONNX 模型文件（保留 models/mlp_verifier.onnx）
*.onnx
!models/mlp_verifier.onnx

# HuggingFace 上传临时目录
hf_upload/

# HuggingFace 缓存
.cache/huggingface/
```