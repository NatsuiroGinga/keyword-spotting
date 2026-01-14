---
name: add-main-and-push-hf
overview: 修复main.py中的模型路径问题，复制到hf_upload目录，提交变更并推送到HuggingFace私有仓库
todos:
  - id: explore-main
    content: 使用[subagent:code-explorer]查找main.py中所有模型路径引用
    status: completed
  - id: verify-hf-auth
    content: 使用[mcp:hf-mcp-server]验证HuggingFace认证状态
    status: completed
  - id: fix-model-paths
    content: 修改main.py中的模型文件名为简化名称
    status: completed
    dependencies:
      - explore-main
  - id: fix-default-dir
    content: 修改main.py中默认model-dir为./kws_finetune_v3
    status: completed
    dependencies:
      - explore-main
  - id: copy-to-hf
    content: 复制修改后的main.py到hf_upload目录
    status: completed
    dependencies:
      - fix-model-paths
      - fix-default-dir
  - id: push-to-hf
    content: 在hf_upload目录提交变更并推送到HuggingFace仓库
    status: completed
    dependencies:
      - copy-to-hf
      - verify-hf-auth
---

## Product Overview

修复关键词唤醒(KWS)项目中main.py的模型路径配置问题，确保代码能正确引用HuggingFace仓库中的模型文件，并将修改后的代码同步到hf_upload目录后推送到HuggingFace私有仓库。

## Core Features

- 修复模型文件路径：将长文件名（如encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx）改为简化名称（如encoder.int8.onnx）
- 修改默认model-dir参数：从exp/kws_finetune_v3改为./kws_finetune_v3
- 同步文件到hf_upload目录：复制修改后的main.py
- 推送到HuggingFace：提交变更并推送到Heehobino/streaming-kws私有仓库

## Tech Stack

- 语言：Python
- 模型格式：ONNX (int8量化)
- 版本控制：Git + HuggingFace Hub

## Implementation Details

### 需要修改的文件

```
keyword-spotting/
├── main.py                    # 需修改：模型路径和默认目录
└── hf_upload/
    └── main.py                # 需更新：复制修改后的版本
```

### 关键代码修改

**模型路径修改**：将以下长文件名改为简化名称

| 原文件名 | 新文件名 |
| --- | --- |
| encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx | encoder.int8.onnx |
| decoder-epoch-12-avg-2-chunk-16-left-64.onnx | decoder.onnx |
| joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx | joiner.int8.onnx |


**默认目录修改**：

```python
# 修改前
default="exp/kws_finetune_v3"

# 修改后
default="./kws_finetune_v3"
```

### HuggingFace推送流程

1. 在hf_upload目录执行git add
2. 提交变更（commit message描述修改内容）
3. 推送到Heehobino/streaming-kws仓库

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose：探索main.py文件内容，确认所有需要修改的模型路径位置
- Expected outcome：获取完整的模型路径引用列表，确保修改无遗漏

### MCP

- **hf-mcp-server**
- Purpose：使用hf_whoami验证HuggingFace认证状态，确保有权限推送到私有仓库
- Expected outcome：确认用户Heehobino已认证，可以执行推送操作