# HuggingFace 上传指南

## 概述

本文档说明如何将流式KWS系统部署到HuggingFace Hub，以及如何从HuggingFace下载和使用模型。

## 已完成的上传

- **仓库ID**: `Heehobino/streaming-kws`
- **状态**: 🔐 私有
- **访问链接**: [https://huggingface.co/Heehobino/streaming-kws](https://huggingface.co/Heehobino/streaming-kws)
- **上传时间**: 2024-01-13

## 上传内容清单

### 模型文件 (Models)
- ✅ `kws_finetune_v3/encoder.int8.onnx` (4.22 MB) - Zipformer编码器
- ✅ `kws_finetune_v3/decoder.int8.onnx` (174 KB) - Zipformer解码器
- ✅ `kws_finetune_v3/joiner.int8.onnx` (65 KB) - Zipformer连接器
- ✅ `models/mlp_verifier.onnx` (12.7 KB) - MLP验证器
- ✅ `kws_finetune_v3/tokens.txt` - 词表文件
- ✅ `kws_finetune_v3/keywords.txt` - 关键词配置

### 推理代码 (Source Code)
- ✅ `src/audio/capture.py` - 音频采集模块
- ✅ `src/audio/feature.py` - MFCC特征提取
- ✅ `src/models/mlp_verifier.py` - MLP推理模块
- ✅ `src/pipeline/kws_stream.py` - 流式KWS管道
- ✅ `src/utils/config.py` - 配置管理

### 文档
- ✅ `README.md` - 完整使用文档
- ✅ `MODEL_CARD.md` - 模型元数据和详情
- ✅ `requirements.txt` - Python依赖列表

## 上传工具

### 脚本位置
```
tools/
├── export_mlp_onnx.py      # MLP模型导出工具
├── create_hf_repo.py       # 创建HF仓库
└── upload_to_hf.py         # 上传文件到HF
```

### 使用上传工具

#### 1. 创建新仓库

```bash
python tools/create_hf_repo.py \
    --username your_hf_username \
    --repo-name your-repo-name
```

#### 2. 准备上传文件

```bash
# 文件已自动组织在 hf_upload/ 目录
ls -la hf_upload/
```

#### 3. 执行上传（干运行）

```bash
python tools/upload_to_hf.py \
    --repo-id your_username/your_repo_name \
    --local-dir hf_upload \
    --dry-run
```

#### 4. 执行实际上传

```bash
python tools/upload_to_hf.py \
    --repo-id your_username/your_repo_name \
    --local-dir hf_upload
```

## 从HuggingFace下载和使用

### 方法1：使用huggingface_hub库

```python
from huggingface_hub import snapshot_download

# 下载模型到本地
model_dir = snapshot_download(
    repo_id="Heehobino/streaming-kws",
    local_dir="./models/streaming-kws"
)

print(f"Models downloaded to: {model_dir}")
```

### 方法2：使用Git克隆

```bash
# 需要安装git-lfs
git lfs install

# 克隆私有仓库（需要HuggingFace访问令牌）
git clone https://huggingface.co/Heehobino/streaming-kws

cd streaming-kws
```

### 方法3：直接从仓库运行

```python
from src.pipeline.kws_stream import StreamingKWSPipeline

# 直接指定HuggingFace仓库
pipeline = StreamingKWSPipeline(
    zipformer_encoder="hf://Heehobino/streaming-kws/kws_finetune_v3/encoder.int8.onnx",
    zipformer_decoder="hf://Heehobino/streaming-kws/kws_finetune_v3/decoder.int8.onnx",
    zipformer_joiner="hf://Heehobino/streaming-kws/kws_finetune_v3/joiner.int8.onnx",
    mlp_model="hf://Heehobino/streaming-kws/models/mlp_verifier.onnx",
    tokens_file="hf://Heehobino/streaming-kws/kws_finetune_v3/tokens.txt",
    keywords_file="hf://Heehobino/streaming-kws/kws_finetune_v3/keywords.txt"
)
```

## 访问权限管理

### 分享私有仓库

1. 访问仓库页面
2. 点击 "Settings" → "Collaborators"
3. 输入要添加的用户名
4. 选择权限等级：
   - **Read**: 仅查看
   - **Write**: 查看和修改
   - **Admin**: 完全控制

### 生成访问令牌

1. 访问 https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择权限范围（建议 `read` + `write`）
4. 复制令牌并妥善保管

## 常见问题

### Q: 如何更新仓库中的模型?

```bash
# 方法1: 使用upload_to_hf.py重新上传
python tools/upload_to_hf.py \
    --repo-id Heehobino/streaming-kws \
    --local-dir hf_upload

# 方法2: 使用git push
cd streaming-kws
git add .
git commit -m "Update models"
git push
```

### Q: 如何让其他人访问私有仓库?

1. 邀请用户为协作者（见上面的权限管理）
2. 或生成读取令牌并与他人分享（不推荐）

### Q: 模型文件太大，上传失败?

HuggingFace支持单个文件最大50GB。如果上传失败：
1. 检查网络连接
2. 使用 `--dry-run` 测试
3. 分批上传不同文件

### Q: 如何删除仓库?

1. 访问仓库设置页面
2. 滚到页面底部
3. 点击 "Delete this repo"（需要输入仓库名确认）

## 部署工作流

### 完整上传流程

```bash
# 1. 导出MLP模型到ONNX
python tools/export_mlp_onnx.py \
    --model-path experiments/multi_stage_ablation/models/mlp_verifier.pt \
    --output-path models/mlp_verifier.onnx

# 2. 准备上传文件（自动化脚本）
# 仓库结构:
# hf_upload/
# ├── kws_finetune_v3/
# ├── models/
# ├── src/
# ├── README.md
# └── MODEL_CARD.md

# 3. 创建HuggingFace仓库
python tools/create_hf_repo.py \
    --username your_username \
    --repo-name streaming-kws

# 4. 上传到HuggingFace
python tools/upload_to_hf.py \
    --repo-id your_username/streaming-kws \
    --local-dir hf_upload

# 5. 验证上传（可选）
python -c "
from huggingface_hub import snapshot_download
model_dir = snapshot_download('your_username/streaming-kws')
print(f'Successfully downloaded to: {model_dir}')
"
```

## 最佳实践

1. **版本控制**: 在README中记录模型版本号和发布日期
2. **文档完整**: 保持MODEL_CARD和README最新
3. **访问控制**: 定期审查仓库的协作者权限
4. **备份**: 在GitHub或本地保留备份
5. **测试**: 在完整上传前进行干运行测试

## 相关资源

- [HuggingFace Hub文档](https://huggingface.co/docs/hub)
- [huggingface_hub库](https://huggingface.co/docs/hub/security-git-credentials)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Sherpa-ONNX](https://k2-fsa.github.io/sherpa/onnx/)
