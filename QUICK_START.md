# 快速开始 - HuggingFace部署

## 📦 已完成上传

✅ **仓库**: [Heehobino/streaming-kws](https://huggingface.co/Heehobino/streaming-kws) (私有)  
✅ **内容**: Zipformer V3 模型 + MLP验证器 + 推理代码  
✅ **大小**: 4.48 MB (INT8量化)

## 🚀 快速使用

### 下载模型
```bash
python -c "
from huggingface_hub import snapshot_download
model_dir = snapshot_download('Heehobino/streaming-kws')
print(f'✓ 模型已下载到: {model_dir}')
"
```

### 运行流式KWS
```bash
python main.py --model-dir kws_finetune_v3
```

## 📋 部署工具

### 一键部署新仓库
```bash
python tools/kws_deploy_skill.py \
    --username your_username \
    --repo-name streaming-kws \
    --action full
```

### 分步操作
```bash
# 1. 准备文件
python tools/kws_deploy_skill.py \
    --username your_username \
    --action prepare

# 2. 创建仓库
python tools/kws_deploy_skill.py \
    --username your_username \
    --action create

# 3. 上传文件
python tools/kws_deploy_skill.py \
    --username your_username \
    --action upload

# 4. 验证上传
python tools/kws_deploy_skill.py \
    --username your_username \
    --action verify

# 5. 下载测试
python tools/kws_deploy_skill.py \
    --username your_username \
    --action download
```

## 📚 文档

- [HuggingFace部署完整指南](./docs/hf_upload_guide.md)
- [部署总结](./DEPLOYMENT_SUMMARY.md)
- [Windows部署指南](./docs/windows_deployment.md)
- [模型说明](./hf_upload/README.md)

## 🔧 依赖

```bash
pip install -r requirements.txt
pip install huggingface_hub  # 如果没有
```

## 📝 文件结构

```
keyword-spotting/
├── tools/
│   ├── export_mlp_onnx.py       # MLP导出
│   ├── create_hf_repo.py        # 创建仓库
│   ├── upload_to_hf.py          # 上传工具
│   └── kws_deploy_skill.py      # 完整技能
├── hf_upload/                   # 预准备目录
│   ├── kws_finetune_v3/         # Zipformer模型
│   ├── models/mlp_verifier.onnx # MLP模型
│   ├── src/                     # 推理代码
│   └── README.md
├── docs/
│   ├── hf_upload_guide.md
│   └── windows_deployment.md
├── DEPLOYMENT_SUMMARY.md
└── QUICK_START.md
```

## 💡 常见用法

### 分享模型给他人
```python
from huggingface_hub import HfApi
api = HfApi()
api.add_user_to_repo(
    repo_id="Heehobino/streaming-kws",
    username="friend_username",
    permission="read"
)
```

### 更新模型
```bash
python tools/upload_to_hf.py \
    --repo-id Heehobino/streaming-kws \
    --local-dir hf_upload
```

### 创建模型版本
```bash
python tools/kws_deploy_skill.py \
    --username Heehobino \
    --repo-name streaming-kws-v2 \
    --action full
```

## ✨ 关键特性

| 特性 | 说明 |
|------|------|
| 两阶段检测 | Zipformer初筛 + MLP验证 |
| INT8量化 | 模型大小减少85% |
| 流式处理 | 低延迟实时推理 |
| ONNX格式 | 跨平台兼容 |
| 私有部署 | 离线运行无隐私风险 |

---

更多信息请查看完整文档 📖
