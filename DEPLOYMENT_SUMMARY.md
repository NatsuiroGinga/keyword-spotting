# KWS系统HuggingFace部署总结

## 部署完成 ✅

### 基本信息
- **日期**: 2024-01-13
- **用户**: Heehobino
- **仓库**: [Heehobino/streaming-kws](https://huggingface.co/Heehobino/streaming-kws)
- **状态**: 🔐 私有

## 上传内容

### 模型文件 (4.48 MB)
| 文件 | 大小 | 说明 |
|------|------|------|
| encoder.int8.onnx | 4.22 MB | Zipformer编码器 (INT8量化) |
| decoder.int8.onnx | 174 KB | Zipformer解码器 (INT8量化) |
| joiner.int8.onnx | 65 KB | Zipformer连接器 (INT8量化) |
| mlp_verifier.onnx | 12.7 KB | MLP验证器 (INT8量化) |
| tokens.txt | - | 词表文件 |
| keywords.txt | - | 关键词配置 |

### 推理代码
- `src/audio/capture.py` - 麦克风音频采集
- `src/audio/feature.py` - MFCC特征提取
- `src/models/mlp_verifier.py` - MLP推理模块
- `src/pipeline/kws_stream.py` - 流式KWS管道
- `src/utils/config.py` - 配置管理

### 文档
- `README.md` - 完整使用指南 (6.7 KB)
- `MODEL_CARD.md` - 模型元数据 (6.3 KB)
- `requirements.txt` - Python依赖

## 部署工具

### 可用的部署脚本

#### 1. MLP导出工具
```bash
python tools/export_mlp_onnx.py \
    --model-path experiments/multi_stage_ablation/models/mlp_verifier.pt \
    --output-path models/mlp_verifier.onnx
```

#### 2. HF仓库创建工具
```bash
python tools/create_hf_repo.py \
    --username your_username \
    --repo-name your-repo-name
```

#### 3. 文件上传工具
```bash
python tools/upload_to_hf.py \
    --repo-id your_username/repo_name \
    --local-dir hf_upload \
    --dry-run  # 可选：先预览
```

#### 4. 可复用部署技能
```bash
# 完整部署
python tools/kws_deploy_skill.py \
    --username your_username \
    --repo-name streaming-kws \
    --action full

# 或单独执行步骤
python tools/kws_deploy_skill.py \
    --username your_username \
    --action prepare   # 准备上传目录
python tools/kws_deploy_skill.py \
    --username your_username \
    --action upload    # 上传文件
python tools/kws_deploy_skill.py \
    --username your_username \
    --action verify    # 验证上传
python tools/kws_deploy_skill.py \
    --username your_username \
    --action download  # 下载并测试
```

## 使用模型

### 方法1: 直接从HuggingFace下载

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download("Heehobino/streaming-kws")
print(f"Models in: {model_dir}")
```

### 方法2: 使用StreamingKWSPipeline

```python
from src.pipeline.kws_stream import StreamingKWSPipeline

pipeline = StreamingKWSPipeline(
    zipformer_encoder="kws_finetune_v3/encoder.int8.onnx",
    zipformer_decoder="kws_finetune_v3/decoder.int8.onnx",
    zipformer_joiner="kws_finetune_v3/joiner.int8.onnx",
    mlp_model="models/mlp_verifier.onnx",
    tokens_file="kws_finetune_v3/tokens.txt",
    keywords_file="kws_finetune_v3/keywords.txt"
)

# 推理
result = pipeline.process_frame(audio_frame)
```

### 方法3: 命令行使用

```bash
python main.py --model-dir kws_finetune_v3
```

## 部署体系结构

```
项目根目录
├── tools/                          # 部署工具
│   ├── export_mlp_onnx.py         # MLP导出
│   ├── create_hf_repo.py          # 创建仓库
│   ├── upload_to_hf.py            # 上传文件
│   └── kws_deploy_skill.py        # 完整部署技能
│
├── hf_upload/                      # 上传目录（预准备）
│   ├── kws_finetune_v3/           # Zipformer模型
│   ├── models/                     # MLP模型
│   ├── src/                        # 推理代码
│   ├── README.md
│   └── MODEL_CARD.md
│
├── docs/
│   ├── hf_upload_guide.md         # HF部署指南
│   └── windows_deployment.md      # Windows部署指南
│
├── DEPLOYMENT_SUMMARY.md          # 本文件
└── requirements.txt
```

## 关键特性

### 两阶段检测架构
- **第一阶段**: Zipformer快速筛选（FAR 72%）
- **第二阶段**: MLP验证确认（FAR 1.3%）
- **总体性能**: 端对端延迟 ~100ms

### 优化特性
- ✅ INT8量化（大幅减小模型大小）
- ✅ 流式处理（支持实时音频）
- ✅ ONNX格式（跨平台兼容）
- ✅ 私有部署（离线使用无隐私风险）

## 许可和归属

- **许可**: Apache License 2.0
- **模型**: Zipformer + MLP Verifier
- **框架**: Sherpa-ONNX + ONNX Runtime
- **数据**: WeNet Speech Corpus (Chinese)

## 后续步骤

### 1. 分享模型
```python
# 在HuggingFace网页上添加协作者
# 或通过API：
from huggingface_hub import HfApi
api = HfApi()
api.add_user_to_repo(
    repo_id="Heehobino/streaming-kws",
    username="collaborator_username",
    permission="read"  # 或 "write"
)
```

### 2. 更新模型
```bash
# 修改本地文件后重新上传
python tools/upload_to_hf.py \
    --repo-id Heehobino/streaming-kws \
    --local-dir hf_upload
```

### 3. 创建新版本
```bash
# 创建新的模型版本仓库
python tools/kws_deploy_skill.py \
    --username Heehobino \
    --repo-name streaming-kws-v2 \
    --action full
```

## 相关文档

- [HF上传详细指南](./docs/hf_upload_guide.md)
- [Windows部署指南](./docs/windows_deployment.md)
- [模型README](./hf_upload/README.md)
- [模型卡片](./hf_upload/MODEL_CARD.md)

## 故障排查

### 上传失败
1. 检查HuggingFace令牌: `huggingface-cli login`
2. 验证网络连接
3. 使用 `--dry-run` 预览

### 无法下载模型
1. 检查权限设置
2. 验证令牌有 `read` 权限
3. 检查仓库是否私有

### 推理性能慢
1. 确保使用INT8量化模型
2. 检查CPU或GPU配置
3. 优化批处理大小

## 支持

- 问题报告: GitHub Issues
- 文档: 本项目docs/目录
- 模型下载: [HuggingFace Hub](https://huggingface.co/Heehobino/streaming-kws)

---

**最后更新**: 2024-01-13  
**部署版本**: v1.0  
**状态**: ✅ 生产就绪
