#!/bin/bash
# 从epoch-30断点续训V4 KWS模型
# 修复OOV问题后的续训脚本

set -e

export CUDA_VISIBLE_DEVICES="0"

BASE_DIR=/data/workspace/llm/keyword-spotting
EXP_DIR=${BASE_DIR}/experiments/baseline_streaming/exp_v4
MANIFEST_DIR=${BASE_DIR}/experiments/baseline_streaming/manifests
LANG_DIR=${BASE_DIR}/data/lang_partial_tone

export PYTHONPATH=${BASE_DIR}/icefall:$PYTHONPATH

cd ${BASE_DIR}/icefall/egs/wenetspeech/KWS

echo "=========================================="
echo "从epoch-30断点续训V4 KWS模型"
echo "=========================================="
echo "检查点目录: ${EXP_DIR}"
echo "数据目录: ${MANIFEST_DIR}"
echo ""

# 检查epoch-30是否存在
if [ ! -f "${EXP_DIR}/epoch-30.pt" ]; then
    echo "错误: 找不到 epoch-30.pt"
    exit 1
fi

echo "从 epoch-31 开始续训..."
echo "目标: 训练直到loss稳定 (最多100 epochs)"
echo ""

# 断点续训: 使用 --continue-finetune 从现有检查点继续
/data/workspace/llm/anaconda3/envs/kws-train/bin/python ./zipformer/finetune.py \
    --world-size 1 \
    --num-epochs 100 \
    --start-epoch 31 \
    --continue-finetune 1 \
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
    --base-lr 0.00005 \
    --lr-epochs 50 \
    --lr-batches 50000 \
    --max-duration 200 \
    --bucketing-sampler 1 \
    --num-buckets 5 \
    --num-workers 2 \
    2>&1 | tee -a ${EXP_DIR}/train_resume.log

echo ""
echo "=========================================="
echo "续训完成！"
echo "=========================================="
