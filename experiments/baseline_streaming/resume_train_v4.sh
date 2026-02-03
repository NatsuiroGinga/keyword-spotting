#!/bin/bash
# 从epoch-30断点续训V4 KWS模型
# 支持收敛自动停止

set -e

export CUDA_VISIBLE_DEVICES="0"

BASE_DIR=/data/workspace/llm/keyword-spotting
MANIFEST_DIR=${BASE_DIR}/experiments/baseline_streaming/manifests
LANG_DIR=${BASE_DIR}/data/lang_partial_tone
EXP_DIR=${BASE_DIR}/experiments/baseline_streaming/exp_v4

export PYTHONPATH=${BASE_DIR}/icefall:$PYTHONPATH

cd ${BASE_DIR}/icefall/egs/wenetspeech/KWS

# 查找最新的epoch
LAST_EPOCH=$(ls ${EXP_DIR}/epoch-*.pt 2>/dev/null | sort -V | tail -1 | grep -oP 'epoch-\K[0-9]+')
if [ -z "$LAST_EPOCH" ]; then
    echo "错误: 找不到已有的checkpoint"
    exit 1
fi

START_EPOCH=$((LAST_EPOCH + 1))
NUM_EPOCHS=100  # 设置足够大，由早停决定实际结束点

echo "=========================================="
echo "V4模型断点续训"
echo "=========================================="
echo "继续训练从: epoch-${LAST_EPOCH}.pt"
echo "起始epoch: ${START_EPOCH}"
echo "最大epochs: ${NUM_EPOCHS}"
echo "输出目录: ${EXP_DIR}"
echo ""

/data/workspace/llm/anaconda3/envs/kws-train/bin/python ./zipformer/finetune.py \
    --world-size 1 \
    --num-epochs ${NUM_EPOCHS} \
    --start-epoch ${START_EPOCH} \
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
    --base-lr 0.0001 \
    --lr-epochs 50 \
    --lr-batches 50000 \
    --max-duration 200 \
    --bucketing-sampler 1 \
    --num-buckets 5 \
    --num-workers 2 \
    2>&1 | tee -a ${EXP_DIR}/train.log

echo ""
echo "=========================================="
echo "训练完成！"
echo "模型保存在: ${EXP_DIR}"
echo "=========================================="
