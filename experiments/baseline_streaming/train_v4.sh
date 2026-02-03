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

# 创建输出目录
mkdir -p ${EXP_DIR}

cd ${BASE_DIR}/icefall/egs/wenetspeech/KWS

echo "=========================================="
echo "从真实人声数据微调V4 KWS模型"
echo "=========================================="
echo "预训练模型: ${PRETRAINED_CKPT}"
echo "数据目录: ${MANIFEST_DIR}"
echo "输出目录: ${EXP_DIR}"
echo "训练样本: 282 (60正+222负)"
echo ""

# 使用较小的学习率，因为真实数据量较少
# 增加epoch数以充分学习
/data/workspace/llm/anaconda3/envs/kws-train/bin/python ./zipformer/finetune.py \
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

echo ""
echo "=========================================="
echo "训练完成！"
echo "模型保存在: ${EXP_DIR}"
echo "=========================================="
