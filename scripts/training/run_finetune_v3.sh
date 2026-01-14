#!/bin/bash
# KWS Fine-tuning Script V3 for "你好真真" wake word
# Using Kokoro TTS-generated data with improved negative samples
# Based on sherpa-onnx-kws-zipformer-wenetspeech-3.3M

set -e

# Configuration
export CUDA_VISIBLE_DEVICES="0"

# Use kws-train environment with full path
PYTHON=/data/workspace/llm/anaconda3/envs/kws-train/bin/python

# Paths
BASE_DIR="/data/workspace/llm/keyword-spotting"
ICEFALL_DIR="${BASE_DIR}/icefall/egs/wenetspeech/KWS"
EXP_DIR="${BASE_DIR}/exp/kws_finetune_v3"
LANG_DIR="${BASE_DIR}/data/lang_partial_tone"
MANIFEST_DIR="${BASE_DIR}/data/manifests_v3"
PRETRAINED_CKPT="${BASE_DIR}/icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt"

export PYTHONPATH=${BASE_DIR}/icefall:$PYTHONPATH

# Create experiment directory
mkdir -p ${EXP_DIR}

cd ${ICEFALL_DIR}

echo "========================================"
echo "Starting KWS fine-tuning V3"
echo "========================================"
echo "Pretrained checkpoint: ${PRETRAINED_CKPT}"
echo "Manifest directory: ${MANIFEST_DIR}"
echo "Experiment directory: ${EXP_DIR}"
echo "Data: 1000 positive + 4440 negative samples"
echo "========================================"

${PYTHON} ./zipformer/finetune.py \
    --world-size 1 \
    --num-epochs 30 \
    --start-epoch 1 \
    --exp-dir ${EXP_DIR} \
    --lang-dir ${LANG_DIR} \
    --manifest-dir ${MANIFEST_DIR} \
    --pinyin-type partial_with_tone \
    --use-fp16 1 \
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
    --base-lr 0.0003 \
    --lr-epochs 100 \
    --lr-batches 100000 \
    --finetune-ckpt ${PRETRAINED_CKPT} \
    --max-duration 500 \
    --bucketing-sampler 1 \
    --num-buckets 10 \
    --num-workers 2

echo "========================================"
echo "Fine-tuning V3 completed!"
echo "Checkpoints saved to: ${EXP_DIR}"
echo "========================================"
