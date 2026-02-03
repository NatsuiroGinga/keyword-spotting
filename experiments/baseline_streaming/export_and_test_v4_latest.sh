#!/bin/bash
# 导出最新V4模型(epoch-98)并测试

set -e
export CUDA_VISIBLE_DEVICES="0"

BASE_DIR=/data/workspace/llm/keyword-spotting
EXP_DIR=${BASE_DIR}/experiments/baseline_streaming/exp_v4
LANG_DIR=${BASE_DIR}/data/lang_partial_tone

export PYTHONPATH=${BASE_DIR}/icefall:$PYTHONPATH
cd ${BASE_DIR}/icefall/egs/wenetspeech/KWS

BEST_EPOCH=98

echo "=========================================="
echo "导出epoch-${BEST_EPOCH}模型到ONNX"
echo "=========================================="

/data/workspace/llm/anaconda3/envs/kws-train/bin/python ./zipformer/export-onnx-streaming.py \
    --exp-dir ${EXP_DIR} \
    --lang-dir ${LANG_DIR} \
    --epoch ${BEST_EPOCH} \
    --avg 1 \
    --decoder-dim 320 \
    --joiner-dim 320 \
    --num-encoder-layers "1,1,1,1,1,1" \
    --feedforward-dim "192,192,192,192,192,192" \
    --encoder-dim "128,128,128,128,128,128" \
    --encoder-unmasked-dim "128,128,128,128,128,128" \
    --causal 1 \
    --chunk-size 16 \
    --left-context-frames 64

echo ""
echo "导出完成！"
