#!/bin/bash
# 导出V4模型为ONNX格式

set -e

BASE_DIR=/data/workspace/llm/keyword-spotting
EXP_DIR=${BASE_DIR}/experiments/baseline_streaming/exp_v4
LANG_DIR=${BASE_DIR}/data/lang_partial_tone
TOKENS=${LANG_DIR}/tokens.txt

export PYTHONPATH=${BASE_DIR}/icefall:$PYTHONPATH

cd ${BASE_DIR}/icefall/egs/wenetspeech/KWS

echo "=========================================="
echo "导出V4模型为ONNX"
echo "=========================================="

# 找到最后一个epoch
LAST_EPOCH=$(ls ${EXP_DIR}/epoch-*.pt 2>/dev/null | sort -V | tail -1 | grep -oP 'epoch-\K[0-9]+')
if [ -z "$LAST_EPOCH" ]; then
    echo "错误: 找不到训练检查点"
    exit 1
fi
echo "使用epoch: ${LAST_EPOCH}"

/data/workspace/llm/anaconda3/envs/kws-train/bin/python ./zipformer/export-onnx-streaming.py \
    --exp-dir ${EXP_DIR} \
    --tokens ${TOKENS} \
    --epoch ${LAST_EPOCH} \
    --avg 1 \
    --use-averaged-model 0 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --decoder-dim 320 \
    --joiner-dim 320 \
    --num-encoder-layers "1,1,1,1,1,1" \
    --feedforward-dim "192,192,192,192,192,192" \
    --encoder-dim "128,128,128,128,128,128" \
    --encoder-unmasked-dim "128,128,128,128,128,128" \
    --causal 1

echo ""
echo "INT8量化..."
cd ${EXP_DIR}

# Find and quantize ONNX files
for model_type in encoder decoder joiner; do
    ONNX_FILE=$(ls -1 ${model_type}-epoch-*.onnx 2>/dev/null | grep -v int8 | head -1)
    if [ -n "$ONNX_FILE" ]; then
        echo "量化 $ONNX_FILE"
        /data/workspace/llm/anaconda3/envs/kws-train/bin/python -c "
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('${ONNX_FILE}', '${ONNX_FILE%.onnx}.int8.onnx', weight_type=QuantType.QInt8)
print('Done: ${ONNX_FILE%.onnx}.int8.onnx')
"
    fi
done

# 复制tokens和keywords
cp ${TOKENS} ${EXP_DIR}/
echo "n ǐ h ǎo zh ēn zh ēn @你好真真" > ${EXP_DIR}/keywords.txt

echo ""
echo "=========================================="
echo "导出完成！"
echo "模型文件在: ${EXP_DIR}"
echo "=========================================="
ls -la ${EXP_DIR}/*.onnx 2>/dev/null || echo "没有找到ONNX文件"
