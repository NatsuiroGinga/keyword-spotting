#!/bin/bash
# Export KWS model V2 to ONNX and quantize

set -e

PYTHON=/data/workspace/llm/anaconda3/envs/kws-train/bin/python
BASE_DIR="/data/workspace/llm/keyword-spotting"
EXP_DIR="${BASE_DIR}/exp/kws_finetune_v2"
ICEFALL_DIR="${BASE_DIR}/icefall/egs/wenetspeech/KWS"
TOKENS="${BASE_DIR}/data/lang_partial_tone/tokens.txt"

export PYTHONPATH=${BASE_DIR}/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

cd ${ICEFALL_DIR}

echo "Step 1: Exporting to ONNX (streaming)..."
${PYTHON} ./zipformer/export-onnx-streaming.py \
    --exp-dir ${EXP_DIR} \
    --tokens ${TOKENS} \
    --epoch 20 \
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

echo "Step 2: Quantizing to INT8..."
cd ${EXP_DIR}

# Find the generated ONNX files
ENCODER_ONNX=$(ls -1 encoder-epoch-*.onnx 2>/dev/null | grep -v int8 | head -1)
DECODER_ONNX=$(ls -1 decoder-epoch-*.onnx 2>/dev/null | grep -v int8 | head -1)
JOINER_ONNX=$(ls -1 joiner-epoch-*.onnx 2>/dev/null | grep -v int8 | head -1)

if [ -n "$ENCODER_ONNX" ]; then
    echo "Quantizing encoder: $ENCODER_ONNX"
    ${PYTHON} -c "
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('${ENCODER_ONNX}', '${ENCODER_ONNX%.onnx}.int8.onnx', weight_type=QuantType.QUInt8)
print('Encoder quantized successfully')
"
fi

if [ -n "$DECODER_ONNX" ]; then
    echo "Quantizing decoder: $DECODER_ONNX"
    ${PYTHON} -c "
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('${DECODER_ONNX}', '${DECODER_ONNX%.onnx}.int8.onnx', weight_type=QuantType.QUInt8)
print('Decoder quantized successfully')
"
fi

if [ -n "$JOINER_ONNX" ]; then
    echo "Quantizing joiner: $JOINER_ONNX"
    ${PYTHON} -c "
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('${JOINER_ONNX}', '${JOINER_ONNX%.onnx}.int8.onnx', weight_type=QuantType.QUInt8)
print('Joiner quantized successfully')
"
fi

echo "Step 3: Creating keywords.txt..."
echo "n ǐ h ǎo zh ēn zh ēn @你好真真" > ${EXP_DIR}/keywords.txt

echo "Step 4: Copying tokens.txt..."
cp ${TOKENS} ${EXP_DIR}/

echo "Export complete!"
echo "ONNX models saved to: ${EXP_DIR}"
ls -la ${EXP_DIR}/*.onnx 2>/dev/null || echo "No ONNX files found"
