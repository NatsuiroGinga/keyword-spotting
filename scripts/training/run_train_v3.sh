#!/bin/bash
set -euo pipefail

# Training configuration
export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH=$PYTHONPATH:/data/workspace/llm/keyword-spotting/icefall

cd /data/workspace/llm/keyword-spotting/icefall/egs/wenetspeech/KWS/zipformer

# Training parameters
world_size=1
num_epochs=25
start_epoch=1
use_fp16=1
exp_dir=/data/workspace/llm/keyword-spotting/exp/kws_finetune_v3
max_duration=400
base_lr=1e-4

# Base model path
base_model=/data/workspace/llm/keyword-spotting/icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt

# Create experiment directory
mkdir -p $exp_dir

# Copy base model if needed
if [ ! -f "$exp_dir/epoch-0.pt" ] && [ -f "$base_model" ]; then
    echo "Copying base model to experiment directory..."
    cp "$base_model" "$exp_dir/epoch-0.pt"
fi

# Run training
python ./finetune.py \
  --world-size $world_size \
  --num-epochs $num_epochs \
  --start-epoch $start_epoch \
  --use-fp16 $use_fp16 \
  --exp-dir $exp_dir \
  --max-duration $max_duration \
  --base-lr $base_lr \
  --lr-batches 5000 \
  --lr-epochs 3 \
  --seed 42 \
  --save-every-n 2000 \
  --keep-last-k 10 \
  --average-period 200 \
  --use-averaged-model true \
  2>&1 | tee $exp_dir/train.log

echo "Training completed! Check logs in $exp_dir/train.log"
