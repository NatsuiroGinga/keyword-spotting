#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="${PYTHONPATH:-}:/data/workspace/llm/keyword-spotting/icefall"

cd /data/workspace/llm/keyword-spotting/icefall/egs/wenetspeech/KWS/zipformer

# 训练参数
world_size=1
num_epochs=20
start_epoch=1
use_fp16=1
exp_dir=/data/workspace/llm/keyword-spotting/exp/kws_finetune_v3
max_duration=300
base_lr=5e-5

# 创建实验目录
mkdir -p $exp_dir/log

# 基础模型路径
base_model=/data/workspace/llm/keyword-spotting/icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt

echo "Starting training with the following parameters:"
echo "  Epochs: $num_epochs"
echo "  Learning rate: $base_lr"
echo "  Max duration: $max_duration"
echo "  Experiment dir: $exp_dir"
echo "  Base model: $base_model"

# 运行微调
python ./finetune.py \
  --world-size $world_size \
  --num-epochs $num_epochs \
  --start-epoch $start_epoch \
  --use-fp16 $use_fp16 \
  --exp-dir $exp_dir \
  --max-duration $max_duration \
  --base-lr $base_lr \
  --lr-batches 3000 \
  --lr-epochs 2 \
  --seed 42 \
  --save-every-n 1000 \
  --keep-last-k 5 \
  --average-period 100 \
  --tensorboard true \
  --finetune-ckpt $base_model \
  --use-custom-kws-data true \
  --manifest-dir /data/workspace/llm/keyword-spotting/data/manifests_v3 \
  --lang-dir /data/workspace/llm/keyword-spotting/data/lang_partial_tone \
  2>&1 | tee $exp_dir/log/train.log

echo "Training completed! Logs saved to $exp_dir/log/train.log"
