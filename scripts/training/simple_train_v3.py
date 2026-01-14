#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple KWS Training Script V3
直接使用现有的icefall训练框架进行微调
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data_structure():
    """准备icefall期望的数据结构"""
    base_dir = Path("/data/workspace/llm/keyword-spotting")
    kws_dir = base_dir / "icefall/egs/wenetspeech/KWS"
    
    # 创建data/fbank目录
    fbank_dir = kws_dir / "data/fbank"
    fbank_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制我们的V3 manifests
    manifests_v3_dir = base_dir / "data/manifests_v3"
    
    if manifests_v3_dir.exists():
        # 复制训练数据
        train_cuts_src = manifests_v3_dir / "kws_recordings_train_v3.jsonl.gz"
        train_supervisions_src = manifests_v3_dir / "kws_supervisions_train_v3.jsonl.gz"
        
        if train_cuts_src.exists():
            # 重命名为icefall期望的格式
            shutil.copy2(train_cuts_src, fbank_dir / "nihaowenwen_cuts_train.jsonl.gz")
            logging.info(f"Copied training cuts to {fbank_dir / 'nihaowenwen_cuts_train.jsonl.gz'}")
        
        if train_supervisions_src.exists():
            shutil.copy2(train_supervisions_src, fbank_dir / "nihaowenwen_supervisions_train.jsonl.gz")
            logging.info(f"Copied training supervisions")
        
        # 创建验证和测试数据（使用训练数据的子集）
        # 这里简化处理，实际应该有独立的验证集
        if train_cuts_src.exists():
            shutil.copy2(train_cuts_src, fbank_dir / "nihaowenwen_cuts_dev.jsonl.gz")
            shutil.copy2(train_cuts_src, fbank_dir / "nihaowenwen_cuts_test.jsonl.gz")
            logging.info("Created dev and test sets (copies of training set for now)")
    
    # 复制语言模型文件
    lang_src = base_dir / "data/lang_partial_tone"
    lang_dst = kws_dir / "data/lang_partial_tone"
    
    if lang_src.exists() and not lang_dst.exists():
        shutil.copytree(lang_src, lang_dst)
        logging.info(f"Copied language model to {lang_dst}")
    
    # 创建标记文件
    (fbank_dir / ".cn_speech_commands.done").touch()
    
    return kws_dir

def create_finetune_script():
    """创建微调脚本"""
    script_content = '''#!/bin/bash
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

# 复制基础模型
base_model=/data/workspace/llm/keyword-spotting/icefall-kws-zipformer-wenetspeech-20240219/exp/pretrained.pt
if [ -f "$base_model" ] && [ ! -f "$exp_dir/epoch-0.pt" ]; then
    echo "Copying base model..."
    cp "$base_model" "$exp_dir/epoch-0.pt"
fi

echo "Starting training with the following parameters:"
echo "  Epochs: $num_epochs"
echo "  Learning rate: $base_lr"
echo "  Max duration: $max_duration"
echo "  Experiment dir: $exp_dir"

# 运行微调
python ./finetune.py \\
  --world-size $world_size \\
  --num-epochs $num_epochs \\
  --start-epoch $start_epoch \\
  --use-fp16 $use_fp16 \\
  --exp-dir $exp_dir \\
  --max-duration $max_duration \\
  --base-lr $base_lr \\
  --lr-batches 3000 \\
  --lr-epochs 2 \\
  --seed 42 \\
  --save-every-n 1000 \\
  --keep-last-k 5 \\
  --average-period 100 \\
  --use-averaged-model true \\
  --tensorboard true \\
  2>&1 | tee $exp_dir/log/train.log

echo "Training completed! Logs saved to $exp_dir/log/train.log"
'''
    
    script_path = Path("/data/workspace/llm/keyword-spotting/scripts/run_simple_train_v3.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path

def run_training():
    """运行训练"""
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("Simple KWS Training V3")
    logging.info("=" * 60)
    
    # 准备数据结构
    logging.info("Preparing data structure...")
    kws_dir = prepare_data_structure()
    
    # 创建训练脚本
    logging.info("Creating training script...")
    script_path = create_finetune_script()
    
    logging.info(f"Training script created: {script_path}")
    logging.info("Starting training...")
    
    # 运行训练
    try:
        result = subprocess.run(
            [str(script_path)],
            cwd=kws_dir,
            capture_output=False,  # 让输出直接显示
            text=True,
            timeout=7200  # 2小时超时
        )
        
        if result.returncode == 0:
            logging.info("Training completed successfully!")
            
            # 显示结果
            exp_dir = Path("/data/workspace/llm/keyword-spotting/exp/kws_finetune_v3")
            if exp_dir.exists():
                logging.info(f"Model checkpoints saved in: {exp_dir}")
                
                # 列出生成的文件
                checkpoints = list(exp_dir.glob("*.pt"))
                if checkpoints:
                    logging.info("Generated checkpoints:")
                    for cp in sorted(checkpoints):
                        logging.info(f"  {cp.name}")
                
                log_file = exp_dir / "log/train.log"
                if log_file.exists():
                    logging.info(f"Training log: {log_file}")
            
            return True
        else:
            logging.error(f"Training failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("Training timed out after 2 hours")
        return False
    except Exception as e:
        logging.error(f"Training failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = run_training()
    sys.exit(0 if success else 1)