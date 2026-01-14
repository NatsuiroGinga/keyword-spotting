#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KWS Model Training Script V3
Fine-tuning script based on existing icefall KWS training.

Usage:
cd /data/workspace/llm/keyword-spotting
export PYTHONPATH=$PYTHONPATH:/data/workspace/llm/keyword-spotting/icefall
python scripts/train_kws_v3.py
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup paths
BASE_DIR = Path("/data/workspace/llm/keyword-spotting")
ICEFALL_DIR = BASE_DIR / "icefall"
KWS_DIR = ICEFALL_DIR / "egs/wenetspeech/KWS/zipformer"

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def prepare_training_environment():
    """Prepare the training environment and data."""
    logging.info("Preparing training environment...")
    
    # Ensure directories exist
    exp_dir = BASE_DIR / "exp/kws_finetune_v3"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy manifests to expected location
    manifests_v3_dir = BASE_DIR / "data/manifests_v3"
    target_manifests_dir = BASE_DIR / "data/fbank"
    target_manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symbolic links or copy files
    if manifests_v3_dir.exists():
        for manifest_file in manifests_v3_dir.glob("*.jsonl.gz"):
            target_file = target_manifests_dir / manifest_file.name
            if not target_file.exists():
                os.symlink(manifest_file, target_file)
                logging.info(f"Linked {manifest_file} -> {target_file}")
    
    # Prepare language model files
    lang_dir = BASE_DIR / "data/lang_partial_tone"
    if not lang_dir.exists():
        logging.warning(f"Language directory {lang_dir} not found")
    
    return exp_dir

def create_training_script():
    """Create the actual training script using icefall framework."""
    
    script_content = '''#!/bin/bash
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
python ./finetune.py \\
  --world-size $world_size \\
  --num-epochs $num_epochs \\
  --start-epoch $start_epoch \\
  --use-fp16 $use_fp16 \\
  --exp-dir $exp_dir \\
  --max-duration $max_duration \\
  --base-lr $base_lr \\
  --lr-batches 5000 \\
  --lr-epochs 3 \\
  --seed 42 \\
  --save-every-n 2000 \\
  --keep-last-k 10 \\
  --average-period 200 \\
  --use-averaged-model true \\
  2>&1 | tee $exp_dir/train.log

echo "Training completed! Check logs in $exp_dir/train.log"
'''
    
    script_path = BASE_DIR / "scripts/run_train_v3.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    logging.info(f"Created training script: {script_path}")
    return script_path

def run_training():
    """Execute the training process."""
    logging.info("Starting KWS model training V3...")
    
    # Prepare environment
    exp_dir = prepare_training_environment()
    
    # Create and run training script
    script_path = create_training_script()
    
    logging.info(f"Experiment directory: {exp_dir}")
    logging.info(f"Training script: {script_path}")
    
    # Execute training
    try:
        result = subprocess.run([str(script_path)], 
                              cwd=BASE_DIR, 
                              capture_output=True, 
                              text=True, 
                              timeout=3600*4)  # 4 hour timeout
        
        if result.returncode == 0:
            logging.info("Training completed successfully!")
            logging.info("STDOUT:")
            logging.info(result.stdout)
        else:
            logging.error("Training failed!")
            logging.error("STDERR:")
            logging.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("Training timed out after 4 hours")
        return False
    except Exception as e:
        logging.error(f"Training failed with exception: {e}")
        return False
    
    return True

def main():
    """Main function."""
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("KWS Model Training V3")
    logging.info("=" * 60)
    
    success = run_training()
    
    if success:
        logging.info("Training process completed successfully!")
        
        # Show next steps
        exp_dir = BASE_DIR / "exp/kws_finetune_v3"
        logging.info(f"Model saved in: {exp_dir}")
        logging.info("Next steps:")
        logging.info("1. Export model to ONNX format")
        logging.info("2. Implement delayed decision inference logic")
        logging.info("3. Run comprehensive evaluation")
    else:
        logging.error("Training process failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())