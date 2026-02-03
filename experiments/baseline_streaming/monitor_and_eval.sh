#!/bin/bash
# 监控训练进度并在完成后自动导出和评估

BASE_DIR=/data/workspace/llm/keyword-spotting
EXP_DIR=${BASE_DIR}/experiments/baseline_streaming/exp_v4
LOG_FILE=${EXP_DIR}/train.log

echo "=========================================="
echo "训练监控脚本"
echo "=========================================="

# 检查训练是否在运行
check_training() {
    pgrep -f "finetune.py.*exp_v4" > /dev/null
    return $?
}

# 获取当前epoch
get_current_epoch() {
    if [ -f "$LOG_FILE" ]; then
        grep "Epoch" "$LOG_FILE" | tail -1 | grep -oP 'Epoch \K[0-9]+' || echo "0"
    else
        echo "0"
    fi
}

# 获取最新验证损失
get_latest_val_loss() {
    if [ -f "$LOG_FILE" ]; then
        grep "validation:" "$LOG_FILE" | tail -1 | grep -oP 'loss=\K[0-9.]+' || echo "N/A"
    else
        echo "N/A"
    fi
}

# 监控循环
while true; do
    if check_training; then
        epoch=$(get_current_epoch)
        val_loss=$(get_latest_val_loss)
        echo "[$(date '+%H:%M:%S')] 训练中... Epoch: $epoch, Val Loss: $val_loss"
        sleep 60
    else
        echo "[$(date '+%H:%M:%S')] 训练已结束"
        break
    fi
done

# 训练结束后自动导出ONNX
echo ""
echo "开始导出ONNX模型..."
bash ${BASE_DIR}/experiments/baseline_streaming/export_v4_onnx.sh

echo ""
echo "开始对比测试..."
cd ${BASE_DIR}
/data/workspace/llm/anaconda3/envs/kws-train/bin/python experiments/baseline_streaming/run_comparison.py

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="
