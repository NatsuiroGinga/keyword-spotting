#!/bin/bash
# 带收敛检测的V4断点续训
# 当检测到收敛时自动停止训练

set -e

BASE_DIR=/data/workspace/llm/keyword-spotting
EXP_DIR=${BASE_DIR}/experiments/baseline_streaming/exp_v4
LOG_FILE=${EXP_DIR}/train.log
CONVERGENCE_SCRIPT=${BASE_DIR}/experiments/baseline_streaming/convergence_monitor.py
RESUME_SCRIPT=${BASE_DIR}/experiments/baseline_streaming/resume_train_v4.sh

# 收敛检测参数
PATIENCE=5           # 连续5个epoch
THRESHOLD=0.005      # 下降率<0.5%
CHECK_INTERVAL=60    # 检查间隔60秒
MAX_WAIT=180         # 最大等待180分钟

echo "=========================================="
echo "V4模型断点续训（带收敛检测）"
echo "=========================================="
echo "收敛条件: 连续${PATIENCE}个epoch, 下降率<$(echo "${THRESHOLD}*100" | bc)%"
echo "最大训练时间: ${MAX_WAIT}分钟"
echo ""

# 记录日志起始行数（用于后续只解析新增内容）
if [ -f "$LOG_FILE" ]; then
    INITIAL_LINES=$(wc -l < "$LOG_FILE")
else
    INITIAL_LINES=0
fi

# 启动训练（后台运行）
echo "启动训练进程..."
chmod +x ${RESUME_SCRIPT}
nohup bash ${RESUME_SCRIPT} > /dev/null 2>&1 &
TRAIN_PID=$!
echo "训练进程PID: ${TRAIN_PID}"

# 等待日志文件生成
sleep 10

# 定义清理函数
cleanup() {
    echo ""
    echo "停止训练进程..."
    if kill -0 $TRAIN_PID 2>/dev/null; then
        kill $TRAIN_PID 2>/dev/null || true
        # 等待进程结束
        wait $TRAIN_PID 2>/dev/null || true
    fi
    echo "训练进程已停止"
}

# 捕获退出信号
trap cleanup EXIT

# 启动收敛监控
echo ""
echo "启动收敛监控..."
/data/workspace/llm/anaconda3/envs/kws-train/bin/python ${CONVERGENCE_SCRIPT} \
    --log-path ${LOG_FILE} \
    --patience ${PATIENCE} \
    --threshold ${THRESHOLD} \
    --interval ${CHECK_INTERVAL} \
    --max-wait ${MAX_WAIT}

# 获取监控结果
MONITOR_EXIT=$?

# 停止训练
echo ""
echo "收敛监控结束，停止训练进程..."
if kill -0 $TRAIN_PID 2>/dev/null; then
    kill $TRAIN_PID 2>/dev/null || true
fi

# 找到最佳epoch并导出
echo ""
echo "=========================================="
echo "训练结束，准备导出最佳模型"
echo "=========================================="

# 找到最佳验证loss的epoch
BEST_EPOCH=$(ls ${EXP_DIR}/epoch-*.pt 2>/dev/null | sort -V | tail -1 | grep -oP 'epoch-\K[0-9]+')

if [ -n "$BEST_EPOCH" ]; then
    echo "最新epoch: ${BEST_EPOCH}"
    echo "最佳验证模型: ${EXP_DIR}/best-valid-loss.pt"
    
    # 提示后续操作
    echo ""
    echo "后续操作:"
    echo "  1. 导出ONNX: bash experiments/baseline_streaming/export_v4_onnx.sh"
    echo "  2. 评估性能: python experiments/baseline_streaming/run_full_evaluation.py"
fi

echo ""
echo "完成！"
