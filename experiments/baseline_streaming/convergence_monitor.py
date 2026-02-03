#!/usr/bin/env python3
"""
收敛检测监控器

监控训练日志中的validation loss，判断是否达到收敛条件。
收敛条件：连续N个epoch的loss下降率均小于阈值。
"""

import re
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from collections import deque


class ConvergenceDetector:
    """收敛检测器"""
    
    def __init__(
        self,
        patience: int = 5,
        threshold: float = 0.005,  # 0.5%下降率阈值
        min_epochs: int = 5,       # 最少训练epoch数
    ):
        """
        Args:
            patience: 连续多少个epoch下降率小于阈值时判定为收敛
            threshold: 下降率阈值（小于此值视为趋于平稳）
            min_epochs: 最少训练epoch数，低于此值不判定收敛
        """
        self.patience = patience
        self.threshold = threshold
        self.min_epochs = min_epochs
        self.loss_history: List[Tuple[int, float]] = []
    
    def update(self, epoch: int, val_loss: float) -> bool:
        """
        更新loss历史并检测是否收敛
        
        Args:
            epoch: 当前epoch
            val_loss: 验证损失
            
        Returns:
            是否达到收敛条件
        """
        self.loss_history.append((epoch, val_loss))
        
        # 至少需要min_epochs个epoch
        if len(self.loss_history) < self.min_epochs:
            return False
        
        # 至少需要patience+1个点来计算patience个下降率
        if len(self.loss_history) < self.patience + 1:
            return False
        
        return self._check_convergence()
    
    def _check_convergence(self) -> bool:
        """检查最近patience个epoch的下降率是否均小于threshold"""
        recent = self.loss_history[-(self.patience + 1):]
        decrease_rates = []
        
        for i in range(1, len(recent)):
            prev_loss = recent[i-1][1]
            curr_loss = recent[i][1]
            
            if prev_loss > 0:
                # 正的下降率表示loss在减小
                rate = (prev_loss - curr_loss) / prev_loss
                decrease_rates.append(rate)
        
        # 如果所有下降率都小于阈值（包括负值，即loss上升），判定为收敛
        converged = all(abs(r) < self.threshold for r in decrease_rates)
        
        return converged
    
    def get_best_epoch(self) -> Tuple[int, float]:
        """获取最佳epoch（loss最低的epoch）"""
        if not self.loss_history:
            return -1, float('inf')
        return min(self.loss_history, key=lambda x: x[1])
    
    def get_summary(self) -> str:
        """获取当前状态摘要"""
        if len(self.loss_history) < 2:
            return "数据不足"
        
        recent = self.loss_history[-min(self.patience + 1, len(self.loss_history)):]
        first_loss = recent[0][1]
        last_loss = recent[-1][1]
        total_decrease = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0
        
        best_epoch, best_loss = self.get_best_epoch()
        
        return (f"最近{len(recent)}个epoch: loss从{first_loss:.4f}到{last_loss:.4f} "
                f"(变化{total_decrease:+.2f}%), 最佳epoch={best_epoch} (loss={best_loss:.4f})")


def parse_log_file(log_path: Path) -> List[Tuple[int, float]]:
    """从训练日志解析validation loss"""
    pattern = r"Epoch (\d+).*validation: loss=([\d.]+)"
    results = []
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                results.append((epoch, loss))
    
    return results


def monitor_training(
    log_path: Path,
    patience: int = 5,
    threshold: float = 0.005,
    check_interval: int = 30,
    max_wait_minutes: int = 120,
) -> Tuple[bool, int, float]:
    """
    监控训练进程，等待收敛
    
    Args:
        log_path: 训练日志路径
        patience: 收敛检测patience
        threshold: 收敛阈值
        check_interval: 检查间隔（秒）
        max_wait_minutes: 最大等待时间（分钟）
        
    Returns:
        (是否收敛, 最佳epoch, 最佳loss)
    """
    detector = ConvergenceDetector(patience=patience, threshold=threshold)
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    last_epoch = 0
    
    print(f"开始监控训练日志: {log_path}")
    print(f"收敛条件: 连续{patience}个epoch, 下降率<{threshold*100}%")
    print(f"最大等待时间: {max_wait_minutes}分钟")
    print()
    
    while True:
        if not log_path.exists():
            print(f"等待日志文件生成...")
            time.sleep(check_interval)
            continue
        
        # 解析日志
        history = parse_log_file(log_path)
        
        # 更新检测器
        for epoch, loss in history:
            if epoch > last_epoch:
                converged = detector.update(epoch, loss)
                print(f"[Epoch {epoch}] val_loss={loss:.4f} - {detector.get_summary()}")
                last_epoch = epoch
                
                if converged:
                    best_epoch, best_loss = detector.get_best_epoch()
                    print()
                    print("=" * 60)
                    print(f"✓ 检测到收敛!")
                    print(f"  最佳epoch: {best_epoch}")
                    print(f"  最佳loss: {best_loss:.4f}")
                    print("=" * 60)
                    return True, best_epoch, best_loss
        
        # 检查超时
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            print()
            print("=" * 60)
            print(f"✗ 达到最大等待时间 ({max_wait_minutes}分钟)")
            best_epoch, best_loss = detector.get_best_epoch()
            print(f"  当前最佳epoch: {best_epoch}")
            print(f"  当前最佳loss: {best_loss:.4f}")
            print("=" * 60)
            return False, best_epoch, best_loss
        
        # 等待下一次检查
        time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description="训练收敛检测监控器")
    parser.add_argument("--log-path", type=str, required=True,
                        help="训练日志文件路径")
    parser.add_argument("--patience", type=int, default=5,
                        help="收敛检测patience（默认5）")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="收敛阈值（默认0.5%）")
    parser.add_argument("--interval", type=int, default=30,
                        help="检查间隔秒数（默认30）")
    parser.add_argument("--max-wait", type=int, default=120,
                        help="最大等待分钟数（默认120）")
    parser.add_argument("--once", action="store_true",
                        help="只检查一次，不持续监控")
    
    args = parser.parse_args()
    
    log_path = Path(args.log_path)
    
    if args.once:
        # 单次检查模式
        if not log_path.exists():
            print(f"日志文件不存在: {log_path}")
            return
        
        history = parse_log_file(log_path)
        detector = ConvergenceDetector(patience=args.patience, threshold=args.threshold)
        
        for epoch, loss in history:
            converged = detector.update(epoch, loss)
        
        print(f"已处理 {len(history)} 个epoch")
        print(detector.get_summary())
        best_epoch, best_loss = detector.get_best_epoch()
        print(f"最佳epoch: {best_epoch}, 最佳loss: {best_loss:.4f}")
        print(f"是否收敛: {'是' if converged else '否'}")
    else:
        # 持续监控模式
        converged, best_epoch, best_loss = monitor_training(
            log_path=log_path,
            patience=args.patience,
            threshold=args.threshold,
            check_interval=args.interval,
            max_wait_minutes=args.max_wait,
        )


if __name__ == "__main__":
    main()
