#!/usr/bin/env python3
"""
消融实验：对比原始预训练模型 vs V3 微调模型

实验设计：
1. 控制变量：相同的测试数据集、相同的评估指标、相同的阈值搜索范围
2. 自变量：模型类型（原始预训练 / V3 微调）
3. 因变量：FAR、FRR、准确率、F1、RTF

实验假设：
- H0: 原始模型与V3模型在唤醒词检测任务上无显著差异
- H1: V3微调模型在目标唤醒词检测上显著优于原始模型
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import sherpa_onnx
import soundfile as sf


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 模型配置
    v3_model_dir: str = "exp/kws_finetune_v3"
    pretrained_model_dir: str = "exp/kws_finetune"  # V1 作为基线
    
    # 测试数据
    test_dir: str = "data/all"
    
    # 评估参数
    threshold_range: Tuple[float, float, float] = (0.0, 1.0, 0.05)  # (min, max, step)
    keywords_score_range: Tuple[float, float, float] = (0.5, 3.0, 0.5)
    
    # 正样本关键词（文件名包含这些词视为正样本）
    positive_keywords: List[str] = field(default_factory=lambda: ["你好真真", "你好珍珍"])
    
    # 音频参数
    sample_rate: int = 16000
    chunk_ms: int = 100  # 流式块大小(ms)
    
    # 输出配置
    output_dir: str = "experiments/streaming_cnn_kws/ablation_results"


@dataclass
class EvaluationResult:
    """单次评估结果"""
    model_name: str
    threshold: float
    keywords_score: float
    
    # 核心指标
    tp: int = 0  # True Positive
    fp: int = 0  # False Positive
    tn: int = 0  # True Negative
    fn: int = 0  # False Negative
    
    # 计算指标
    far: float = 0.0  # False Accept Rate = FP / (FP + TN)
    frr: float = 0.0  # False Reject Rate = FN / (FN + TP)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    
    # 性能指标
    total_audio_duration: float = 0.0  # 秒
    total_processing_time: float = 0.0  # 秒
    rtf: float = 0.0  # Real-Time Factor
    
    # 详细结果
    false_accepts: List[str] = field(default_factory=list)
    false_rejects: List[str] = field(default_factory=list)
    
    def compute_metrics(self):
        """计算衍生指标"""
        # FAR = FP / (FP + TN)，即负样本被误判为正的比例
        if self.fp + self.tn > 0:
            self.far = self.fp / (self.fp + self.tn)
        
        # FRR = FN / (FN + TP)，即正样本被漏检的比例
        if self.fn + self.tp > 0:
            self.frr = self.fn / (self.fn + self.tp)
        
        # Precision = TP / (TP + FP)
        if self.tp + self.fp > 0:
            self.precision = self.tp / (self.tp + self.fp)
        
        # Recall = TP / (TP + FN) = 1 - FRR
        if self.tp + self.fn > 0:
            self.recall = self.tp / (self.tp + self.fn)
        
        # F1 Score
        if self.precision + self.recall > 0:
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        
        # Accuracy
        total = self.tp + self.fp + self.tn + self.fn
        if total > 0:
            self.accuracy = (self.tp + self.tn) / total
        
        # RTF
        if self.total_audio_duration > 0:
            self.rtf = self.total_processing_time / self.total_audio_duration


class StreamingKWSEvaluator:
    """流式 KWS 评估器"""
    
    def __init__(self, model_dir: Path, config: ExperimentConfig):
        self.model_dir = model_dir
        self.config = config
        self.kws = None
        
    def _find_onnx_files(self) -> Tuple[str, str, str]:
        """查找 ONNX 文件"""
        encoder_files = list(self.model_dir.glob("encoder*.int8.onnx"))
        decoder_files = list(self.model_dir.glob("decoder*.int8.onnx"))
        joiner_files = list(self.model_dir.glob("joiner*.int8.onnx"))
        
        if not encoder_files or not decoder_files or not joiner_files:
            raise FileNotFoundError(f"ONNX files not found in {self.model_dir}")
        
        return str(encoder_files[0]), str(decoder_files[0]), str(joiner_files[0])
    
    def load(self, keywords_score: float = 1.5, keywords_threshold: float = 0.25):
        """加载模型"""
        encoder, decoder, joiner = self._find_onnx_files()
        
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=str(self.model_dir / "tokens.txt"),
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=str(self.model_dir / "keywords.txt"),
            num_threads=2,
            keywords_score=keywords_score,
            keywords_threshold=keywords_threshold,
        )
    
    def evaluate_file(self, audio_path: Path) -> Tuple[bool, float]:
        """
        流式评估单个音频文件
        
        Returns:
            (detected, processing_time)
        """
        # 读取音频
        audio, sr = sf.read(str(audio_path), dtype='float32')
        if sr != self.config.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
        
        # 创建流
        stream = self.kws.create_stream()
        
        # 流式处理
        chunk_size = int(self.config.sample_rate * self.config.chunk_ms / 1000)
        detected = False
        
        start_time = time.time()
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            stream.accept_waveform(self.config.sample_rate, chunk)
            
            while self.kws.is_ready(stream):
                self.kws.decode_stream(stream)
            
            result = self.kws.get_result(stream)
            if result:
                detected = True
                break
        
        processing_time = time.time() - start_time
        audio_duration = len(audio) / self.config.sample_rate
        
        return detected, processing_time, audio_duration


class AblationExperiment:
    """消融实验主类"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent
        self.results = {}
        
    def _is_positive_sample(self, filename: str) -> bool:
        """判断是否为正样本"""
        for kw in self.config.positive_keywords:
            if kw in filename:
                return True
        return False
    
    def _load_test_files(self) -> Tuple[List[Path], List[Path]]:
        """加载测试文件，分为正负样本"""
        test_dir = self.project_root / self.config.test_dir
        positive_files = []
        negative_files = []
        
        for audio_file in sorted(test_dir.glob("*.wav")):
            if self._is_positive_sample(audio_file.name):
                positive_files.append(audio_file)
            else:
                negative_files.append(audio_file)
        
        return positive_files, negative_files
    
    def evaluate_model(
        self,
        model_dir: Path,
        model_name: str,
        positive_files: List[Path],
        negative_files: List[Path],
        keywords_score: float,
        threshold: float,
    ) -> EvaluationResult:
        """评估单个模型配置"""
        
        evaluator = StreamingKWSEvaluator(model_dir, self.config)
        evaluator.load(keywords_score=keywords_score, keywords_threshold=threshold)
        
        result = EvaluationResult(
            model_name=model_name,
            threshold=threshold,
            keywords_score=keywords_score,
        )
        
        # 评估正样本
        for audio_file in positive_files:
            detected, proc_time, duration = evaluator.evaluate_file(audio_file)
            result.total_processing_time += proc_time
            result.total_audio_duration += duration
            
            if detected:
                result.tp += 1
            else:
                result.fn += 1
                result.false_rejects.append(audio_file.name)
        
        # 评估负样本
        for audio_file in negative_files:
            detected, proc_time, duration = evaluator.evaluate_file(audio_file)
            result.total_processing_time += proc_time
            result.total_audio_duration += duration
            
            if detected:
                result.fp += 1
                result.false_accepts.append(audio_file.name)
            else:
                result.tn += 1
        
        result.compute_metrics()
        return result
    
    def run_threshold_search(
        self,
        model_dir: Path,
        model_name: str,
        positive_files: List[Path],
        negative_files: List[Path],
        keywords_score: float = 1.5,
    ) -> List[EvaluationResult]:
        """对单个模型进行阈值搜索"""
        
        results = []
        min_th, max_th, step = self.config.threshold_range
        
        for threshold in np.arange(min_th, max_th + step, step):
            threshold = round(threshold, 2)
            print(f"  Testing {model_name} @ threshold={threshold:.2f}, score={keywords_score:.1f}")
            
            result = self.evaluate_model(
                model_dir, model_name, positive_files, negative_files,
                keywords_score, threshold
            )
            results.append(result)
            
            print(f"    FAR={result.far*100:.2f}%, FRR={result.frr*100:.2f}%, "
                  f"F1={result.f1*100:.2f}%, RTF={result.rtf:.4f}")
        
        return results
    
    def run_full_ablation(self):
        """运行完整的消融实验"""
        
        print("=" * 60)
        print("消融实验：原始模型 vs V3 微调模型")
        print("=" * 60)
        
        # 加载测试文件
        positive_files, negative_files = self._load_test_files()
        print(f"\n测试集统计：")
        print(f"  正样本: {len(positive_files)} 个文件")
        print(f"  负样本: {len(negative_files)} 个文件")
        print(f"  总计: {len(positive_files) + len(negative_files)} 个文件")
        
        # 定义实验模型
        models = {
            "V1_Baseline": self.project_root / self.config.pretrained_model_dir,
            "V3_Finetuned": self.project_root / self.config.v3_model_dir,
        }
        
        # 检查 V2 模型是否存在
        v2_path = self.project_root / "exp/kws_finetune_v2"
        if v2_path.exists():
            models["V2_Negative"] = v2_path
        
        all_results = {}
        
        # 对每个模型进行评估
        for model_name, model_dir in models.items():
            print(f"\n{'='*40}")
            print(f"评估模型: {model_name}")
            print(f"模型路径: {model_dir}")
            print(f"{'='*40}")
            
            # 使用固定的 keywords_score 进行阈值搜索
            results = self.run_threshold_search(
                model_dir, model_name,
                positive_files, negative_files,
                keywords_score=1.5
            )
            all_results[model_name] = results
        
        self.results = all_results
        return all_results
    
    def find_optimal_threshold(
        self,
        results: List[EvaluationResult],
        far_target: float = 0.10,
        frr_target: float = 0.05,
    ) -> Optional[EvaluationResult]:
        """
        找到满足目标的最优阈值
        优先级：FAR < target 且 FRR < target 的最高 F1
        """
        valid_results = [r for r in results if r.far <= far_target and r.frr <= frr_target]
        
        if valid_results:
            return max(valid_results, key=lambda x: x.f1)
        
        # 如果没有同时满足的，找 FAR 最接近目标的
        return min(results, key=lambda x: abs(x.far - far_target))
    
    def generate_report(self) -> str:
        """生成实验报告"""
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("消融实验报告：原始模型 vs V3 微调模型")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # 实验设置
        report_lines.append("## 1. 实验设置")
        report_lines.append("")
        report_lines.append("### 1.1 控制变量")
        report_lines.append(f"- 测试数据集: {self.config.test_dir}")
        report_lines.append(f"- 正样本关键词: {self.config.positive_keywords}")
        report_lines.append(f"- 音频采样率: {self.config.sample_rate} Hz")
        report_lines.append(f"- 流式块大小: {self.config.chunk_ms} ms")
        report_lines.append(f"- 阈值搜索范围: {self.config.threshold_range}")
        report_lines.append("")
        
        report_lines.append("### 1.2 自变量")
        report_lines.append("- 模型类型：V1 基线模型 / V2 负样本增强 / V3 微调模型")
        report_lines.append("")
        
        report_lines.append("### 1.3 因变量（评估指标）")
        report_lines.append("- FAR (False Accept Rate): 误唤醒率，负样本被误判为正的比例")
        report_lines.append("- FRR (False Reject Rate): 漏检率，正样本被漏检的比例")
        report_lines.append("- Precision: 精确率")
        report_lines.append("- Recall: 召回率")
        report_lines.append("- F1 Score: F1 分数")
        report_lines.append("- RTF: 实时因子")
        report_lines.append("")
        
        # 各模型最优结果
        report_lines.append("## 2. 实验结果")
        report_lines.append("")
        
        optimal_results = {}
        for model_name, results in self.results.items():
            optimal = self.find_optimal_threshold(results)
            optimal_results[model_name] = optimal
        
        # 结果表格
        report_lines.append("### 2.1 各模型最优配置对比")
        report_lines.append("")
        report_lines.append("| 模型 | 阈值 | FAR | FRR | Precision | Recall | F1 | RTF |")
        report_lines.append("|------|------|-----|-----|-----------|--------|----|----|")
        
        for model_name, result in optimal_results.items():
            if result:
                report_lines.append(
                    f"| {model_name} | {result.threshold:.2f} | "
                    f"{result.far*100:.2f}% | {result.frr*100:.2f}% | "
                    f"{result.precision*100:.2f}% | {result.recall*100:.2f}% | "
                    f"{result.f1*100:.2f}% | {result.rtf:.4f} |"
                )
        
        report_lines.append("")
        
        # FAR-FRR 权衡分析
        report_lines.append("### 2.2 FAR-FRR 权衡曲线数据")
        report_lines.append("")
        
        for model_name, results in self.results.items():
            report_lines.append(f"**{model_name}:**")
            report_lines.append("")
            report_lines.append("| 阈值 | FAR | FRR | F1 |")
            report_lines.append("|------|-----|-----|-----|")
            
            for r in results[::4]:  # 每4个取1个
                report_lines.append(
                    f"| {r.threshold:.2f} | {r.far*100:.2f}% | "
                    f"{r.frr*100:.2f}% | {r.f1*100:.2f}% |"
                )
            report_lines.append("")
        
        # 误报分析
        report_lines.append("### 2.3 误报样本分析")
        report_lines.append("")
        
        for model_name, result in optimal_results.items():
            if result and result.false_accepts:
                report_lines.append(f"**{model_name} 误报样本 (Top 10):**")
                for fa in result.false_accepts[:10]:
                    report_lines.append(f"- {fa}")
                report_lines.append("")
        
        # 漏检分析
        report_lines.append("### 2.4 漏检样本分析")
        report_lines.append("")
        
        for model_name, result in optimal_results.items():
            if result and result.false_rejects:
                report_lines.append(f"**{model_name} 漏检样本:**")
                for fr in result.false_rejects:
                    report_lines.append(f"- {fr}")
                report_lines.append("")
        
        # 结论
        report_lines.append("## 3. 统计分析与结论")
        report_lines.append("")
        
        if "V1_Baseline" in optimal_results and "V3_Finetuned" in optimal_results:
            v1 = optimal_results["V1_Baseline"]
            v3 = optimal_results["V3_Finetuned"]
            
            if v1 and v3:
                far_improvement = (v1.far - v3.far) * 100
                frr_change = (v3.frr - v1.frr) * 100
                f1_improvement = (v3.f1 - v1.f1) * 100
                
                report_lines.append("### 3.1 V3 相对于 V1 的改进")
                report_lines.append(f"- FAR 降低: {far_improvement:.2f}%")
                report_lines.append(f"- FRR 变化: {'+' if frr_change > 0 else ''}{frr_change:.2f}%")
                report_lines.append(f"- F1 提升: {f1_improvement:.2f}%")
                report_lines.append("")
        
        report_lines.append("### 3.2 结论")
        report_lines.append("")
        
        # 自动生成结论
        best_model = max(optimal_results.items(), key=lambda x: x[1].f1 if x[1] else 0)
        report_lines.append(f"基于实验结果，**{best_model[0]}** 在唤醒词检测任务上表现最佳：")
        if best_model[1]:
            report_lines.append(f"- F1 Score: {best_model[1].f1*100:.2f}%")
            report_lines.append(f"- FAR: {best_model[1].far*100:.2f}%")
            report_lines.append(f"- FRR: {best_model[1].frr*100:.2f}%")
        
        return "\n".join(report_lines)
    
    def save_results(self):
        """保存结果到文件"""
        output_dir = self.project_root / self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存详细 JSON 结果
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = [
                {
                    "threshold": r.threshold,
                    "keywords_score": r.keywords_score,
                    "tp": r.tp, "fp": r.fp, "tn": r.tn, "fn": r.fn,
                    "far": r.far, "frr": r.frr,
                    "precision": r.precision, "recall": r.recall,
                    "f1": r.f1, "accuracy": r.accuracy, "rtf": r.rtf,
                }
                for r in results
            ]
        
        json_path = output_dir / f"ablation_results_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # 保存报告
        report = self.generate_report()
        report_path = output_dir / f"ablation_report_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n结果已保存:")
        print(f"  - JSON: {json_path}")
        print(f"  - 报告: {report_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="消融实验：原始模型 vs V3 微调模型")
    parser.add_argument("--test-dir", default="data/all", help="测试数据目录")
    parser.add_argument("--output-dir", default="experiments/streaming_cnn_kws/ablation_results")
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument("--threshold-max", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        threshold_range=(args.threshold_min, args.threshold_max, args.threshold_step),
    )
    
    experiment = AblationExperiment(config)
    experiment.run_full_ablation()
    report = experiment.save_results()
    
    print("\n" + "=" * 70)
    print("实验报告摘要:")
    print("=" * 70)
    print(report)


if __name__ == "__main__":
    main()
