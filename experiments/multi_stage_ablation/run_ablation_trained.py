#!/usr/bin/env python3
"""
使用训练好的验证器运行消融实验
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import AblationConfig, AblationResult
from stage1.prefix_detector import PrefixDetector
from stage2 import DTWVerifier, CNNVerifier, MLPVerifier, BaseVerifier
from utils import load_audio, extract_suffix


class TrainedAblationExperiment:
    """使用训练好的模型进行消融实验"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.stage1_detector: Optional[PrefixDetector] = None
        self.verifiers: Dict[str, BaseVerifier] = {}
        self.results: Dict[str, AblationResult] = {}
    
    def setup(self) -> None:
        """初始化实验"""
        print("=" * 60)
        print("多阶段关键词检测消融实验（训练模型版）")
        print("=" * 60)
        
        # 初始化阶段1检测器
        print("\n[1/2] 初始化阶段1检测器...")
        self.stage1_detector = PrefixDetector(
            model_dir=self.config.model_dir,
            threshold=self.config.stage1_threshold,
            use_int8=self.config.stage1_use_int8
        )
        self.stage1_detector.load_model()
        print("阶段1检测器初始化完成")
        
        # 初始化阶段2验证器
        print("\n[2/2] 初始化阶段2验证器...")
        
        # DTW - 调整阈值
        dtw_verifier = DTWVerifier(
            threshold=200.0,  # 放宽阈值
            n_mfcc=13,
            template_dir=self.config.dtw_template_dir
        )
        self.verifiers["dtw"] = dtw_verifier
        print("  - dtw: 已初始化 (threshold=200)")
        
        # CNN - 使用训练好的模型
        cnn_path = Path(__file__).parent / "models" / "cnn_verifier.pt"
        if cnn_path.exists():
            cnn_verifier = CNNVerifier(threshold=0.5, model_path=str(cnn_path))
            self.verifiers["cnn"] = cnn_verifier
            print(f"  - cnn: 已初始化 (使用训练模型)")
        else:
            print(f"  - cnn: 跳过 (模型文件不存在)")
        
        # MLP - 使用训练好的模型
        mlp_path = Path(__file__).parent / "models" / "mlp_verifier.pt"
        if mlp_path.exists():
            mlp_verifier = MLPVerifier(threshold=0.5, model_path=str(mlp_path))
            self.verifiers["mlp"] = mlp_verifier
            print(f"  - mlp: 已初始化 (使用训练模型)")
        else:
            print(f"  - mlp: 跳过 (模型文件不存在)")
        
        print(f"\n共初始化 {len(self.verifiers)} 个验证器")
    
    def get_audio_files(self) -> tuple:
        """获取测试音频文件列表"""
        test_path = Path(self.config.test_data_path)
        positive_dir = test_path / self.config.positive_dir
        negative_dir = test_path / self.config.negative_dir
        
        positive_files = sorted(positive_dir.glob("*.wav"))
        negative_files = sorted(negative_dir.glob("*.wav"))
        
        return positive_files, negative_files
    
    def run_baseline(self) -> AblationResult:
        """运行基线评估"""
        print("\n" + "-" * 60)
        print("评估基线（仅阶段1）")
        print("-" * 60)
        
        positive_files, negative_files = self.get_audio_files()
        
        result = AblationResult(verifier_name="baseline")
        result.total_positive = len(positive_files)
        result.total_negative = len(negative_files)
        
        total_time = 0
        total_audio_duration = 0
        
        for audio_path in positive_files:
            samples, sr = load_audio(str(audio_path))
            audio_duration_ms = len(samples) / sr * 1000
            total_audio_duration += audio_duration_ms
            
            detected, _, _, proc_time = self.stage1_detector.detect_file(str(audio_path))
            total_time += proc_time
            if detected:
                result.stage1_passed_positive += 1
                result.true_positive += 1
            else:
                result.false_negative += 1
        
        for audio_path in negative_files:
            samples, sr = load_audio(str(audio_path))
            audio_duration_ms = len(samples) / sr * 1000
            total_audio_duration += audio_duration_ms
            
            detected, _, _, proc_time = self.stage1_detector.detect_file(str(audio_path))
            total_time += proc_time
            if detected:
                result.stage1_passed_negative += 1
                result.false_positive += 1
            else:
                result.true_negative += 1
        
        total_samples = len(positive_files) + len(negative_files)
        result.avg_process_time_ms = total_time / total_samples
        result.avg_audio_duration_ms = total_audio_duration / total_samples
        
        self.results["baseline"] = result
        self._print_result(result)
        
        return result
    
    def run_with_verifier(self, verifier_name: str) -> AblationResult:
        """使用指定验证器运行评估"""
        if verifier_name not in self.verifiers:
            return None
        
        print("\n" + "-" * 60)
        print(f"评估方案: 阶段1 + {verifier_name.upper()}")
        print("-" * 60)
        
        verifier = self.verifiers[verifier_name]
        if not verifier.is_loaded():
            verifier.load_model()
        
        positive_files, negative_files = self.get_audio_files()
        
        result = AblationResult(verifier_name=verifier_name)
        result.total_positive = len(positive_files)
        result.total_negative = len(negative_files)
        
        total_time = 0
        total_audio_duration = 0
        
        # 正样本
        print(f"评估正样本 ({len(positive_files)} 个)...")
        for audio_path in positive_files:
            start_time = time.perf_counter()
            
            samples, sr = load_audio(str(audio_path))
            audio_duration_ms = len(samples) / sr * 1000
            total_audio_duration += audio_duration_ms
            
            detected, _, _ = self.stage1_detector.detect(samples, sr)
            
            final_accepted = False
            if detected:
                result.stage1_passed_positive += 1
                suffix = extract_suffix(
                    samples, sr,
                    start_ratio=self.config.suffix_start_ratio,
                    min_duration_ms=self.config.suffix_min_duration_ms,
                    max_duration_ms=self.config.suffix_max_duration_ms
                )
                accepted, _ = verifier.verify(suffix, sr)
                if accepted:
                    result.stage2_passed_positive += 1
                    final_accepted = True
            
            total_time += (time.perf_counter() - start_time) * 1000
            
            if final_accepted:
                result.true_positive += 1
            else:
                result.false_negative += 1
        
        # 负样本
        print(f"评估负样本 ({len(negative_files)} 个)...")
        for audio_path in negative_files:
            start_time = time.perf_counter()
            
            samples, sr = load_audio(str(audio_path))
            audio_duration_ms = len(samples) / sr * 1000
            total_audio_duration += audio_duration_ms
            
            detected, _, _ = self.stage1_detector.detect(samples, sr)
            
            final_accepted = False
            if detected:
                result.stage1_passed_negative += 1
                suffix = extract_suffix(
                    samples, sr,
                    start_ratio=self.config.suffix_start_ratio,
                    min_duration_ms=self.config.suffix_min_duration_ms,
                    max_duration_ms=self.config.suffix_max_duration_ms
                )
                accepted, _ = verifier.verify(suffix, sr)
                if accepted:
                    result.stage2_passed_negative += 1
                    final_accepted = True
            
            total_time += (time.perf_counter() - start_time) * 1000
            
            if final_accepted:
                result.false_positive += 1
            else:
                result.true_negative += 1
        
        total_samples = len(positive_files) + len(negative_files)
        result.avg_process_time_ms = total_time / total_samples
        result.avg_audio_duration_ms = total_audio_duration / total_samples
        
        self.results[verifier_name] = result
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: AblationResult) -> None:
        """打印结果"""
        print(f"\n结果 [{result.verifier_name}]:")
        print(f"  阶段1通过: 正样本 {result.stage1_passed_positive}/{result.total_positive}, "
              f"负样本 {result.stage1_passed_negative}/{result.total_negative}")
        if result.verifier_name != "baseline":
            s1_pos = result.stage1_passed_positive if result.stage1_passed_positive > 0 else 1
            s1_neg = result.stage1_passed_negative if result.stage1_passed_negative > 0 else 1
            print(f"  阶段2通过: 正样本 {result.stage2_passed_positive}/{result.stage1_passed_positive}, "
                  f"负样本 {result.stage2_passed_negative}/{result.stage1_passed_negative}")
        print(f"  FRR: {result.frr*100:.2f}%")
        print(f"  FAR: {result.far*100:.2f}%")
        print(f"  准确率: {result.accuracy*100:.2f}%")
        print(f"  F1: {result.f1*100:.2f}%")
        print(f"  平均时间: {result.avg_process_time_ms:.2f}ms")
        print(f"  RTF: {result.rtf:.4f}")
        
        if result.meets_target(self.config):
            print(f"  ✓ 达到目标")
        else:
            print(f"  ✗ 未达目标 (目标: FRR≤{self.config.target_frr*100}%, FAR≤{self.config.target_far*100}%)")
    
    def run_all(self) -> Dict[str, AblationResult]:
        """运行所有评估"""
        self.run_baseline()
        for name in self.verifiers:
            try:
                self.run_with_verifier(name)
            except Exception as e:
                print(f"\n{name} 评估失败: {e}")
                import traceback
                traceback.print_exc()
        return self.results
    
    def generate_report(self) -> str:
        """生成报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        lines = []
        lines.append("=" * 80)
        lines.append("多阶段关键词检测消融实验报告（训练模型版）")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        lines.append("\n## 配置")
        lines.append(f"- 正样本: {self.config.positive_samples}")
        lines.append(f"- 负样本: {self.config.negative_samples}")
        lines.append(f"- 目标: FRR≤{self.config.target_frr*100}%, FAR≤{self.config.target_far*100}%")
        
        lines.append("\n## 结果对比")
        lines.append("")
        lines.append(f"{'方案':<12} {'FRR':>8} {'FAR':>8} {'Acc':>8} {'F1':>8} {'Time':>10} {'RTF':>8} {'达标':>6}")
        lines.append("-" * 80)
        
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: (x[1].far, x[1].frr)
        )
        
        for name, result in sorted_results:
            meets = "✓" if result.meets_target(self.config) else "✗"
            lines.append(
                f"{name:<12} "
                f"{result.frr*100:>7.2f}% "
                f"{result.far*100:>7.2f}% "
                f"{result.accuracy*100:>7.2f}% "
                f"{result.f1*100:>7.2f}% "
                f"{result.avg_process_time_ms:>9.2f}ms "
                f"{result.rtf:>8.4f} "
                f"{meets:>6}"
            )
        
        lines.append("-" * 80)
        
        # 找最优方案
        best = None
        for name, result in sorted_results:
            if result.meets_target(self.config):
                if best is None or result.far < best[1].far:
                    best = (name, result)
        
        lines.append("\n## 结论")
        if best:
            baseline_far = self.results["baseline"].far
            improvement = (baseline_far - best[1].far) * 100
            lines.append(f"\n**最优方案**: {best[0]}")
            lines.append(f"- FRR: {best[1].frr*100:.2f}%")
            lines.append(f"- FAR: {best[1].far*100:.2f}%")
            lines.append(f"- FAR降低: {improvement:.2f}% (相比基线)")
        else:
            lines.append("\n**无方案达到目标**")
            lines.append("建议调整阈值或收集更多训练数据")
        
        report = "\n".join(lines)
        
        # 保存
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = results_dir / f"trained_ablation_report_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存: {report_path}")
        
        json_path = results_dir / f"trained_ablation_results_{timestamp}.json"
        json_data = {name: result.to_dict() for name, result in self.results.items()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str,
                        default="/data/workspace/llm/keyword-spotting/exp/kws_finetune_v3")
    parser.add_argument("--test-data", type=str,
                        default="/data/workspace/llm/audio-classification/dataset/kws_test_data_merged")
    
    args = parser.parse_args()
    
    config = AblationConfig(
        model_dir=args.model_dir,
        test_data_path=args.test_data,
    )
    
    experiment = TrainedAblationExperiment(config)
    experiment.setup()
    experiment.run_all()
    report = experiment.generate_report()
    print("\n" + report)


if __name__ == "__main__":
    main()
