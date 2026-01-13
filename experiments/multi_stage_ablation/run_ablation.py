#!/usr/bin/env python3
"""
多阶段关键词检测消融实验主入口

评估不同阶段2验证方案对FAR/FRR的影响
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

import numpy as np

from config import AblationConfig, AblationResult, VerifierResult
from stage1.prefix_detector import PrefixDetector
from stage2 import DTWVerifier, ASRVerifier, CNNVerifier, MLPVerifier, BaseVerifier
from utils import load_audio, extract_suffix, format_metrics_table, MetricsResult


class AblationExperiment:
    """消融实验类"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.stage1_detector: Optional[PrefixDetector] = None
        self.verifiers: Dict[str, BaseVerifier] = {}
        self.results: Dict[str, AblationResult] = {}
    
    def setup(self) -> None:
        """初始化实验"""
        print("=" * 60)
        print("多阶段关键词检测消融实验")
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
        
        for verifier_name in self.config.verifiers:
            verifier = self._create_verifier(verifier_name)
            if verifier:
                self.verifiers[verifier_name] = verifier
                print(f"  - {verifier_name}: 已初始化")
        
        print(f"\n共初始化 {len(self.verifiers)} 个验证器")
    
    def _create_verifier(self, name: str) -> Optional[BaseVerifier]:
        """创建验证器实例"""
        if name == "dtw":
            return DTWVerifier(
                threshold=self.config.dtw_threshold,
                n_mfcc=self.config.dtw_n_mfcc,
                template_dir=self.config.dtw_template_dir
            )
        elif name == "asr":
            return ASRVerifier(
                model_name=self.config.asr_model_name,
                target_text=self.config.asr_target_text,
                threshold=self.config.asr_threshold
            )
        elif name == "cnn":
            return CNNVerifier(
                threshold=self.config.cnn_threshold,
                model_path=self.config.cnn_model_path if self.config.cnn_model_path else None
            )
        elif name == "mlp":
            return MLPVerifier(
                threshold=self.config.mlp_threshold,
                model_path=self.config.mlp_model_path if self.config.mlp_model_path else None
            )
        else:
            print(f"  - {name}: 未知验证器类型")
            return None
    
    def get_audio_files(self) -> tuple:
        """获取测试音频文件列表"""
        test_path = Path(self.config.test_data_path)
        
        positive_dir = test_path / self.config.positive_dir
        negative_dir = test_path / self.config.negative_dir
        
        positive_files = sorted(positive_dir.glob("*.wav"))
        negative_files = sorted(negative_dir.glob("*.wav"))
        
        return positive_files, negative_files
    
    def run_baseline(self) -> AblationResult:
        """运行基线评估（仅阶段1）"""
        print("\n" + "-" * 60)
        print("评估基线（仅阶段1，无阶段2验证）")
        print("-" * 60)
        
        positive_files, negative_files = self.get_audio_files()
        
        result = AblationResult(verifier_name="baseline")
        result.total_positive = len(positive_files)
        result.total_negative = len(negative_files)
        
        total_time = 0
        total_samples = 0
        
        # 评估正样本
        print(f"\n评估正样本 ({len(positive_files)} 个)...")
        for audio_path in positive_files:
            detected, keyword, conf, proc_time = self.stage1_detector.detect_file(str(audio_path))
            total_time += proc_time
            total_samples += 1
            
            if detected:
                result.stage1_passed_positive += 1
                result.true_positive += 1
            else:
                result.false_negative += 1
        
        # 评估负样本
        print(f"评估负样本 ({len(negative_files)} 个)...")
        for audio_path in negative_files:
            detected, keyword, conf, proc_time = self.stage1_detector.detect_file(str(audio_path))
            total_time += proc_time
            total_samples += 1
            
            if detected:
                result.stage1_passed_negative += 1
                result.false_positive += 1
            else:
                result.true_negative += 1
        
        result.avg_process_time_ms = total_time / total_samples if total_samples > 0 else 0
        
        self.results["baseline"] = result
        self._print_result(result)
        
        return result
    
    def run_with_verifier(self, verifier_name: str) -> AblationResult:
        """使用指定验证器运行评估"""
        if verifier_name not in self.verifiers:
            print(f"验证器 {verifier_name} 未初始化")
            return None
        
        print("\n" + "-" * 60)
        print(f"评估方案: 阶段1 + {verifier_name.upper()} 验证器")
        print("-" * 60)
        
        verifier = self.verifiers[verifier_name]
        
        # 加载验证器模型
        if not verifier.is_loaded():
            print(f"加载 {verifier_name} 验证器...")
            verifier.load_model()
        
        positive_files, negative_files = self.get_audio_files()
        
        result = AblationResult(verifier_name=verifier_name)
        result.total_positive = len(positive_files)
        result.total_negative = len(negative_files)
        
        total_time = 0
        total_samples = 0
        
        # 评估正样本
        print(f"\n评估正样本 ({len(positive_files)} 个)...")
        for i, audio_path in enumerate(positive_files):
            start_time = time.perf_counter()
            
            # 阶段1检测
            samples, sr = load_audio(str(audio_path))
            detected, keyword, conf = self.stage1_detector.detect(samples, sr)
            
            final_accepted = False
            
            if detected:
                result.stage1_passed_positive += 1
                
                # 阶段2验证
                suffix = extract_suffix(
                    samples, sr,
                    start_ratio=self.config.suffix_start_ratio,
                    min_duration_ms=self.config.suffix_min_duration_ms,
                    max_duration_ms=self.config.suffix_max_duration_ms
                )
                
                accepted, stage2_conf = verifier.verify(suffix, sr)
                
                if accepted:
                    result.stage2_passed_positive += 1
                    final_accepted = True
            
            proc_time = (time.perf_counter() - start_time) * 1000
            total_time += proc_time
            total_samples += 1
            
            if final_accepted:
                result.true_positive += 1
            else:
                result.false_negative += 1
            
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(positive_files)}")
        
        # 评估负样本
        print(f"\n评估负样本 ({len(negative_files)} 个)...")
        for i, audio_path in enumerate(negative_files):
            start_time = time.perf_counter()
            
            # 阶段1检测
            samples, sr = load_audio(str(audio_path))
            detected, keyword, conf = self.stage1_detector.detect(samples, sr)
            
            final_accepted = False
            
            if detected:
                result.stage1_passed_negative += 1
                
                # 阶段2验证
                suffix = extract_suffix(
                    samples, sr,
                    start_ratio=self.config.suffix_start_ratio,
                    min_duration_ms=self.config.suffix_min_duration_ms,
                    max_duration_ms=self.config.suffix_max_duration_ms
                )
                
                accepted, stage2_conf = verifier.verify(suffix, sr)
                
                if accepted:
                    result.stage2_passed_negative += 1
                    final_accepted = True
            
            proc_time = (time.perf_counter() - start_time) * 1000
            total_time += proc_time
            total_samples += 1
            
            if final_accepted:
                result.false_positive += 1
            else:
                result.true_negative += 1
            
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(negative_files)}")
        
        result.avg_process_time_ms = total_time / total_samples if total_samples > 0 else 0
        
        self.results[verifier_name] = result
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: AblationResult) -> None:
        """打印单个结果"""
        print(f"\n结果 [{result.verifier_name}]:")
        print(f"  阶段1通过率: 正样本 {result.stage1_passed_positive}/{result.total_positive}, "
              f"负样本 {result.stage1_passed_negative}/{result.total_negative}")
        if result.verifier_name != "baseline":
            print(f"  阶段2通过率: 正样本 {result.stage2_passed_positive}/{result.stage1_passed_positive}, "
                  f"负样本 {result.stage2_passed_negative}/{result.stage1_passed_negative}")
        print(f"  FRR: {result.frr*100:.2f}%")
        print(f"  FAR: {result.far*100:.2f}%")
        print(f"  准确率: {result.accuracy*100:.2f}%")
        print(f"  平均处理时间: {result.avg_process_time_ms:.2f}ms")
        
        if result.meets_target(self.config):
            print(f"  ✓ 达到目标 (FRR≤{self.config.target_frr*100}%, FAR≤{self.config.target_far*100}%)")
        else:
            print(f"  ✗ 未达目标 (目标: FRR≤{self.config.target_frr*100}%, FAR≤{self.config.target_far*100}%)")
    
    def run_all(self) -> Dict[str, AblationResult]:
        """运行所有评估"""
        # 基线评估
        self.run_baseline()
        
        # 各验证器评估
        for verifier_name in self.verifiers:
            try:
                self.run_with_verifier(verifier_name)
            except Exception as e:
                print(f"\n{verifier_name} 验证器评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def generate_report(self) -> str:
        """生成对比报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        lines = []
        lines.append("=" * 80)
        lines.append("多阶段关键词检测消融实验报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        
        lines.append("\n## 实验配置")
        lines.append(f"- 测试数据: {self.config.test_data_path}")
        lines.append(f"- 正样本数: {self.config.positive_samples}")
        lines.append(f"- 负样本数: {self.config.negative_samples}")
        lines.append(f"- 目标FRR: ≤{self.config.target_frr*100}%")
        lines.append(f"- 目标FAR: ≤{self.config.target_far*100}%")
        
        lines.append("\n## 结果对比")
        lines.append("")
        
        # 表头
        lines.append(f"{'方案':<15} {'FRR':>8} {'FAR':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Time(ms)':>10} {'达标':>6}")
        lines.append("-" * 90)
        
        # 按FAR排序
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: (x[1].far, x[1].frr)
        )
        
        for name, result in sorted_results:
            meets = "✓" if result.meets_target(self.config) else "✗"
            lines.append(
                f"{name:<15} "
                f"{result.frr*100:>7.2f}% "
                f"{result.far*100:>7.2f}% "
                f"{result.accuracy*100:>7.2f}% "
                f"{result.precision*100:>7.2f}% "
                f"{result.recall*100:>7.2f}% "
                f"{result.f1*100:>7.2f}% "
                f"{result.avg_process_time_ms:>10.2f} "
                f"{meets:>6}"
            )
        
        lines.append("-" * 90)
        
        # 分析
        lines.append("\n## 分析")
        
        best_result = None
        for name, result in sorted_results:
            if result.meets_target(self.config):
                if best_result is None or result.far < best_result[1].far:
                    best_result = (name, result)
        
        if best_result:
            lines.append(f"\n**最优方案**: {best_result[0]}")
            lines.append(f"- FRR: {best_result[1].frr*100:.2f}%")
            lines.append(f"- FAR: {best_result[1].far*100:.2f}%")
            lines.append(f"- 相比基线FAR降低: {(self.results['baseline'].far - best_result[1].far)*100:.2f}%")
        else:
            lines.append("\n**无方案达到目标指标**")
            lines.append("建议:")
            lines.append("1. 调整阶段2验证器阈值")
            lines.append("2. 训练CNN/MLP验证器")
            lines.append("3. 收集更多训练数据")
        
        # 详细结果
        lines.append("\n## 详细结果")
        for name, result in self.results.items():
            lines.append(f"\n### {name}")
            lines.append(f"- 阶段1通过: 正样本 {result.stage1_passed_positive}/{result.total_positive} "
                        f"({result.stage1_passed_positive/result.total_positive*100:.1f}%), "
                        f"负样本 {result.stage1_passed_negative}/{result.total_negative} "
                        f"({result.stage1_passed_negative/result.total_negative*100:.1f}%)")
            if name != "baseline":
                lines.append(f"- 阶段2通过: 正样本 {result.stage2_passed_positive}/{result.stage1_passed_positive}, "
                            f"负样本 {result.stage2_passed_negative}/{result.stage1_passed_negative}")
            lines.append(f"- TP={result.true_positive}, FN={result.false_negative}, "
                        f"FP={result.false_positive}, TN={result.true_negative}")
        
        report = "\n".join(lines)
        
        # 保存报告
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = results_dir / f"ablation_report_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存: {report_path}")
        
        # 保存JSON结果
        json_path = results_dir / f"ablation_results_{timestamp}.json"
        json_data = {name: result.to_dict() for name, result in self.results.items()}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"JSON结果已保存: {json_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="多阶段关键词检测消融实验")
    parser.add_argument("--model-dir", type=str, 
                        default="/data/workspace/llm/keyword-spotting/exp/kws_finetune_v3",
                        help="V3模型目录")
    parser.add_argument("--test-data", type=str,
                        default="/data/workspace/llm/audio-classification/dataset/kws_test_data_merged",
                        help="测试数据目录")
    parser.add_argument("--verifiers", type=str, nargs="+",
                        default=["dtw", "cnn", "mlp"],
                        help="要评估的验证器列表 (dtw, asr, cnn, mlp)")
    parser.add_argument("--skip-asr", action="store_true",
                        help="跳过ASR验证器（加载较慢）")
    parser.add_argument("--target-frr", type=float, default=0.05,
                        help="目标FRR (默认5%%)")
    parser.add_argument("--target-far", type=float, default=0.10,
                        help="目标FAR (默认10%%)")
    
    args = parser.parse_args()
    
    # 配置
    config = AblationConfig(
        model_dir=args.model_dir,
        test_data_path=args.test_data,
        verifiers=args.verifiers,
        target_frr=args.target_frr,
        target_far=args.target_far,
    )
    
    if args.skip_asr and "asr" in config.verifiers:
        config.verifiers.remove("asr")
    
    # 运行实验
    experiment = AblationExperiment(config)
    experiment.setup()
    experiment.run_all()
    
    # 生成报告
    report = experiment.generate_report()
    print("\n" + report)


if __name__ == "__main__":
    main()
