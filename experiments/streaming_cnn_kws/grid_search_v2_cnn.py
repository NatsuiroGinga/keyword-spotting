#!/usr/bin/env python3
"""
网格搜索：V2 + CNN 验证器的最优配置
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Dict
import time
import warnings
warnings.filterwarnings("ignore")

import sherpa_onnx
import soundfile as sf

from models.cnn_verifier import CNNVerifier, CNNConfig
from features.feature_extractor import FeatureExtractor, FeatureConfig, SuffixExtractor

PROJECT_ROOT = Path(__file__).parent.parent.parent


class GridSearchEvaluator:
    """网格搜索评估器"""
    
    def __init__(
        self,
        kws_model_dir: Path,
        cnn_model_path: Path,
        test_dir: Path,
        positive_keywords: List[str]
    ):
        self.kws_model_dir = kws_model_dir
        self.cnn_model_path = cnn_model_path
        self.test_dir = test_dir
        self.positive_keywords = positive_keywords
        
        # 加载测试文件
        self.audio_files, self.labels = self._load_dataset()
        
        # CNN 相关
        self.cnn_model = None
        self.feature_extractor = None
        self.suffix_extractor = None
        
    def _load_dataset(self):
        """加载数据集"""
        files = []
        labels = []
        
        for audio_file in sorted(self.test_dir.glob("*.wav")):
            is_positive = any(kw in audio_file.name for kw in self.positive_keywords)
            files.append(audio_file)
            labels.append(1 if is_positive else 0)
        
        return files, labels
    
    def _load_cnn(self):
        """加载 CNN 模型"""
        checkpoint = torch.load(self.cnn_model_path, map_location='cpu')
        
        config = CNNConfig(**checkpoint.get('config', {}))
        self.cnn_model = CNNVerifier(config)
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn_model.eval()
        
        feature_config = FeatureConfig(
            n_mfcc=config.input_dim,
            target_frames=config.target_frames
        )
        self.feature_extractor = FeatureExtractor(feature_config)
        self.suffix_extractor = SuffixExtractor()
        
        self.cnn_threshold = checkpoint.get('best_threshold', 0.5)
    
    def _create_kws(self, kws_threshold: float, keywords_score: float):
        """创建 KWS 检测器"""
        encoder_files = list(self.kws_model_dir.glob("encoder*.int8.onnx"))
        decoder_files = list(self.kws_model_dir.glob("decoder*.int8.onnx"))
        joiner_files = list(self.kws_model_dir.glob("joiner*.int8.onnx"))
        
        return sherpa_onnx.KeywordSpotter(
            tokens=str(self.kws_model_dir / "tokens.txt"),
            encoder=str(encoder_files[0]),
            decoder=str(decoder_files[0]),
            joiner=str(joiner_files[0]),
            keywords_file=str(self.kws_model_dir / "keywords.txt"),
            num_threads=2,
            keywords_score=keywords_score,
            keywords_threshold=kws_threshold,
        )
    
    def evaluate_config(
        self,
        kws_threshold: float,
        keywords_score: float,
        cnn_threshold: float,
    ) -> Dict:
        """评估单个配置"""
        
        kws = self._create_kws(kws_threshold, keywords_score)
        
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for audio_file, label in zip(self.audio_files, self.labels):
            # 读取音频
            audio, sr = sf.read(str(audio_file), dtype='float32')
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # KWS 流式检测
            stream = kws.create_stream()
            chunk_size = 1600  # 100ms
            kws_detected = False
            kws_trigger_pos = 0
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                stream.accept_waveform(16000, chunk)
                
                while kws.is_ready(stream):
                    kws.decode_stream(stream)
                
                result = kws.get_result(stream)
                if result:
                    kws_detected = True
                    kws_trigger_pos = i + chunk_size
                    break
            
            # CNN 验证
            final_detected = False
            if kws_detected:
                # 提取后缀音频
                suffix_audio = self.suffix_extractor.extract(audio[:kws_trigger_pos])
                if suffix_audio is not None and len(suffix_audio) > 0:
                    # 提取特征（固定长度）
                    features = self.feature_extractor.extract_mfcc_fixed(suffix_audio)
                    if features is not None:
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        
                        with torch.no_grad():
                            cnn_score = self.cnn_model(features_tensor).item()
                        
                        if cnn_score >= cnn_threshold:
                            final_detected = True
                else:
                    # 后缀提取失败，根据 KWS 置信度判断
                    final_detected = kws_threshold >= 0.5
            
            # 统计
            if label == 1:
                if final_detected:
                    tp += 1
                else:
                    fn += 1
            else:
                if final_detected:
                    fp += 1
                else:
                    tn += 1
        
        # 计算指标
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'kws_threshold': kws_threshold,
            'keywords_score': keywords_score,
            'cnn_threshold': cnn_threshold,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'far': far, 'frr': frr,
            'precision': precision, 'recall': recall,
            'f1': f1, 'accuracy': accuracy
        }
    
    def run_grid_search(
        self,
        kws_thresholds: List[float],
        keywords_scores: List[float],
        cnn_thresholds: List[float],
    ) -> List[Dict]:
        """运行网格搜索"""
        
        print("加载 CNN 模型...")
        self._load_cnn()
        
        print(f"测试集: {len(self.audio_files)} 个文件")
        print(f"  正样本: {sum(self.labels)}")
        print(f"  负样本: {len(self.labels) - sum(self.labels)}")
        print()
        
        total = len(kws_thresholds) * len(keywords_scores) * len(cnn_thresholds)
        print(f"网格搜索: {total} 个配置")
        print()
        
        results = []
        count = 0
        
        for kws_th in kws_thresholds:
            for kw_score in keywords_scores:
                for cnn_th in cnn_thresholds:
                    count += 1
                    result = self.evaluate_config(kws_th, kw_score, cnn_th)
                    results.append(result)
                    
                    print(f"[{count}/{total}] KWS_th={kws_th:.2f}, KW_score={kw_score:.1f}, CNN_th={cnn_th:.2f}")
                    print(f"    FAR={result['far']*100:.2f}%, FRR={result['frr']*100:.2f}%, F1={result['f1']*100:.2f}%")
        
        return results
    
    def find_best_configs(self, results: List[Dict], far_target: float = 0.10, frr_target: float = 0.05):
        """找到最优配置"""
        
        # 同时满足目标的配置
        valid = [r for r in results if r['far'] <= far_target and r['frr'] <= frr_target]
        
        if valid:
            best = max(valid, key=lambda x: x['f1'])
            print("\n最优配置（同时满足 FAR<10% 和 FRR<5%）:")
        else:
            # 找 FAR < target 中 FRR 最低的
            far_valid = [r for r in results if r['far'] <= far_target]
            if far_valid:
                best = min(far_valid, key=lambda x: x['frr'])
                print(f"\n最优配置（满足 FAR<{far_target*100:.0f}%，FRR 最低）:")
            else:
                # 找 FAR 最接近目标的
                best = min(results, key=lambda x: abs(x['far'] - far_target))
                print(f"\n最优配置（FAR 最接近 {far_target*100:.0f}%）:")
        
        print(f"  KWS 阈值: {best['kws_threshold']:.2f}")
        print(f"  Keywords Score: {best['keywords_score']:.1f}")
        print(f"  CNN 阈值: {best['cnn_threshold']:.2f}")
        print(f"  FAR: {best['far']*100:.2f}%")
        print(f"  FRR: {best['frr']*100:.2f}%")
        print(f"  F1: {best['f1']*100:.2f}%")
        print(f"  Accuracy: {best['accuracy']*100:.2f}%")
        
        return best


def main():
    # 路径
    kws_model_dir = PROJECT_ROOT / "exp" / "kws_finetune_v2"
    cnn_model_path = Path(__file__).parent / "outputs" / "cnn_verifier_latest.pt"
    test_dir = PROJECT_ROOT / "data" / "all"
    
    # 检查文件
    if not kws_model_dir.exists():
        print(f"错误: KWS 模型目录不存在: {kws_model_dir}")
        return
    if not cnn_model_path.exists():
        print(f"错误: CNN 模型不存在: {cnn_model_path}")
        return
    
    print("=" * 60)
    print("V2 + CNN 网格搜索")
    print("=" * 60)
    print(f"KWS 模型: {kws_model_dir}")
    print(f"CNN 模型: {cnn_model_path}")
    print()
    
    evaluator = GridSearchEvaluator(
        kws_model_dir=kws_model_dir,
        cnn_model_path=cnn_model_path,
        test_dir=test_dir,
        positive_keywords=["你好真真", "你好珍珍"]
    )
    
    # 网格搜索参数
    kws_thresholds = [0.3, 0.4, 0.5, 0.55, 0.6]
    keywords_scores = [1.0, 1.5, 2.0]
    cnn_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = evaluator.run_grid_search(kws_thresholds, keywords_scores, cnn_thresholds)
    
    # 保存结果
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"grid_search_v2_cnn_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    # 找最优配置
    best = evaluator.find_best_configs(results)
    
    # 打印 FAR < 10% 的所有配置
    print("\n" + "=" * 60)
    print("所有 FAR < 10% 的配置:")
    print("=" * 60)
    far_valid = sorted([r for r in results if r['far'] < 0.10], key=lambda x: x['frr'])
    
    print(f"{'KWS_th':>8} {'KW_score':>8} {'CNN_th':>8} {'FAR':>8} {'FRR':>8} {'F1':>8}")
    print("-" * 56)
    for r in far_valid[:15]:
        print(f"{r['kws_threshold']:>8.2f} {r['keywords_score']:>8.1f} {r['cnn_threshold']:>8.2f} "
              f"{r['far']*100:>7.2f}% {r['frr']*100:>7.2f}% {r['f1']*100:>7.2f}%")


if __name__ == "__main__":
    main()
