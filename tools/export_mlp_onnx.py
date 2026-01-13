#!/usr/bin/env python3
"""
MLP验证器ONNX导出工具

将PyTorch格式的MLP验证器模型导出为ONNX格式，用于嵌入式部署。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """MLP分类器（与训练时结构一致）"""
    
    def __init__(self, input_dim: int = 650):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)


def export_mlp_to_onnx(
    model_path: str,
    output_path: str,
    input_dim: int = 650,
    opset_version: int = 13,
    verify: bool = True
) -> None:
    """
    导出MLP验证器为ONNX格式
    
    Args:
        model_path: PyTorch模型路径 (.pt)
        output_path: ONNX输出路径 (.onnx)
        input_dim: 输入维度 (n_mfcc * target_frames = 13 * 50 = 650)
        opset_version: ONNX opset版本
        verify: 是否验证导出的模型
    """
    print(f"加载PyTorch模型: {model_path}")
    
    # 构建模型并加载权重
    model = SimpleMLP(input_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # 创建dummy input
    dummy_input = torch.randn(1, input_dim)
    
    # 导出ONNX
    print(f"导出ONNX模型: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    
    print(f"ONNX模型已保存: {output_path}")
    
    if verify:
        verify_onnx_model(model, output_path, input_dim)


def verify_onnx_model(
    pytorch_model: nn.Module,
    onnx_path: str,
    input_dim: int
) -> None:
    """验证ONNX模型与PyTorch模型输出一致性"""
    import onnx
    import onnxruntime as ort
    
    print("\n验证ONNX模型...")
    
    # 检查ONNX模型有效性
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX模型结构有效")
    
    # 创建测试输入
    test_input = np.random.randn(1, input_dim).astype(np.float32)
    
    # PyTorch推理
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()
    
    # ONNX Runtime推理
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # 比较输出
    diff = np.abs(pytorch_output - ort_output).max()
    print(f"✓ PyTorch vs ONNX 最大差异: {diff:.2e}")
    
    if diff < 1e-5:
        print("✓ 验证通过：ONNX模型输出与PyTorch一致")
    else:
        print("⚠ 警告：输出差异较大，请检查模型")
    
    # 打印模型信息
    print(f"\n模型信息:")
    print(f"  - 输入: {ort_session.get_inputs()[0].name}, shape: {ort_session.get_inputs()[0].shape}")
    print(f"  - 输出: {ort_session.get_outputs()[0].name}, shape: {ort_session.get_outputs()[0].shape}")
    
    # 文件大小
    onnx_size = Path(onnx_path).stat().st_size / 1024
    print(f"  - 文件大小: {onnx_size:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="导出MLP验证器为ONNX格式")
    parser.add_argument(
        "--model-path",
        type=str,
        default="experiments/multi_stage_ablation/models/mlp_verifier.pt",
        help="PyTorch模型路径"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/mlp_verifier.onnx",
        help="ONNX输出路径"
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=650,
        help="输入维度 (n_mfcc * target_frames)"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=13,
        help="ONNX opset版本"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过验证步骤"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.model_path).exists():
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 创建输出目录
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 导出模型
    export_mlp_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        input_dim=args.input_dim,
        opset_version=args.opset_version,
        verify=not args.no_verify
    )


if __name__ == "__main__":
    main()
