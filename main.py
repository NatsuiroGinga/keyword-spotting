#!/usr/bin/env python3
"""
流式关键词识别主程序

"你好真真"唤醒词实时识别系统

使用方法:
    python main.py --model-dir ./kws_finetune_v3

功能:
    1. 从麦克风实时采集音频
    2. 使用Zipformer进行流式关键词检测
    3. 使用MLP验证器进行二次确认
    4. 检测到唤醒词时输出提示
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="流式关键词识别 - 你好真真",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认配置运行
    python main.py --model-dir ./kws_finetune_v3
    
    # 禁用MLP验证器
    python main.py --model-dir ./kws_finetune_v3 --no-mlp
    
    # 调整检测阈值
    python main.py --model-dir ./kws_finetune_v3 --kws-threshold 0.3 --mlp-threshold 0.6
    
    # 列出音频设备
    python main.py --list-devices
"""
    )
    
    # 模型配置
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./kws_finetune_v3",
        help="模型目录路径"
    )
    parser.add_argument(
        "--mlp-model",
        type=str,
        default="models/mlp_verifier.onnx",
        help="MLP验证器模型路径"
    )
    
    # 关键词配置
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=["你好真真"],
        help="要检测的关键词列表"
    )
    parser.add_argument(
        "--kws-score",
        type=float,
        default=1.5,
        help="关键词加分权重（越高越容易触发）"
    )
    parser.add_argument(
        "--kws-threshold",
        type=float,
        default=0.25,
        help="KWS触发阈值（越低越容易触发）"
    )
    
    # MLP配置
    parser.add_argument(
        "--mlp-threshold",
        type=float,
        default=0.5,
        help="MLP验证阈值"
    )
    parser.add_argument(
        "--no-mlp",
        action="store_true",
        help="禁用MLP二阶段验证"
    )
    
    # 音频配置
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="音频设备索引"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出所有音频设备并退出"
    )
    
    # 其他配置
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="推理线程数"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "coreml"],
        help="计算提供者"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息"
    )
    
    return parser


def list_audio_devices():
    """列出所有音频设备"""
    try:
        import sounddevice as sd
        print("可用音频设备:")
        print(sd.query_devices())
    except ImportError:
        print("错误: 需要安装 sounddevice: pip install sounddevice")
        sys.exit(1)


def on_keyword_detected(result):
    """关键词检测回调"""
    print("\n" + "=" * 50)
    print(f"🎤 检测到唤醒词: {result.keyword}")
    print(f"   时间: {result.timestamp:.2f}s")
    if result.mlp_confidence is not None:
        print(f"   MLP置信度: {result.mlp_confidence:.3f}")
    print("=" * 50 + "\n")


def run_streaming_kws(args):
    """运行流式关键词识别"""
    try:
        import sounddevice as sd
    except ImportError:
        print("错误: 需要安装 sounddevice: pip install sounddevice")
        sys.exit(1)
    
    # 添加src到路径
    sys.path.insert(0, str(Path(__file__).parent))
    
    from src.utils.config import KWSConfig
    from src.pipeline.kws_stream import StreamingKWSPipeline
    
    # 创建配置
    model_dir = Path(args.model_dir)
    
    config = KWSConfig(
        # 模型路径
        encoder_path=str(model_dir / "encoder.int8.onnx"),
        decoder_path=str(model_dir / "decoder.int8.onnx"),
        joiner_path=str(model_dir / "joiner.int8.onnx"),
        tokens_path=str(model_dir / "tokens.txt"),
        keywords_file=str(model_dir / "keywords.txt"),
        
        # 关键词配置
        keywords=args.keywords,
        keywords_score=args.kws_score,
        keywords_threshold=args.kws_threshold,
        
        # MLP配置
        mlp_model_path=args.mlp_model,
        mlp_threshold=args.mlp_threshold,
        mlp_enabled=not args.no_mlp,
        
        # 推理配置
        num_threads=args.num_threads,
        provider=args.provider,
    )
    
    # 检查模型文件
    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    # 创建管道
    print("正在初始化流式KWS系统...")
    pipeline = StreamingKWSPipeline(config)
    
    try:
        pipeline.load()
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        sys.exit(1)
    
    # 设置回调
    pipeline.set_on_detection(on_keyword_detected)
    
    # 音频参数
    sample_rate = config.sample_rate
    chunk_duration_ms = config.chunk_duration_ms
    samples_per_read = int(sample_rate * chunk_duration_ms / 1000)
    
    print("\n" + "=" * 50)
    print("🎙️  流式关键词识别已启动")
    print(f"   关键词: {', '.join(args.keywords)}")
    print(f"   采样率: {sample_rate}Hz")
    print(f"   MLP验证: {'启用' if not args.no_mlp else '禁用'}")
    print("   按 Ctrl+C 退出")
    print("=" * 50 + "\n")
    
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=samples_per_read,
            device=args.device
        ) as stream:
            print("正在监听...")
            
            while True:
                # 读取音频
                audio, _ = stream.read(samples_per_read)
                audio = audio.reshape(-1)
                
                # 处理音频块
                result = pipeline.process_chunk(audio)
                
                # 如果verbose模式，显示进度
                if args.verbose and pipeline.detection_count > 0:
                    print(f"\r检测次数: {pipeline.detection_count}", end="", flush=True)
                    
    except KeyboardInterrupt:
        print("\n\n程序已退出")
        print(f"总检测次数: {pipeline.detection_count}")


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 列出设备模式
    if args.list_devices:
        list_audio_devices()
        return
    
    # 运行流式KWS
    run_streaming_kws(args)


if __name__ == "__main__":
    main()
