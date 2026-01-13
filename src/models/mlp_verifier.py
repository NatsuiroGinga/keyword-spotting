"""
MLP验证器ONNX推理模块

使用ONNX Runtime进行MLP验证器推理。
"""
import numpy as np
from pathlib import Path
from typing import Tuple


class MLPVerifierONNX:
    """
    MLP验证器（ONNX Runtime推理）
    
    用于对候选关键词进行二次验证，降低误报率。
    """
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        providers: list = None
    ):
        """
        初始化MLP验证器
        
        Args:
            model_path: ONNX模型路径
            threshold: 分类阈值
            providers: ONNX Runtime执行提供者列表
        """
        self.model_path = model_path
        self.threshold = threshold
        self.providers = providers or ["CPUExecutionProvider"]
        
        self._session = None
        self._input_name = None
        self._output_name = None
        
    def load(self) -> None:
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("需要安装 onnxruntime: pip install onnxruntime")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 创建推理会话
        self._session = ort.InferenceSession(
            self.model_path,
            providers=self.providers
        )
        
        # 获取输入输出名称
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        
        print(f"MLP验证器已加载: {self.model_path}")
        print(f"  - 输入: {self._input_name}")
        print(f"  - 输出: {self._output_name}")
        print(f"  - 阈值: {self.threshold}")
    
    def predict(self, features: np.ndarray) -> float:
        """
        预测置信度
        
        Args:
            features: 特征向量 (input_dim,) 或 (batch, input_dim)
            
        Returns:
            置信度分数 (0-1)
        """
        if self._session is None:
            self.load()
        
        # 确保输入维度正确
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 确保类型正确
        features = features.astype(np.float32)
        
        # 推理
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: features}
        )
        
        return outputs[0][0, 0]
    
    def verify(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        验证特征是否为目标关键词
        
        Args:
            features: 特征向量
            
        Returns:
            (是否通过验证, 置信度分数)
        """
        confidence = self.predict(features)
        is_accepted = confidence >= self.threshold
        return is_accepted, confidence
    
    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._session is not None
