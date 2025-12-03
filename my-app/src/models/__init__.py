"""模型模块
包含各种 LLM 模型和 Embedding 模型的配置和初始化
"""

from .llm_config import get_tongyi_model, get_siliconflow_model
from .embedding_config import get_siliconflow_embeddings

__all__ = [
    "get_tongyi_model",
    "get_siliconflow_model",
    "get_siliconflow_embeddings",
]
