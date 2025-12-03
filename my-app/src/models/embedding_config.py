"""Embedding 配置模块
负责各种嵌入模型的初始化和配置
"""

import os
from langchain_openai import OpenAIEmbeddings
from loguru import logger


def get_siliconflow_embeddings(
    model_name: str = "BAAI/bge-large-zh-v1.5",
    check_embedding_ctx_length: bool = False
):
    """获取硅基流动 Embedding 模型实例
    
    Args:
        model_name: 模型名称，默认使用 "BAAI/bge-large-zh-v1.5"
                   其他可选模型：
                   - "BAAI/bge-m3" (多语言)
                   - "BAAI/bge-large-en-v1.5" (英文)
                   - "sentence-transformers/all-MiniLM-L6-v2" (轻量级)
        check_embedding_ctx_length: 是否检查嵌入上下文长度，有些模型不支持自动检查，建议关闭
        
    Returns:
        配置好的 OpenAIEmbeddings 实例（兼容硅基流动 API）
        
    Raises:
        ValueError: 如果缺少 SILICONFLOW_API_KEY 环境变量
        
    Examples:
        >>> embeddings = get_siliconflow_embeddings()
        >>> # 嵌入单个查询
        >>> vector = embeddings.embed_query("LangChain 是一个框架")
        >>> print(f"向量维度: {len(vector)}")
        >>> 
        >>> # 嵌入多个文档
        >>> docs = ["文档1", "文档2", "文档3"]
        >>> vectors = embeddings.embed_documents(docs)
        >>> print(f"已处理 {len(vectors)} 个文档")
    """
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        error_msg = "缺少 SILICONFLOW_API_KEY 环境变量，请在 .env 文件中配置"
        logger.error(f"[get_siliconflow_embeddings] {error_msg}")
        raise ValueError(error_msg)
    
    logger.info(
        f"[get_siliconflow_embeddings] 初始化硅基流动嵌入模型，"
        f"model={model_name}, check_ctx_length={check_embedding_ctx_length}"
    )
    
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=check_embedding_ctx_length,
    )
    
    return embeddings
