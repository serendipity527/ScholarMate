"""LLM 配置模块
负责各种大语言模型的初始化和配置
"""

import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI
from loguru import logger


def get_tongyi_model(streaming: bool = True, temperature: float = 0.7):
    """获取通义千问模型实例
    
    Args:
        streaming: 是否启用流式输出
        temperature: 温度参数，控制输出的随机性（0-1）
        
    Returns:
        配置好的 ChatTongyi 模型实例
        
    Raises:
        ValueError: 如果缺少 TONGYI_API_KEY 环境变量
    """
    api_key = os.getenv("TONGYI_API_KEY")
    if not api_key:
        error_msg = "缺少 TONGYI_API_KEY 环境变量，请在 .env 文件中配置"
        logger.error(f"[get_tongyi_model] {error_msg}")
        raise ValueError(error_msg)
    
    logger.info(f"[get_tongyi_model] 初始化通义千问模型，streaming={streaming}, temperature={temperature}")
    
    model = ChatTongyi(
        streaming=streaming,
        temperature=temperature,
        api_key=api_key,
    )
    
    return model


def get_siliconflow_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    streaming: bool = True,
    temperature: float = 0.7
):
    """获取硅基流动模型实例
    
    Args:
        model_name: 模型名称，例如 "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct" 等
        streaming: 是否启用流式输出
        temperature: 温度参数，控制输出的随机性（0-1）
        
    Returns:
        配置好的 ChatOpenAI 模型实例（兼容硅基流动 API）
        
    Raises:
        ValueError: 如果缺少 SILICONFLOW_API_KEY 环境变量
    """
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        error_msg = "缺少 SILICONFLOW_API_KEY 环境变量，请在 .env 文件中配置"
        logger.error(f"[get_siliconflow_model] {error_msg}")
        raise ValueError(error_msg)
    
    logger.info(
        f"[get_siliconflow_model] 初始化硅基流动模型，"
        f"model={model_name}, streaming={streaming}, temperature={temperature}"
    )
    
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        streaming=streaming,
        temperature=temperature,
    )
    
    return model
