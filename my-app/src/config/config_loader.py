"""配置加载器
负责从 YAML 文件加载配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def get_config_path(config_name: str = "prompts.yaml") -> Path:
    """获取配置文件路径
    
    Args:
        config_name: 配置文件名
        
    Returns:
        配置文件的完整路径
    """
    # 获取当前文件所在目录
    config_dir = Path(__file__).parent
    config_path = config_dir / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    return config_path


def load_config(config_name: str = "prompts.yaml") -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_name: 配置文件名
        
    Returns:
        配置字典
    """
    config_path = get_config_path(config_name)
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"[config_loader] 成功加载配置文件: {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"[config_loader] 加载配置文件失败: {config_path}, 错误: {str(e)}")
        raise


def load_prompt(prompt_key: str, config_name: str = "prompts.yaml") -> str:
    """加载指定的提示词
    
    Args:
        prompt_key: 提示词的键名，如 'research_assistant_prompt'
        config_name: 配置文件名
        
    Returns:
        提示词字符串
        
    Raises:
        KeyError: 如果指定的提示词不存在
    """
    config = load_config(config_name)
    
    if prompt_key not in config:
        available_keys = list(config.keys())
        raise KeyError(
            f"提示词 '{prompt_key}' 不存在。可用的提示词: {available_keys}"
        )
    
    prompt = config[prompt_key]
    logger.debug(f"[config_loader] 加载提示词: {prompt_key}")
    
    return prompt


def load_all_prompts(config_name: str = "prompts.yaml") -> Dict[str, str]:
    """加载所有提示词
    
    Args:
        config_name: 配置文件名
        
    Returns:
        包含所有提示词的字典
    """
    config = load_config(config_name)
    logger.info(f"[config_loader] 加载了 {len(config)} 个提示词")
    
    return config


def reload_config(config_name: str = "prompts.yaml") -> Dict[str, Any]:
    """重新加载配置文件（用于配置热更新）
    
    Args:
        config_name: 配置文件名
        
    Returns:
        配置字典
    """
    logger.info(f"[config_loader] 重新加载配置文件: {config_name}")
    return load_config(config_name)


# 默认加载研究助手提示词的快捷函数
def get_research_assistant_prompt() -> str:
    """获取研究助手的系统提示词
    
    Returns:
        研究助手的系统提示词
    """
    return load_prompt("research_assistant_prompt")


def get_paper_analysis_prompt() -> str:
    """获取论文分析的提示词
    
    Returns:
        论文分析的提示词
    """
    return load_prompt("paper_analysis_prompt")


def get_citation_network_prompt() -> str:
    """获取引用网络分析的提示词
    
    Returns:
        引用网络分析的提示词
    """
    return load_prompt("citation_network_prompt")
