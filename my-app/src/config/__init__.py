"""配置模块
用于加载和管理应用配置
"""

from .config_loader import load_prompt, load_all_prompts, get_config_path

__all__ = [
    "load_prompt",
    "load_all_prompts",
    "get_config_path",
]
