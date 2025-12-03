"""Agent 模块
包含智能体的定义和意图分类
"""

import os
import sys
from pathlib import Path

# 将 src 目录添加到 Python 路径中
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from langchain.agents import create_agent

# 导入工具函数
from tools import (
    search_papers_openalex,
    tavily_search,
)

# 导入模型配置
from models import get_tongyi_model, get_siliconflow_model

# 导入配置加载器
from config import load_prompt


# 初始化 LLM 模型
# 从环境变量选择使用的模型，默认为 tongyi
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "tongyi").lower()

if LLM_PROVIDER == "siliconflow":
    # 使用硅基流动模型
    model_name = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    model = get_siliconflow_model(
        model_name=model_name,
        streaming=True,
        temperature=0.7
    )
elif LLM_PROVIDER == "tongyi":
    # 使用通义千问模型
    model = get_tongyi_model(streaming=True, temperature=0.7)
else:
    raise ValueError(
        f"不支持的 LLM_PROVIDER: {LLM_PROVIDER}。"
        f"支持的选项: tongyi, siliconflow"
    )

# 加载系统提示词
system_prompt = load_prompt("research_assistant_prompt")

# 创建 Agent
agent = create_agent(
    model=model,
    tools=[
        search_papers_openalex,
        tavily_search,
    ],
    system_prompt=system_prompt,
)
