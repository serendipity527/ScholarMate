"""Agent 模块
包含智能体的定义和意图分类
"""

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
    search_papers_ArXiv,
    search_papers_semantic_scholar,
    search_papers_aggregated,
    tavily_search,
    tavily_extract,
    tavily_crawl,
    tavily_map,
    analyze_citation_network,
)

# 导入模型配置
from models import get_tongyi_model

# 导入配置加载器
from config import load_prompt


# 初始化 LLM 模型
model = get_tongyi_model(streaming=True, temperature=0.7)

# 加载系统提示词
system_prompt = load_prompt("research_assistant_prompt")

# 创建 Agent
agent = create_agent(
    model=model,
    tools=[
        search_papers_openalex,
        search_papers_ArXiv,
        search_papers_semantic_scholar,
        search_papers_aggregated,
        tavily_search,
        tavily_extract,
        tavily_crawl,
        tavily_map,
        analyze_citation_network,
    ],
    system_prompt=system_prompt,
)
