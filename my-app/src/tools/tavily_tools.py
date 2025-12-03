"""Tavily 工具
包含 Tavily 基本搜索功能
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from loguru import logger
from typing import Optional, Literal, List
from tavily import TavilyClient
import os


# ==================== Tavily Search ====================


class TavilySearchInput(BaseModel):
    """Tavily 搜索输入参数"""
    
    query: str = Field(
        description="搜索查询字符串"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="返回的最大结果数量，默认为 5"
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="搜索深度。'basic' 返回快速结果，'advanced' 返回更详细但较慢的结果"
    )
    topic: Literal["general", "news", "finance"] = Field(
        default="general",
        description="搜索主题类别"
    )
    time_range: Optional[Literal["day", "week", "month", "year", "d", "w", "m", "y"]] = Field(
        default=None,
        description="时间范围过滤"
    )
    include_answer: bool = Field(
        default=False,
        description="是否包含 AI 生成的答案"
    )
    include_raw_content: bool = Field(
        default=False,
        description="是否包含原始内容"
    )
    include_images: bool = Field(
        default=False,
        description="是否包含图片"
    )
    include_domains: List[str] = Field(
        default_factory=list,
        description="包含的域名列表"
    )
    exclude_domains: List[str] = Field(
        default_factory=list,
        description="排除的域名列表"
    )


@tool(args_schema=TavilySearchInput)
def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
    topic: Literal["general", "news", "finance"] = "general",
    time_range: Optional[Literal["day", "week", "month", "year", "d", "w", "m", "y"]] = None,
    include_answer: bool = False,
    include_raw_content: bool = False,
    include_images: bool = False,
    include_domains: List[str] = None,
    exclude_domains: List[str] = None,
) -> str:
    """使用 Tavily 搜索引擎进行网络搜索，专为 AI 代理优化。
    
    适用于获取实时、准确的网络信息，支持多种主题和时间范围过滤。
    
    Args:
        query: 搜索查询字符串
        max_results: 返回的最大结果数量，默认 5，范围 1-20
        search_depth: 搜索深度，'basic' 或 'advanced'
        topic: 搜索主题，'general'、'news' 或 'finance'
        time_range: 时间范围，如 'day'、'week'、'month'、'year'
        include_answer: 是否包含 AI 生成的答案
        include_raw_content: 是否包含原始内容
        include_images: 是否包含图片
        include_domains: 包含的域名列表
        exclude_domains: 排除的域名列表
        
    Returns:
        格式化的搜索结果字符串
    """
    logger.info(f"[tavily_search] 开始搜索，查询：{query}，最大结果数：{max_results}")
    
    try:
        # 获取 API key
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("未设置 TAVILY_API_KEY 环境变量")
        
        # 创建客户端
        client = TavilyClient(api_key=api_key)
        
        # 准备参数
        kwargs = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
        }
        
        if time_range:
            kwargs["time_range"] = time_range
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
        
        # 执行搜索
        response = client.search(**kwargs)
        
        # 格式化输出
        results = response.get("results", [])
        if not results:
            output = f"未找到与 '{query}' 相关的结果。"
            logger.warning("[tavily_search] 未找到结果")
        else:
            output = f"找到 {len(results)} 条相关结果：\n\n"
            
            # 如果有 AI 答案，先显示
            if include_answer and response.get("answer"):
                output += f"**AI 答案：**\n{response['answer']}\n\n---\n\n"
            
            # 显示搜索结果
            for idx, result in enumerate(results, 1):
                title = result.get("title", "未知标题")
                url = result.get("url", "")
                content = result.get("content", "")
                score = result.get("score", 0)
                
                output += f"{idx}. **{title}**\n"
                output += f"   - URL: {url}\n"
                output += f"   - 相关性评分: {score:.2f}\n"
                output += f"   - 内容摘要: {content}\n"
                
                if include_raw_content and result.get("raw_content"):
                    raw = result["raw_content"]
                    output += f"   - 原始内容: {raw[:200]}...\n"
                
                output += "\n"
            
            # 如果有图片
            if include_images and response.get("images"):
                output += "\n**相关图片：**\n"
                for img_url in response["images"][:5]:  # 最多显示 5 张
                    output += f"- {img_url}\n"
            
            logger.info(f"[tavily_search] 成功返回 {len(results)} 条结果")
        
        return output
        
    except Exception as e:
        error_msg = f"Tavily 搜索时发生错误: {str(e)}"
        logger.error(f"[tavily_search] {error_msg}")
        logger.exception(e)
        return error_msg


