"""Tavily 工具
包含 Tavily 搜索、提取、爬取和映射功能
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from loguru import logger
from typing import Optional, Literal, List, Union
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


# ==================== Tavily Extract ====================


class TavilyExtractInput(BaseModel):
    """Tavily 内容提取输入参数"""
    
    urls: Union[str, List[str]] = Field(
        description="要提取内容的 URL 或 URL 列表"
    )
    include_images: bool = Field(
        default=False,
        description="是否包含图片"
    )
    extract_depth: Literal["basic", "advanced"] = Field(
        default="advanced",
        description="提取深度。'basic' 快速提取，'advanced' 详细提取"
    )
    format: Literal["markdown", "text"] = Field(
        default="markdown",
        description="内容格式"
    )


@tool(args_schema=TavilyExtractInput)
def tavily_extract(
    urls: Union[str, List[str]],
    include_images: bool = False,
    extract_depth: Literal["basic", "advanced"] = "advanced",
    format: Literal["markdown", "text"] = "markdown",
) -> str:
    """从指定的 URL 提取网页内容。
    
    使用 Tavily 的内容提取功能，可以从单个或多个 URL 提取结构化内容。
    
    Args:
        urls: 要提取内容的 URL 或 URL 列表
        include_images: 是否包含图片
        extract_depth: 提取深度，'basic' 或 'advanced'
        format: 内容格式，'markdown' 或 'text'
        
    Returns:
        格式化的提取内容字符串
    """
    # 确保 urls 是列表
    if isinstance(urls, str):
        urls = [urls]
    
    logger.info(f"[tavily_extract] 开始提取内容，URL 数量：{len(urls)}")
    
    try:
        # 获取 API key
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("未设置 TAVILY_API_KEY 环境变量")
        
        # 创建客户端
        client = TavilyClient(api_key=api_key)
        
        # 执行提取
        response = client.extract(
            urls=urls,
            include_images=include_images,
            extract_depth=extract_depth,
            format=format,
        )
        
        # 格式化输出
        results = response.get("results", [])
        failed_results = response.get("failed_results", [])
        
        output = "内容提取完成：\n\n"
        
        # 成功的结果
        if results:
            output += f"**成功提取 {len(results)} 个 URL：**\n\n"
            for idx, result in enumerate(results, 1):
                url = result.get("url", "")
                raw_content = result.get("raw_content", "")
                
                output += f"{idx}. **URL:** {url}\n"
                
                # 显示内容预览（前 500 字符）
                if raw_content:
                    preview = raw_content[:500] + "..." if len(raw_content) > 500 else raw_content
                    output += f"   **内容预览：**\n{preview}\n"
                
                # 显示图片
                if include_images and result.get("images"):
                    images = result["images"]
                    output += f"   **图片 ({len(images)} 张)：**\n"
                    for img_url in images[:3]:  # 最多显示 3 张
                        output += f"   - {img_url}\n"
                
                output += "\n"
        
        # 失败的结果
        if failed_results:
            output += f"\n**提取失败 {len(failed_results)} 个 URL：**\n"
            for idx, result in enumerate(failed_results, 1):
                url = result.get("url", "")
                error = result.get("error", "未知错误")
                output += f"{idx}. {url} - 错误：{error}\n"
        
        if not results and not failed_results:
            output = "未提取到任何内容。"
            logger.warning("[tavily_extract] 未提取到内容")
        else:
            logger.info(f"[tavily_extract] 成功提取 {len(results)} 个，失败 {len(failed_results)} 个")
        
        return output
        
    except Exception as e:
        error_msg = f"Tavily 内容提取时发生错误: {str(e)}"
        logger.error(f"[tavily_extract] {error_msg}")
        logger.exception(e)
        return error_msg


# ==================== Tavily Crawl ====================


class TavilyCrawlInput(BaseModel):
    """Tavily 爬取输入参数"""
    
    url: str = Field(
        description="要爬取的起始 URL"
    )
    max_depth: int = Field(
        default=1,
        ge=1,
        description="最大爬取深度"
    )
    max_breadth: int = Field(
        default=20,
        ge=1,
        description="每层最大爬取宽度"
    )
    limit: int = Field(
        default=50,
        ge=1,
        description="最大爬取页面数"
    )
    instructions: Optional[str] = Field(
        default=None,
        description="爬取指令，指导爬虫关注特定内容"
    )
    select_paths: Optional[List[str]] = Field(
        default=None,
        description="选择要包含的路径正则表达式列表"
    )
    exclude_paths: Optional[List[str]] = Field(
        default=None,
        description="排除的路径正则表达式列表"
    )
    allow_external: bool = Field(
        default=True,
        description="是否允许爬取外部链接"
    )
    include_images: bool = Field(
        default=False,
        description="是否包含图片"
    )
    extract_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="提取深度"
    )
    format: Literal["markdown", "text"] = Field(
        default="markdown",
        description="内容格式"
    )


@tool(args_schema=TavilyCrawlInput)
def tavily_crawl(
    url: str,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    instructions: Optional[str] = None,
    select_paths: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    allow_external: bool = True,
    include_images: bool = False,
    extract_depth: Literal["basic", "advanced"] = "basic",
    format: Literal["markdown", "text"] = "markdown",
) -> str:
    """基于图的网站遍历工具，可以并行探索数百条路径。
    
    使用 Tavily Crawl 进行智能网站爬取，内置提取和智能发现功能。
    
    Args:
        url: 要爬取的起始 URL
        max_depth: 最大爬取深度
        max_breadth: 每层最大爬取宽度
        limit: 最大爬取页面数
        instructions: 爬取指令，指导爬虫关注特定内容
        select_paths: 选择要包含的路径正则表达式列表
        exclude_paths: 排除的路径正则表达式列表
        allow_external: 是否允许爬取外部链接
        include_images: 是否包含图片
        extract_depth: 提取深度，'basic' 或 'advanced'
        format: 内容格式，'markdown' 或 'text'
        
    Returns:
        格式化的爬取结果字符串
    """
    logger.info(f"[tavily_crawl] 开始爬取，URL：{url}，深度：{max_depth}，宽度：{max_breadth}")
    
    try:
        # 获取 API key
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("未设置 TAVILY_API_KEY 环境变量")
        
        # 创建客户端
        client = TavilyClient(api_key=api_key)
        
        # 准备参数
        kwargs = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "allow_external": allow_external,
            "include_images": include_images,
            "extract_depth": extract_depth,
            "format": format,
        }
        
        if instructions:
            kwargs["instructions"] = instructions
        if select_paths:
            kwargs["select_paths"] = select_paths
        if exclude_paths:
            kwargs["exclude_paths"] = exclude_paths
        
        # 执行爬取
        response = client.crawl(**kwargs)
        
        # 格式化输出
        base_url = response.get("base_url", url)
        results = response.get("results", [])
        
        if not results:
            output = "未爬取到任何页面。"
            logger.warning("[tavily_crawl] 未爬取到页面")
        else:
            output = f"爬取完成，基础 URL：{base_url}\n\n"
            output += f"**共爬取 {len(results)} 个页面：**\n\n"
            
            for idx, result in enumerate(results, 1):
                page_url = result.get("url", "")
                raw_content = result.get("raw_content", "")
                
                output += f"{idx}. **URL:** {page_url}\n"
                
                # 显示内容预览（前 300 字符）
                if raw_content:
                    preview = raw_content[:300] + "..." if len(raw_content) > 300 else raw_content
                    output += f"   **内容预览：**\n{preview}\n"
                
                # 显示图片
                if include_images and result.get("images"):
                    images = result["images"]
                    output += f"   **图片 ({len(images)} 张)：**\n"
                    for img_url in images[:2]:  # 最多显示 2 张
                        output += f"   - {img_url}\n"
                
                output += "\n"
            
            logger.info(f"[tavily_crawl] 成功爬取 {len(results)} 个页面")
        
        return output
        
    except Exception as e:
        error_msg = f"Tavily 爬取时发生错误: {str(e)}"
        logger.error(f"[tavily_crawl] {error_msg}")
        logger.exception(e)
        return error_msg


# ==================== Tavily Map ====================


class TavilyMapInput(BaseModel):
    """Tavily 映射输入参数"""
    
    url: str = Field(
        description="要映射的起始 URL"
    )
    max_depth: int = Field(
        default=1,
        ge=1,
        description="最大映射深度"
    )
    max_breadth: int = Field(
        default=20,
        ge=1,
        description="每层最大映射宽度"
    )
    limit: int = Field(
        default=50,
        ge=1,
        description="最大映射页面数"
    )
    instructions: Optional[str] = Field(
        default=None,
        description="映射指令，指导映射关注特定路径"
    )
    select_paths: Optional[List[str]] = Field(
        default=None,
        description="选择要包含的路径正则表达式列表"
    )
    exclude_paths: Optional[List[str]] = Field(
        default=None,
        description="排除的路径正则表达式列表"
    )
    allow_external: bool = Field(
        default=True,
        description="是否允许映射外部链接"
    )


@tool(args_schema=TavilyMapInput)
def tavily_map(
    url: str,
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    instructions: Optional[str] = None,
    select_paths: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    allow_external: bool = True,
) -> str:
    """像图一样遍历网站，生成全面的站点地图。
    
    使用 Tavily Map 进行智能网站映射，可以并行探索数百条路径。
    
    Args:
        url: 要映射的起始 URL
        max_depth: 最大映射深度
        max_breadth: 每层最大映射宽度
        limit: 最大映射页面数
        instructions: 映射指令，指导映射关注特定路径
        select_paths: 选择要包含的路径正则表达式列表
        exclude_paths: 排除的路径正则表达式列表
        allow_external: 是否允许映射外部链接
        
    Returns:
        格式化的站点地图字符串
    """
    logger.info(f"[tavily_map] 开始映射，URL：{url}，深度：{max_depth}，宽度：{max_breadth}")
    
    try:
        # 获取 API key
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("未设置 TAVILY_API_KEY 环境变量")
        
        # 创建客户端
        client = TavilyClient(api_key=api_key)
        
        # 准备参数
        kwargs = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "allow_external": allow_external,
        }
        
        if instructions:
            kwargs["instructions"] = instructions
        if select_paths:
            kwargs["select_paths"] = select_paths
        if exclude_paths:
            kwargs["exclude_paths"] = exclude_paths
        
        # 执行映射 (注意: SDK 中方法名是 map，不是 mapping)
        response = client.map(**kwargs)
        
        # 格式化输出
        base_url = response.get("base_url", url)
        results = response.get("results", [])
        
        if not results:
            output = "未找到任何页面。"
            logger.warning("[tavily_map] 未找到页面")
        else:
            output = f"站点地图生成完成，基础 URL：{base_url}\n\n"
            output += f"**共发现 {len(results)} 个页面：**\n\n"
            
            for idx, page_url in enumerate(results, 1):
                output += f"{idx}. {page_url}\n"
            
            logger.info(f"[tavily_map] 成功映射 {len(results)} 个页面")
        
        return output
        
    except Exception as e:
        error_msg = f"Tavily 映射时发生错误: {str(e)}"
        logger.error(f"[tavily_map] {error_msg}")
        logger.exception(e)
        return error_msg
