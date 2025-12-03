"""论文搜索工具
只包含 OpenAlex 论文搜索功能
"""

from langchain.tools import tool
from pydantic import BaseModel, Field
from loguru import logger
import requests
from typing import Optional, Literal
import re
import os


def _translate_query_to_english(query: str) -> tuple[str, bool]:
    """将查询翻译为英文（如果需要）
    
    Args:
        query: 原始查询字符串
        
    Returns:
        tuple: (翻译后的查询, 是否进行了翻译)
    """
    # 检测是否包含中文字符
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
    
    if not has_chinese:
        logger.info(f"[_translate_query] 查询已是英文，无需翻译: {query}")
        return query, False
    
    logger.info(f"[_translate_query] 检测到中文查询，准备翻译: {query}")
    
    try:
        # 动态导入模型，避免循环依赖
        from models import get_siliconflow_model, get_tongyi_model
        
        # 根据环境变量选择模型
        llm_provider = os.getenv("LLM_PROVIDER", "tongyi").lower()
        
        if llm_provider == "siliconflow":
            model_name = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")
            model = get_siliconflow_model(
                model_name=model_name,
                streaming=False,
                temperature=0.1  # 降低温度以获得更确定的翻译
            )
        else:
            model = get_tongyi_model(
                streaming=False,
                temperature=0.1
            )
        
        # 构建翻译提示词
        translation_prompt = f"""请将以下中文学术搜索查询翻译为准确的英文学术术语。
只返回翻译结果，不要包含任何解释。保持专业学术用语。

中文查询: {query}
英文翻译:"""
        
        # 调用模型进行翻译
        response = model.invoke(translation_prompt)
        translated_query = response.content.strip()
        
        # 清理可能的引号和多余空格
        translated_query = translated_query.strip('"\'')
        translated_query = ' '.join(translated_query.split())
        
        logger.info(f"[_translate_query] 翻译成功: '{query}' -> '{translated_query}'")
        return translated_query, True
        
    except Exception as e:
        logger.error(f"[_translate_query] 翻译失败: {str(e)}，使用原始查询")
        logger.exception(e)
        # 翻译失败时返回原始查询
        return query, False


class OpenAlexSearchInput(BaseModel):
    """OpenAlex 论文搜索的输入参数"""

    query: str = Field(
        description="""搜索关键词或短语，会在论文标题、摘要和全文中搜索。
        
示例：
- 'machine learning'（机器学习）
- 'CRISPR gene editing'（基因编辑）
- 'climate change impact'（气候变化影响）

提示：使用英文关键词效果最佳，可以使用多个词组合"""
    )

    max_results: int = Field(
        default=10,
        description="""返回的最大论文数量。
        
默认：10 篇
建议范围：5-50 篇
最大：200 篇

用户说法映射：
- '几篇' / '一些' → 5-10
- '很多' / '尽可能多' → 50-100
- 未提及 → 使用默认 10""",
        ge=1,
        le=200,
    )

    sort_by: Literal["relevance", "publication_date", "cited_by_count"] = Field(
        default="relevance",
        description="""结果排序方式。
        
选项说明：
- 'relevance'（相关性）：按与搜索词的匹配度排序【默认】
- 'publication_date'（发表日期）：按发表时间从新到旧排序
- 'cited_by_count'（引用次数）：按被引用次数从高到低排序

用户说法映射：
- '最相关的' / '匹配度高的' → relevance
- '最新的' / '最近的' / '最近发表的' → publication_date  
- '最有影响力' / '被引用最多' / '高引用' → cited_by_count
- 未提及 → 使用默认 relevance""",
    )

    publication_year: Optional[str] = Field(
        default=None,
        description="""发表年份筛选，支持多种格式。
        
格式说明：
- '2023'：精确匹配 2023 年
- '>2020'：2021 年及以后（2021, 2022, 2023...）
- '<2020'：2019 年及之前
- '2020-2023'：2020 到 2023 年之间（包含边界）

用户说法映射：
- '最近' / '近期' / '最近几年' → '>2021' 或 '>2022'
- '2023年' / '去年' → '2023'
- '2020年到2023年' / '近三年' → '2020-2023'
- '2020年之后' → '>2020'
- 未提及 → None（不限制年份）

注意：当前年份是 2024 年，请据此计算相对时间""",
    )

    open_access_only: bool = Field(
        default=False,
        description="""是否仅返回开放获取（Open Access, OA）论文。
        
- True：只返回可以免费下载全文的论文
- False：返回所有论文（包括需要订阅的）【默认】

用户说法映射：
- '免费' / '免费下载' / '开放获取' / 'OA' / '能下载的' → True
- '不限' / 未提及 → False""",
    )

    cited_by_count_min: Optional[int] = Field(
        default=None,
        description="""最小引用次数筛选，只返回被引用次数达到此值的论文。
        
用于筛选高影响力的论文。

用户说法映射：
- '高引用' / '有影响力的' → 50 或 100
- '被引用很多' → 100
- '至少被引用X次' → X
- '引用次数超过X' → X
- 未提及 → None（不限制引用次数）

参考值：
- 10+：有一定影响力
- 50+：较高影响力  
- 100+：高影响力
- 500+：非常高影响力""",
    )


@tool(args_schema=OpenAlexSearchInput)
def search_papers_openalex(
    query: str,
    max_results: int = 10,
    sort_by: Literal["relevance", "publication_date", "cited_by_count"] = "relevance",
    publication_year: Optional[str] = None,
    open_access_only: bool = False,
    cited_by_count_min: Optional[int] = None,
) -> str:
    """搜索 OpenAlex 学术论文数据库 - 240M+ 篇学术文献，覆盖所有学科领域。

    OpenAlex 是开放的学术数据库，包含期刊文章、会议论文、预印本等。提供引用数据、
    开放获取状态、主题分类等丰富信息。适合各学科领域的论文搜索和学术研究。

    Args:
        query: 搜索关键词，在标题、摘要、全文中搜索
        max_results: 返回论文数量，默认 10 篇，最大 200 篇
        sort_by: 排序方式，默认相关性
        publication_year: 年份筛选，支持单年、范围、比较运算
        open_access_only: 是否仅返回可免费访问的开放获取论文
        cited_by_count_min: 最小引用次数，用于筛选高影响力论文
    """
    # 保存原始查询用于日志
    original_query = query
    
    logger.info(
        f"[search_papers_openalex] 搜索论文 - 原始查询: {query}, 数量: {max_results}, 排序: {sort_by}"
    )
    
    # 自动翻译中文查询为英文
    translated_query, was_translated = _translate_query_to_english(query)
    query = translated_query  # 使用翻译后的查询
    
    if was_translated:
        logger.info(f"[search_papers_openalex] 查询已翻译: '{original_query}' -> '{query}'")
    
    # 打印搜索参数
    logger.info(f"[search_papers_openalex] 搜索参数: query={query}, max_results={max_results}, sort_by={sort_by}")

    try:
        # 映射排序参数到 OpenAlex API 格式
        sort_mapping = {
            "relevance": "relevance_score:desc",
            "publication_date": "publication_date:desc",
            "cited_by_count": "cited_by_count:desc",
        }
        api_sort = sort_mapping.get(sort_by, "relevance_score:desc")

        # 构建过滤条件列表（OpenAlex 使用逗号分隔的 filter 参数）
        filters = []

        # 年份筛选
        if publication_year:
            filters.append(f"publication_year:{publication_year}")

        # 开放获取筛选
        if open_access_only:
            filters.append("is_oa:true")

        # 最小引用次数筛选
        if cited_by_count_min is not None:
            filters.append(f"cited_by_count:>{cited_by_count_min - 1}")

        # 构建 API 请求参数
        base_url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per_page": max_results,  # OpenAlex 最佳实践：使用更大的 per_page 值
            "sort": api_sort,
            "mailto": "347699233@qq.com",  # Polite pool: 10 req/sec (vs 1 req/sec)
        }

        # 添加过滤条件（如果有）
        if filters:
            params["filter"] = ",".join(filters)

        logger.debug(f"[search_papers_openalex] API 参数: {params}")

        # 发送 HTTP 请求（超时 30 秒，按文档建议）
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        # 解析 JSON 响应
        data = response.json()
        results = data.get("results", [])
        meta = data.get("meta", {})
        total_count = meta.get("count", 0)

        # 格式化输出结果
        if not results:
            output = f"❌ 未找到与 '{query}' 相关的论文。\n"
            if was_translated:
                output += f"💡 原始查询: {original_query}\n"
                output += f"🔄 翻译后: {query}\n"
            output += "建议：尝试更通用的关键词或检查拼写。"
            logger.warning("[search_papers_openalex] 未找到结果")
        else:
            # 构建搜索摘要
            filter_desc = []
            if publication_year:
                filter_desc.append(f"年份: {publication_year}")
            if open_access_only:
                filter_desc.append("仅开放获取")
            if cited_by_count_min:
                filter_desc.append(f"引用≥{cited_by_count_min}")

            filter_text = f" ({', '.join(filter_desc)})" if filter_desc else ""
            output = f"📚 找到 {len(results)} 篇论文{filter_text}\n"
            
            # 如果进行了翻译，显示翻译信息
            if was_translated:
                output += f"💡 原始查询: {original_query} → 英文: {query}\n"
            
            output += f"📊 数据库总计: {total_count:,} 篇相关文献\n\n"

            for idx, work in enumerate(results, 1):
                # 提取基本信息
                title = work.get("title") or work.get("display_name", "未知标题")
                doi = work.get("doi", "")
                pub_year = work.get("publication_year", "")
                pub_date = work.get("publication_date", "")
                cited_count = work.get("cited_by_count", 0)
                openalex_id = work.get("id", "")
                work_type = work.get("type", "").replace("-", " ").title()

                # 提取作者信息（只显示前 3 位）
                authorships = work.get("authorships", [])
                if authorships:
                    author_names = [
                        auth.get("author", {}).get("display_name", "未知")
                        for auth in authorships[:3]
                    ]
                    if len(authorships) > 3:
                        authors = (
                            ", ".join(author_names) + f" 等 ({len(authorships)} 位作者)"
                        )
                    else:
                        authors = ", ".join(author_names)
                else:
                    authors = "未知作者"

                # 提取期刊/来源信息
                primary_location = work.get("primary_location") or {}
                source = primary_location.get("source") or {}
                source_name = source.get("display_name", "")

                # 提取开放获取信息
                open_access = work.get("open_access") or {}
                oa_url = open_access.get("oa_url", "")
                oa_status = open_access.get("oa_status", "closed")
                is_oa = open_access.get("is_oa", False)

                # 提取研究主题（前 2 个）
                topics = work.get("topics") or []
                topic_names = [
                    t.get("display_name", "")
                    for t in topics[:2]
                    if t.get("display_name")
                ]

                # 构建输出 - 使用更清晰的格式
                output += f"## {idx}. {title}\n\n"

                # 基本信息
                output += f"**👥 作者:** {authors}\n"
                if source_name:
                    output += f"**📖 来源:** {source_name}\n"
                if pub_year:
                    output += f"**📅 发表:** {pub_year}"
                    if pub_date and pub_date != pub_year:
                        output += f" ({pub_date})"
                    output += "\n"
                if work_type:
                    output += f"**📄 类型:** {work_type}\n"

                # 影响力指标
                output += f"**📊 引用次数:** {cited_count:,}\n"

                # 研究主题
                if topic_names:
                    output += f"**🔬 研究主题:** {', '.join(topic_names)}\n"

                # 链接和访问
                if doi:
                    # 移除 DOI URL 前缀，只保留 DOI
                    doi_clean = doi.replace("https://doi.org/", "")
                    output += f"**🔗 DOI:** [{doi_clean}]({doi})\n"

                # 开放获取状态
                if is_oa and oa_url:
                    oa_emoji = "🔓"
                    oa_text = {
                        "gold": "金色开放获取",
                        "green": "绿色开放获取",
                        "hybrid": "混合开放获取",
                        "bronze": "铜色开放获取",
                    }.get(oa_status, "开放获取")
                    output += f"**{oa_emoji} 全文访问:** [{oa_text}]({oa_url})\n"
                else:
                    output += "**🔒 访问:** 需要订阅或付费\n"

                # OpenAlex 链接
                if openalex_id:
                    output += f"**🔍 详情:** {openalex_id}\n"

                output += "\n"

            logger.info(f"[search_papers_openalex] 成功返回 {len(results)} 篇论文")

        return output

    except requests.exceptions.Timeout:
        error_msg = "⏱️ OpenAlex API 请求超时（>30秒）\n建议：请稍后重试，或尝试更具体的搜索条件。"
        logger.error("[search_papers_openalex] 请求超时")
        return error_msg

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            error_msg = "🚫 已达到 API 速率限制\n建议：请等待 1 分钟后重试。"
        elif e.response.status_code == 404:
            error_msg = "❌ 未找到资源\n建议：检查搜索参数是否正确。"
        elif e.response.status_code >= 500:
            error_msg = f"⚠️ OpenAlex 服务器错误 ({e.response.status_code})\n建议：这是临时问题，请稍后重试。"
        else:
            error_msg = f"❌ API 请求失败: HTTP {e.response.status_code}\n{str(e)}"
        logger.error(f"[search_papers_openalex] HTTP错误: {e.response.status_code}")
        return error_msg

    except requests.exceptions.RequestException as e:
        error_msg = f"🌐 网络连接错误: {str(e)}\n建议：检查网络连接或稍后重试。"
        logger.error(f"[search_papers_openalex] 网络错误: {str(e)}")
        return error_msg

    except Exception as e:
        error_msg = f"❌ 搜索过程发生未知错误: {str(e)}\n建议：请联系技术支持。"
        logger.error("[search_papers_openalex] 未知错误")
        logger.exception(e)
        return error_msg
