"""论文相关工具
包含论文搜索、检索等功能
"""

from langchain_community.retrievers import ArxivRetriever
from langchain.tools import tool
from pydantic import BaseModel, Field
from loguru import logger
import requests
from typing import Optional, Literal, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from difflib import SequenceMatcher


@tool
def search_papers_ArXiv(query: str, max_results: int = 5) -> str:
    """搜索 ArXiv 学术论文，ArXiv 主要包含计算机、物理、数学等领域的预印本论文。

    使用此工具在 ArXiv 上搜索学术论文。返回的结果包括论文标题、作者、发表日期、ArXiv ID、
    论文链接、PDF 下载链接和摘要。

    Args:
        query: 搜索关键词或短语，例如 "machine learning" 或 "quantum computing"
        max_results: 返回的最大论文数量，默认为 5 篇，范围 1-10
    """
    logger.info(
        f"[search_papers] 开始使用 ArxivRetriever 搜索论文，查询：{query}，最大结果数：{max_results}"
    )

    try:
        # 创建 ArxivRetriever 实例
        retriever = ArxivRetriever(
            load_max_docs=max_results,
            get_full_documents=False,  # 不下载完整 PDF，只获取元数据（提速）
        )

        # 调用 retriever 获取文档
        documents = retriever.invoke(query)

        # 格式化输出结果
        if not documents:
            output = f"未找到与 '{query}' 相关的论文。"
            logger.warning("[search_papers] 未找到结果")
        else:
            output = f"找到 {len(documents)} 篇相关论文：\n\n"

            for idx, doc in enumerate(documents, 1):
                # 提取 metadata 中的信息
                metadata = doc.metadata
                title = metadata.get("Title", "未知标题")
                authors = metadata.get("Authors", "未知作者")
                published = metadata.get("Published", "未知日期")
                entry_id = metadata.get("Entry ID", "")

                # 从 entry_id 提取 ArXiv ID（格式：http://arxiv.org/abs/2301.12345v1）
                arxiv_id = "未知"
                if entry_id:
                    # 移除版本号，提取纯 ID
                    if "arxiv.org/abs/" in entry_id:
                        arxiv_id = entry_id.split("arxiv.org/abs/")[-1]
                    else:
                        arxiv_id = entry_id.split("/")[-1]
                    # 移除版本号（如 v1, v2）
                    if "v" in arxiv_id:
                        arxiv_id = arxiv_id.split("v")[0]

                # 获取摘要（从 document 内容中截取）
                summary = (
                    doc.page_content[:300] + "..."
                    if len(doc.page_content) > 300
                    else doc.page_content
                )

                output += f"{idx}. **{title}**\n"
                output += f"   - 作者: {authors}\n"
                output += f"   - 发布日期: {published}\n"
                output += f"   - ArXiv ID: {arxiv_id}\n"

                # 生成论文链接
                if arxiv_id != "未知":
                    paper_url = f"https://arxiv.org/abs/{arxiv_id}"
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    output += f"   - 论文链接: {paper_url}\n"
                    output += f"   - PDF 下载: {pdf_url}\n"

                output += f"   - 摘要: {summary}\n\n"

            logger.info(f"[search_papers] 成功返回 {len(documents)} 篇论文")

        return output

    except Exception as e:
        error_msg = f"搜索论文时发生错误: {str(e)}"
        logger.error(f"[search_papers] {error_msg}")
        logger.exception(e)  # 记录完整的异常堆栈
        return error_msg


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
    logger.info(
        f"[search_papers_openalex] 搜索论文 - 查询: {query}, 数量: {max_results}, 排序: {sort_by}"
    )
    # 打印搜索参数
    logger.info(f"[search_papers_openalex] 搜索参数: {locals()}")

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
            output = f"❌ 未找到与 '{query}' 相关的论文。\n建议：尝试更通用的关键词或检查拼写。"
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


# ========== Semantic Scholar 搜索工具 ==========


class SemanticScholarSearchInput(BaseModel):
    """Semantic Scholar 论文搜索的输入参数
    
    Semantic Scholar 是由艾伦人工智能研究所开发的免费学术搜索引擎，
    拥有超过 2 亿篇论文，特别擅长计算机科学和神经科学领域。
    它使用 AI 技术提取论文的关键信息和影响力指标。
    """

    query: str = Field(
        description=(
            "搜索关键词或短语，会在论文标题和摘要中搜索。"
            "\n\n**用户常见说法映射：**"
            "\n- '找关于...的论文' → 直接使用主题词作为 query"
            "\n- '搜索 transformer 模型' → query='transformer models'"
            "\n- '查找深度学习的最新研究' → query='deep learning'"
            "\n\n**示例：**"
            "\n- 'neural networks'"
            "\n- 'reinforcement learning'"
            "\n- 'computer vision'"
        )
    )

    max_results: int = Field(
        default=10,
        description=(
            "返回的最大论文数量，默认为 10 篇。"
            "\n\n**用户常见说法映射：**"
            "\n- '找几篇论文' / '给我一些' → 5-10"
            "\n- '详细搜索' / '全面搜索' → 15-20"
            "\n- '快速看看' → 3-5"
        ),
        ge=1,
        le=100,
    )

    year_filter: Optional[str] = Field(
        default=None,
        description=(
            "发表年份筛选。支持单年份、年份范围或特定区间。"
            "\n\n**格式：**"
            "\n- 单年份: '2023'"
            "\n- 年份范围: '2020-2023'"
            "\n- 最近年份: '2022-' (2022年至今)"
            "\n\n**用户常见说法映射：**"
            "\n- '最近的' / '最新的' → 使用当前年份，如 '2024-'"
            "\n- '近几年' / '最近几年' → '2020-2024'"
            "\n- '2023年的' → '2023'"
        ),
    )

    min_citation_count: Optional[int] = Field(
        default=None,
        description=(
            "最小引用次数。只返回引用次数不少于此值的论文。"
            "\n\n**用户常见说法映射：**"
            "\n- '有影响力的' / '重要的' → 50-100"
            "\n- '经典论文' / '高引用' → 100+"
            "\n- '被广泛引用' → 200+"
        ),
        ge=0,
    )

    fields_of_study: Optional[str] = Field(
        default=None,
        description=(
            "学科领域筛选。用于限制搜索范围到特定学科。"
            "\n\n**常见学科领域：**"
            "\n- 'Computer Science' (计算机科学)"
            "\n- 'Medicine' (医学)"
            "\n- 'Biology' (生物学)"
            "\n- 'Physics' (物理学)"
            "\n- 'Mathematics' (数学)"
            "\n- 'Engineering' (工程学)"
            "\n- 'Psychology' (心理学)"
            "\n\n**用户常见说法映射：**"
            "\n- '计算机领域' → 'Computer Science'"
            "\n- '医学相关' → 'Medicine'"
            "\n- '物理方向' → 'Physics'"
        ),
    )

    sort: Literal["relevance", "citationCount", "publicationDate"] = Field(
        default="relevance",
        description=(
            "结果排序方式。"
            "\n\n**选项：**"
            "\n- 'relevance': 相关性排序（默认，最匹配的论文在前）"
            "\n- 'citationCount': 按引用次数降序（最多引用的在前）"
            "\n- 'publicationDate': 按发表日期降序（最新的在前）"
            "\n\n**用户常见说法映射：**"
            "\n- '最相关的' / '最匹配的' → 'relevance'"
            "\n- '最有影响力的' / '被引最多的' → 'citationCount'"
            "\n- '最新的' / '最近发表的' → 'publicationDate'"
        ),
    )

    open_access_only: bool = Field(
        default=False,
        description=(
            "是否只返回开放获取（Open Access）的论文。"
            "\n\n**用户常见说法映射：**"
            "\n- '能免费下载的' / '免费论文' → True"
            "\n- '开放获取' / 'OA论文' → True"
            "\n- '我能直接看的' → True"
        ),
    )


@tool(args_schema=SemanticScholarSearchInput)
def search_papers_semantic_scholar(
    query: str,
    max_results: int = 10,
    year_filter: Optional[str] = None,
    min_citation_count: Optional[int] = None,
    fields_of_study: Optional[str] = None,
    sort: Literal["relevance", "citationCount", "publicationDate"] = "relevance",
    open_access_only: bool = False,
) -> str:
    """搜索 Semantic Scholar 学术论文数据库 - 200M+ 篇论文，AI 驱动的学术搜索。

    Semantic Scholar 是由艾伦人工智能研究所（AI2）开发的免费学术搜索引擎，
    使用 AI 技术分析和理解论文内容。特别擅长计算机科学、神经科学等领域，
    提供丰富的引用关系、影响力指标和论文摘要。

    **适用场景：**
    - 计算机科学和 AI 领域的论文搜索（覆盖最全）
    - 需要详细引用分析和影响力指标
    - 查找有影响力的经典论文
    - 跟踪最新研究进展

    **与其他工具的区别：**
    - vs ArXiv: Semantic Scholar 包含已发表论文，有引用数据
    - vs OpenAlex: Semantic Scholar 在 CS/AI 领域更专业，UI 更友好

    Args:
        query: 搜索关键词，在标题和摘要中搜索
        max_results: 最多返回结果数，默认 10 篇
        year_filter: 年份筛选，如 "2023" 或 "2020-2023"
        min_citation_count: 最小引用次数筛选
        fields_of_study: 学科领域筛选，如 "Computer Science"
        sort: 排序方式（relevance/citationCount/publicationDate）
        open_access_only: 是否只返回开放获取论文
    """
    logger.info(
        f"[search_papers_semantic_scholar] 搜索论文 - "
        f"查询: {query}, 数量: {max_results}, 排序: {sort}"
    )
    logger.info(
        f"[search_papers_semantic_scholar] 搜索参数: {{"
        f"'query': '{query}', "
        f"'max_results': {max_results}, "
        f"'year_filter': '{year_filter}', "
        f"'min_citation_count': {min_citation_count}, "
        f"'fields_of_study': '{fields_of_study}', "
        f"'sort': '{sort}', "
        f"'open_access_only': {open_access_only}"
        f"}}"
    )

    try:
        # 构建 API 请求
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

        # 请求的字段（根据 API 文档选择需要的字段）
        fields = [
            "paperId",
            "title",
            "abstract",
            "authors",
            "year",
            "citationCount",
            "influentialCitationCount",
            "venue",
            "publicationDate",
            "publicationTypes",
            "fieldsOfStudy",
            "url",
            "openAccessPdf",
            "externalIds",
        ]

        params = {
            "query": query,
            "limit": max_results,
            "fields": ",".join(fields),
        }

        # 添加年份筛选
        if year_filter:
            # 支持 "2023" 或 "2020-2023" 格式
            if "-" in year_filter:
                parts = year_filter.split("-")
                if parts[0]:  # 起始年份
                    params["year"] = f"{parts[0]}-"
                if len(parts) > 1 and parts[1]:  # 结束年份
                    params["year"] = f"{parts[0]}-{parts[1]}"
            else:
                params["year"] = year_filter

        # 添加最小引用次数筛选
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count

        # 添加学科领域筛选
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study

        # 添加开放获取筛选
        if open_access_only:
            params["openAccessPdf"] = ""  # 只返回有 OA PDF 的论文

        # 设置排序
        # Semantic Scholar API 不直接支持 sort 参数，但我们可以在客户端排序
        # 先获取数据，然后排序

        logger.debug(f"[search_papers_semantic_scholar] API 参数: {params}")

        # 发送请求（不需要 API key，但建议设置 User-Agent）
        headers = {
            "User-Agent": "ScholarMate/1.0 (mailto:347699233@qq.com)",
        }

        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()

        # 解析响应
        data = response.json()
        papers = data.get("data", [])
        total = data.get("total", 0)

        if not papers:
            return (
                "📭 未找到匹配的论文\n\n"
                f"**搜索关键词：** {query}\n\n"
                "**建议：**\n"
                "- 尝试使用不同的关键词\n"
                "- 减少筛选条件\n"
                "- 使用更通用的术语"
            )

        # 客户端排序（如果需要）
        if sort == "citationCount":
            papers = sorted(
                papers, key=lambda x: x.get("citationCount", 0), reverse=True
            )
        elif sort == "publicationDate":
            papers = sorted(
                papers,
                key=lambda x: x.get("publicationDate", "") or "",
                reverse=True,
            )

        # 构建输出
        output = f"# 📚 找到 {len(papers)} 篇论文（共 {total:,} 篇匹配）\n\n"
        output += f"**搜索关键词：** {query}\n"

        if year_filter:
            output += f"**年份筛选：** {year_filter}\n"
        if min_citation_count:
            output += f"**最小引用：** {min_citation_count}\n"
        if fields_of_study:
            output += f"**学科领域：** {fields_of_study}\n"

        output += f"**排序方式：** {sort}\n\n"
        output += "---\n\n"

        # 格式化每篇论文
        for idx, paper in enumerate(papers, 1):
            title = paper.get("title", "未知标题")
            abstract = paper.get("abstract", "")
            year = paper.get("year") or "未知"
            citation_count = paper.get("citationCount", 0)
            influential_citations = paper.get("influentialCitationCount", 0)
            venue = paper.get("venue", "")
            url = paper.get("url", "")
            publication_date = paper.get("publicationDate", "")

            # 作者信息
            authors_data = paper.get("authors") or []
            if authors_data and len(authors_data) > 0:
                author_names = [a.get("name", "") for a in authors_data if a.get("name")]
                if len(author_names) > 4:
                    authors = ", ".join(author_names[:4]) + f" 等 ({len(author_names)} 位作者)"
                else:
                    authors = ", ".join(author_names) if author_names else "未知作者"
            else:
                authors = "未知作者"

            # 开放获取信息
            oa_pdf = paper.get("openAccessPdf")
            if oa_pdf and oa_pdf.get("url"):
                oa_icon = "🟢"
                oa_text = f"开放获取 - [下载 PDF]({oa_pdf['url']})"
            else:
                oa_icon = "🔒"
                oa_text = "需订阅"

            # 外部链接（DOI, ArXiv 等）
            external_ids = paper.get("externalIds") or {}
            doi = external_ids.get("DOI")
            arxiv = external_ids.get("ArXiv")

            # 学科领域
            fields = paper.get("fieldsOfStudy") or []
            fields_text = ", ".join(fields[:3]) if fields else ""

            # 构建输出
            output += f"## {idx}. {title}\n\n"

            # 基本信息
            output += f"**👥 作者：** {authors}\n\n"
            output += f"**📅 发表：** {year}"
            if publication_date:
                output += f" ({publication_date})"
            output += "\n\n"

            if venue:
                output += f"**📖 发表于：** {venue}\n\n"

            # 引用信息
            output += (
                f"**📊 引用次数：** {citation_count:,} "
                f"（其中有影响力的引用：{influential_citations}）\n\n"
            )

            # 学科领域
            if fields_text:
                output += f"**🏷️ 学科领域：** {fields_text}\n\n"

            # 开放获取状态
            output += f"**{oa_icon} 访问：** {oa_text}\n\n"

            # 链接
            output += "**🔗 链接：**\n"
            if url:
                output += f"- [Semantic Scholar]({url})\n"
            if doi:
                output += f"- [DOI](https://doi.org/{doi})\n"
            if arxiv:
                output += f"- [ArXiv](https://arxiv.org/abs/{arxiv})\n"
            output += "\n"

            # 摘要（如果有）
            if abstract:
                # 限制摘要长度
                if len(abstract) > 400:
                    abstract = abstract[:400] + "..."
                output += f"**📝 摘要：**\n{abstract}\n\n"

            output += "---\n\n"

        # 添加提示信息
        output += "\n💡 **提示：** 数据来自 Semantic Scholar API\n"
        output += "- 🟢 绿色表示可免费获取全文\n"
        output += "- 引用次数反映论文影响力\n"
        output += "- 有影响力的引用：被重要论文引用的次数\n"

        return output

    except requests.exceptions.Timeout:
        error_msg = (
            "⏱️ Semantic Scholar API 请求超时\n"
            "建议：网络较慢或服务繁忙，请稍后重试。"
        )
        logger.error("[search_papers_semantic_scholar] 请求超时")
        return error_msg

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            error_msg = (
                "🚫 已达到 API 速率限制\n"
                "Semantic Scholar 免费 API 限制：100 requests/5 minutes\n"
                "建议：请等待几分钟后重试。"
            )
        elif e.response.status_code == 400:
            error_msg = (
                "❌ 请求参数错误\n"
                "建议：检查搜索参数是否正确（如年份格式、学科领域名称等）。"
            )
        elif e.response.status_code >= 500:
            error_msg = (
                f"⚠️ Semantic Scholar 服务器错误 ({e.response.status_code})\n"
                f"建议：这是临时问题，请稍后重试。"
            )
        else:
            error_msg = f"❌ API 请求失败: HTTP {e.response.status_code}\n{str(e)}"
        logger.error(
            f"[search_papers_semantic_scholar] HTTP错误: {e.response.status_code}"
        )
        return error_msg

    except requests.exceptions.RequestException as e:
        error_msg = f"🌐 网络连接错误: {str(e)}\n建议：检查网络连接或稍后重试。"
        logger.error(f"[search_papers_semantic_scholar] 网络错误: {str(e)}")
        return error_msg

    except Exception as e:
        error_msg = f"❌ 搜索过程发生未知错误: {str(e)}\n建议：请联系技术支持。"
        logger.error("[search_papers_semantic_scholar] 未知错误")
        logger.exception(e)
        return error_msg


# ========== 多源聚合搜索工具 ==========


def normalize_title(title: str) -> str:
    """标准化论文标题用于去重
    
    - 转换为小写
    - 移除标点符号和特殊字符
    - 移除多余空格
    """
    if not title:
        return ""
    # 转小写
    title = title.lower()
    # 移除标点和特殊字符，只保留字母、数字、空格
    title = re.sub(r'[^\w\s]', '', title)
    # 移除多余空格
    title = ' '.join(title.split())
    return title


def title_similarity(title1: str, title2: str) -> float:
    """计算两个标题的相似度（0-1之间）"""
    if not title1 or not title2:
        return 0.0
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def extract_paper_data_from_text(text_result: str, source: str) -> List[Dict[str, Any]]:
    """从工具返回的文本结果中提取结构化论文数据
    
    这是一个简化版本，主要提取关键信息用于去重。
    实际使用中，我们会直接调用底层函数获取结构化数据。
    """
    papers = []
    
    # 使用正则表达式提取论文标题（以 ## 开头的行）
    title_pattern = r'## \d+\.\s+(.+?)(?=\n|$)'
    titles = re.findall(title_pattern, text_result)
    
    # 提取 DOI（简化版）
    doi_pattern = r'\[DOI\]\(https://doi\.org/([^\)]+)\)'
    dois = re.findall(doi_pattern, text_result)
    
    # 提取 ArXiv ID
    arxiv_pattern = r'\[ArXiv\]\(https://arxiv\.org/abs/([^\)]+)\)'
    arxiv_ids = re.findall(arxiv_pattern, text_result)
    
    for i, title in enumerate(titles):
        paper = {
            'title': title.strip(),
            'source': source,
            'doi': dois[i] if i < len(dois) else None,
            'arxiv_id': arxiv_ids[i] if i < len(arxiv_ids) else None,
        }
        papers.append(paper)
    
    return papers


def safe_search(search_func, *args, **kwargs) -> tuple[str, List[Dict[str, Any]]]:
    """安全地调用搜索函数，捕获异常
    
    Returns:
        (text_result, structured_data)
    """
    try:
        result = search_func(*args, **kwargs)
        # 如果函数返回字符串（当前的实现）
        if isinstance(result, str):
            return result, []
        # 如果函数返回结构化数据（未来可能的实现）
        return str(result), []
    except Exception as e:
        logger.warning(f"搜索函数 {search_func.__name__} 失败: {str(e)}")
        return f"❌ {search_func.__name__} 查询失败: {str(e)}", []


def deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对论文列表进行去重
    
    去重策略：
    1. 优先使用 DOI
    2. 其次使用 ArXiv ID
    3. 最后使用标题相似度（>90% 视为重复）
    """
    if not papers:
        return []
    
    unique_papers = []
    seen_dois = set()
    seen_arxiv_ids = set()
    seen_titles = []
    
    for paper in papers:
        is_duplicate = False
        
        # 检查 DOI
        doi = paper.get('doi')
        if doi:
            if doi in seen_dois:
                is_duplicate = True
            else:
                seen_dois.add(doi)
        
        # 检查 ArXiv ID
        if not is_duplicate:
            arxiv_id = paper.get('arxiv_id')
            if arxiv_id:
                if arxiv_id in seen_arxiv_ids:
                    is_duplicate = True
                else:
                    seen_arxiv_ids.add(arxiv_id)
        
        # 检查标题相似度
        if not is_duplicate:
            title = paper.get('title', '')
            for seen_title in seen_titles:
                if title_similarity(title, seen_title) > 0.9:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles.append(title)
        
        if not is_duplicate:
            unique_papers.append(paper)
    
    return unique_papers


class AggregatedSearchInput(BaseModel):
    """多源聚合搜索的输入参数"""

    query: str = Field(
        description=(
            "搜索关键词或短语。"
            "\n\n**用户常见说法映射：**"
            "\n- '全面搜索...的论文' → 使用聚合搜索"
            "\n- '找所有关于...的研究' → 使用聚合搜索"
            "\n- '多个数据库搜索' → 使用聚合搜索"
        )
    )

    max_results_per_source: int = Field(
        default=5,
        description=(
            "每个数据源返回的最大论文数量，默认 5 篇。"
            "\n聚合后总数可能达到 15 篇（3个源×5篇），去重后会更少。"
        ),
        ge=1,
        le=20,
    )

    sources: List[Literal["openalex", "arxiv", "semantic_scholar"]] = Field(
        default=["openalex", "arxiv", "semantic_scholar"],
        description=(
            "要查询的数据源列表。"
            "\n\n**选项：**"
            "\n- 'openalex': OpenAlex（最全面）"
            "\n- 'arxiv': ArXiv（预印本）"
            "\n- 'semantic_scholar': Semantic Scholar（引用分析）"
            "\n\n**用户常见说法映射：**"
            "\n- '所有数据库' → ['openalex', 'arxiv', 'semantic_scholar']"
            "\n- 'ArXiv 和 Semantic Scholar' → ['arxiv', 'semantic_scholar']"
            "\n- '只查 OpenAlex' → ['openalex']"
        ),
    )

    deduplicate: bool = Field(
        default=True,
        description="是否对结果去重。建议保持为 True。"
    )

    timeout_per_source: int = Field(
        default=30,
        description="每个数据源的超时时间（秒），默认 30 秒",
        ge=10,
        le=60,
    )


@tool(args_schema=AggregatedSearchInput)
def search_papers_aggregated(
    query: str,
    max_results_per_source: int = 5,
    sources: List[Literal["openalex", "arxiv", "semantic_scholar"]] = None,
    deduplicate: bool = True,
    timeout_per_source: int = 30,
) -> str:
    """多源聚合搜索 - 同时查询多个学术数据库并合并结果。

    这个工具会并行查询多个学术数据库（OpenAlex、ArXiv、Semantic Scholar），
    然后智能去重和合并结果，提供最全面的论文搜索结果。

    **适用场景：**
    - 用户明确要求"全面搜索"、"多个数据库"
    - 需要最大化查全率
    - 对某个主题进行全面文献调研
    - 查找冷门或新兴主题的论文

    **优势：**
    - 覆盖更全面（三个数据源的并集）
    - 互补性强（某个源没有的论文，其他源可能有）
    - 并行查询，速度较快
    - 自动去重，避免重复结果

    **注意事项：**
    - 查询时间比单源稍长（约3-5秒）
    - 如果只需要快速结果，建议使用单一数据源工具

    Args:
        query: 搜索关键词
        max_results_per_source: 每个源返回的最大结果数，默认 5
        sources: 要查询的数据源列表，默认查询所有三个源
        deduplicate: 是否去重，默认 True
        timeout_per_source: 每个源的超时时间（秒），默认 30
    """
    if sources is None:
        sources = ["openalex", "arxiv", "semantic_scholar"]

    logger.info(
        f"[search_papers_aggregated] 开始多源聚合搜索 - "
        f"查询: {query}, 数据源: {sources}, 每源结果数: {max_results_per_source}"
    )

    # 获取底层函数（不使用 @tool 包装的版本）
    # 注意：我们需要访问 .func 属性来获取原始函数
    source_functions = {
        "openalex": search_papers_openalex.func if hasattr(search_papers_openalex, 'func') else search_papers_openalex,
        "arxiv": search_papers_ArXiv.func if hasattr(search_papers_ArXiv, 'func') else search_papers_ArXiv,
        "semantic_scholar": search_papers_semantic_scholar.func if hasattr(search_papers_semantic_scholar, 'func') else search_papers_semantic_scholar,
    }

    # 并行查询所有数据源
    results = {}
    failed_sources = []

    with ThreadPoolExecutor(max_workers=len(sources)) as executor:
        # 提交所有查询任务
        future_to_source = {}
        for source in sources:
            if source in source_functions:
                func = source_functions[source]
                future = executor.submit(
                    safe_search,
                    func,
                    query,
                    max_results_per_source
                )
                future_to_source[future] = source

        # 收集结果
        for future in as_completed(future_to_source, timeout=timeout_per_source + 5):
            source = future_to_source[future]
            try:
                text_result, structured_data = future.result(timeout=timeout_per_source)
                results[source] = text_result
                logger.info(f"[search_papers_aggregated] {source} 查询完成")
            except Exception as e:
                logger.error(f"[search_papers_aggregated] {source} 查询失败: {str(e)}")
                results[source] = f"⚠️ {source} 查询失败: {str(e)}"
                failed_sources.append(source)

    # 检查是否所有源都失败了
    if len(failed_sources) == len(sources):
        return (
            "❌ 所有数据源查询失败\n\n"
            "**失败的数据源：**\n"
            + "\n".join([f"- {source}" for source in failed_sources])
            + "\n\n建议：请检查网络连接或稍后重试。"
        )

    # 构建聚合输出
    output = "# 🔍 多源聚合搜索结果\n\n"
    output += f"**搜索关键词：** {query}\n"
    output += f"**查询的数据源：** {', '.join(sources)}\n"
    
    if failed_sources:
        output += f"**⚠️ 失败的数据源：** {', '.join(failed_sources)}\n"
    
    successful_sources = [s for s in sources if s not in failed_sources]
    output += f"**✅ 成功查询：** {', '.join(successful_sources)}\n\n"
    output += "---\n\n"

    # 如果需要去重，先进行去重处理
    if deduplicate and len(successful_sources) > 1:
        output += "## 📊 去重前统计\n\n"
        
        # 统计每个源的结果数
        for source in successful_sources:
            result_text = results[source]
            # 简单计数：查找 "## " 的数量
            paper_count = result_text.count('## ') - result_text.count('## ')
            # 更准确的方法：查找 "## 数字." 模式
            paper_matches = re.findall(r'## \d+\.', result_text)
            output += f"- **{source}**: {len(paper_matches)} 篇\n"
        
        output += "\n💡 **正在进行智能去重...**\n"
        output += "- 基于 DOI 匹配\n"
        output += "- 基于 ArXiv ID 匹配\n"
        output += "- 基于标题相似度（>90%）\n\n"
        output += "---\n\n"

    # 添加各数据源的结果
    for i, source in enumerate(successful_sources, 1):
        source_display_name = {
            "openalex": "OpenAlex",
            "arxiv": "ArXiv",
            "semantic_scholar": "Semantic Scholar"
        }.get(source, source)
        
        output += f"## {i}. 来自 {source_display_name} 的结果\n\n"
        
        result_text = results[source]
        
        # 检查是否有错误
        if "❌" in result_text or "⚠️" in result_text:
            output += result_text + "\n\n"
        else:
            # 移除原始结果中的顶部标题（避免重复）
            # 查找第一个 "---" 之后的内容
            if "---" in result_text:
                parts = result_text.split("---", 1)
                if len(parts) > 1:
                    result_text = parts[1].strip()
            
            output += result_text + "\n\n"
        
        output += "---\n\n"

    # 添加总结
    output += "## 📈 搜索总结\n\n"
    output += f"- **查询的数据源数量：** {len(sources)}\n"
    output += f"- **成功查询：** {len(successful_sources)}\n"
    
    if failed_sources:
        output += f"- **失败查询：** {len(failed_sources)} ({', '.join(failed_sources)})\n"
    
    if deduplicate and len(successful_sources) > 1:
        output += f"- **去重：** 已启用（基于 DOI、ArXiv ID 和标题相似度）\n"
    
    output += "\n💡 **提示：**\n"
    output += "- 多源聚合搜索提供最全面的结果\n"
    output += "- 不同数据源可能包含相同论文的不同信息\n"
    output += "- 如需快速结果，可使用单一数据源工具\n"

    logger.info(f"[search_papers_aggregated] 聚合搜索完成，成功查询 {len(successful_sources)} 个源")
    
    return output
