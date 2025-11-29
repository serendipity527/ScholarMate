"""引用网络分析工具
使用 Semantic Scholar API 分析论文的引用关系
"""

import requests
import time
from typing import List, Dict, Optional
from langchain.tools import tool
from pydantic import BaseModel, Field
from loguru import logger


# ================== 配置 ==================

# Semantic Scholar API 配置
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_RATE_LIMIT = 1.0  # 每秒最多1个请求

# 字段配置：需要获取的论文字段
PAPER_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "authors",
    "citationCount",
    "influentialCitationCount",
    "venue",
    "publicationDate",
    "isOpenAccess",
    "openAccessPdf",
    "url",
]


# ================== 速率限制器 ==================

class RateLimiter:
    """API 请求速率限制器"""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.last_call = 0.0
    
    def wait(self):
        """等待以满足速率限制"""
        now = time.time()
        time_since_last_call = now - self.last_call
        if time_since_last_call < 1.0 / self.calls_per_second:
            sleep_time = 1.0 / self.calls_per_second - time_since_last_call
            time.sleep(sleep_time)
        self.last_call = time.time()


# 全局速率限制器
rate_limiter = RateLimiter(calls_per_second=S2_RATE_LIMIT)


# ================== 论文识别器 ==================

def identify_paper(paper_input: str) -> Optional[Dict]:
    """识别论文，支持多种输入格式
    
    Args:
        paper_input: 论文标识（标题/DOI/ArXiv ID/S2 ID/URL）
        
    Returns:
        论文信息字典，如果未找到返回 None
    """
    logger.info(f"[identify_paper] 识别论文: {paper_input[:100]}")
    
    # 1. 尝试作为 Semantic Scholar ID
    if paper_input.startswith("CorpusId:") or len(paper_input) == 40:
        paper = get_paper_by_id(paper_input)
        if paper:
            logger.info("[identify_paper] ✅ 通过 S2 ID 找到")
            return paper
    
    # 2. 优先检查 ArXiv ID（包括 DOI 中的 ArXiv）
    arxiv_id = extract_arxiv_id(paper_input)
    if arxiv_id:
        logger.debug(f"[identify_paper] 尝试 ArXiv ID: {arxiv_id}")
        paper = search_paper_by_arxiv(arxiv_id)
        if paper:
            logger.info("[identify_paper] ✅ 通过 ArXiv ID 找到")
            return paper
    
    # 3. 尝试作为 DOI（但排除 arXiv DOI，因为已经在步骤2处理）
    if "10." in paper_input and "arxiv" not in paper_input.lower():
        doi = extract_doi(paper_input)
        if doi:
            logger.debug(f"[identify_paper] 尝试 DOI: {doi}")
            paper = search_paper_by_doi(doi)
            if paper:
                logger.info("[identify_paper] ✅ 通过 DOI 找到")
                return paper
    
    # 4. 最后尝试标题搜索（但要排除明显的ID格式）
    # 如果输入看起来像 ID 而不是标题，跳过标题搜索
    if not (paper_input.startswith("10.") or paper_input.startswith("http")):
        logger.debug("[identify_paper] 尝试标题搜索")
        paper = search_paper_by_title(paper_input)
        if paper:
            logger.info("[identify_paper] ✅ 通过标题搜索找到")
            return paper
    
    logger.warning(f"[identify_paper] ❌ 未找到论文: {paper_input}")
    return None


def extract_arxiv_id(text: str) -> Optional[str]:
    """从文本中提取 ArXiv ID"""
    import re
    # 匹配格式：1234.5678 或 arXiv:1234.5678
    patterns = [
        r'(?:arXiv:)?(\d{4}\.\d{4,5})',
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_doi(text: str) -> Optional[str]:
    """从文本中提取 DOI"""
    import re
    # 匹配 DOI 格式
    match = re.search(r'10\.\d{4,}/[^\s]+', text)
    if match:
        return match.group(0)
    return None


def get_paper_by_id(paper_id: str) -> Optional[Dict]:
    """通过 Semantic Scholar ID 获取论文"""
    rate_limiter.wait()
    url = f"{S2_API_BASE}/paper/{paper_id}"
    params = {"fields": ",".join(PAPER_FIELDS)}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        logger.warning(f"[get_paper_by_id] 状态码 {response.status_code}")
    except Exception as e:
        logger.error(f"[get_paper_by_id] 错误: {e}")
    
    return None


def search_paper_by_arxiv(arxiv_id: str) -> Optional[Dict]:
    """通过 ArXiv ID 搜索论文"""
    rate_limiter.wait()
    url = f"{S2_API_BASE}/paper/arXiv:{arxiv_id}"
    params = {"fields": ",".join(PAPER_FIELDS)}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"[search_paper_by_arxiv] 错误: {e}")
    
    return None


def search_paper_by_doi(doi: str) -> Optional[Dict]:
    """通过 DOI 搜索论文"""
    rate_limiter.wait()
    url = f"{S2_API_BASE}/paper/DOI:{doi}"
    params = {"fields": ",".join(PAPER_FIELDS)}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"[search_paper_by_doi] 错误: {e}")
    
    return None


def search_paper_by_title(title: str) -> Optional[Dict]:
    """通过标题搜索论文"""
    rate_limiter.wait()
    url = f"{S2_API_BASE}/paper/search"
    params = {
        "query": title,
        "fields": ",".join(PAPER_FIELDS),
        "limit": 1,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                return data["data"][0]
    except Exception as e:
        logger.error(f"[search_paper_by_title] 错误: {e}")
    
    return None


# ================== 引用数据获取器 ==================

def get_references(paper_id: str, limit: int = 100) -> List[Dict]:
    """获取论文的参考文献列表
    
    Args:
        paper_id: Semantic Scholar 论文 ID
        limit: 最多获取的数量
        
    Returns:
        参考文献列表
    """
    logger.info(f"[get_references] 获取参考文献: {paper_id}, limit={limit}")
    rate_limiter.wait()
    
    url = f"{S2_API_BASE}/paper/{paper_id}/references"
    params = {
        "fields": ",".join(PAPER_FIELDS),
        "limit": limit,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            data_list = data.get("data") or []
            references = [item["citedPaper"] for item in data_list if item.get("citedPaper")]
            logger.info(f"[get_references] 获取到 {len(references)} 篇参考文献")
            return references
        else:
            logger.warning(f"[get_references] API 返回状态码: {response.status_code}")
    except Exception as e:
        logger.error(f"[get_references] 错误: {e}")
    
    return []


def get_citations(paper_id: str, limit: int = 100) -> List[Dict]:
    """获取引用该论文的文献列表
    
    Args:
        paper_id: Semantic Scholar 论文 ID
        limit: 最多获取的数量
        
    Returns:
        引用论文列表
    """
    logger.info(f"[get_citations] 获取引用论文: {paper_id}, limit={limit}")
    rate_limiter.wait()
    
    url = f"{S2_API_BASE}/paper/{paper_id}/citations"
    params = {
        "fields": ",".join(PAPER_FIELDS),
        "limit": limit,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            data_list = data.get("data") or []
            citations = [item["citingPaper"] for item in data_list if item.get("citingPaper")]
            logger.info(f"[get_citations] 获取到 {len(citations)} 篇引用论文")
            return citations
        else:
            logger.warning(f"[get_citations] API 返回状态码: {response.status_code}")
    except Exception as e:
        logger.error(f"[get_citations] 错误: {e}")
    
    return []


# ================== 重要性评分器 ==================

def calculate_reference_score(paper: Dict) -> float:
    """计算参考文献的重要性分数
    
    评分维度：
    - 被引次数（40分）
    - 影响力引用（30分）
    - 发表年份（15分，早期经典加分）
    - 发表期刊/会议（15分，顶会加分）
    
    Args:
        paper: 论文信息字典
        
    Returns:
        重要性分数（0-100）
    """
    score = 0.0
    
    # 1. 被引次数（归一化到40分）
    citation_count = paper.get("citationCount") or 0
    score += min(citation_count / 1000, 1.0) * 40
    
    # 2. 影响力引用（归一化到30分）
    influential_count = paper.get("influentialCitationCount") or 0
    score += min(influential_count / 100, 1.0) * 30
    
    # 3. 发表年份（早期经典加分）
    year = paper.get("year")
    if year:
        if year < 2010:
            score += 15  # 经典论文
        elif year < 2015:
            score += 10
        elif year < 2020:
            score += 5
    
    # 4. 顶会加分
    venue = paper.get("venue", "").lower()
    top_venues = [
        "cvpr", "iccv", "eccv",  # 计算机视觉
        "iclr", "neurips", "icml",  # 机器学习
        "acl", "emnlp", "naacl",  # NLP
        "aaai", "ijcai",  # AI
        "kdd", "www", "sigir",  # 数据挖掘
    ]
    if any(v in venue for v in top_venues):
        score += 15
    
    return round(score, 1)


def calculate_citation_score(paper: Dict) -> float:
    """计算引用论文的重要性分数（SOTA评分）
    
    评分维度：
    - 发表年份（40分，近期优先）
    - 被引次数（30分）
    - 发表期刊/会议（20分，顶会加分）
    - 开放获取（10分）
    
    Args:
        paper: 论文信息字典
        
    Returns:
        SOTA 分数（0-100）
    """
    score = 0.0
    
    # 1. 发表年份（近期优先）
    year = paper.get("year")
    if year:
        current_year = 2024  # 可以改为动态获取
        if year >= current_year:
            score += 40
        elif year >= current_year - 1:
            score += 30
        elif year >= current_year - 2:
            score += 20
        elif year >= current_year - 3:
            score += 10
    
    # 2. 被引次数（归一化到30分）
    citation_count = paper.get("citationCount") or 0
    score += min(citation_count / 50, 1.0) * 30
    
    # 3. 顶会加分
    venue = paper.get("venue", "").lower()
    top_venues = [
        "cvpr", "iccv", "eccv",
        "iclr", "neurips", "icml",
        "acl", "emnlp", "naacl",
        "aaai", "ijcai",
        "kdd", "www", "sigir",
    ]
    if any(v in venue for v in top_venues):
        score += 20
    
    # 4. 开放获取加分
    if paper.get("isOpenAccess", False):
        score += 10
    
    return round(score, 1)


def rank_papers(papers: List[Dict], mode: str = "reference") -> List[Dict]:
    """对论文列表进行排序
    
    Args:
        papers: 论文列表
        mode: 排序模式，"reference"（参考文献）或 "citation"（引用论文）
        
    Returns:
        排序后的论文列表（带有 importance_score 字段）
    """
    # 计算分数
    for paper in papers:
        if mode == "reference":
            paper["importance_score"] = calculate_reference_score(paper)
        else:
            paper["importance_score"] = calculate_citation_score(paper)
    
    # 按分数降序排序
    papers.sort(key=lambda x: x["importance_score"], reverse=True)
    
    return papers


# ================== 格式化输出 ==================

def format_paper_info(paper: Dict, index: int = None) -> str:
    """格式化单篇论文信息
    
    Args:
        paper: 论文信息字典
        index: 序号（可选）
        
    Returns:
        格式化的字符串
    """
    title = paper.get("title", "无标题")
    authors = paper.get("authors", [])
    author_names = [a.get("name", "") for a in authors[:3]]
    author_str = ", ".join(author_names)
    if len(authors) > 3:
        author_str += f" 等 {len(authors)} 位作者"
    
    year = paper.get("year", "N/A")
    citation_count = paper.get("citationCount") or 0
    venue = paper.get("venue", "N/A")
    url = paper.get("url", "N/A")
    score = paper.get("importance_score") or 0
    
    # 开放获取标记
    oa_mark = "🔓" if paper.get("isOpenAccess", False) else ""
    
    prefix = f"### {index}. " if index else "### "
    
    return f"""{prefix}{title} {oa_mark}
- **作者**: {author_str}
- **年份**: {year}
- **被引**: {citation_count} 次
- **期刊/会议**: {venue}
- **重要性评分**: {score}/100
- **链接**: {url}

"""


def format_citation_network(
    center_paper: Dict,
    references: List[Dict],
    citations: List[Dict]
) -> str:
    """格式化引用网络分析结果
    
    Args:
        center_paper: 中心论文
        references: 参考文献列表
        citations: 引用论文列表
        
    Returns:
        格式化的 Markdown 字符串
    """
    output = f"""# 📊 引用网络分析

## 🎯 目标论文

**{center_paper.get('title', '无标题')}**

"""
    
    # 论文基本信息
    authors = center_paper.get("authors", [])
    if authors:
        author_names = [a.get("name", "") for a in authors[:5]]
        output += f"- **作者**: {', '.join(author_names)}"
        if len(authors) > 5:
            output += f" 等 {len(authors)} 位作者"
        output += "\n"
    
    output += f"""- **年份**: {center_paper.get('year', 'N/A')}
- **被引次数**: {center_paper.get('citationCount', 0)} 次
- **影响力引用**: {center_paper.get('influentialCitationCount', 0)} 次
- **期刊/会议**: {center_paper.get('venue', 'N/A')}
- **链接**: {center_paper.get('url', 'N/A')}

---

## 📚 前世：重要参考文献（Top {len(references)}）

这些是该论文引用的最重要的参考文献，代表了该研究的理论基础：

"""
    
    # 参考文献列表
    for idx, ref in enumerate(references, 1):
        output += format_paper_info(ref, idx)
    
    output += f"""---

## 🚀 今生：引用它的 SOTA 论文（Top {len(citations)}）

这些是引用了该论文的最新高质量研究，代表了该方向的最新进展：

"""
    
    # 引用论文列表
    for idx, cite in enumerate(citations, 1):
        output += format_paper_info(cite, idx)
    
    # 添加简单的统计信息
    # 计算平均被引次数（避免除以0和None）
    avg_citations_ref = (
        sum(r.get('citationCount') or 0 for r in references) / len(references)
        if len(references) > 0 else 0
    )
    avg_citations_cite = (
        sum(c.get('citationCount') or 0 for c in citations) / len(citations)
        if len(citations) > 0 else 0
    )
    
    output += f"""---

## 📈 统计摘要

- **参考文献总数**: 检测到 {len(references)} 篇重要参考文献
- **引用论文总数**: 检测到 {len(citations)} 篇引用论文
- **平均被引次数（参考文献）**: {avg_citations_ref:.1f} 次
- **平均被引次数（引用论文）**: {avg_citations_cite:.1f} 次
- **开放获取论文数**: {sum(1 for p in references + citations if p.get('isOpenAccess', False))} 篇

💡 **提示**: 标有 🔓 的论文可以免费获取全文。

"""
    
    # 添加引用关系可视化图（Mermaid）
    if references or citations:
        output += generate_mermaid_graph(center_paper, references[:5], citations[:5])
    
    return output


def generate_mermaid_graph(
    center_paper: Dict,
    top_references: List[Dict],
    top_citations: List[Dict]
) -> str:
    """生成 Mermaid 引用关系图
    
    Args:
        center_paper: 中心论文
        top_references: Top N 参考文献
        top_citations: Top N 引用论文
        
    Returns:
        Mermaid 图表的 Markdown 代码
    """
    center_title = center_paper.get('title', '目标论文')[:40]
    center_year = center_paper.get('year', 'N/A')
    
    graph = """---

## 🔗 引用关系可视化

```mermaid
graph TB
    Center["📄 {center_title}...<br/>({center_year})"]
    
    subgraph refs["📚 前世 - 重要参考文献"]
""".format(center_title=center_title, center_year=center_year)
    
    # 添加参考文献节点
    for idx, ref in enumerate(top_references):
        ref_title = ref.get('title', '参考文献')[:30]
        ref_year = ref.get('year', 'N/A')
        ref_citations = ref.get('citationCount') or 0
        
        # 根据被引次数设置节点样式
        if ref_citations > 10000:
            node_style = "Ref{idx}[/🌟 {title}...<br/>({year}, {cites}引)/]"
        elif ref_citations > 1000:
            node_style = "Ref{idx}[/{title}...<br/>({year}, {cites}引)/]"
        else:
            node_style = "Ref{idx}[{title}...<br/>({year})]"
        
        graph += "        " + node_style.format(
            idx=idx,
            title=ref_title,
            year=ref_year,
            cites=ref_citations
        ) + "\n"
        graph += f"        Ref{idx} -->|引用| Center\n"
    
    graph += "    end\n\n    subgraph cites[\"🚀 今生 - SOTA 引用论文\"]\n"
    
    # 添加引用论文节点
    for idx, cite in enumerate(top_citations):
        cite_title = cite.get('title', '引用论文')[:30]
        cite_year = cite.get('year', 'N/A')
        cite_citations = cite.get('citationCount') or 0
        
        # 根据年份设置节点样式
        if cite_year and cite_year >= 2023:
            node_style = "Cite{idx}[\\🔥 {title}...<br/>({year}, {cites}引)\\]"
        elif cite_year and cite_year >= 2020:
            node_style = "Cite{idx}[\\{title}...<br/>({year}, {cites}引)\\]"
        else:
            node_style = "Cite{idx}[{title}...<br/>({year})]"
        
        graph += "        " + node_style.format(
            idx=idx,
            title=cite_title,
            year=cite_year,
            cites=cite_citations
        ) + "\n"
        graph += f"        Center -->|被引用| Cite{idx}\n"
    
    graph += """    end
    
    style Center fill:#f9f,stroke:#333,stroke-width:4px
    style refs fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    style cites fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

**图例说明**：
- 📄 中心节点：目标论文
- 📚 左侧：该论文引用的重要参考文献
- 🚀 右侧：引用了该论文的 SOTA 研究
- 🌟 高被引论文（>10,000次）
- 🔥 最新论文（2023年+）

"""
    
    return graph


# ================== LangChain 工具 ==================

class CitationNetworkInput(BaseModel):
    """引用网络分析工具的输入参数"""
    
    paper_identifier: str = Field(
        description="论文标识，支持多种格式：论文标题、DOI、ArXiv ID、Semantic Scholar ID 或 URL"
    )
    max_references: int = Field(
        default=5,
        description="返回的最大参考文献数量，默认5篇"
    )
    max_citations: int = Field(
        default=5,
        description="返回的最大引用论文数量，默认5篇"
    )


@tool(args_schema=CitationNetworkInput)
def analyze_citation_network(
    paper_identifier: str,
    max_references: int = 5,
    max_citations: int = 5
) -> str:
    """分析论文的引用网络，找出重要的参考文献和引用它的SOTA论文。
    
    这个工具可以帮助用户快速了解一篇论文的学术影响力和研究脉络：
    - **前世**：这篇论文引用的重要参考文献（理论基础）
    - **今生**：引用了这篇论文的最新SOTA研究（最新进展）
    
    使用场景：
    - 快速了解某个研究方向的经典论文
    - 追踪研究主题的最新进展
    - 发现相关的高质量论文
    - 构建文献综述
    
    Args:
        paper_identifier: 论文标识（标题/DOI/ArXiv ID/URL）
        max_references: 返回的最大参考文献数量
        max_citations: 返回的最大引用论文数量
    
    Returns:
        格式化的引用网络分析结果
        
    示例：
        analyze_citation_network("Attention is All You Need")
        analyze_citation_network("1706.03762")  # ArXiv ID
        analyze_citation_network("10.48550/arXiv.1706.03762")  # DOI
    """
    try:
        # 1. 识别论文
        logger.info(f"[analyze_citation_network] 开始分析: {paper_identifier}")
        paper = identify_paper(paper_identifier)
        
        if not paper:
            return f"❌ 未找到论文: {paper_identifier}\n\n请检查输入是否正确，支持以下格式：\n- 论文标题\n- ArXiv ID（如：1706.03762）\n- DOI\n- Semantic Scholar URL"
        
        paper_id = paper.get("paperId")
        paper_title = paper.get("title", "未知")
        logger.info(f"[analyze_citation_network] 找到论文: {paper_title}")
        
        # 2. 并行获取引用数据
        references = get_references(paper_id, limit=100)
        citations = get_citations(paper_id, limit=100)
        
        if not references and not citations:
            return f"⚠️ 找到论文但无引用数据: {paper_title}\n\n这可能是一篇非常新的论文或数据库中暂无引用信息。"
        
        # 3. 筛选和排序
        if references:
            references = rank_papers(references, mode="reference")
            references = references[:max_references]
        
        if citations:
            citations = rank_papers(citations, mode="citation")
            citations = citations[:max_citations]
        
        # 4. 格式化输出
        result = format_citation_network(paper, references, citations)
        
        logger.info(f"[analyze_citation_network] 分析完成: {len(references)} 篇参考文献, {len(citations)} 篇引用论文")
        return result
        
    except Exception as e:
        error_msg = f"❌ 分析引用网络时出错: {str(e)}"
        logger.error(f"[analyze_citation_network] {error_msg}")
        return error_msg
