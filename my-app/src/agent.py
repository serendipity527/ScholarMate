from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import ArxivRetriever
import os
from enum import Enum
from loguru import logger
from typing import List, Dict

# 测试 loguru 是否正常工作
logger.info("Agent 模块加载完成，loguru 日志系统正常")

class IntentType(Enum):
    # 意图类型
    NORMAL_CONVERSATION = "普通对话"  # 日常聊天、简单问答、非学术性讨论
    PAPER_SEARCH = "论文搜索"  # 搜索、查找、检索学术论文


def search_papers(query: str, max_results: int = 5) -> str:
    """使用 LangChain ArxivRetriever 搜索 ArXiv 学术论文
    
    Args:
        query: 搜索关键词，支持 ArXiv 高级搜索语法
        max_results: 最多返回结果数量，默认5篇（避免输出过长）
        
    Returns:
        格式化的搜索结果字符串，包含论文标题、作者、摘要等信息
    """
    logger.info(f"[search_papers] 开始使用 ArxivRetriever 搜索论文，查询：{query}，最大结果数：{max_results}")
    
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
            logger.warning(f"[search_papers] 未找到结果")
        else:
            output = f"找到 {len(documents)} 篇相关论文：\n\n"
            
            for idx, doc in enumerate(documents, 1):
                # 提取 metadata 中的信息
                metadata = doc.metadata
                title = metadata.get('Title', '未知标题')
                authors = metadata.get('Authors', '未知作者')
                published = metadata.get('Published', '未知日期')
                entry_id = metadata.get('Entry ID', '')
                
                # 从 entry_id 提取 ArXiv ID（格式：http://arxiv.org/abs/2301.12345v1）
                arxiv_id = '未知'
                if entry_id:
                    # 移除版本号，提取纯 ID
                    if 'arxiv.org/abs/' in entry_id:
                        arxiv_id = entry_id.split('arxiv.org/abs/')[-1]
                    else:
                        arxiv_id = entry_id.split('/')[-1]
                    # 移除版本号（如 v1, v2）
                    if 'v' in arxiv_id:
                        arxiv_id = arxiv_id.split('v')[0]
                
                # 获取摘要（从 document 内容中截取）
                summary = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                
                output += f"{idx}. **{title}**\n"
                output += f"   - 作者: {authors}\n"
                output += f"   - 发布日期: {published}\n"
                output += f"   - ArXiv ID: {arxiv_id}\n"
                
                # 生成论文链接
                if arxiv_id != '未知':
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


def send_email(to: str, subject: str, body: str):
    """发送邮件"""
    logger.info(f"[send_email] 调用发送邮件功能，收件人：{to}，主题：{subject}")
    # 模拟邮件发送
    result = f"发送邮件到 {to}，主题：{subject}"
    logger.info(f"[send_email] 返回结果：{result}")
    return result


TONGYI_API_KEY = os.getenv("TONGYI_API_KEY")
if not TONGYI_API_KEY:
    raise ValueError("缺少 TONGYI_API_KEY 环境变量")

model = ChatTongyi(
    streaming=True,
    api_key=TONGYI_API_KEY,
)

agent = create_agent(
    model=model,
    tools=[search_papers, send_email],
    system_prompt="你是一个智能助手，具备意图识别功能，可以进行普通对话、论文搜索和发送邮件。",
)
