"""测试 OpenAlex 搜索功能"""

from tools import search_papers_openalex
from loguru import logger

# 配置日志
logger.add("logs/test_openalex.log", rotation="10 MB")

def test_basic_search():
    """测试基本搜索功能"""
    print("=" * 80)
    print("测试 1: 基本搜索 - 'deep learning'")
    print("=" * 80)
    
    result = search_papers_openalex("deep learning", max_results=3)
    print(result)
    print("\n")

def test_sort_by_citations():
    """测试按引用次数排序"""
    print("=" * 80)
    print("测试 2: 按引用次数排序 - 'transformer'")
    print("=" * 80)
    
    result = search_papers_openalex(
        "transformer", 
        max_results=3,
        sort_by="cited_by_count:desc"
    )
    print(result)
    print("\n")

def test_recent_papers():
    """测试获取最新论文"""
    print("=" * 80)
    print("测试 3: 最新论文 - 'large language model'")
    print("=" * 80)
    
    result = search_papers_openalex(
        "large language model",
        max_results=3,
        sort_by="publication_date:desc"
    )
    print(result)
    print("\n")

if __name__ == "__main__":
    logger.info("开始测试 OpenAlex 搜索功能")
    
    try:
        test_basic_search()
        test_sort_by_citations()
        test_recent_papers()
        
        logger.info("所有测试完成！")
        print("✅ 所有测试完成！")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        logger.exception(e)
        print(f"❌ 测试失败: {e}")
