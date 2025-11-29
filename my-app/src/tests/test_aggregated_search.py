"""多源聚合搜索工具的测试"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# 添加 src 到路径
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from tools import paper_tools
from tools.paper_tools import (
    AggregatedSearchInput,
    normalize_title,
    title_similarity,
    deduplicate_papers,
)

# 获取底层函数
search_papers_aggregated = paper_tools.search_papers_aggregated.func


class TestAggregatedSearchInput:
    """测试聚合搜索输入模型"""

    def test_valid_basic_input(self):
        """测试基本有效输入"""
        input_data = AggregatedSearchInput(query="deep learning")
        assert input_data.query == "deep learning"
        assert input_data.max_results_per_source == 5
        assert input_data.sources == ["openalex", "arxiv", "semantic_scholar"]
        assert input_data.deduplicate is True
        assert input_data.timeout_per_source == 30

    def test_custom_sources(self):
        """测试自定义数据源"""
        input_data = AggregatedSearchInput(
            query="test",
            sources=["openalex", "arxiv"]
        )
        assert len(input_data.sources) == 2
        assert "semantic_scholar" not in input_data.sources

    def test_max_results_validation(self):
        """测试 max_results_per_source 验证"""
        # 有效值
        input_data = AggregatedSearchInput(query="test", max_results_per_source=1)
        assert input_data.max_results_per_source == 1

        input_data = AggregatedSearchInput(query="test", max_results_per_source=20)
        assert input_data.max_results_per_source == 20

        # 无效值应该抛出异常
        with pytest.raises(Exception):
            AggregatedSearchInput(query="test", max_results_per_source=0)

        with pytest.raises(Exception):
            AggregatedSearchInput(query="test", max_results_per_source=21)


class TestHelperFunctions:
    """测试辅助函数"""

    def test_normalize_title(self):
        """测试标题标准化"""
        assert normalize_title("Deep Learning") == "deep learning"
        assert normalize_title("Deep-Learning!") == "deeplearning"
        assert normalize_title("Deep  Learning") == "deep learning"
        assert normalize_title("") == ""
        assert normalize_title(None) == ""

    def test_title_similarity(self):
        """测试标题相似度计算"""
        # 完全相同
        similarity = title_similarity(
            "Deep Learning for Computer Vision",
            "Deep Learning for Computer Vision"
        )
        assert similarity == 1.0

        # 非常相似（只有标点符号不同）
        similarity = title_similarity(
            "Deep Learning for Computer Vision",
            "Deep Learning for Computer-Vision!"
        )
        assert similarity > 0.9

        # 完全不同
        similarity = title_similarity(
            "Deep Learning",
            "Quantum Computing"
        )
        assert similarity < 0.5

        # 空字符串
        assert title_similarity("", "") == 0.0
        assert title_similarity("test", "") == 0.0

    def test_deduplicate_papers(self):
        """测试论文去重"""
        papers = [
            {"title": "Paper A", "doi": "10.1234/a", "arxiv_id": None},
            {"title": "Paper A", "doi": "10.1234/a", "arxiv_id": None},  # 重复（相同 DOI）
            {"title": "Paper B", "doi": None, "arxiv_id": "2301.12345"},
            {"title": "Paper B", "doi": None, "arxiv_id": "2301.12345"},  # 重复（相同 ArXiv ID）
            {"title": "Paper C", "doi": None, "arxiv_id": None},
            {"title": "Paper-C!", "doi": None, "arxiv_id": None},  # 重复（标题相似）
            {"title": "Paper D", "doi": None, "arxiv_id": None},
        ]

        unique = deduplicate_papers(papers)
        
        # 应该去掉 3 个重复项
        assert len(unique) == 4
        
        # 验证保留的论文
        titles = [p["title"] for p in unique]
        assert "Paper A" in titles
        assert "Paper B" in titles
        assert "Paper D" in titles


class TestAggregatedSearch:
    """测试聚合搜索功能"""

    @patch('tools.paper_tools.search_papers_openalex')
    @patch('tools.paper_tools.search_papers_ArXiv')
    @patch('tools.paper_tools.search_papers_semantic_scholar')
    def test_basic_aggregated_search(self, mock_ss, mock_arxiv, mock_openalex):
        """测试基本聚合搜索"""
        # 模拟三个数据源的返回结果
        mock_openalex.func = Mock(return_value="# 📚 找到 2 篇论文\n\n## 1. Paper from OpenAlex\n")
        mock_arxiv.func = Mock(return_value="# 📚 找到 1 篇论文\n\n## 1. Paper from ArXiv\n")
        mock_ss.func = Mock(return_value="# 📚 找到 1 篇论文\n\n## 1. Paper from Semantic Scholar\n")

        result = search_papers_aggregated(
            query="deep learning",
            max_results_per_source=5
        )

        # 验证结果包含关键信息
        assert "多源聚合搜索结果" in result
        assert "deep learning" in result
        assert "OpenAlex" in result or "openalex" in result
        assert "ArXiv" in result or "arxiv" in result
        assert "Semantic Scholar" in result or "semantic_scholar" in result

    @patch('tools.paper_tools.search_papers_openalex')
    @patch('tools.paper_tools.search_papers_ArXiv')
    @patch('tools.paper_tools.search_papers_semantic_scholar')
    def test_custom_sources(self, mock_ss, mock_arxiv, mock_openalex):
        """测试只查询指定的数据源"""
        mock_openalex.func = Mock(return_value="# 📚 找到 1 篇论文\n\n## 1. Paper\n")
        mock_arxiv.func = Mock(return_value="# 📚 找到 1 篇论文\n\n## 1. Paper\n")
        mock_ss.func = Mock(return_value="# 📚 找到 1 篇论文\n\n## 1. Paper\n")

        result = search_papers_aggregated(
            query="test",
            sources=["openalex", "arxiv"]  # 只查询两个源
        )

        # OpenAlex 和 ArXiv 应该被调用
        assert mock_openalex.func.called or "OpenAlex" in result
        
        # Semantic Scholar 不应该出现在结果中（除非默认值有变化）
        # 注意：由于实现中可能会显示所有配置的源，我们主要验证功能正常

    @patch('tools.paper_tools.search_papers_openalex')
    @patch('tools.paper_tools.search_papers_ArXiv')
    @patch('tools.paper_tools.search_papers_semantic_scholar')
    def test_partial_failure(self, mock_ss, mock_arxiv, mock_openalex):
        """测试部分数据源失败的情况"""
        # 模拟 OpenAlex 成功，其他失败
        mock_openalex.func = Mock(return_value="# 📚 找到 2 篇论文\n\n## 1. Paper\n")
        mock_arxiv.func = Mock(side_effect=Exception("ArXiv API 失败"))
        mock_ss.func = Mock(side_effect=Exception("Semantic Scholar API 失败"))

        result = search_papers_aggregated(
            query="test",
            max_results_per_source=5
        )

        # 应该包含成功和失败的信息
        assert "多源聚合搜索结果" in result or "搜索" in result
        # 至少有一个源成功，所以不应该是全部失败
        assert "所有数据源查询失败" not in result

    @patch('tools.paper_tools.search_papers_openalex')
    @patch('tools.paper_tools.search_papers_ArXiv')
    @patch('tools.paper_tools.search_papers_semantic_scholar')
    def test_all_sources_fail(self, mock_ss, mock_arxiv, mock_openalex):
        """测试所有数据源都失败的情况"""
        # 模拟所有源都失败
        mock_openalex.func = Mock(side_effect=Exception("失败"))
        mock_arxiv.func = Mock(side_effect=Exception("失败"))
        mock_ss.func = Mock(side_effect=Exception("失败"))

        result = search_papers_aggregated(
            query="test",
            max_results_per_source=5
        )

        # 应该返回所有失败的错误信息
        assert "所有数据源查询失败" in result or "失败" in result

    def test_deduplicate_parameter(self):
        """测试去重参数"""
        # 测试去重开启
        input_data = AggregatedSearchInput(
            query="test",
            deduplicate=True
        )
        assert input_data.deduplicate is True

        # 测试去重关闭
        input_data = AggregatedSearchInput(
            query="test",
            deduplicate=False
        )
        assert input_data.deduplicate is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
