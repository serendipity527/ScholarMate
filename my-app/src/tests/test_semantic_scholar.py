"""Semantic Scholar 工具的基本测试"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# 添加 src 到路径
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from tools import paper_tools
from tools.paper_tools import SemanticScholarSearchInput

# 获取底层函数
search_papers_semantic_scholar = paper_tools.search_papers_semantic_scholar.func


class TestSemanticScholarSearchInput:
    """测试 Semantic Scholar 输入模型的参数验证"""

    def test_valid_basic_input(self):
        """测试基本有效输入"""
        input_data = SemanticScholarSearchInput(query="machine learning")
        assert input_data.query == "machine learning"
        assert input_data.max_results == 10
        assert input_data.sort == "relevance"

    def test_valid_all_parameters(self):
        """测试所有参数"""
        input_data = SemanticScholarSearchInput(
            query="deep learning",
            max_results=20,
            year_filter="2020-2023",
            min_citation_count=50,
            fields_of_study="Computer Science",
            sort="citationCount",
            open_access_only=True,
        )
        assert input_data.query == "deep learning"
        assert input_data.max_results == 20
        assert input_data.year_filter == "2020-2023"
        assert input_data.min_citation_count == 50
        assert input_data.fields_of_study == "Computer Science"
        assert input_data.sort == "citationCount"
        assert input_data.open_access_only is True

    def test_max_results_validation(self):
        """测试 max_results 边界值"""
        # 有效值
        input_data = SemanticScholarSearchInput(query="test", max_results=1)
        assert input_data.max_results == 1

        input_data = SemanticScholarSearchInput(query="test", max_results=100)
        assert input_data.max_results == 100

        # 无效值（小于1）应该抛出异常
        with pytest.raises(Exception):
            SemanticScholarSearchInput(query="test", max_results=0)

        # 无效值（大于100）应该抛出异常
        with pytest.raises(Exception):
            SemanticScholarSearchInput(query="test", max_results=101)


class TestSearchPapersSemanticScholar:
    """测试 Semantic Scholar 搜索功能"""

    @patch('tools.paper_tools.requests.get')
    def test_basic_search(self, mock_get):
        """测试基本搜索"""
        # 模拟 API 响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 100,
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Test Paper",
                    "abstract": "This is a test abstract.",
                    "authors": [{"name": "Author One"}],
                    "year": 2023,
                    "citationCount": 10,
                    "influentialCitationCount": 2,
                    "venue": "Test Conference",
                    "publicationDate": "2023-05-15",
                    "fieldsOfStudy": ["Computer Science"],
                    "url": "https://semanticscholar.org/paper/abc123",
                    "openAccessPdf": None,
                    "externalIds": {"DOI": "10.1234/test"},
                }
            ],
        }
        mock_get.return_value = mock_response

        result = search_papers_semantic_scholar(query="machine learning")

        # 验证结果
        assert "找到 1 篇论文" in result
        assert "Test Paper" in result
        assert "Author One" in result
        assert "10" in result  # 引用次数
        assert mock_get.called

    @patch('tools.paper_tools.requests.get')
    def test_search_with_filters(self, mock_get):
        """测试带筛选条件的搜索"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 10,
            "data": [
                {
                    "paperId": "def456",
                    "title": "Influential Paper",
                    "abstract": "Important research.",
                    "authors": [{"name": "Dr. Smith"}],
                    "year": 2022,
                    "citationCount": 150,
                    "influentialCitationCount": 30,
                    "venue": "Nature",
                    "publicationDate": "2022-03-01",
                    "fieldsOfStudy": ["Computer Science", "AI"],
                    "url": "https://semanticscholar.org/paper/def456",
                    "openAccessPdf": {"url": "https://example.com/paper.pdf"},
                    "externalIds": {"DOI": "10.5678/test", "ArXiv": "2203.12345"},
                }
            ],
        }
        mock_get.return_value = mock_response

        result = search_papers_semantic_scholar(
            query="deep learning",
            year_filter="2020-2023",
            min_citation_count=100,
            fields_of_study="Computer Science",
            sort="citationCount",
        )

        # 验证结果包含期望的信息
        assert "Influential Paper" in result
        assert "Dr. Smith" in result
        assert "150" in result
        assert "开放获取" in result
        assert mock_get.called

        # 验证 API 调用参数
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["query"] == "deep learning"
        assert params["minCitationCount"] == 100
        assert params["fieldsOfStudy"] == "Computer Science"

    @patch('tools.paper_tools.requests.get')
    def test_search_no_results(self, mock_get):
        """测试无结果的搜索"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"total": 0, "data": []}
        mock_get.return_value = mock_response

        result = search_papers_semantic_scholar(query="nonexistent query xyz123")

        assert "未找到匹配的论文" in result
        assert "建议" in result

    @patch('tools.paper_tools.requests.get')
    def test_timeout_error(self, mock_get):
        """测试超时错误"""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        result = search_papers_semantic_scholar(query="test")

        assert "超时" in result
        assert "建议" in result

    @patch('tools.paper_tools.requests.get')
    def test_rate_limit_error(self, mock_get):
        """测试速率限制错误"""
        import requests

        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.side_effect = requests.exceptions.HTTPError(response=mock_response)

        result = search_papers_semantic_scholar(query="test")

        assert "速率限制" in result
        assert "100 requests/5 minutes" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
