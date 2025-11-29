"""论文搜索工具的全方位测试

测试覆盖：
1. 参数验证测试
2. API 调用测试
3. 错误处理测试
4. 数据格式化测试
5. 边界值测试
"""

# 标准库导入
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# 第三方库导入
import pytest
from pydantic import ValidationError
import requests

# 将 src 目录添加到路径
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 项目导入
from tools import paper_tools
from tools.paper_tools import OpenAlexSearchInput

# 获取底层函数（因为 @tool 装饰器会包装成 StructuredTool 对象）
search_papers_openalex = paper_tools.search_papers_openalex.func
search_papers_ArXiv = paper_tools.search_papers_ArXiv.func


# ========== 1. 参数验证测试 ==========

class TestOpenAlexSearchInput:
    """测试 Pydantic 输入模型的参数验证"""

    def test_valid_basic_input(self):
        """测试基本有效输入"""
        input_data = OpenAlexSearchInput(query="machine learning")
        assert input_data.query == "machine learning"
        assert input_data.max_results == 10  # 默认值
        assert input_data.sort_by == "relevance"  # 默认值
        assert input_data.publication_year is None
        assert input_data.open_access_only is False
        assert input_data.cited_by_count_min is None

    def test_valid_all_parameters(self):
        """测试所有参数都设置的情况"""
        input_data = OpenAlexSearchInput(
            query="deep learning",
            max_results=20,
            sort_by="cited_by_count",
            publication_year="2023",
            open_access_only=True,
            cited_by_count_min=50
        )
        assert input_data.query == "deep learning"
        assert input_data.max_results == 20
        assert input_data.sort_by == "cited_by_count"
        assert input_data.publication_year == "2023"
        assert input_data.open_access_only is True
        assert input_data.cited_by_count_min == 50

    def test_max_results_validation(self):
        """测试 max_results 参数的边界验证"""
        # 测试最小值
        input_data = OpenAlexSearchInput(query="test", max_results=1)
        assert input_data.max_results == 1

        # 测试最大值
        input_data = OpenAlexSearchInput(query="test", max_results=200)
        assert input_data.max_results == 200

        # 测试超出范围（应该抛出错误）
        with pytest.raises(ValidationError):
            OpenAlexSearchInput(query="test", max_results=0)

        with pytest.raises(ValidationError):
            OpenAlexSearchInput(query="test", max_results=201)

    def test_sort_by_validation(self):
        """测试 sort_by 参数只接受指定值"""
        valid_values = ["relevance", "publication_date", "cited_by_count"]
        
        for value in valid_values:
            input_data = OpenAlexSearchInput(query="test", sort_by=value)
            assert input_data.sort_by == value

        # 测试无效值
        with pytest.raises(ValidationError):
            OpenAlexSearchInput(query="test", sort_by="invalid_sort")

    def test_publication_year_formats(self):
        """测试 publication_year 支持多种格式"""
        test_cases = [
            "2023",           # 单年
            ">2020",          # 大于
            "<2020",          # 小于
            "2020-2023",      # 范围
        ]
        
        for year_format in test_cases:
            input_data = OpenAlexSearchInput(
                query="test",
                publication_year=year_format
            )
            assert input_data.publication_year == year_format

    def test_required_query_parameter(self):
        """测试 query 参数是必填的"""
        with pytest.raises(ValidationError):
            OpenAlexSearchInput()


# ========== 2. API 调用模拟测试 ==========

class TestSearchPapersOpenAlexAPI:
    """测试 search_papers_openalex 函数的 API 调用"""

    @pytest.fixture
    def mock_success_response(self):
        """模拟成功的 API 响应"""
        return {
            "meta": {
                "count": 100,
                "db_response_time_ms": 42,
                "page": 1,
                "per_page": 10
            },
            "results": [
                {
                    "id": "https://openalex.org/W123456789",
                    "title": "Deep Learning for Medical Image Analysis",
                    "display_name": "Deep Learning for Medical Image Analysis",
                    "doi": "https://doi.org/10.1234/example",
                    "publication_year": 2023,
                    "publication_date": "2023-05-15",
                    "cited_by_count": 150,
                    "type": "journal-article",
                    "authorships": [
                        {
                            "author": {
                                "display_name": "张三"
                            }
                        },
                        {
                            "author": {
                                "display_name": "李四"
                            }
                        }
                    ],
                    "primary_location": {
                        "source": {
                            "display_name": "Nature Medicine"
                        }
                    },
                    "open_access": {
                        "is_oa": True,
                        "oa_status": "gold",
                        "oa_url": "https://example.com/paper.pdf"
                    },
                    "topics": [
                        {"display_name": "Machine Learning"},
                        {"display_name": "Medical Imaging"}
                    ]
                }
            ]
        }

    @patch('tools.paper_tools.requests.get')
    def test_basic_search(self, mock_get, mock_success_response):
        """测试基本搜索功能"""
        # 设置 mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_success_response
        mock_get.return_value = mock_response

        # 调用函数
        result = search_papers_openalex(query="deep learning")

        # 验证
        assert "📚 找到 1 篇论文" in result
        assert "Deep Learning for Medical Image Analysis" in result
        assert "张三" in result
        assert "Nature Medicine" in result
        assert "150" in result  # 引用次数
        assert "金色开放获取" in result

        # 验证 API 调用参数
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://api.openalex.org/works"
        assert call_args[1]["params"]["search"] == "deep learning"

    @patch('tools.paper_tools.requests.get')
    def test_search_with_all_filters(self, mock_get, mock_success_response):
        """测试使用所有筛选参数的搜索"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_success_response
        mock_get.return_value = mock_response

        result = search_papers_openalex(
            query="CRISPR",
            max_results=20,
            sort_by="cited_by_count",
            publication_year="2023",
            open_access_only=True,
            cited_by_count_min=50
        )

        # 验证输出包含筛选信息
        assert "年份: 2023" in result
        assert "仅开放获取" in result
        assert "引用≥50" in result

        # 验证 API 调用参数
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["search"] == "CRISPR"
        assert params["per_page"] == 20
        assert params["sort"] == "cited_by_count:desc"
        assert "publication_year:2023" in params["filter"]
        assert "is_oa:true" in params["filter"]
        assert "cited_by_count:>49" in params["filter"]

    @patch('tools.paper_tools.requests.get')
    def test_search_no_results(self, mock_get):
        """测试搜索无结果的情况"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"count": 0},
            "results": []
        }
        mock_get.return_value = mock_response

        result = search_papers_openalex(query="nonexistent_topic_xyz123")

        assert "❌ 未找到" in result
        assert "建议：尝试更通用的关键词" in result

    @patch('tools.paper_tools.requests.get')
    def test_multiple_authors_display(self, mock_get):
        """测试多作者显示逻辑"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"count": 1},
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test Paper",
                    "display_name": "Test Paper",
                    "cited_by_count": 10,
                    "authorships": [
                        {"author": {"display_name": f"Author {i}"}}
                        for i in range(1, 6)  # 5 位作者
                    ],
                    "open_access": {"is_oa": False},
                    "topics": []
                }
            ]
        }
        mock_get.return_value = mock_response

        result = search_papers_openalex(query="test")

        # 应该只显示前 3 位作者，然后显示总数
        assert "Author 1" in result
        assert "Author 2" in result
        assert "Author 3" in result
        assert "等 (5 位作者)" in result
        assert "Author 4" not in result
        assert "Author 5" not in result


# ========== 3. 错误处理测试 ==========

class TestErrorHandling:
    """测试各种错误情况的处理"""

    @patch('tools.paper_tools.requests.get')
    def test_timeout_error(self, mock_get):
        """测试请求超时错误"""
        mock_get.side_effect = requests.exceptions.Timeout()

        result = search_papers_openalex(query="test")

        assert "⏱️" in result
        assert "超时" in result
        assert "建议" in result

    @patch('tools.paper_tools.requests.get')
    def test_rate_limit_error(self, mock_get):
        """测试速率限制错误（403）"""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response
        mock_get.return_value.raise_for_status.side_effect = \
            requests.exceptions.HTTPError(response=mock_response)

        result = search_papers_openalex(query="test")

        assert "🚫" in result
        assert "速率限制" in result

    @patch('tools.paper_tools.requests.get')
    def test_not_found_error(self, mock_get):
        """测试 404 错误"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        mock_get.return_value.raise_for_status.side_effect = \
            requests.exceptions.HTTPError(response=mock_response)

        result = search_papers_openalex(query="test")

        assert "❌" in result
        assert "未找到资源" in result

    @patch('tools.paper_tools.requests.get')
    def test_server_error(self, mock_get):
        """测试服务器错误（500+）"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        mock_get.return_value.raise_for_status.side_effect = \
            requests.exceptions.HTTPError(response=mock_response)

        result = search_papers_openalex(query="test")

        assert "⚠️" in result
        assert "服务器错误" in result
        assert "500" in result

    @patch('tools.paper_tools.requests.get')
    def test_network_error(self, mock_get):
        """测试网络连接错误"""
        mock_get.side_effect = requests.exceptions.ConnectionError(
            "Network unreachable"
        )

        result = search_papers_openalex(query="test")

        assert "🌐" in result
        assert "网络连接错误" in result

    @patch('tools.paper_tools.requests.get')
    def test_unexpected_error(self, mock_get):
        """测试未预期的错误"""
        mock_get.side_effect = Exception("Unexpected error")

        result = search_papers_openalex(query="test")

        assert "❌" in result
        assert "未知错误" in result


# ========== 4. 数据格式化测试 ==========

class TestOutputFormatting:
    """测试输出格式化"""

    @patch('tools.paper_tools.requests.get')
    def test_doi_formatting(self, mock_get):
        """测试 DOI 链接格式化"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"count": 1},
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test",
                    "doi": "https://doi.org/10.1234/test",
                    "cited_by_count": 0,
                    "authorships": [],
                    "open_access": {"is_oa": False},
                    "topics": []
                }
            ]
        }
        mock_get.return_value = mock_response

        result = search_papers_openalex(query="test")

        # DOI 应该被清理（移除 https://doi.org/ 前缀）
        assert "10.1234/test" in result
        assert "[10.1234/test](https://doi.org/10.1234/test)" in result

    @patch('tools.paper_tools.requests.get')
    def test_open_access_status_display(self, mock_get):
        """测试开放获取状态显示"""
        test_cases = [
            ("gold", "金色开放获取"),
            ("green", "绿色开放获取"),
            ("hybrid", "混合开放获取"),
            ("bronze", "铜色开放获取"),
        ]

        for oa_status, expected_text in test_cases:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "meta": {"count": 1},
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "title": "Test",
                        "cited_by_count": 0,
                        "authorships": [],
                        "open_access": {
                            "is_oa": True,
                            "oa_status": oa_status,
                            "oa_url": "https://example.com/paper.pdf"
                        },
                        "topics": []
                    }
                ]
            }
            mock_get.return_value = mock_response

            result = search_papers_openalex(query="test")
            assert expected_text in result
            assert "🔓" in result

    @patch('tools.paper_tools.requests.get')
    def test_citation_count_formatting(self, mock_get):
        """测试引用次数的千分位格式化"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"count": 1},
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Highly Cited Paper",
                    "cited_by_count": 123456,  # 大数字
                    "authorships": [],
                    "open_access": {"is_oa": False},
                    "topics": []
                }
            ]
        }
        mock_get.return_value = mock_response

        result = search_papers_openalex(query="test")

        # 应该包含千分位分隔符
        assert "123,456" in result

    @patch('tools.paper_tools.requests.get')
    def test_handle_none_values_in_api_response(self, mock_get):
        """测试处理 API 返回 None 值的情况（防御性编程）"""
        # 模拟 API 返回包含 None 值的响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "meta": {"count": 1},
            "results": [
                {
                    "id": "https://openalex.org/W123",
                    "title": "Test Paper",
                    "cited_by_count": 10,
                    "authorships": [],
                    "primary_location": None,  # None 值
                    "open_access": None,  # None 值
                    "topics": None  # None 值
                }
            ]
        }
        mock_get.return_value = mock_response

        # 应该不抛出异常，正常处理
        result = search_papers_openalex(query="test")

        assert "Test Paper" in result
        assert "未知作者" in result
        # 应该包含默认的访问状态
        assert "🔒" in result or "访问" in result


# ========== 5. 集成测试（需要真实 API）==========

class TestRealAPIIntegration:
    """真实 API 集成测试（需要网络连接）
    
    注意：这些测试会调用真实的 OpenAlex API
    默认会跳过，使用以下方式运行：
    - pytest -m integration  # 只运行集成测试
    - pytest --run-integration  # 运行所有测试包括集成测试
    """

    @pytest.mark.integration
    @pytest.mark.skip(reason="集成测试默认跳过，使用 --run-integration 或 -m integration 运行")
    def test_real_search_basic(self):
        """测试真实的基本搜索"""
        result = search_papers_openalex(
            query="machine learning",
            max_results=3
        )
        
        assert "📚 找到" in result
        assert "篇论文" in result
        # 应该至少有一些结果
        assert "未找到" not in result

    @pytest.mark.integration
    @pytest.mark.skip(reason="集成测试默认跳过，使用 --run-integration 或 -m integration 运行")
    def test_real_search_with_filters(self):
        """测试真实的带筛选条件的搜索"""
        result = search_papers_openalex(
            query="deep learning",
            max_results=5,
            publication_year="2023",
            open_access_only=True,
            sort_by="cited_by_count"
        )
        
        assert "📚 找到" in result
        assert "年份: 2023" in result
        assert "仅开放获取" in result


# ========== 6. ArXiv 工具测试 ==========

class TestSearchPapersArXiv:
    """测试 ArXiv 搜索工具"""

    @patch('tools.paper_tools.ArxivRetriever')
    def test_arxiv_basic_search(self, mock_retriever_class):
        """测试 ArXiv 基本搜索"""
        # 模拟 ArxivRetriever 返回
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.metadata = {
            'Title': 'Test ArXiv Paper',
            'Authors': 'John Doe, Jane Smith',
            'Published': '2023-01-15',
            'Entry ID': 'http://arxiv.org/abs/2301.12345v1'
        }
        mock_doc.page_content = "This is a test abstract."
        mock_retriever.invoke.return_value = [mock_doc]
        mock_retriever_class.return_value = mock_retriever

        result = search_papers_ArXiv(query="quantum computing", max_results=5)

        assert "Test ArXiv Paper" in result
        assert "John Doe" in result
        assert "2301.12345" in result
        assert "https://arxiv.org/abs/2301.12345" in result

    @patch('tools.paper_tools.ArxivRetriever')
    def test_arxiv_no_results(self, mock_retriever_class):
        """测试 ArXiv 无结果情况"""
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_retriever_class.return_value = mock_retriever

        result = search_papers_ArXiv(query="xyz123nonexistent")

        assert "未找到" in result


# ========== 配置 pytest ==========

if __name__ == "__main__":
    # 允许直接运行测试文件
    pytest.main([__file__, "-v", "--tb=short"])
