"""Tavily 工具的全方位测试

测试覆盖：
1. 参数验证测试
2. API 调用测试
3. 错误处理测试
4. 数据格式化测试
5. 边界值测试
"""

# 标准库导入
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# 第三方库导入
import pytest
from pydantic import ValidationError

# 将 src 目录添加到路径
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# 项目导入
from tools import tavily_tools
from tools.tavily_tools import (
    TavilySearchInput,
    TavilyExtractInput,
    TavilyCrawlInput,
    TavilyMapInput,
)

# 获取底层函数（因为 @tool 装饰器会包装成 StructuredTool 对象）
tavily_search = tavily_tools.tavily_search.func
tavily_extract = tavily_tools.tavily_extract.func
tavily_crawl = tavily_tools.tavily_crawl.func
tavily_map = tavily_tools.tavily_map.func


# ========== 1. 参数验证测试 ==========


class TestTavilySearchInput:
    """测试 Tavily Search 输入模型的参数验证"""

    def test_valid_basic_input(self):
        """测试基本有效输入"""
        input_data = TavilySearchInput(query="machine learning")
        assert input_data.query == "machine learning"
        assert input_data.max_results == 5  # 默认值
        assert input_data.search_depth == "basic"  # 默认值
        assert input_data.topic == "general"  # 默认值
        assert input_data.time_range is None
        assert input_data.include_answer is False
        assert input_data.include_raw_content is False
        assert input_data.include_images is False

    def test_valid_all_parameters(self):
        """测试所有参数都设置的情况"""
        input_data = TavilySearchInput(
            query="AI research",
            max_results=10,
            search_depth="advanced",
            topic="general",
            time_range="week",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            include_domains=["arxiv.org"],
            exclude_domains=["spam.com"],
        )
        assert input_data.query == "AI research"
        assert input_data.max_results == 10
        assert input_data.search_depth == "advanced"
        assert input_data.include_answer is True

    def test_max_results_validation(self):
        """测试 max_results 参数的边界验证"""
        # 测试最小值
        input_data = TavilySearchInput(query="test", max_results=1)
        assert input_data.max_results == 1

        # 测试最大值
        input_data = TavilySearchInput(query="test", max_results=20)
        assert input_data.max_results == 20

        # 测试超出范围（应该抛出错误）
        with pytest.raises(ValidationError):
            TavilySearchInput(query="test", max_results=0)

        with pytest.raises(ValidationError):
            TavilySearchInput(query="test", max_results=21)

    def test_search_depth_validation(self):
        """测试 search_depth 参数只接受指定值"""
        valid_values = ["basic", "advanced"]

        for value in valid_values:
            input_data = TavilySearchInput(query="test", search_depth=value)
            assert input_data.search_depth == value

        # 测试无效值
        with pytest.raises(ValidationError):
            TavilySearchInput(query="test", search_depth="invalid")

    def test_topic_validation(self):
        """测试 topic 参数只接受指定值"""
        valid_values = ["general", "news", "finance"]

        for value in valid_values:
            input_data = TavilySearchInput(query="test", topic=value)
            assert input_data.topic == value

        # 测试无效值
        with pytest.raises(ValidationError):
            TavilySearchInput(query="test", topic="invalid")


class TestTavilyExtractInput:
    """测试 Tavily Extract 输入模型的参数验证"""

    def test_valid_single_url(self):
        """测试单个 URL 输入"""
        input_data = TavilyExtractInput(urls="https://example.com")
        assert input_data.urls == "https://example.com"
        assert input_data.include_images is False
        assert input_data.extract_depth == "advanced"
        assert input_data.format == "markdown"

    def test_valid_multiple_urls(self):
        """测试多个 URL 输入"""
        urls = ["https://example1.com", "https://example2.com"]
        input_data = TavilyExtractInput(urls=urls)
        assert input_data.urls == urls

    def test_extract_depth_validation(self):
        """测试 extract_depth 参数只接受指定值"""
        valid_values = ["basic", "advanced"]

        for value in valid_values:
            input_data = TavilyExtractInput(
                urls="https://example.com", extract_depth=value
            )
            assert input_data.extract_depth == value

        # 测试无效值
        with pytest.raises(ValidationError):
            TavilyExtractInput(urls="https://example.com", extract_depth="invalid")

    def test_format_validation(self):
        """测试 format 参数只接受指定值"""
        valid_values = ["markdown", "text"]

        for value in valid_values:
            input_data = TavilyExtractInput(urls="https://example.com", format=value)
            assert input_data.format == value

        # 测试无效值
        with pytest.raises(ValidationError):
            TavilyExtractInput(urls="https://example.com", format="invalid")


class TestTavilyCrawlInput:
    """测试 Tavily Crawl 输入模型的参数验证"""

    def test_valid_basic_input(self):
        """测试基本有效输入"""
        input_data = TavilyCrawlInput(url="https://example.com")
        assert input_data.url == "https://example.com"
        assert input_data.max_depth == 1
        assert input_data.max_breadth == 20
        assert input_data.limit == 50
        assert input_data.allow_external is True

    def test_depth_validation(self):
        """测试 max_depth 参数的边界验证"""
        # 测试最小值
        input_data = TavilyCrawlInput(url="https://example.com", max_depth=1)
        assert input_data.max_depth == 1

        # 测试超出范围（应该抛出错误）
        with pytest.raises(ValidationError):
            TavilyCrawlInput(url="https://example.com", max_depth=0)


class TestTavilyMapInput:
    """测试 Tavily Map 输入模型的参数验证"""

    def test_valid_basic_input(self):
        """测试基本有效输入"""
        input_data = TavilyMapInput(url="https://example.com")
        assert input_data.url == "https://example.com"
        assert input_data.max_depth == 1
        assert input_data.max_breadth == 20
        assert input_data.limit == 50
        assert input_data.allow_external is True


# ========== 2. API 调用测试（使用 Mock） ==========


class TestTavilySearchAPI:
    """测试 Tavily Search API 调用"""

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_successful_search(self, mock_client_class):
        """测试成功的搜索调用"""
        # 模拟 API 响应
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Test Article",
                    "url": "https://example.com/article",
                    "content": "This is a test article about AI.",
                    "score": 0.95,
                }
            ],
            "query": "AI research",
            "response_time": 1.23,
        }

        # 调用函数
        result = tavily_search(query="AI research", max_results=5)

        # 验证
        assert "找到 1 条相关结果" in result
        assert "Test Article" in result
        assert "https://example.com/article" in result
        mock_client.search.assert_called_once()

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_search_with_answer(self, mock_client_class):
        """测试包含 AI 答案的搜索"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Test",
                    "url": "https://example.com",
                    "content": "Test content",
                    "score": 0.9,
                }
            ],
            "query": "test query",
            "answer": "This is the AI generated answer.",
            "response_time": 1.0,
        }

        result = tavily_search(query="test query", include_answer=True)

        assert "AI 答案" in result
        assert "This is the AI generated answer" in result

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        """测试缺少 API key 的情况"""
        result = tavily_search(query="test")
        assert "未设置 TAVILY_API_KEY 环境变量" in result

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_no_results(self, mock_client_class):
        """测试没有结果的情况"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search.return_value = {"results": [], "query": "test"}

        result = tavily_search(query="test")
        assert "未找到与 'test' 相关的结果" in result


class TestTavilyExtractAPI:
    """测试 Tavily Extract API 调用"""

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_successful_extract(self, mock_client_class):
        """测试成功的提取调用"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.extract.return_value = {
            "results": [
                {
                    "url": "https://example.com",
                    "raw_content": "This is the extracted content from the webpage.",
                }
            ],
            "failed_results": [],
            "response_time": 1.5,
        }

        result = tavily_extract(urls="https://example.com")

        assert "成功提取 1 个 URL" in result
        assert "https://example.com" in result
        assert "extracted content" in result
        mock_client.extract.assert_called_once()

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_extract_with_failures(self, mock_client_class):
        """测试部分失败的提取"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.extract.return_value = {
            "results": [
                {
                    "url": "https://example1.com",
                    "raw_content": "Content 1",
                }
            ],
            "failed_results": [
                {
                    "url": "https://example2.com",
                    "error": "Connection timeout",
                }
            ],
            "response_time": 2.0,
        }

        result = tavily_extract(urls=["https://example1.com", "https://example2.com"])

        assert "成功提取 1 个 URL" in result
        assert "提取失败 1 个 URL" in result
        assert "Connection timeout" in result

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_extract_multiple_urls(self, mock_client_class):
        """测试提取多个 URL"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        urls = ["https://example1.com", "https://example2.com"]
        tavily_extract(urls=urls)

        # 验证传入的 URLs 是列表
        call_args = mock_client.extract.call_args
        assert call_args[1]["urls"] == urls


class TestTavilyCrawlAPI:
    """测试 Tavily Crawl API 调用"""

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_successful_crawl(self, mock_client_class):
        """测试成功的爬取调用"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.crawl.return_value = {
            "base_url": "https://example.com",
            "results": [
                {
                    "url": "https://example.com/page1",
                    "raw_content": "Page 1 content",
                },
                {
                    "url": "https://example.com/page2",
                    "raw_content": "Page 2 content",
                },
            ],
            "response_time": 5.0,
        }

        result = tavily_crawl(url="https://example.com", max_depth=2)

        assert "共爬取 2 个页面" in result
        assert "https://example.com/page1" in result
        assert "https://example.com/page2" in result
        mock_client.crawl.assert_called_once()

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_crawl_with_instructions(self, mock_client_class):
        """测试带指令的爬取"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.crawl.return_value = {
            "base_url": "https://docs.example.com",
            "results": [
                {
                    "url": "https://docs.example.com/api",
                    "raw_content": "API documentation",
                }
            ],
            "response_time": 3.0,
        }

        tavily_crawl(url="https://docs.example.com", instructions="Find API docs")

        # 验证 instructions 被传递
        call_args = mock_client.crawl.call_args
        assert call_args[1]["instructions"] == "Find API docs"


class TestTavilyMapAPI:
    """测试 Tavily Map API 调用"""

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_successful_map(self, mock_client_class):
        """测试成功的映射调用"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.map.return_value = {
            "base_url": "https://example.com",
            "results": [
                "https://example.com/page1",
                "https://example.com/page2",
                "https://example.com/page3",
            ],
            "response_time": 4.0,
        }

        result = tavily_map(url="https://example.com", max_depth=2)

        assert "共发现 3 个页面" in result
        assert "https://example.com/page1" in result
        assert "https://example.com/page2" in result
        assert "https://example.com/page3" in result
        mock_client.map.assert_called_once()

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_map_with_filters(self, mock_client_class):
        """测试带过滤器的映射"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.map.return_value = {
            "base_url": "https://example.com",
            "results": ["https://example.com/docs/api"],
            "response_time": 2.0,
        }

        tavily_map(
            url="https://example.com",
            select_paths=["/docs/.*"],
            exclude_paths=["/private/.*"],
        )

        # 验证过滤器被传递
        call_args = mock_client.map.call_args
        assert call_args[1]["select_paths"] == ["/docs/.*"]
        assert call_args[1]["exclude_paths"] == ["/private/.*"]


# ========== 3. 错误处理测试 ==========


class TestErrorHandling:
    """测试错误处理"""

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_search_api_error(self, mock_client_class):
        """测试 API 错误处理"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search.side_effect = Exception("API Error")

        result = tavily_search(query="test")
        assert "Tavily 搜索时发生错误" in result
        assert "API Error" in result

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_extract_api_error(self, mock_client_class):
        """测试提取 API 错误处理"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.extract.side_effect = Exception("Extract Error")

        result = tavily_extract(urls="https://example.com")
        assert "Tavily 内容提取时发生错误" in result
        assert "Extract Error" in result


# ========== 4. 集成测试（需要真实 API key，标记为 skip） ==========


@pytest.mark.skip(reason="需要真实的 TAVILY_API_KEY 才能运行")
class TestRealAPIIntegration:
    """集成测试（需要真实 API）"""

    def test_real_search(self):
        """测试真实的搜索请求"""
        result = tavily_search(
            query="Python programming", max_results=3, search_depth="basic"
        )
        assert "找到" in result or "未找到" in result

    def test_real_extract(self):
        """测试真实的提取请求"""
        result = tavily_extract(urls="https://www.python.org")
        assert "提取完成" in result or "错误" in result


# ========== 5. 参数组合测试 ==========


class TestParameterCombinations:
    """测试参数组合"""

    @patch("tools.tavily_tools.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
    def test_search_all_parameters(self, mock_client_class):
        """测试所有参数的组合"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.search.return_value = {
            "results": [],
            "query": "test",
        }

        tavily_search(
            query="test",
            max_results=10,
            search_depth="advanced",
            topic="news",
            time_range="week",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
            include_domains=["example.com"],
            exclude_domains=["spam.com"],
        )

        # 验证所有参数都被传递
        call_args = mock_client.search.call_args
        assert call_args[1]["query"] == "test"
        assert call_args[1]["max_results"] == 10
        assert call_args[1]["search_depth"] == "advanced"
        assert call_args[1]["topic"] == "news"
        assert call_args[1]["time_range"] == "week"
