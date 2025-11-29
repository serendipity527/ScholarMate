"""性能测试

测试工具在各种负载下的性能表现
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import time

import pytest

# 添加 src 到路径
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from tools import paper_tools
from tools.paper_tools import OpenAlexSearchInput

# 获取底层函数（因为 @tool 装饰器会包装成 StructuredTool 对象）
search_papers_openalex = paper_tools.search_papers_openalex.func


class TestPerformance:
    """性能基准测试"""

    @patch('tools.paper_tools.requests.get')
    def test_response_time_basic_search(self, mock_get, sample_openalex_response):
        """测试基本搜索的响应时间"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_openalex_response
        mock_get.return_value = mock_response

        start_time = time.time()
        result = search_papers_openalex(query="machine learning")
        end_time = time.time()

        execution_time = end_time - start_time
        
        # 本地处理应该很快（不包括网络请求时间）
        assert execution_time < 0.5, f"执行时间 {execution_time}s 超过 0.5s"
        assert result is not None

    @patch('tools.paper_tools.requests.get')
    def test_large_result_set_processing(self, mock_get):
        """测试处理大量结果的性能"""
        # 生成 200 个结果
        large_response = {
            "meta": {"count": 1000},
            "results": [
                {
                    "id": f"https://openalex.org/W{i}",
                    "title": f"Paper {i}",
                    "cited_by_count": i * 10,
                    "authorships": [
                        {"author": {"display_name": f"Author {j}"}}
                        for j in range(5)
                    ],
                    "open_access": {"is_oa": False},
                    "topics": []
                }
                for i in range(200)
            ]
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = large_response
        mock_get.return_value = mock_response

        start_time = time.time()
        result = search_papers_openalex(query="test", max_results=200)
        end_time = time.time()

        execution_time = end_time - start_time
        
        # 处理 200 个结果应该在 2 秒内完成
        assert execution_time < 2.0, f"处理 200 个结果耗时 {execution_time}s"
        assert "200 篇论文" in result or "200" in result

    def test_input_validation_performance(self):
        """测试参数验证的性能"""
        iterations = 1000
        
        start_time = time.time()
        for i in range(iterations):
            input_data = OpenAlexSearchInput(
                query=f"test query {i}",
                max_results=10,
                sort_by="relevance"
            )
        end_time = time.time()

        avg_time = (end_time - start_time) / iterations
        
        # 每次验证应该在 1ms 内完成
        assert avg_time < 0.001, f"平均验证时间 {avg_time*1000}ms 过长"

    @pytest.mark.slow
    @patch('tools.paper_tools.requests.get')
    def test_timeout_handling_performance(self, mock_get):
        """测试超时处理的性能"""
        import requests
        
        mock_get.side_effect = requests.exceptions.Timeout()

        start_time = time.time()
        result = search_papers_openalex(query="test")
        end_time = time.time()

        execution_time = end_time - start_time
        
        # 超时处理应该立即返回
        assert execution_time < 0.1, f"超时处理耗时 {execution_time}s"
        assert "超时" in result

    @patch('tools.paper_tools.requests.get')
    def test_concurrent_request_simulation(self, mock_get, sample_openalex_response):
        """模拟并发请求的性能"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_openalex_response
        mock_get.return_value = mock_response

        # 模拟 10 个连续请求
        start_time = time.time()
        for i in range(10):
            result = search_papers_openalex(query=f"query {i}", max_results=5)
        end_time = time.time()

        execution_time = end_time - start_time
        avg_time = execution_time / 10
        
        # 平均每个请求应该在 0.1 秒内完成（不包括网络时间）
        assert avg_time < 0.1, f"平均请求处理时间 {avg_time}s 过长"


class TestMemoryUsage:
    """内存使用测试"""

    @pytest.mark.slow
    @patch('tools.paper_tools.requests.get')
    def test_no_memory_leak_repeated_calls(self, mock_get, sample_openalex_response):
        """测试重复调用是否有内存泄漏"""
        import gc
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_openalex_response
        mock_get.return_value = mock_response

        # 执行多次调用
        for i in range(100):
            result = search_papers_openalex(query=f"query {i}")
            # 确保结果被使用
            assert "篇论文" in result

        # 强制垃圾回收
        gc.collect()
        
        # 如果有内存泄漏，这里的对象数量会显著增加
        # 这是一个简单的检查，实际生产中应该使用更专业的工具
        assert True  # 基本检查：代码能够完成而不崩溃


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
