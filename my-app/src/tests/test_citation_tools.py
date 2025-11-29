"""引用网络分析工具测试"""

import sys
from pathlib import Path

# 将 src 目录添加到路径
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from unittest.mock import patch, MagicMock
from tools.citation_tools import (
    identify_paper,
    extract_arxiv_id,
    extract_doi,
    get_references,
    get_citations,
    calculate_reference_score,
    calculate_citation_score,
    rank_papers,
    format_paper_info,
    analyze_citation_network,
)


class TestPaperIdentification:
    """测试论文识别功能"""

    def test_extract_arxiv_id(self):
        """测试提取 ArXiv ID"""
        assert extract_arxiv_id("1706.03762") == "1706.03762"
        assert extract_arxiv_id("arXiv:1706.03762") == "1706.03762"
        assert extract_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
        assert extract_arxiv_id("no arxiv id here") is None

    def test_extract_doi(self):
        """测试提取 DOI"""
        assert extract_doi("10.48550/arXiv.1706.03762") == "10.48550/arXiv.1706.03762"
        assert extract_doi("DOI: 10.1234/test.5678") == "10.1234/test.5678"
        assert extract_doi("no doi here") is None

    @patch("tools.citation_tools.requests.get")
    def test_identify_paper_by_arxiv(self, mock_get):
        """测试通过 ArXiv ID 识别论文"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "paperId": "test_id",
            "title": "Test Paper",
            "year": 2017,
        }
        mock_get.return_value = mock_response

        result = identify_paper("1706.03762")
        assert result is not None
        assert result["title"] == "Test Paper"

    @patch("tools.citation_tools.requests.get")
    def test_identify_paper_not_found(self, mock_get):
        """测试未找到论文"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = identify_paper("nonexistent")
        assert result is None


class TestCitationDataFetching:
    """测试引用数据获取功能"""

    @patch("tools.citation_tools.requests.get")
    def test_get_references(self, mock_get):
        """测试获取参考文献"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"citedPaper": {"paperId": "ref1", "title": "Reference 1"}},
                {"citedPaper": {"paperId": "ref2", "title": "Reference 2"}},
            ]
        }
        mock_get.return_value = mock_response

        references = get_references("test_paper_id", limit=10)
        assert len(references) == 2
        assert references[0]["title"] == "Reference 1"

    @patch("tools.citation_tools.requests.get")
    def test_get_citations(self, mock_get):
        """测试获取引用论文"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"citingPaper": {"paperId": "cite1", "title": "Citation 1"}},
                {"citingPaper": {"paperId": "cite2", "title": "Citation 2"}},
            ]
        }
        mock_get.return_value = mock_response

        citations = get_citations("test_paper_id", limit=10)
        assert len(citations) == 2
        assert citations[0]["title"] == "Citation 1"

    @patch("tools.citation_tools.requests.get")
    def test_get_references_empty(self, mock_get):
        """测试获取空的参考文献列表"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response

        references = get_references("test_paper_id")
        assert len(references) == 0

    @patch("tools.citation_tools.requests.get")
    def test_get_citations_error(self, mock_get):
        """测试获取引用论文时出错"""
        mock_get.side_effect = Exception("Network error")

        citations = get_citations("test_paper_id")
        assert len(citations) == 0


class TestImportanceScoring:
    """测试重要性评分功能"""

    def test_calculate_reference_score_high(self):
        """测试高分参考文献评分"""
        paper = {
            "citationCount": 5000,  # 高被引
            "influentialCitationCount": 500,  # 高影响力
            "year": 2005,  # 早期经典
            "venue": "CVPR",  # 顶会
        }
        score = calculate_reference_score(paper)
        assert score > 80  # 应该得到高分

    def test_calculate_reference_score_low(self):
        """测试低分参考文献评分"""
        paper = {
            "citationCount": 10,
            "influentialCitationCount": 1,
            "year": 2023,
            "venue": "Unknown Conference",
        }
        score = calculate_reference_score(paper)
        assert score < 30  # 应该得到低分

    def test_calculate_citation_score_sota(self):
        """测试 SOTA 论文评分（近期+高质量）"""
        paper = {
            "citationCount": 100,
            "year": 2024,  # 最新
            "venue": "ICLR",  # 顶会
            "isOpenAccess": True,  # 开放获取
        }
        score = calculate_citation_score(paper)
        assert score > 90  # 应该得到高分

    def test_calculate_citation_score_old(self):
        """测试旧论文评分（不是 SOTA）"""
        paper = {
            "citationCount": 50,
            "year": 2015,  # 较旧
            "venue": "Workshop",
            "isOpenAccess": False,
        }
        score = calculate_citation_score(paper)
        assert score < 50  # 不应该得到高分

    def test_rank_papers_reference_mode(self):
        """测试参考文献排序"""
        papers = [
            {"citationCount": 100, "influentialCitationCount": 10, "year": 2020, "venue": "CVPR"},
            {"citationCount": 5000, "influentialCitationCount": 500, "year": 2010, "venue": "CVPR"},
            {"citationCount": 50, "influentialCitationCount": 5, "year": 2022, "venue": "Workshop"},
        ]
        
        ranked = rank_papers(papers, mode="reference")
        
        # 检查是否按分数降序排序
        assert ranked[0]["importance_score"] >= ranked[1]["importance_score"]
        assert ranked[1]["importance_score"] >= ranked[2]["importance_score"]
        
        # 第二篇（高被引经典）应该排最前
        assert ranked[0]["citationCount"] == 5000

    def test_rank_papers_citation_mode(self):
        """测试引用论文排序"""
        papers = [
            {"citationCount": 50, "year": 2020, "venue": "Workshop", "isOpenAccess": False},
            {"citationCount": 100, "year": 2024, "venue": "ICLR", "isOpenAccess": True},
            {"citationCount": 200, "year": 2015, "venue": "CVPR", "isOpenAccess": False},
        ]
        
        ranked = rank_papers(papers, mode="citation")
        
        # 第二篇（2024年+顶会+OA）应该排最前
        assert ranked[0]["year"] == 2024


class TestFormatting:
    """测试格式化输出功能"""

    def test_format_paper_info(self):
        """测试格式化单篇论文信息"""
        paper = {
            "title": "Test Paper",
            "authors": [
                {"name": "Author A"},
                {"name": "Author B"},
            ],
            "year": 2023,
            "citationCount": 100,
            "venue": "ICLR",
            "url": "https://example.com",
            "importance_score": 85.5,
            "isOpenAccess": True,
        }
        
        result = format_paper_info(paper, index=1)
        
        assert "Test Paper" in result
        assert "Author A" in result
        assert "2023" in result
        assert "100" in result
        assert "ICLR" in result
        assert "85.5" in result
        assert "🔓" in result  # 开放获取标记

    def test_format_paper_info_many_authors(self):
        """测试格式化多作者论文"""
        paper = {
            "title": "Test Paper",
            "authors": [{"name": f"Author {i}"} for i in range(10)],
            "year": 2023,
            "citationCount": 50,
            "venue": "Conference",
            "url": "https://example.com",
            "importance_score": 75.0,
            "isOpenAccess": False,
        }
        
        result = format_paper_info(paper)
        
        assert "10 位作者" in result
        assert "Author 0" in result
        assert "🔓" not in result  # 不是开放获取


class TestAnalyzeCitationNetwork:
    """测试完整的引用网络分析工具"""

    @patch("tools.citation_tools.identify_paper")
    @patch("tools.citation_tools.get_references")
    @patch("tools.citation_tools.get_citations")
    def test_analyze_citation_network_success(
        self, mock_citations, mock_references, mock_identify
    ):
        """测试成功分析引用网络"""
        # Mock 识别论文
        mock_identify.return_value = {
            "paperId": "test_id",
            "title": "Attention is All You Need",
            "authors": [{"name": "Vaswani"}],
            "year": 2017,
            "citationCount": 50000,
            "influentialCitationCount": 5000,
            "venue": "NeurIPS",
            "url": "https://example.com",
        }
        
        # Mock 参考文献
        mock_references.return_value = [
            {
                "paperId": "ref1",
                "title": "Neural Machine Translation",
                "authors": [{"name": "Bahdanau"}],
                "year": 2014,
                "citationCount": 10000,
                "influentialCitationCount": 1000,
                "venue": "ICLR",
                "url": "https://example.com/ref1",
                "isOpenAccess": True,
            }
        ]
        
        # Mock 引用论文
        mock_citations.return_value = [
            {
                "paperId": "cite1",
                "title": "BERT",
                "authors": [{"name": "Devlin"}],
                "year": 2018,
                "citationCount": 30000,
                "influentialCitationCount": 3000,
                "venue": "NAACL",
                "url": "https://example.com/cite1",
                "isOpenAccess": True,
            }
        ]
        
        result = analyze_citation_network.invoke({
            "paper_identifier": "Attention is All You Need",
            "max_references": 10,
            "max_citations": 10
        })
        
        assert "📊 引用网络分析" in result
        assert "Attention is All You Need" in result
        assert "前世：重要参考文献" in result
        assert "今生：引用它的 SOTA 论文" in result
        assert "Neural Machine Translation" in result
        assert "BERT" in result

    @patch("tools.citation_tools.identify_paper")
    def test_analyze_citation_network_not_found(self, mock_identify):
        """测试论文未找到的情况"""
        mock_identify.return_value = None
        
        result = analyze_citation_network.invoke({"paper_identifier": "Nonexistent Paper"})
        
        assert "❌ 未找到论文" in result

    @patch("tools.citation_tools.identify_paper")
    @patch("tools.citation_tools.get_references")
    @patch("tools.citation_tools.get_citations")
    def test_analyze_citation_network_no_data(
        self, mock_citations, mock_references, mock_identify
    ):
        """测试无引用数据的情况"""
        mock_identify.return_value = {
            "paperId": "test_id",
            "title": "Very New Paper",
            "authors": [{"name": "Author"}],
            "year": 2024,
            "citationCount": 0,
            "venue": "arXiv",
            "url": "https://example.com",
        }
        mock_references.return_value = []
        mock_citations.return_value = []
        
        result = analyze_citation_network.invoke({"paper_identifier": "Very New Paper"})
        
        assert "⚠️" in result or "无引用数据" in result


class TestEdgeCases:
    """测试边界情况"""

    def test_calculate_score_missing_fields(self):
        """测试缺少字段的论文评分"""
        paper = {}  # 空字典
        
        score = calculate_reference_score(paper)
        assert score >= 0  # 应该返回有效分数
        
        score = calculate_citation_score(paper)
        assert score >= 0

    def test_format_paper_info_missing_fields(self):
        """测试格式化缺少字段的论文"""
        paper = {"title": "Minimal Paper"}
        
        result = format_paper_info(paper)
        
        assert "Minimal Paper" in result
        assert "N/A" in result  # 缺失字段应显示 N/A

    def test_rank_empty_list(self):
        """测试排序空列表"""
        papers = []
        ranked = rank_papers(papers, mode="reference")
        assert len(ranked) == 0

    def test_format_citation_network_empty_lists(self):
        """测试格式化空的引用列表（避免除以0错误）"""
        from tools.citation_tools import format_citation_network
        
        center_paper = {
            "title": "Test Paper",
            "authors": [{"name": "Author"}],
            "year": 2023,
            "citationCount": 0,
            "venue": "Conference",
            "url": "https://example.com",
        }
        
        # 空的参考文献和引用论文列表
        result = format_citation_network(center_paper, [], [])
        
        # 应该能成功格式化，不会出现除以0错误
        assert "📊 引用网络分析" in result
        assert "Test Paper" in result
        assert "平均被引次数（参考文献）" in result
        assert "0.0 次" in result  # 确保计算了平均值
        assert "开放获取论文数" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
