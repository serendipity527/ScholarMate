"""Pytest 配置和共享 fixtures

这个文件包含：
- 共享的 pytest fixtures
- 测试配置
- 命令行选项
"""

import pytest


def pytest_addoption(parser):
    """添加自定义命令行选项"""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="运行集成测试（需要网络连接和真实 API 调用）"
    )


def pytest_configure(config):
    """Pytest 配置钩子"""
    config.addinivalue_line(
        "markers", 
        "integration: 需要网络连接的集成测试"
    )


@pytest.fixture
def sample_openalex_response():
    """提供标准的 OpenAlex API 响应样本"""
    return {
        "meta": {
            "count": 100,
            "db_response_time_ms": 42,
            "page": 1,
            "per_page": 10
        },
        "results": [
            {
                "id": "https://openalex.org/W2741809807",
                "title": "Deep Learning for Image Recognition",
                "display_name": "Deep Learning for Image Recognition",
                "doi": "https://doi.org/10.1038/nature12345",
                "publication_year": 2023,
                "publication_date": "2023-05-15",
                "cited_by_count": 250,
                "relevance_score": 0.95,
                "type": "journal-article",
                "authorships": [
                    {"author": {"display_name": "Alice Johnson"}},
                    {"author": {"display_name": "Bob Smith"}},
                    {"author": {"display_name": "Charlie Brown"}},
                    {"author": {"display_name": "David Lee"}},
                ],
                "primary_location": {
                    "source": {
                        "display_name": "Nature",
                        "issn_l": "0028-0836"
                    }
                },
                "open_access": {
                    "is_oa": True,
                    "oa_status": "gold",
                    "oa_url": "https://www.nature.com/articles/nature12345.pdf"
                },
                "topics": [
                    {
                        "id": "https://openalex.org/T10001",
                        "display_name": "Artificial Intelligence"
                    },
                    {
                        "id": "https://openalex.org/T10002",
                        "display_name": "Computer Vision"
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_arxiv_document():
    """提供标准的 ArXiv 文档样本"""
    from unittest.mock import Mock
    
    doc = Mock()
    doc.metadata = {
        'Title': 'Attention Is All You Need',
        'Authors': 'Vaswani, Ashish; Shazeer, Noam; Parmar, Niki',
        'Published': '2017-06-12',
        'Entry ID': 'http://arxiv.org/abs/1706.03762v5'
    }
    doc.page_content = (
        "The dominant sequence transduction models are based on complex "
        "recurrent or convolutional neural networks that include an encoder "
        "and a decoder. The best performing models also connect the encoder "
        "and decoder through an attention mechanism."
    )
    return doc


@pytest.fixture
def empty_openalex_response():
    """提供空的 OpenAlex API 响应（无结果）"""
    return {
        "meta": {
            "count": 0,
            "db_response_time_ms": 15,
            "page": 1,
            "per_page": 10
        },
        "results": []
    }


@pytest.fixture
def mock_requests_get(mocker):
    """提供 requests.get 的 mock"""
    return mocker.patch('tools.paper_tools.requests.get')


@pytest.fixture
def mock_arxiv_retriever(mocker):
    """提供 ArxivRetriever 的 mock"""
    return mocker.patch('tools.paper_tools.ArxivRetriever')
