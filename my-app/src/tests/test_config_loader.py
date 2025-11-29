"""配置加载器测试"""

import sys
from pathlib import Path

# 将 src 目录添加到路径
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from config.config_loader import (
    load_config,
    load_prompt,
    load_all_prompts,
    get_config_path,
    get_research_assistant_prompt,
)


class TestConfigLoader:
    """测试配置加载器"""

    def test_get_config_path(self):
        """测试获取配置文件路径"""
        config_path = get_config_path("prompts.yaml")
        assert config_path.exists()
        assert config_path.name == "prompts.yaml"

    def test_load_config(self):
        """测试加载配置文件"""
        config = load_config("prompts.yaml")
        assert isinstance(config, dict)
        assert len(config) > 0
        assert "research_assistant_prompt" in config

    def test_load_prompt(self):
        """测试加载单个提示词"""
        prompt = load_prompt("research_assistant_prompt")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "科研助手" in prompt

    def test_load_prompt_not_exist(self):
        """测试加载不存在的提示词"""
        with pytest.raises(KeyError) as exc_info:
            load_prompt("non_existent_prompt")
        assert "non_existent_prompt" in str(exc_info.value)

    def test_load_all_prompts(self):
        """测试加载所有提示词"""
        prompts = load_all_prompts()
        assert isinstance(prompts, dict)
        assert len(prompts) >= 3  # 至少有3个提示词
        assert "research_assistant_prompt" in prompts
        assert "paper_analysis_prompt" in prompts
        assert "citation_network_prompt" in prompts

    def test_get_research_assistant_prompt(self):
        """测试快捷函数获取研究助手提示词"""
        prompt = get_research_assistant_prompt()
        assert isinstance(prompt, str)
        assert "科研助手" in prompt
        assert "工具能力" in prompt
        assert "论文搜索工具" in prompt


class TestPromptContent:
    """测试提示词内容"""

    def test_research_assistant_prompt_structure(self):
        """测试研究助手提示词结构"""
        prompt = load_prompt("research_assistant_prompt")
        
        # 检查关键部分是否存在
        assert "工具能力" in prompt
        assert "论文搜索工具" in prompt
        assert "网络搜索工具" in prompt
        assert "参数映射指南" in prompt
        assert "核心原则" in prompt

    def test_research_assistant_prompt_tools(self):
        """测试研究助手提示词包含所有工具"""
        prompt = load_prompt("research_assistant_prompt")
        
        # 检查论文搜索工具
        assert "search_papers_openalex" in prompt
        assert "search_papers_ArXiv" in prompt
        assert "search_papers_semantic_scholar" in prompt
        assert "search_papers_aggregated" in prompt
        
        # 检查 Tavily 工具
        assert "tavily_search" in prompt
        assert "tavily_extract" in prompt
        assert "tavily_crawl" in prompt
        assert "tavily_map" in prompt

    def test_paper_analysis_prompt(self):
        """测试论文分析提示词"""
        prompt = load_prompt("paper_analysis_prompt")
        assert "论文分析" in prompt
        assert len(prompt) > 0

    def test_citation_network_prompt(self):
        """测试引用网络分析提示词"""
        prompt = load_prompt("citation_network_prompt")
        assert "引用网络" in prompt
        assert len(prompt) > 0


class TestPromptModification:
    """测试提示词可以被修改"""

    def test_yaml_format_allows_easy_modification(self):
        """测试 YAML 格式便于修改"""
        config_path = get_config_path("prompts.yaml")
        
        # 读取原始内容
        with open(config_path, "r", encoding="utf-8") as f:
            original_content = f.read()
        
        # 验证是 YAML 格式
        assert "research_assistant_prompt:" in original_content
        assert "|" in original_content  # YAML 多行字符串标记
        
        # 验证有注释
        assert "#" in original_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
