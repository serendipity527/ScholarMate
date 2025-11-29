# 论文搜索工具测试套件

## 📋 测试概览

本测试套件为 `paper_tools.py` 中的论文搜索工具提供全方位测试覆盖。

### 测试文件结构

```
tests/
├── __init__.py              # 包初始化
├── conftest.py              # 共享 fixtures 和配置
├── test_paper_tools.py      # 主要功能测试
├── test_performance.py      # 性能和负载测试
└── README.md               # 本文件
```

## 🧪 测试类型

### 1. **单元测试** (`test_paper_tools.py`)

#### ✅ 参数验证测试 (`TestOpenAlexSearchInput`)
- 测试 Pydantic 模型的输入验证
- 边界值测试（max_results: 1-200）
- 枚举值验证（sort_by 选项）
- 必填参数检查

#### ✅ API 调用测试 (`TestSearchPapersOpenAlexAPI`)
- 基本搜索功能
- 带筛选条件的复杂搜索
- 无结果处理
- 多作者显示逻辑
- API 参数正确性验证

#### ✅ 错误处理测试 (`TestErrorHandling`)
- 超时错误（Timeout）
- 速率限制错误（403）
- 资源未找到错误（404）
- 服务器错误（500+）
- 网络连接错误
- 未预期的异常

#### ✅ 输出格式化测试 (`TestOutputFormatting`)
- DOI 链接格式化
- 开放获取状态显示（金/绿/混合/铜色）
- 引用次数千分位格式化
- Markdown 格式正确性

#### ✅ ArXiv 工具测试 (`TestSearchPapersArXiv`)
- 基本搜索功能
- 元数据提取
- ArXiv ID 解析

### 2. **集成测试** (`TestRealAPIIntegration`)
- 真实 API 调用测试（需要网络）
- 端到端工作流验证

### 3. **性能测试** (`test_performance.py`)
- 响应时间基准测试
- 大数据集处理性能
- 并发请求模拟
- 内存泄漏检测

## 🚀 运行测试

### 安装依赖

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装测试依赖
pip install pytest pytest-cov pytest-mock
```

### 运行所有单元测试

```bash
# 在项目根目录
cd /home/darwin/projects/ScholarMate/my-app

# 运行所有测试
pytest src/tests/test_paper_tools.py -v

# 或者简写
pytest -v
```

### 运行特定测试类

```bash
# 只运行参数验证测试
pytest src/tests/test_paper_tools.py::TestOpenAlexSearchInput -v

# 只运行错误处理测试
pytest src/tests/test_paper_tools.py::TestErrorHandling -v
```

### 运行特定测试方法

```bash
# 运行单个测试
pytest src/tests/test_paper_tools.py::TestOpenAlexSearchInput::test_valid_basic_input -v
```

### 运行性能测试

```bash
# 运行性能测试（包含慢速测试）
pytest src/tests/test_performance.py -v -m slow
```

### 运行集成测试（需要网络）

```bash
# 运行真实 API 测试
pytest src/tests/test_paper_tools.py::TestRealAPIIntegration -v --run-integration
```

### 生成测试覆盖率报告

```bash
# 生成 HTML 覆盖率报告
pytest --cov=src/tools --cov-report=html --cov-report=term

# 查看报告
open htmlcov/index.html  # macOS
# 或
xdg-open htmlcov/index.html  # Linux
```

## 📊 测试覆盖率目标

| 组件 | 目标覆盖率 | 当前状态 |
|------|-----------|---------|
| `search_papers_openalex` | 95%+ | ✅ |
| `search_papers_ArXiv` | 90%+ | ✅ |
| `OpenAlexSearchInput` | 100% | ✅ |
| 错误处理 | 100% | ✅ |

## 🏷️ 测试标记

使用 pytest 标记来选择性运行测试：

```bash
# 只运行集成测试
pytest -m integration

# 排除慢速测试
pytest -m "not slow"

# 只运行单元测试
pytest -m unit

# 运行 API 相关测试
pytest -m api
```

## 📝 编写新测试

### 测试命名规范

- 测试类：`Test<功能名>`
- 测试方法：`test_<测试场景描述>`
- 使用清晰的描述性名称

### 示例：添加新测试

```python
class TestNewFeature:
    """测试新功能的描述"""

    def test_feature_basic_case(self):
        """测试基本用例"""
        # Arrange（准备）
        input_data = OpenAlexSearchInput(query="test")
        
        # Act（执行）
        result = search_papers_openalex(**input_data.dict())
        
        # Assert（断言）
        assert result is not None
        assert "论文" in result
```

## 🐛 调试测试

### 查看详细输出

```bash
# 显示 print 输出
pytest -v -s

# 显示完整的错误堆栈
pytest -v --tb=long

# 在第一个失败时停止
pytest -x
```

### 使用 pdb 调试

```bash
# 在失败时进入调试器
pytest --pdb

# 在测试开始时进入调试器
pytest --trace
```

### 查看测试日志

```bash
# 显示日志输出
pytest -v --log-cli-level=DEBUG
```

## 📈 持续集成

### GitHub Actions 配置示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-mock
      - name: Run tests
        run: pytest --cov=src/tools --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## 🔧 故障排除

### 常见问题

#### 1. 导入错误
```bash
# 确保 PYTHONPATH 正确
export PYTHONPATH="${PYTHONPATH}:/path/to/ScholarMate/my-app/src"
```

#### 2. Mock 不生效
```python
# 确保 patch 路径正确
@patch('tools.paper_tools.requests.get')  # ✅ 正确
@patch('requests.get')  # ❌ 错误
```

#### 3. Fixture 未找到
```bash
# 确保 conftest.py 在正确位置
tests/
├── conftest.py  # ✅ 这里
└── test_paper_tools.py
```

## 📚 参考资源

- [Pytest 官方文档](https://docs.pytest.org/)
- [unittest.mock 文档](https://docs.python.org/3/library/unittest.mock.html)
- [Pydantic 测试指南](https://docs.pydantic.dev/latest/usage/devtools/)
- [测试最佳实践](https://docs.pytest.org/en/latest/goodpractices.html)

## ✅ 测试检查清单

运行测试前确保：

- [ ] 已激活虚拟环境
- [ ] 已安装所有测试依赖
- [ ] 已更新代码改动
- [ ] 已添加新功能的测试
- [ ] 所有测试通过
- [ ] 代码覆盖率达标

## 🎯 下一步计划

- [ ] 添加 mutation 测试
- [ ] 集成代码质量检查（pylint, flake8）
- [ ] 添加 API 录制/回放功能（VCR.py）
- [ ] 性能基准测试自动化
- [ ] 添加压力测试

---

**最后更新：** 2024-11-29  
**维护者：** ScholarMate Team
