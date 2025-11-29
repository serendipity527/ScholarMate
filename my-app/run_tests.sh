#!/bin/bash

# 测试运行脚本
# 使用方法: ./run_tests.sh [选项]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   ScholarMate 测试套件${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠️  警告: 虚拟环境未激活${NC}"
    echo -e "${YELLOW}正在尝试激活虚拟环境...${NC}"
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo -e "${GREEN}✅ 虚拟环境已激活${NC}"
    else
        echo -e "${RED}❌ 错误: 未找到虚拟环境${NC}"
        exit 1
    fi
fi

# 检查依赖
echo -e "${BLUE}检查测试依赖...${NC}"
python -c "import pytest" 2>/dev/null || {
    echo -e "${YELLOW}安装测试依赖...${NC}"
    pip install pytest pytest-cov pytest-mock
}

# 解析命令行参数
TEST_TYPE=${1:-all}

case $TEST_TYPE in
    all)
        echo -e "${GREEN}🧪 运行所有测试${NC}"
        pytest src/tests/ -v --tb=short
        ;;
    
    unit)
        echo -e "${GREEN}🧪 运行单元测试${NC}"
        pytest src/tests/test_paper_tools.py -v -m "not integration"
        ;;
    
    integration)
        echo -e "${GREEN}🧪 运行集成测试（需要网络）${NC}"
        pytest src/tests/test_paper_tools.py::TestRealAPIIntegration -v --run-integration
        ;;
    
    performance)
        echo -e "${GREEN}🧪 运行性能测试${NC}"
        pytest src/tests/test_performance.py -v
        ;;
    
    coverage)
        echo -e "${GREEN}🧪 运行测试并生成覆盖率报告${NC}"
        pytest src/tests/ --cov=src/tools --cov-report=html --cov-report=term
        echo -e "${GREEN}📊 覆盖率报告已生成: htmlcov/index.html${NC}"
        ;;
    
    quick)
        echo -e "${GREEN}⚡ 快速测试（跳过慢速测试）${NC}"
        pytest src/tests/ -v -m "not slow and not integration" --tb=short
        ;;
    
    watch)
        echo -e "${GREEN}👀 监视模式（文件改动时自动运行）${NC}"
        echo -e "${YELLOW}安装 pytest-watch...${NC}"
        pip install pytest-watch
        ptw src/tests/ -- -v --tb=short
        ;;
    
    debug)
        echo -e "${GREEN}🐛 调试模式${NC}"
        pytest src/tests/ -v -s --tb=long --log-cli-level=DEBUG
        ;;
    
    help)
        echo "使用方法: ./run_tests.sh [选项]"
        echo ""
        echo "选项:"
        echo "  all          - 运行所有测试（默认）"
        echo "  unit         - 只运行单元测试"
        echo "  integration  - 运行集成测试（需要网络）"
        echo "  performance  - 运行性能测试"
        echo "  coverage     - 生成测试覆盖率报告"
        echo "  quick        - 快速测试（跳过慢速测试）"
        echo "  watch        - 监视模式（自动重运行）"
        echo "  debug        - 调试模式（详细输出）"
        echo "  help         - 显示此帮助信息"
        echo ""
        echo "示例:"
        echo "  ./run_tests.sh           # 运行所有测试"
        echo "  ./run_tests.sh unit      # 只运行单元测试"
        echo "  ./run_tests.sh coverage  # 生成覆盖率报告"
        exit 0
        ;;
    
    *)
        echo -e "${RED}❌ 错误: 未知选项 '$TEST_TYPE'${NC}"
        echo "使用 './run_tests.sh help' 查看可用选项"
        exit 1
        ;;
esac

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ 所有测试通过！${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ 测试失败${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
