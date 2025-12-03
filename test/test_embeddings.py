"""测试嵌入模型
演示如何使用封装后的 embedding 模型
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 已加载环境变量文件: {env_path}")
else:
    print(f"⚠️  未找到 .env 文件: {env_path}")

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(project_root / "my-app" / "src"))

from models import get_siliconflow_embeddings  # noqa: E402


def test_embedding_query():
    """测试嵌入单个查询"""
    print("\n" + "="*50)
    print("测试 1: 嵌入单个查询")
    print("="*50)
    
    # 初始化 embeddings 模型
    embeddings = get_siliconflow_embeddings()
    
    # 嵌入单个查询
    text = "LangChain 是一个用于构建 LLM 应用的框架。"
    query_result = embeddings.embed_query(text)
    
    print(f"✨ 查询文本: {text}")
    print(f"✨ 嵌入向量维度: {len(query_result)}")
    print(f"✨ 向量前5位: {query_result[:5]}")


def test_embedding_documents():
    """测试嵌入多个文档"""
    print("\n" + "="*50)
    print("测试 2: 嵌入多个文档")
    print("="*50)
    
    # 初始化 embeddings 模型
    embeddings = get_siliconflow_embeddings()
    
    # 嵌入文档列表
    docs = [
        "我喜欢 Python 编程。",
        "今天天气不错。",
        "人工智能正在改变世界。"
    ]
    doc_results = embeddings.embed_documents(docs)
    
    print(f"📚 已处理文档数量: {len(doc_results)}")
    for i, doc in enumerate(docs):
        print(f"📄 文档 {i+1}: {doc}")
        print(f"   向量维度: {len(doc_results[i])}")
        print(f"   向量前3位: {doc_results[i][:3]}")


def test_different_model():
    """测试使用不同的嵌入模型"""
    print("\n" + "="*50)
    print("测试 3: 使用不同的嵌入模型")
    print("="*50)
    
    # 使用多语言模型
    embeddings = get_siliconflow_embeddings(model_name="BAAI/bge-m3")
    
    text = "This is a multilingual embedding model."
    query_result = embeddings.embed_query(text)
    
    print("✨ 模型: BAAI/bge-m3 (多语言)")
    print(f"✨ 查询文本: {text}")
    print(f"✨ 嵌入向量维度: {len(query_result)}")


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_embedding_query()
        test_embedding_documents()
        # test_different_model()  # 如果需要测试其他模型，取消注释
        
        print("\n" + "="*50)
        print("✅ 所有测试完成！")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
