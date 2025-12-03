# 模型配置模块

本模块提供各种 LLM 和 Embedding 模型的统一配置和初始化接口。

## 目录结构

```
models/
├── __init__.py           # 模块导出
├── llm_config.py         # LLM 模型配置
├── embedding_config.py   # Embedding 模型配置
└── README.md            # 本文档
```

## 可用模型

### LLM 模型

#### 1. 通义千问 (Tongyi)

```python
from models import get_tongyi_model

# 获取通义千问模型实例
model = get_tongyi_model(
    streaming=True,      # 是否启用流式输出
    temperature=0.7      # 温度参数 (0-1)
)
```

**环境变量要求**: `TONGYI_API_KEY`

#### 2. 硅基流动 (SiliconFlow)

```python
from models import get_siliconflow_model

# 获取硅基流动模型实例
model = get_siliconflow_model(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # 模型名称
    streaming=True,                          # 是否启用流式输出
    temperature=0.7                          # 温度参数 (0-1)
)
```

**环境变量要求**: `SILICONFLOW_API_KEY`

**可用模型**:
- `Qwen/Qwen2.5-7B-Instruct` (默认)
- `Qwen/Qwen2.5-72B-Instruct`
- 更多模型请参考硅基流动官方文档

### Embedding 模型

#### 硅基流动 Embedding

```python
from models import get_siliconflow_embeddings

# 获取 embedding 模型实例
embeddings = get_siliconflow_embeddings(
    model_name="BAAI/bge-large-zh-v1.5",  # 模型名称
    check_embedding_ctx_length=False       # 是否检查上下文长度
)

# 嵌入单个查询
query_vector = embeddings.embed_query("这是一个查询文本")
print(f"向量维度: {len(query_vector)}")

# 嵌入多个文档
docs = ["文档1", "文档2", "文档3"]
doc_vectors = embeddings.embed_documents(docs)
print(f"已处理 {len(doc_vectors)} 个文档")
```

**环境变量要求**: `SILICONFLOW_API_KEY`

**推荐模型**:
- `BAAI/bge-large-zh-v1.5` - 中文，1024维 (默认)
- `BAAI/bge-m3` - 多语言，1024维
- `BAAI/bge-large-en-v1.5` - 英文，1024维
- `sentence-transformers/all-MiniLM-L6-v2` - 轻量级，384维

## 环境变量配置

在项目根目录的 `.env` 文件中配置所需的 API Key：

```bash
# 通义千问 API Key
TONGYI_API_KEY=your_tongyi_api_key

# 硅基流动 API Key
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

## 使用示例

### 完整示例：聊天应用

```python
from models import get_siliconflow_model

# 初始化模型
model = get_siliconflow_model()

# 发送消息
response = model.invoke("介绍一下 LangChain 框架")
print(response.content)
```

### 完整示例：文档向量化

```python
from models import get_siliconflow_embeddings

# 初始化 embedding 模型
embeddings = get_siliconflow_embeddings()

# 准备文档
documents = [
    "人工智能是计算机科学的一个分支。",
    "机器学习是人工智能的核心技术。",
    "深度学习推动了AI的快速发展。"
]

# 向量化
vectors = embeddings.embed_documents(documents)

# 使用向量进行相似度计算等操作
for i, vector in enumerate(vectors):
    print(f"文档 {i+1} 向量维度: {len(vector)}")
```

## 测试

运行测试脚本验证模型配置：

```bash
# 测试 embedding 模型
python test/test_embeddings.py
```

## 注意事项

1. **API Key 安全**: 永远不要将 API Key 硬编码在代码中或提交到版本控制系统
2. **模型选择**: 根据任务需求选择合适的模型（中文/英文/多语言、轻量级/高精度）
3. **上下文长度**: 对于长文本，注意分块处理以避免超出模型的最大上下文长度
4. **并发限制**: 注意 API 的并发调用限制和速率限制

## 日志

所有模型初始化都会通过 `loguru` 记录日志，便于调试和监控：

```
[get_siliconflow_embeddings] 初始化硅基流动嵌入模型，model=BAAI/bge-large-zh-v1.5, check_ctx_length=False
```
