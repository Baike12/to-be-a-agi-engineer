### ingestion pipline
- 子步骤可以缓存，下次同样的操作可以直接从缓存读取
```python
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
import nest_asyncio
nest_asyncio.apply() # 只在Jupyter笔记环境中需要此操作，否则会报错

chroma_client.reset() # 为演示方便，实际不用每次 reset
chroma_collection = chroma_client.create_collection("ingestion_demo")

# 创建 Vector Store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=300, chunk_overlap=100), # 按句子切分
        TitleExtractor(), # 利用 LLM 对文本生成标题
        OpenAIEmbedding(), # 将文本向量化
    ],
    vector_store=vector_store,
)

documents = SimpleDirectoryReader(
    "./data", 
    required_exts=[".pdf"],
    file_extractor={".pdf": PyMuPDFReader()}
).load_data()

# 计时
with Timer():
    # Ingest directly into a vector db
    pipeline.run(documents=documents)

# 创建索引
index = VectorStoreIndex.from_vector_store(vector_store)

# 获取 retriever
vector_retriever = index.as_retriever(similarity_top_k=1)

# 检索
results = vector_retriever.retrieve("Llama2有多少参数")

show_list_obj(results[:1])
```
### 总结
- 加载各种格式的数据
- 数据处理
- 提供对embedding的封装，然后建索引，提高相关数据检索效率
- 提供了pipeline，使得每一步的结果可以缓存起来
- 检索后提供对rerank的封装
- 提供直接从向量中检索的对话，以及流式对话
- 对prompt、llm和embedding都提供了封装