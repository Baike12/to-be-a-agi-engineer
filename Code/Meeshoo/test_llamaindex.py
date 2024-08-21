import json
from pydantic.v1 import BaseModel

def show_json(data):
    """用于展示json数据"""
    if isinstance(data, str):
        obj = json.loads(data)
        print(json.dumps(obj, indent=4))
    elif isinstance(data, dict) or isinstance(data, list):
        print(json.dumps(data, indent=4))
    elif issubclass(type(data), BaseModel):
        print(json.dumps(data.dict(), indent=4, ensure_ascii=False))

def show_list_obj(data):
    """用于展示一组对象"""
    if isinstance(data, list):
        for item in data:
            show_json(item)
    else:
        raise ValueError("Input is not a list")


from llama_index.core import SimpleDirectoryReader
# from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import TokenTextSplitter
def read_pdf_file():
    reader = SimpleDirectoryReader(
            input_dir=r"D:\tmp", # 目标目录
            recursive=False, # 是否递归遍历子目录
            required_exts=[".pdf"], # (可选)只读取指定后缀的文件
            # file_extractor={".pdf":PyMuPDFReader()}
        )
    documents = reader.load_data()
    show_json(documents[0])
    show_json(documents[1])



    node_parser = TokenTextSplitter(
        chunk_size=100,  # 每个 chunk 的最大长度
        chunk_overlap=50  # chunk 之间重叠长度
    )

    nodes = node_parser.get_nodes_from_documents(
        documents, show_progress=False
    )
    show_json(nodes[0])


from llama_index.core import Document
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import MarkdownNodeParser
from pathlib import Path

def read_md_file():
    md_docs = FlatReader().load_data(Path("D:/Note/embeding.md"))
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(md_docs)

    show_json(nodes[2])
    show_json(nodes[3])


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter

# 加载 pdf 文档
documents = SimpleDirectoryReader(
    r"D:\tmp",
    required_exts=[".pdf"],
    # file_extractor={".pdf": PyMuPDFReader()}
).load_data()

# 定义 Node Parser
node_parser = TokenTextSplitter(chunk_size=300, chunk_overlap=100)
# 切分文档
nodes = node_parser.get_nodes_from_documents(documents)

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
# 导入开源embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    # model_name="BAAI/bge-small-en-v1.5"
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(r"D:\ChromaData")
# chroma_collection = chroma_client.get_collection("dtemp")
def add_to_vecdb():
    chroma_collection = chroma_client.get_collection("dtemp")
    # 创建 Vector Store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # Storage Context 是 Vector Store 的存储容器，用于存储文本、index、向量等数据
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print(storage_context)
    # 创建 index：通过 Storage Context 关联到自定义的 Vector Store
    index = VectorStoreIndex(nodes, storage_context=storage_context)# 向量存储索引,加速索引
    # 获取 retriever
    vector_retriever = index.as_retriever(similarity_top_k=2)
    # 检索
    results = vector_retriever.retrieve("buffer of thought")
    show_list_obj(results)
import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"耗时 {self.interval*1000} ms")


from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY
# from openai import OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
# llm = OpenAI(model="gpt-4o-mini", max_tokens=512)
# llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)
from llama_index.embeddings.openai import OpenAIEmbedding
# import nest_asyncio
# nest_asyncio.apply()
def ingestion_pipeline():
    chroma_collection = chroma_client.get_collection("dtmp_pipeline")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=300, chunk_overlap=100),
            # TitleExtractor(),
            # OpenAIEmbedding(), # 将文本向量化
            HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        ],
        vector_store=vector_store,
    )

    documents1 = SimpleDirectoryReader(
        r"D:\tmp",
        required_exts=[".pdf"],
        # file_extractor={".pdf": PyMuPDFReader()}
    ).load_data()
    with Timer():
        pipeline.run(documents=documents1)
    index=VectorStoreIndex.from_vector_store(vector_store)
    vector_retrieve = index.as_retriever(similarity_top_k=1)
    results = vector_retrieve.retrieve("BoT有什么用")
    show_list_obj(results[:1])

ingestion_pipeline()
from llama_index.core.postprocessor import SentenceTransformerRerank

def rerank():

    # 检索后排序模型
    postprocessor = SentenceTransformerRerank(
        model="BAAI/bge-reranker-large", top_n=2
    )
    nodes = postprocessor.postprocess_nodes(nodes, query_str="Llama2 有商用许可吗?")

    for i, node in enumerate(nodes):
        print(f"[{i}] {node.text}")












