### 有什么用
- 存储embedding的向量
- 查询相似向量
### 如何用
```python
import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_colee")
# switch `add` to `upsert` to avoid adding the same documents every time
from sentence_transformers import SentenceTransformer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
def get_embedding(text):
    '''获取文本的向量表示'''
    return model.encode(text)
def test_1():
    text = [
            "This is a document about pineapple",
            "This is a document about oranges"
    ]
    collection.add(
        documents=text,
        ids=["id1", "id2"],
        embeddings=get_embedding(text)
    )
    results = collection.query(
        query_texts=["This is a query document about florida"], # Chroma will embed this for you
        n_results=2 # how many results to return
    )

    print(results)

def show_collections():
    collections = chroma_client.list_collections()

    # 打印每个集合的信息
    for collection in collections:
        print(f"Collection name: {collection.name}")
        # 可以获取更多关于集合的信息，例如描述等

show_collections()
```
### 如何实现
#### 存储实现
#### 查询实现