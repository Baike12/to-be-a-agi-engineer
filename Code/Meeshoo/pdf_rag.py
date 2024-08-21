from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''从 PDF 文件中（按指定页码）提取文字'''
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):

        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' '+text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs


# paragraphs = extract_text_from_pdf(r"D:\Book\Ai\llama2.pdf", min_line_length=10)
# print(paragraphs)
from elasticsearch7 import Elasticsearch, helpers
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

import warnings
warnings.simplefilter("ignore")  # 屏蔽 ES 的一些Warnings


def to_keywords(input_string):
    '''（英文）文本只保留关键字'''
    # 使用正则表达式替换所有非字母数字的字符为空格
    no_symbols = re.sub(r'[^a-zA-Z0-9\s]', ' ', input_string)# 去除标点符号
    word_tokens = word_tokenize(no_symbols)# 分词
    # 加载停用词表
    stop_words = set(stopwords.words('english'))# 停用词表
    ps = PorterStemmer()# 词根提取器
    # 去停用词，取词根
    filtered_sentence = [ps.stem(w)
                         for w in word_tokens if not w.lower() in stop_words]#
    return ' '.join(filtered_sentence)


import os, time

# 引入配置文件
ELASTICSEARCH_BASE_URL = os.getenv('ELASTICSEARCH_BASE_URL')
ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')
ELASTICSEARCH_NAME= os.getenv('ELASTICSEARCH_NAME')

# tips: 如果想在本地运行，请在下面一行 print(ELASTICSEARCH_BASE_URL) 获取真实的配置
# o=[ELASTICSEARCH_BASE_URL ,"ELASTICSEARCH_BASE_URL:"]; print(o[1], o[0])
es = Elasticsearch(
    hosts=["http://localhost:9200"],  # 服务地址与端口
    http_auth=("baike", "19980611bkab"),  # 用户名，密码
)

index_name = "llama2"
def llama2_bulk_into_es():
    paragraphs = extract_text_from_pdf(r"D:\Book\Ai\llama2.pdf", min_line_length=10)

    # 3. 如果索引已存在，删除它（仅供演示，实际应用时不需要这步）
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # 4. 创建索引
    es.indices.create(index=index_name)

    actions = [
        {
            "_index": index_name,
            "_source": {
                "keywords": to_keywords(para),
                "text": para
            }
        }
        for para in paragraphs
    ]

    helpers.bulk(es, actions)

    # 灌库是异步的
    time.sleep(2)

# llama2_bulk_into_es()

def search(query_string, top_n=3):
    # ES 的查询语言
    search_query = {
        "match": {
            "keywords": to_keywords(query_string)
        }
    }
    res = es.search(index=index_name, query=search_query, size=top_n)
    return [hit["_source"]["text"] for hit in res["hits"]["hits"]]


# for r in results:
#     print(r+"\n")

from openai import OpenAI
import os
# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

client = OpenAI(base_url="https://api.deepseek.com")
def get_completion(prompt):
    '''封装 openai 接口'''
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content

def build_prompt(prompt_template, **kwargs):
    '''将 Prompt 模板赋值'''
    inputs = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val
    return prompt_template.format(**inputs)

prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。

已知信息:
{context}

用户问：
{query}

如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
请不要输出已知信息中不包含的信息或答案。
请用中文回答用户问题。
"""
def ask(query):
    search_results = search(query, 2)
    prompt = build_prompt(prompt_template, context=search_results, query=query)
    print("===Prompt===")
    print(prompt)

    # 3. 调用 LLM
    response = get_completion(prompt)

    print("===回复===")
    print(response)


user_query1 = "how many parameters does llama 2 have?"
user_query2 = "please tell me the structure of llama2"
# ask(user_query2)
# for res in search_results:
#     print(res+"\n")

import numpy as np
from numpy import dot
from numpy.linalg import norm


def cos_sim(a, b):
    '''余弦距离 -- 越大越相似'''
    return dot(a, b)/(norm(a)*norm(b))

def l2(a, b):
    '''欧氏距离 -- 越小越相似'''
    x = np.asarray(a)-np.asarray(b)
    return norm(x)

from sentence_transformers import SentenceTransformer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
def get_embedding(text):
    '''获取文本的向量表示'''
    return model.encode(text)
# sentences = ["This is the first sentence.", "And this is another one."]
# print(sentences)
# print(get_embedding(sentences).shape)
import chromadb
from chromadb.config import Settings


class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.PersistentClient(r"D:\ChromaData")
        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results

vector_db = MyVectorDBConnector("llama2", get_embedding)
def add_llama2_to_vector_db(pages):
    paragraphs = extract_text_from_pdf(
        r"D:\Book\Ai\llama2.pdf",
        page_numbers=pages,
        min_line_length=10
    )
    print(paragraphs)
    vector_db.add_documents(paragraphs)

# add_llama2_to_vector_db([0])

def query_vector_db(query, top_n):
    '''检索向量数据库'''
    results = vector_db.search(query, top_n)
    return results

user_query = "how many parameters does llama2 have?"
# result = query_vector_db(user_query, 3)
# print(result)

class RAG_Bot:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. 检索
        search_results = self.vector_db.search(user_query, self.n_results)

        # 2. 构建 Prompt
        prompt = build_prompt(
            prompt_template, context=search_results['documents'][0], query=user_query)

        # 3. 调用 LLM
        response = self.llm_api(prompt)
        return response

bot = RAG_Bot(
    vector_db,
    llm_api=get_completion
)

user_query = "llama 2有多少参数?"
response = bot.chat(user_query)
print(response)
















