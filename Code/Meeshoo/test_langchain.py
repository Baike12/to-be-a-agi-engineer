from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())



def single_chat():
    res = deepseek_llm.invoke("halo")
    print(res.content)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
def multi_chat():
    messages = [
        SystemMessage(content="你是一个精通各种语言的代码助手"),
        HumanMessage(content="我是一个程序员"),
        AIMessage(content="欢迎！"),
        HumanMessage(content="我是谁")
    ]
    res = openai_llm.invoke(messages)
    print(res.content)
# multi_chat()

from langchain.prompts import PromptTemplate
def template_chat():
    template = PromptTemplate.from_template("给我讲个关于{subject}的笑话")
    print("===Template===")
    print(template)
    print("===Prompt===")
    prompt = template.format(subject='包公')
    print(prompt)
    ret = deepseek_llm.invoke(prompt)
    # 打印输出
    print(ret.content)
# template_chat()


from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
def mul_chat_template():
    human_prompt = "Translate your answer to {language}."
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages(
         [MessagesPlaceholder("history"), human_message_template]
    )

    human_message = HumanMessage(content="Who is Elon Musk?")
    ai_message = AIMessage(
        content="Elon Musk is a billionaire entrepreneur, inventor, and industrial designer"
    )
    #
    messages = chat_prompt.format_prompt(
        # 对 "history" 和 "language" 赋值
        history=[human_message, ai_message], language="中文"
    )

    print(messages.to_messages())
    result = deepseek_llm.invoke(messages)
    print(result.content)

def template_file():
    template = PromptTemplate.from_file("template.txt")
    print("===Template===")
    print(template)
    print("===Prompt===")
    print(template.format(topic='小狗'))


from pydantic import BaseModel, Field

# 定义你的输出对象
class Date(BaseModel):#
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="BC or AD")

from langchain_core.output_parsers import PydanticOutputParser
json_schema = {
    "title": "Date",
    "description": "Formated date expression",
    "type": "object",
    "properties": {
        "year": {
            "type": "integer",
            "description": "year, YYYY",
        },
        "month": {
            "type": "integer",
            "description": "month, MM",
        },
        "day": {
            "type": "integer",
            "description": "day, DD",
        },
        "era": {
            "type": "string",
            "description": "BC or AD",
        },
    },
}
def struct_output():
    structured_llm = deepseek_llm.with_structured_output(json_schema)# 可以结构化输出

    template1 = """提取用户输入中的日期。用户输入:{query}"""

    prompt = PromptTemplate(
        template=template1,
        input_variables=["query"],
    )

    query = "2023年四月6日天气晴..."
    input_prompt = prompt.format_prompt(query=query)
    print(input_prompt)
    res = structured_llm.invoke(input_prompt)
    print(res)

# struct_output()
from langchain_core.output_parsers import JsonOutputParser
def json_output():
    parser = JsonOutputParser(pydantic_object=Date)
    prompt = PromptTemplate(
        template="提取用户输入中的日期。\n用户输入:{query}\n{format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    query = "2023年四月6日天气晴..."
    input_prompt = prompt.format_prompt(query=query)
    print(input_prompt)
    output = deepseek_llm.invoke(input_prompt)
    print("原始输出:\n"+output.content)

    print("\n解析后:")
    parser.invoke(output)


from langchain_core.tools import tool
import json
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b
def function_calling():
    llm_with_tools = openai_llm.bind_tools([add, multiply])

    query = "3的4倍是多少?"
    messages = [HumanMessage(query)]

    output = llm_with_tools.invoke(messages)

    print(json.dumps(output.tool_calls, indent=4))
    messages.append(output)
    print(messages)
    for tool_call in output.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        print(tool_call)
        tool_msg = selected_tool.invoke(tool_call['args'])
        print(tool_msg)
        messages.append(ToolMessage(tool_msg, tool_call_id=tool_call["id"]))
        print(messages)

    new_output = llm_with_tools.invoke(messages)

    print(new_output.content)


from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_process():
    loader = PyMuPDFLoader(r"D:\tmp\Buffer of Thoughts.pdf")
    pages = loader.load_and_split()

    print(pages[0].page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    paragraphs = text_splitter.create_documents([pages[0].page_content])
    for para in paragraphs:
        print(para.page_content)
        print('-------')

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
    filter_messages,
)

def chat_process():
    messages = [
        SystemMessage("you're a good assistant, you always respond with a joke."),
        HumanMessage("i wonder why it's called langchain"),
        AIMessage(
            'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
        ),
        HumanMessage("and who is harrison chasing anyways"),
        AIMessage(
            "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
        ),
        HumanMessage("what do you call a speechless parrot"),
    ]

    # print(messages)

    messages1 =  trim_messages(
        messages,
        max_tokens=45,
        strategy="last",
        token_counter=openai_llm,
    )
    # print(messages1)
    messages2 = trim_messages(
        messages,
        max_tokens=56,
        strategy="last",
        token_counter=openai_llm,
        include_system=True,
        allow_partial=True,
    )
    print(messages2)

    messages = [
        SystemMessage("you are a good assistant", id="1"),
        HumanMessage("example input", id="2", name="example_user"),
        AIMessage("example output", id="3", name="example_assistant"),
        HumanMessage("real input", id="4", name="bob"),
        AIMessage("real output", id="5", name="alice"),
    ]
    o=[messages ,"messages:"]; print(o[1], o[0])
    messages3 =  filter_messages(messages, include_types="human")
    o=[messages3 ,"messages3:"]; print(o[1], o[0])

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List, Dict, Optional
from enum import Enum
import json

# 输出结构
class SortEnum(str, Enum):
    data = 'data'
    price = 'price'


class OrderingEnum(str, Enum):
    ascend = 'ascend'
    descend = 'descend'


class Semantics(BaseModel):
    name: Optional[str] = Field(description="流量包名称", default=None)
    price_lower: Optional[int] = Field(description="价格下限", default=None)
    price_upper: Optional[int] = Field(description="价格上限", default=None)
    data_lower: Optional[int] = Field(description="流量下限", default=None)
    data_upper: Optional[int] = Field(description="流量上限", default=None)
    sort_by: Optional[SortEnum] = Field(description="按价格或流量排序", default=None)
    ordering: Optional[OrderingEnum] = Field(
        description="升序或降序排列", default=None)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "将用户的输入解析成JSON表示。"),
        ("human", "{text}"),
    ]
)

structured_llm = deepseek_llm.with_structured_output(Semantics)
def out_put_singel():
    runnable = (
        {"text":RunnablePassthrough()}|prompt|structured_llm
    )

    ret = runnable.invoke("不超过100元的流量大的套餐有哪些")
    print(
        json.dumps(
            ret.dict(),
            indent = 4,
            ensure_ascii=False
        )
    )
def out_put_stream():
    runnable = (
        {"text": RunnablePassthrough()} | prompt | deepseek_llm | StrOutputParser()
    )


    # 流式输出
    for s in runnable.stream("不超过100元的流量大的套餐有哪些"):
        print(s, end="", flush=True)


from langchain_core.runnables.utils import ConfigurableField
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage
import os
model = openai_llm.configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="gpt",
    ds=deepseek_llm,
    # claude=claude_model,
)

# Prompt 模板
prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

# LCEL
chain = (
    {"query": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 运行时指定模型 "gpt" or "ernie"
ret = chain.with_config(configurable={"llm": "ds"}).invoke("请自我介绍")

print(ret)




