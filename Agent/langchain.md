### 核心组件
- 模型封装
- 数据连接封装
- 对话历史管理
- 架构封装
	- chain：功能组合
	- agent：根据用户输入，自动规划实现步骤、使用外部工具、完成指定功能
		- tool：调用外部函数
- callbacks
	- 获得查询
	- 从对话历史或者向量数据库查询，填充到prompt
	- 请求发送到llm
	- 解析返回值再输出
#### 模型封装
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")  # 默认是gpt-3.5-turbo
response = llm.invoke("你是谁")
print(response.content)
```
##### 多轮对话变成模版
```python
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

human_prompt = "Translate your answer to {language}."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    # variable_name 是 message placeholder 在模板中的变量名
    # 用于在赋值时使用
    [MessagesPlaceholder("history"), human_message_template]
)

from langchain_core.messages import AIMessage, HumanMessage

human_message = HumanMessage(content="Who is Elon Musk?")
ai_message = AIMessage(
    content="Elon Musk is a billionaire entrepreneur, inventor, and industrial designer"
)

messages = chat_prompt.format_prompt(
    # 对 "history" 和 "language" 赋值
    history=[human_message, ai_message], language="中文"
)

print(messages.to_messages())
```
##### 结构化输出
- 使用pydantic定义输出结构
```python
from pydantic import BaseModel, Field

# 定义你的输出对象
class Date(BaseModel):#
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="BC or AD")

from langchain_core.output_parsers import PydanticOutputParser
structured_llm = deepseek_llm.with_structured_output(Date)# 可以结构化输出

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
```
#### function calling
```python
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

```
### 对话管理
- 筛选、过滤对话
```python
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
    filter_messages,
)

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
```
### chain实现模型选择
```python
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

```