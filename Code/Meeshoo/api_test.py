from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

client = OpenAI(base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        # {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "ahk中可以捕获alt按下，然后按下m，最后alt弹起这一系列动作吗"},
    ],
    stream=False
)

print(response.choices[0].message.content)