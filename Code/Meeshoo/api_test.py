from openai import OpenAI
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())
# client = OpenAI(api_key="sk-jHN7UG4LANTFza1SyEvbrO6rtA3KHXUeKVwanfdC7NUy8dxY",base_url="https://api.fe8.cn/v1")

llm = OpenAI(api_key="sk-MsXZVIkrfqYUuSzd953wSUyFLmjbgzI9bz78NyKSeD3TpNjB",base_url="https://api.fe8.cn/v1")
completion = llm.chat.completions.create(
    model="gpt-4o-mini",
    # model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "halo."
        }
    ]
)

print(completion.choices[0].message)
