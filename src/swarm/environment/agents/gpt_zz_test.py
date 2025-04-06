from openai import OpenAI
import httpx
import os

client_gpt4 = OpenAI(base_url="https://gtfast.xiaoerchaoren.com:8937//v1", 
    api_key=os.environ.get("OPENAI_API_KEY", r"sk-HaSKH3MvXRP9DH8KDa6303E32d4e4aBaA185BaBc8b50BfAa"),
    http_client=httpx.Client(
        base_url="https://gtfast.xiaoerchaoren.com:8937//v1",
        follow_redirects=True,
    ))

completion = client_gpt4.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "are you gpt3.5?"}
  ]
)

print(completion.choices[0].message)
