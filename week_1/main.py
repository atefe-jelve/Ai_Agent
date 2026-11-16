
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY پیدا نشد. فایل .env را بررسی کن."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    
    # model="openai/gpt-oss-20b:free",
    # model="openai/gpt-oss-20b",
    model="openrouter/free",
    # model="poolside/laguna-s-2.1:free",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: OpenRouter connection successful!",
        }
    ],
)

print("Model:", response.model)
print("Response:", response.choices[0].message.content)
print("Status: SUCCESS")
