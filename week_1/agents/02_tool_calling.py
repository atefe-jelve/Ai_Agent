import json
import os

from dotenv import load_dotenv
from openai import OpenAI


# 1. Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY پیدا نشد. فایل .env را بررسی کن."
    )


# 2. Initialize the model client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = os.getenv(
    "OPENROUTER_MODEL",
    "poolside/laguna-s-2.1:free",
)


# 3. Define a Python tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    result = a * b

    print(f"\n[Tool executed] multiply({a}, {b}) = {result}")

    return result


def add(a: float, b: float) -> float:
    """add two numbers and return the result."""
    result = a + b

    print(f"\n[Tool executed] add({a}, {b}) = {result}")

    return result


import requests


def get_product_price(product_id: int) -> dict:
    response = requests.get(
        f"https://api.example.com/products/{product_id}",
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    return {
        "product_id": product_id,
        "price": data["price"],
        "currency": data["currency"],
    }

# 4. Describe the tool for the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number.",
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number.",
                    },
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
        {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number.",
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number.",
                    },
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
        {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Get the price of a product from the product API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The product ID.",
                    }
                },
                "required": ["product_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


# 5. Map tool names to actual Python functions
available_tools = {
    "multiply": multiply,
    "add": add,
    "get_product_price": get_product_price,

}


# 6. Receive user input
user_input = input("You: ").strip()

if not user_input:
    raise ValueError("لطفاً یک درخواست وارد کن.")


# 7. Send the user request and tool definitions to the model
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "When the user asks to multiply two numbers, "
            "use the multiply or add tool. "
            "After receiving the tool result, explain the answer briefly."
        ),
    },
    {
        "role": "user",
        "content": user_input,
    },
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)


# 8. Inspect the model response
assistant_message = response.choices[0].message

if not assistant_message.tool_calls:
    print("\nAgent:", assistant_message.content)
    raise SystemExit


# 9. Process tool calls
messages.append(
    {
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": [
            tool_call.model_dump(exclude_none=True)
            for tool_call in assistant_message.tool_calls
        ],
    }
)

for tool_call in assistant_message.tool_calls:
    tool_name = tool_call.function.name

    tool_arguments = json.loads(
        tool_call.function.arguments
    )

    print(f"\n[Model requested tool] {tool_name}")
    print("[Arguments]", tool_arguments)

    tool_function = available_tools.get(tool_name)

    if tool_function is None:
        tool_result = {
            "error": f"Unknown tool: {tool_name}"
        }
    else:
        tool_result = tool_function(**tool_arguments)

    # 10. Send the tool result back to the model
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(
                tool_result,
                ensure_ascii=False,
            ),
        }
    )


# 11. Ask the model to generate the final answer
final_response = client.chat.completions.create(
    model=model,
    messages=messages,
)

final_answer = final_response.choices[0].message.content

print("\nAgent:", final_answer)