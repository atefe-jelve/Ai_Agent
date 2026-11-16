import os
import json

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# 1. Configuration
# ==========================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "poolside/laguna-s-2.1:free"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


# ==========================================
# 2. Tools
# ==========================================

products = {
    101: {
        "name": "Keyboard",
        "price": 200,
        "currency": "USD",
    },
    102: {
        "name": "Mouse",
        "price": 50,
        "currency": "USD",
    },
}


def get_product_price(product_id: int):
    """Return the price of a product."""

    product = products.get(product_id)

    if product is None:
        return {
            "error": f"Product {product_id} not found"
        }

    return {
        "product_id": product_id,
        "name": product["name"],
        "price": product["price"],
        "currency": product["currency"],
    }


def calculate_discount(price: float, discount_percent: float):
    """Calculate the final price after discount."""

    discount_amount = price * discount_percent / 100

    final_price = price - discount_amount

    return {
        "original_price": price,
        "discount_percent": discount_percent,
        "discount_amount": discount_amount,
        "final_price": final_price,
    }


# ==========================================
# 3. Tool Schemas
# ==========================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Get the price of a product using its product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer"
                    }
                },
                "required": ["product_id"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_discount",
            "description": "Calculate the final price after applying a discount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number"
                    },
                    "discount_percent": {
                        "type": "number"
                    },
                },
                "required": [
                    "price",
                    "discount_percent",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


# ==========================================
# 4. Tool Registry
# ==========================================

available_tools = {
    "get_product_price": get_product_price,
    "calculate_discount": calculate_discount,
}


# ==========================================
# 5. Agent Loop
# ==========================================

def run_agent(user_input: str):

    messages = [
        {
            "role": "system",
            "content": """
You are a helpful shopping assistant.

You can use these tools:

1. get_product_price:
   Get a product's price.

2. calculate_discount:
   Calculate the final price after discount.

If the user asks for a discounted product price:

- First get the original product price.
- Then calculate the discount using the returned price.
- Do not guess product prices.
- Use the tools when necessary.
- Give the user a clear final answer.
""",
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    max_tool_rounds = 5

    for round_number in range(max_tool_rounds):

        print(f"\n--- Agent Round {round_number + 1} ---")

        # Call LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # Add assistant response to conversation history
        messages.append(message.model_dump(exclude_none=True))

        # If there are no tool calls, we have the final answer
        if not message.tool_calls:

            print("\nAgent Final Answer:")
            print(message.content)

            return message.content

        # Execute tool calls
        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            tool_args = json.loads(
                tool_call.function.arguments
            )

            print(f"Tool selected: {tool_name}")
            print(f"Arguments: {tool_args}")

            # Find the Python function
            tool_function = available_tools.get(tool_name)

            if tool_function is None:

                tool_result = {
                    "error": f"Unknown tool: {tool_name}"
                }

            else:

                try:

                    tool_result = tool_function(**tool_args)

                except Exception as error:

                    tool_result = {
                        "error": str(error)
                    }

            print(f"Tool result: {tool_result}")

            # Return tool result to the LLM
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                }
            )

    print("Maximum tool rounds reached.")

    return None


# ==========================================
# 6. User Chat Loop
# ==========================================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:

        print("Goodbye!")
        break

    run_agent(user_input)