
# ==========================================
# 1. Configuration
# ==========================================

# 1. Load environment variables
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

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "poolside/laguna-s-2.1:free",
)

print("Current model:", MODEL)
# ==========================================
# 2. Mock Database
# ==========================================

PRODUCTS = {
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


# ==========================================
# 3. Mock Tools
# ==========================================

def get_product_price(product_id: int) -> dict:
    product = PRODUCTS.get(product_id)

    if product is None:
        return {"error": "Product not found"}

    return {
        "product_id": product_id,
        **product,
    }


def calculate_discount(
    price: float,
    discount_percent: float,
) -> dict:

    if not 0 <= discount_percent <= 100:
        return {"error": "Invalid discount percentage"}

    if price < 0:
        return {"error": "Invalid price"}

    final_price = price * (1 - discount_percent / 100)

    return {
        "original_price": price,
        "discount_percent": discount_percent,
        "final_price": round(final_price, 2),
    }


AVAILABLE_TOOLS = {
    "get_product_price": get_product_price,
    "calculate_discount": calculate_discount,
}


# ==========================================
# 4. Tool Schemas
# ==========================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": (
                "Look up the original price of a product "
                "by its product ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer"},
                },
                "required": ["product_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_discount",
            "description": (
                "Calculate a discounted price using the "
                "original price and discount percentage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "discount_percent": {"type": "number"},
                },
                "required": ["price", "discount_percent"],
                "additionalProperties": False,
            },
        },
    },
]


# ==========================================
# 5. Real Agent Loop with Trace
# ==========================================

def run_agent(user_input: str, max_rounds: int = 5):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a shopping assistant. "
                "Always retrieve a product's price using "
                "get_product_price before calculating "
                "its discount. Never invent prices. "
                "Use calculate_discount for discount calculations. "
                "If a product is missing, explain that it "
                "was not found."
            ),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    trace = []
    final_answer = None
    completed = False

    for round_number in range(1, max_rounds + 1):

        print(f"\n--- Round {round_number} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        messages.append(
            message.model_dump(exclude_none=True)
        )

        # No tool calls: agent has finished
        if not message.tool_calls:
            final_answer = message.content or ""
            completed = bool(final_answer.strip())

            print("Final answer:", final_answer)
            break

        # Execute requested tools
        for call in message.tool_calls:

            tool_name = call.function.name

            try:
                arguments = json.loads(
                    call.function.arguments
                )

                if not isinstance(arguments, dict):
                    raise ValueError(
                        "Tool arguments must be an object"
                    )

                tool_function = AVAILABLE_TOOLS.get(
                    tool_name
                )

                if tool_function is None:
                    result = {
                        "error": f"Unknown tool: {tool_name}"
                    }
                else:
                    result = tool_function(**arguments)

            except (ValueError, TypeError, KeyError) as error:
                arguments = {
                    "raw": call.function.arguments
                }
                result = {"error": str(error)}

            # Record the actual execution
            trace.append({
                "round": round_number,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            })

            print("Tool:", tool_name)
            print("Arguments:", arguments)
            print("Result:", result)

            # Return tool result to the LLM
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return {
        "final_answer": final_answer,
        "trace": trace,
        "completed": completed,
    }


# ==========================================
# 6. Golden Dataset
# ==========================================

TEST_CASES = [
    {
        "name": "Product 101 - 15 percent discount",
        "input": (
            "What is the price of product 101 "
            "after a 15% discount?"
        ),
        "expected_calls": [
            {
                "tool": "get_product_price",
                "arguments": {"product_id": 101},
            },
            {
                "tool": "calculate_discount",
                "arguments": {
                    "price": 200,
                    "discount_percent": 15,
                },
            },
        ],
        "expected_final_price": 170,
    },
    {
        "name": "Product 102 - 20 percent discount",
        "input": (
            "What is the price of product 102 "
            "after a 20% discount?"
        ),
        "expected_calls": [
            {
                "tool": "get_product_price",
                "arguments": {"product_id": 102},
            },
            {
                "tool": "calculate_discount",
                "arguments": {
                    "price": 50,
                    "discount_percent": 20,
                },
            },
        ],
        "expected_final_price": 40,
    },
    {
        "name": "Product not found",
        "input": "What is the price of product 999?",
        "expected_calls": [
            {
                "tool": "get_product_price",
                "arguments": {"product_id": 999},
            },
        ],
        "expected_final_price": None,
    },
]


# ==========================================
# 7. Deterministic Evaluator
# ==========================================

def evaluate(test_case, agent_result):

    trace = agent_result["trace"]

    actual_calls = [
        {
            "tool": step["tool"],
            "arguments": step["arguments"],
        }
        for step in trace
    ]

    expected_calls = test_case["expected_calls"]

    # Checks both tool names, order and arguments
    calls_passed = actual_calls == expected_calls

    # Check calculated tool result
    expected_price = test_case["expected_final_price"]

    discount_steps = [
        step
        for step in trace
        if step["tool"] == "calculate_discount"
    ]

    if expected_price is None:
        price_passed = not discount_steps
    elif len(discount_steps) == 1:
        actual_price = discount_steps[0]["result"].get(
            "final_price"
        )

        price_passed = (
            isinstance(actual_price, (int, float))
            and abs(actual_price - expected_price) < 0.001
        )
    else:
        price_passed = False

    completed = agent_result["completed"]

    passed = (
        calls_passed
        and price_passed
        and completed
    )

    return {
        "passed": passed,
        "calls_passed": calls_passed,
        "price_passed": price_passed,
        "completed": completed,
        "actual_calls": actual_calls,
    }


# ==========================================
# 8. Run Evaluations
# ==========================================

def main():

    print("Model:", MODEL)

    passed_count = 0

    for test_case in TEST_CASES:

        print("\n" + "=" * 50)
        print("TEST:", test_case["name"])

        try:
            agent_result = run_agent(test_case["input"])

            evaluation = evaluate(
                test_case,
                agent_result,
            )

        except Exception as error:
            print("ERROR:", error)
            print("Test could not be completed.")
            continue

        if evaluation["passed"]:
            passed_count += 1
            print("\nPASS")
        else:
            print("\nFAIL")

        print(
            "Tool sequence and arguments:",
            evaluation["calls_passed"],
        )

        print(
            "Tool result correct:",
            evaluation["price_passed"],
        )

        print(
            "Agent completed:",
            evaluation["completed"],
        )

        if not evaluation["passed"]:
            print(
                "Expected calls:",
                test_case["expected_calls"],
            )
            print(
                "Actual calls:",
                evaluation["actual_calls"],
            )

    print("\n" + "=" * 50)
    print(f"Passed: {passed_count}/{len(TEST_CASES)}")


if __name__ == "__main__":
    main()
