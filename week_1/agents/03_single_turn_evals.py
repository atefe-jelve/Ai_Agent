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


# 3. Define tools available to the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
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
            "name": "divide",
            "description": "Divide two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
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
                    "a": {"type": "number"},
                    "b": {"type": "number"},
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
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


# 4. Create the evaluation dataset
test_cases = [
    {
        "input": "What is 12 multiplied by 8?",
        "expected_tool": "multiply",
    },
    # {
    #     "input": "What is 12 divide by 8?",
    #     "expected_tool": "divide",
    # },
    {
        "input": "What is 20 plus 15?",
        "expected_tool": "add",
    },
    {
        "input": "What is the weather in Berlin?",
        "expected_tool": "get_weather",
    },
    {
        "input": "Hello, how are you?",
        "expected_tool": None,
    },
        {
        "input": "Can you help me with these numbers: 12 and 8?",
        "expected_tool": None,
    },
]


# 5. Ask the model to select a tool
def predict_tool(user_input: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Choose the appropriate tool when needed. "
                    "Use multiply for multiplication, add for addition, divide for dividing"
                    "and get_weather for weather requests. "
                    "For greetings or requests that do not need a tool, "
                    "respond directly without calling a tool."
                ),
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message

    if not message.tool_calls:
        return {
            "selected_tool": None,
            "arguments": None,
            "response": message.content,
        }

    selected_calls = [
        {
            "name": call.function.name,
            "arguments": json.loads(call.function.arguments),
        }
        for call in message.tool_calls
    ]

    return {
        "selected_tool": selected_calls[0]["name"],
        "arguments": selected_calls[0]["arguments"],
        "response": message.content,
        "tool_calls": selected_calls,
    }


# 6. Run evaluations
results = []

for test_case in test_cases:
    user_input = test_case["input"]
    expected_tool = test_case["expected_tool"]

    try:
        prediction = predict_tool(user_input)

        selected_tool = prediction["selected_tool"]

        passed = selected_tool == expected_tool

        result = {
            "input": user_input,
            "expected_tool": expected_tool,
            "selected_tool": selected_tool,
            "passed": passed,
        }

        results.append(result)

        status = "PASS" if passed else "FAIL"

        print(f"\n[{status}] {user_input}")
        print(f"Expected: {expected_tool}")
        print(f"Selected: {selected_tool}")

        if prediction.get("arguments") is not None:
            print(
                "Arguments:",
                prediction["arguments"],
            )

        if not passed:
            print(
                "Full tool calls:",
                prediction.get("tool_calls", []),
            )

    except Exception as error:
        print(f"\n[ERROR] {user_input}")
        print(error)


# 7. Calculate accuracy
if results:
    passed_count = sum(
        result["passed"]
        for result in results
    )

    total_count = len(results)

    accuracy = passed_count / total_count * 100

    print("\n" + "=" * 40)
    print(f"Passed: {passed_count}/{total_count}")
    print(f"Tool selection accuracy: {accuracy:.1f}%")
    