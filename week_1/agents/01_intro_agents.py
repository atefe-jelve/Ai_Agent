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


# 2. Configure the AI model client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = os.getenv(
    "OPENROUTER_MODEL",
    "poolside/laguna-s-2.1:free",
)


# 3. Define the agent's instructions
system_message = {
    "role": "system",
    "content": (
        "You are a helpful AI assistant teaching the user "
        "about AI agents and Python programming. "
        "Answer clearly and keep explanations beginner-friendly."
    ),
}


# 4. Initialize conversation memory
messages = [system_message]


# 5. Start the conversation loop
print("AI Agent started!")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Agent stopped.")
        break

    if not user_input:
        continue

    # Save the user's message in memory
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    try:
        # 6. Ask the model to generate a response
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        assistant_message = response.choices[0].message

        if not assistant_message.content:
            print(
                "Agent: The model returned no text response.\n"
            )
            messages.pop()
            continue

        answer = assistant_message.content

        # Save the assistant's response in memory
        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        print(f"\nAgent: {answer}\n")

    except Exception as error:
        # Remove the unsatisfied user message from memory
        messages.pop()

        print(f"\nRequest failed: {error}\n")
        