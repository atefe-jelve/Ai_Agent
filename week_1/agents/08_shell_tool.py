
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=30.0,
    max_retries=1,
)


# ==========================================
# 2. Restricted Workspace
# ==========================================

WORKSPACE = Path(__file__).resolve().parent / "shell_workspace"
WORKSPACE.mkdir(exist_ok=True)


# ==========================================
# 3. Allowed Commands
# ==========================================

ALLOWED_OPERATIONS = {
    "python_version": [sys.executable, "--version"],
    "show_directory": ["/usr/bin/pwd"],
    "list_files": ["/usr/bin/ls", "-la"],
}


# ==========================================
# 4. Shell Executor
# ==========================================

def run_shell(operation: str) -> dict:
    """
    Execute a predefined, read-only command.
    """

    if operation not in ALLOWED_OPERATIONS:
        return {
            "success": False,
            "error": "Operation is not allowed",
        }

    command = ALLOWED_OPERATIONS[operation]

    try:
        result = subprocess.run(
            command,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
            check=False,
        )

        return {
            "success": result.returncode == 0,
            "operation": operation,
            "exit_code": result.returncode,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:1000],
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out",
        }

    except OSError as error:
        return {
            "success": False,
            "error": str(error),
        }


# ==========================================
# 5. Tool Schema
# ==========================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run one predefined read-only command. "
                "Allowed operations: python_version, "
                "show_directory, list_files. "
                "Arbitrary shell commands are not supported."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "python_version",
                            "show_directory",
                            "list_files",
                        ],
                    }
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
        },
    }
]


# ==========================================
# 6. Agent Loop
# ==========================================

def run_agent(user_input: str):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Linux terminal assistant. "
                "You can use run_shell for a limited set "
                "of read-only operations. "
                "Never claim to execute unsupported commands. "
                "Never invent command outputs. "
                "Explain unsupported requests clearly."
            ),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    MAX_ROUNDS = 4

    for round_number in range(1, MAX_ROUNDS + 1):

        print(f"\n--- Agent Round {round_number} ---")

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

        if not message.tool_calls:
            print("\nAgent:", message.content or "")
            return

        for call in message.tool_calls:

            tool_name = call.function.name

            try:
                arguments = json.loads(
                    call.function.arguments
                )

                if not isinstance(arguments, dict):
                    raise ValueError("Invalid arguments")

                if tool_name != "run_shell":
                    result = {"error": "Unknown tool"}
                else:
                    result = run_shell(**arguments)

            except (ValueError, TypeError) as error:
                result = {"error": str(error)}

            print("Tool:", tool_name)
            print("Result:", result)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                ),
            })

    print("Maximum agent rounds reached.")


# ==========================================
# 7. Chat Loop
# ==========================================

def main():

    print("Restricted Shell Agent started!")
    print("Workspace:", WORKSPACE)
    print("Type 'exit' to stop.\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            run_agent(user_input)

        except Exception as error:
            print("Agent error:", error)


if __name__ == "__main__":
    main()
