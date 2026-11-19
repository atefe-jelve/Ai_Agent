
import json
import os
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

# Allowed directory
WORKSPACE = Path(__file__).resolve().parent / "agent_workspace"
WORKSPACE.mkdir(exist_ok=True)

MAX_FILE_SIZE = 4096


# ==========================================
# 2. Path Validation
# ==========================================

def safe_path(filename: str) -> Path:
    """
    Only allow direct files inside agent_workspace.
    """

    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
    ):
        raise ValueError("Invalid filename")

    path = WORKSPACE / filename

    # Reject symlinks, including links outside the workspace
    if path.is_symlink():
        raise ValueError("Symbolic links are not allowed")

    if path.resolve().parent != WORKSPACE.resolve():
        raise ValueError("Access denied: outside workspace")

    return path


# ==========================================
# 3. File System Tools
# ==========================================

def list_files() -> dict:
    """List files in the allowed workspace."""

    files = []

    for path in WORKSPACE.iterdir():
        if path.is_file() and not path.is_symlink():
            files.append(path.name)

    return {
        "files": sorted(files),
    }


def read_file(filename: str) -> dict:
    """Read an existing UTF-8 text file."""

    path = safe_path(filename)

    # O_NOFOLLOW prevents following a final symlink
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(path, flags)

    try:
        with os.fdopen(fd, "rb") as file:
            content = file.read(MAX_FILE_SIZE + 1)
    except:
        raise

    if len(content) > MAX_FILE_SIZE:
        raise ValueError("File is too large")

    return {
        "filename": filename,
        "content": content.decode("utf-8"),
    }


def write_file(filename: str, content: str) -> dict:
    """Create a new UTF-8 text file."""

    path = safe_path(filename)

    if not isinstance(content, str):
        raise ValueError("Content must be text")

    data = content.encode("utf-8")

    if len(data) > MAX_FILE_SIZE:
        raise ValueError("Content exceeds size limit")

    # O_EXCL ensures an existing file is never overwritten
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW

    fd = os.open(path, flags, 0o600)

    with os.fdopen(fd, "wb") as file:
        file.write(data)

    return {
        "status": "created",
        "filename": filename,
        "bytes_written": len(data),
    }


def delete_file(filename: str) -> dict:
    """Delete an existing file only after user approval."""

    path = safe_path(filename)

    if not path.is_file():
        return {"error": "File not found"}

    print(f"\nDELETE REQUEST: {filename}")
    print("This action cannot be undone.")

    approval = input(
        f"Type DELETE {filename} to confirm: "
    ).strip()

    if approval != f"DELETE {filename}":
        return {
            "status": "cancelled",
            "filename": filename,
        }

    # Check again before deleting
    path = safe_path(filename)
    path.unlink()

    return {
        "status": "deleted",
        "filename": filename,
    }


# ==========================================
# 4. Tool Registry
# ==========================================

AVAILABLE_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "delete_file": delete_file,
}


# ==========================================
# 5. Tool Definitions for LLM
# ==========================================

def parameters(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the allowed workspace.",
            "parameters": parameters({}, []),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file in the workspace.",
            "parameters": parameters(
                {"filename": {"type": "string"}},
                ["filename"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a new text file in the workspace. "
                "Cannot overwrite existing files."
            ),
            "parameters": parameters(
                {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                ["filename", "content"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": (
                "Delete a file in the workspace. "
                "Requires explicit user confirmation."
            ),
            "parameters": parameters(
                {"filename": {"type": "string"}},
                ["filename"],
            ),
        },
    },
]


# ==========================================
# 6. Agent Loop
# ==========================================

def run_agent(user_input: str):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a file management assistant. "
                "Use the available tools for filesystem tasks. "
                "Only work inside the allowed workspace. "
                "Never claim a file operation succeeded "
                "unless a tool confirms success. "
                "Do not attempt to access other directories. "
                "If a tool fails, explain the error."
            ),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    MAX_ROUNDS = 5

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

            print("Selected tool:", tool_name)

            try:
                arguments = json.loads(
                    call.function.arguments
                )

                if not isinstance(arguments, dict):
                    raise ValueError("Invalid arguments")

                tool = AVAILABLE_TOOLS.get(tool_name)

                if tool is None:
                    result = {
                        "error": "Unknown tool"
                    }
                else:
                    result = tool(**arguments)

            except (
                ValueError,
                TypeError,
                OSError,
                UnicodeError,
            ) as error:
                result = {"error": str(error)}

            print("Tool result:", result)

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

    print("File System Agent started!")
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
