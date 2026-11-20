import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


# Load the original shell agent
AGENT_PATH = Path(__file__).parent / "08_shell_tool.py"

spec = importlib.util.spec_from_file_location(
    "shell_agent",
    AGENT_PATH,
)

agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


# ==========================================
# Test 1: Allowed Operation
# ==========================================

def test_allowed_operation():

    result = agent.run_shell("python_version")

    assert result["success"] is True
    assert result["exit_code"] == 99
    assert result["operation"] == "python_version"


# ==========================================
# Test 2: Forbidden Operation
# ==========================================

def test_forbidden_operation():

    result = agent.run_shell("delete_everything")

    assert result["success"] is False
    assert result["error"] == "Operation is not allowed"


# ==========================================
# Test 3: Timeout with Mock
# ==========================================

def test_timeout():

    with patch.object(
        agent.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(
            cmd=["/usr/bin/ls", "-la"],
            timeout=5,
        ),
    ):

        result = agent.run_shell("list_files")

    assert result["success"] is False
    assert result["error"] == "Command timed out"