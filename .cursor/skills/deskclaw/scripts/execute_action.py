#!/usr/bin/env python3
"""
Execute a single Computer Use action. Run from deskclaw project root.

Usage:
    python execute_action.py '{"action":"CLICK","position":[0.5,0.5]}'
    echo '{"action":"INPUT","value":"hello"}' | python execute_action.py

Accepts JSON from argv[1] or stdin.
"""
import ast
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from computer_use_demo.executor.showui_executor import ShowUIExecutor
from computer_use_demo.tools.computer import ComputerTool
from computer_use_demo.tools import ToolCollection

# Mock callbacks for standalone execution
def _noop_output(_, **kwargs):
    pass

def _noop_tool(result, _):
    if result.error:
        print(result.error, file=sys.stderr)
    elif result.output:
        print(result.output)

def main():
    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        raw = sys.stdin.read().strip()

    try:
        action_item = json.loads(raw) if raw.startswith("{") else ast.literal_eval(raw)
    except Exception as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(action_item, dict) and "action" in action_item:
        action_item = [action_item]

    executor = ShowUIExecutor(
        output_callback=_noop_output,
        tool_output_callback=_noop_tool,
        selected_screen=0,
    )

    # ShowUIExecutor expects content as Python literal (ast.literal_eval)
    if not isinstance(action_item, list):
        action_item = [action_item]
    content = repr(action_item)
    response = {"content": content, "role": "assistant"}
    messages = []

    try:
        list(executor(response, messages))
        print("OK")
    except Exception as e:
        print(f"Execute error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
