#!/usr/bin/env python3
"""
Capture screenshot. Run from deskclaw project root.
Prints the saved file path to stdout.
"""
import sys
from pathlib import Path

# Add project root for imports (deskclaw/ from .cursor/skills/deskclaw/scripts/)
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from computer_use_demo.tools.screen_capture import get_screenshot

if __name__ == "__main__":
    _, path = get_screenshot(selected_screen=0, resize=True, target_width=1920, target_height=1080)
    print(str(path.resolve()))
