---
name: deskclaw
description: "Computer Use workflow for desktop GUI automation. Use when the user asks to control the computer, automate desktop tasks, click/type/scroll on screen, or perform GUI actions. The main model acts as Planner: receive screenshot, output action JSON, then execute."
---

# DeskClaw — Computer Use Skill

## Overview

This skill enables the main model to perform desktop GUI automation. **You (the model) are the Planner.** The workflow: (1) capture screenshot, (2) you analyze the image and output one action in JSON, (3) execute the action, (4) repeat until task complete.

## Workflow

```
User task → Screenshot (script) → You see image, output action JSON → Execute (script) → repeat or done
```

**Single iteration:**

1. Run `python .cursor/skills/deskclaw/scripts/screenshot.py` — saves screenshot, prints path to stdout
2. Read the screenshot image (use Read tool on the path)
3. Analyze the screen, decide the next action
4. Output exactly one action as JSON (see [references/action-format.md](references/action-format.md))
5. If action is `None`, `FAIL`, or `ASK_USER` → do not execute; handle (stop / ask user). Otherwise run `python .cursor/skills/deskclaw/scripts/execute_action.py '<json>'`
6. If task complete → stop. Otherwise → go to step 1

## Screenshot & Resolution

- **Output size**: 1920×1080 (default). Captured from `get_screenshot(resize=True, target_width=1920, target_height=1080)`.
- **Multi-screen**: `selected_screen=0` = first screen (by x). Screenshot covers only the selected screen.

### COORDINATE SYSTEM（与项目 planner 一致）

All positions use **RELATIVE coordinates** [x, y] where:
- x = horizontal position, 0.0 = left edge, 1.0 = right edge
- y = vertical position, 0.0 = top edge, 1.0 = bottom edge
- Example: [0.5, 0.5] = exact center of screen
- Example: [0.0, 0.0] = top-left corner
- Example: [1.0, 1.0] = bottom-right corner
To estimate: find the element's pixel position in the screenshot, then divide by screenshot width/height.

See [references/resolution.md](references/resolution.md) for implementation details.

## Action Format (Summary)

Output a single JSON object per step:

```json
{"Thinking": "reasoning", "action": "CLICK|INPUT|KEY|...", "value": "text or null", "position": [x,y] or null}
```

- **Coordinates**: RELATIVE coordinates [x, y] in 0-1 range (divide pixel position by screenshot width/height)
- **One action per step** — never combine multiple actions
- **CLICK** before **INPUT** when typing into a field

Full spec: [references/action-format.md](references/action-format.md)

## Scripts

| Script | Usage | Output |
|--------|-------|--------|
| `scripts/screenshot.py` | `python scripts/screenshot.py` | Path to saved PNG (e.g. `./tmp/outputs/screenshot_xxx.png`) |
| `scripts/execute_action.py` | `python scripts/execute_action.py '{"action":"CLICK","position":[0.5,0.5]}'` | Executes action, prints result |

Run from the **deskclaw project root** (where `computer_use_demo/` exists). Requires `pip install -r requirements.txt`.

## Termination

| action | Meaning |
|--------|---------|
| `None` | Task complete. Include `"summary"` with brief steps. |
| `FAIL` | Same step tried 3× without progress — stop. |
| `ASK_USER` | Uncertain — ask user, then continue. |

## Rules

1. Output only valid JSON. No markdown fences, no extra text.
2. **Coordinates MUST be in 0-1 range** (RELATIVE coordinates). Carefully estimate from the screenshot.
3. Before INPUT, always CLICK the target field first.
4. Use KEY for shortcuts (ctrl+c, alt+f4, etc.).
5. If the same action (type, value, position) repeats 3 times, output FAIL.
