# Action Format for Computer Use

Output exactly **one** JSON object per step.

**COORDINATE SYSTEM** (same as project planner): All positions use RELATIVE coordinates [x, y] where x = horizontal (0.0=left, 1.0=right), y = vertical (0.0=top, 1.0=bottom). To estimate: find pixel position in screenshot, then divide by screenshot width/height. Coordinates MUST be in 0-1 range.

## Actions

| action | value | position | Example |
|--------|-------|----------|---------|
| CLICK | "left" / "right" / null | [x, y] | `{"action":"CLICK","value":null,"position":[0.5,0.08]}` |
| DOUBLE_CLICK | "left" / "right" / null | [x, y] | `{"action":"DOUBLE_CLICK","value":null,"position":[0.5,0.3]}` |
| INPUT | text to type | null | `{"action":"INPUT","value":"hello","position":null}` |
| HOVER | null | [x, y] | `{"action":"HOVER","value":null,"position":[0.3,0.1]}` |
| DRAG | null | [[x1,y1],[x2,y2]] | `{"action":"DRAG","value":null,"position":[[0.2,0.4],[0.5,0.6]]}` |
| ENTER | null | null | `{"action":"ENTER","value":null,"position":null}` |
| ESCAPE | null | null | `{"action":"ESCAPE","value":null,"position":null}` |
| KEY | key combo (e.g. "ctrl+c") | null | `{"action":"KEY","value":"ctrl+c","position":null}` |
| SCROLL | "up" / "down" | null | `{"action":"SCROLL","value":"down","position":null}` |
| ASK_USER | question string | null | `{"action":"ASK_USER","value":"点击左边还是右边？","position":null}` |
| None | null | null | `{"action":null,"value":null,"position":null,"summary":"1. 点击xxx\n2. 输入xxx"}` |
| FAIL | reason | null | `{"action":"FAIL","value":"same click 3x","position":null}` |

## Output Format

```json
{
  "Thinking": "Brief reasoning",
  "action": "CLICK",
  "value": null,
  "position": [0.5, 0.5],
  "summary": "Only when action is None"
}
```

## Rules

1. **One action per step** — never combine CLICK + INPUT in one response
2. **CLICK before INPUT** — always click the field first, then INPUT in the next step
3. **KEY for shortcuts** — ctrl+c, alt+f4, command+q (Mac), etc.
4. **None** = task complete, must include `summary` (Chinese steps)
5. **FAIL** = same step 3× without progress
6. **ASK_USER** = uncertain, ask user then continue
7. Output **only** valid JSON, no markdown fences
8. **Coordinates MUST be in 0-1 range** (RELATIVE coordinates). Divide pixel position by screenshot width/height.
