# Screenshot & Resolution Configuration

## Implementation (project)

- **Source**: `computer_use_demo/tools/screen_capture.py` → `get_screenshot()`
- **Executor**: `computer_use_demo/executor/showui_executor.py` → `_get_screen_resolution()` for coordinate mapping

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selected_screen` | 0 | Screen index (0=first by x, 1=second, …). Uses `screeninfo` (Win) / Quartz (macOS) / xrandr (Linux). |
| `resize` | True | If True, resize captured image to target dimensions. |
| `target_width` | 1920 | Output image width (px). |
| `target_height` | 1080 | Output image height (px). |

## Flow

1. **Capture**: Grab selected screen at native resolution (bbox from screeninfo/Quartz/xrandr).
2. **Resize**: `screenshot.resize((target_width, target_height))` → output 1920×1080 PNG.
3. **Execute**: Executor uses `screen_bbox` (native resolution) to map 0–1 coords to pixels: `(x * width, y * height)`.

## Coordinate Mapping

- **Model input**: 1920×1080 image (logical size; display scaling does not change this).
- **Model output**: RELATIVE coordinates `[x, y]` in 0-1 range. To estimate: divide pixel position by screenshot width/height. (Screenshot is 1920×1080.)
- **Executor**: `coord_px = (int(x * screen_width), int(y * screen_height))` using native screen bbox.

So `[0.5, 0.5]` → center of the actual screen, regardless of native resolution.

## Script Defaults

`screenshot.py` calls:

```python
get_screenshot(selected_screen=0, resize=True, target_width=1920, target_height=1080)
```

To change resolution, edit the script or extend it with env vars / CLI args.
