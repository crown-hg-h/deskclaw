import json
import asyncio
import platform
from collections.abc import Callable
from datetime import datetime
from enum import Enum
try:
    from enum import StrEnum  # Python 3.11+
except ImportError:
    class StrEnum(str, Enum):
        pass
from typing import Any, cast, Dict, Callable

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import TextBlock, ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam

from computer_use_demo.tools.screen_capture import get_screenshot
from computer_use_demo.gui_agent.llm_utils.llm_utils import extract_data, encode_image
from computer_use_demo.tools.colorful_text import colorful_text_showui, colorful_text_vlm

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Windows system with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""

MODEL_TO_HF_PATH = {
    "qwen-vl-7b-instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2-vl-2b-instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2.5-vl-3b-instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
}


class LocalVLMPlanner:
    def __init__(
        self,
        model: str, 
        provider: str, 
        system_prompt_suffix: str, 
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        selected_screen: int = 0,
        print_usage: bool = True,
        device: torch.device = torch.device("cpu"),
        target_width: int = 1920,
        target_height: int = 1080,
    ):
        self.device = device
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1344 * 28 * 28
        self.model_name = model
        if model in MODEL_TO_HF_PATH:
            self.hf_path = MODEL_TO_HF_PATH[model]
        else:
            raise ValueError(f"Model {model} not supported for local VLM planner")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.hf_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.hf_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )
        
        self.provider = provider
        self.system_prompt_suffix = system_prompt_suffix
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.selected_screen = selected_screen
        self.output_callback = output_callback
        self.target_width = target_width
        self.target_height = target_height
        self.system_prompt = self._get_system_prompt() + self.system_prompt_suffix

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0

           
    def __call__(self, messages: list):
        
        # drop looping actions msg, byte image etc
        planner_messages = _message_filter_callback(messages)  
        print(f"filtered_messages: {planner_messages}")

        # Take a screenshot
        screenshot, screenshot_path = get_screenshot(selected_screen=self.selected_screen, resize=True, target_width=self.target_width, target_height=self.target_height)
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)
        self.output_callback(f'Screenshot for {colorful_text_vlm}:\n<img src="data:image/png;base64,{image_base64}">',
                             sender="bot")
        
        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(screenshot_path)

        print(f"Sending messages to VLMPlanner: {planner_messages}")

        messages_for_processor = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            },
            {
                "role": "user",
                "content": [
                {"type": "image", "image": screenshot_path, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                {"type": "text", "text": f"Task: {''.join(planner_messages)}"}
            ],
        }]
        
        text = self.processor.apply_chat_template(
            messages_for_processor, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages_for_processor)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        vlm_response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"VLMPlanner response: {vlm_response}")
        
        vlm_response_json = extract_data(vlm_response, "json")

        # vlm_plan_str = '\n'.join([f'{key}: {value}' for key, value in json.loads(response).items()])
        vlm_plan_str = ""
        for key, value in json.loads(vlm_response_json).items():
            if key == "Thinking":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'
        
        self.output_callback(f"{colorful_text_vlm}:\n{vlm_plan_str}", sender="bot")
        
        return vlm_response_json


    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)
        

    def reformat_messages(self, messages: list):
        pass

    def _get_system_prompt(self):
        os_name = platform.system()
        return f"""
You are a GUI automation agent controlling a {os_name} computer.
You see a screenshot of the current screen state, and you must decide the SINGLE next action to perform.
You can only interact with the desktop GUI (no terminal or application menu access).

You may receive history of previous plans and actions. Use them to understand what has already been tried.

=== COORDINATE SYSTEM ===
All positions use RELATIVE coordinates [x, y] where:
- x = horizontal position, 0.0 = left edge, 1.0 = right edge
- y = vertical position, 0.0 = top edge, 1.0 = bottom edge
- Example: [0.5, 0.5] = exact center of screen
To estimate: find the element's pixel position in the screenshot, then divide by screenshot width/height.

=== AVAILABLE ACTIONS (one per response) ===

1. CLICK - Single click at a position (left or right button)
   Required: position [x, y]
   Optional: value "left" (default) or "right"
   Example: {{"action": "CLICK", "value": null, "position": [0.5, 0.08]}}
   Example: {{"action": "CLICK", "value": "right", "position": [0.5, 0.08]}}

2. DOUBLE_CLICK - Double click at a position (left or right button)
   Required: position [x, y]
   Optional: value "left" (default) or "right"
   Example: {{"action": "DOUBLE_CLICK", "value": null, "position": [0.5, 0.3]}}

3. INPUT - Type text (at current cursor position, CLICK the field first)
   Required: value (text string). ASCII only.
   Example: {{"action": "INPUT", "value": "hello world", "position": null}}

4. HOVER - Move mouse without clicking
   Required: position [x, y]
   Example: {{"action": "HOVER", "value": null, "position": [0.3, 0.1]}}

# 5. PRESS - 已注释，用途有限
#    Example: {{"action": "PRESS", "value": null, "position": [0.5, 0.5]}}

6. DRAG - Drag from start to end position
   Required: position [[x1, y1], [x2, y2]] - start and end (0-1 range)
   Use for: dragging files, reordering, selecting text
   Example: {{"action": "DRAG", "value": null, "position": [[0.2, 0.3], [0.6, 0.5]]}}

7. ENTER - Press Enter key
   Example: {{"action": "ENTER", "value": null, "position": null}}

8. ESCAPE - Press Escape key
   Example: {{"action": "ESCAPE", "value": null, "position": null}}

9. KEY - Press key or key combination (use + for combos)
   Required: value (e.g. "ctrl+c", "alt+f4", "tab", "backspace", "ctrl+a")
   Example: {{"action": "KEY", "value": "ctrl+w", "position": null}}

10. SCROLL - Scroll page. Required: value ("up" or "down")
   Example: {{"action": "SCROLL", "value": "down", "position": null}}

11. None - Task completed.
   When action is None, also output "summary": brief Chinese steps (每步一行，1. xxx 2. xxx).
   Example: {{"action": "None", "value": null, "position": null, "summary": "1. 点击地址栏\\n2. 输入网址\\n3. 按 Enter"}}

12. FAIL - Task failed, terminate immediately.
   Use when: the SAME step has been tried 3 times in a row without completing. Check history - if same action repeated 3 times, output FAIL.
   Example: {{"action": "FAIL", "value": "Reason: same step repeated 3 times", "position": null}}

=== OUTPUT FORMAT (JSON only, no markdown) ===
{{
    "Thinking": "brief reasoning",
    "action": "CLICK|DOUBLE_CLICK|INPUT|HOVER|DRAG|ENTER|ESCAPE|KEY|SCROLL|None|FAIL",
    "value": "text/key or null",
    "position": [x, y] or null,
    "summary": "Only when action is None: brief Chinese steps (每步一行)"
}}

=== RULES ===
1. ONE action per response.
2. Coordinates in 0-1 range.
3. CLICK the field before INPUT.
4. Use KEY for shortcuts.
5. If the SAME step has been tried 3 times in a row without completing, use action "FAIL" to terminate.
6. Output ONLY JSON.
""" 

def _message_filter_callback(messages):
    filtered_list = []
    try:
        for msg in messages:
            if msg.get('role') in ['user']:
                if not isinstance(msg["content"], list):
                    msg["content"] = [msg["content"]]
                if isinstance(msg["content"][0], TextBlock):
                    filtered_list.append(str(msg["content"][0].text))  # User message
                elif isinstance(msg["content"][0], str):
                    filtered_list.append(msg["content"][0])  # User message
                else:
                    print("[_message_filter_callback]: drop message", msg)
                    continue                
                
            else:
                print("[_message_filter_callback]: drop message", msg)
                continue
            
    except Exception as e:
        print("[_message_filter_callback]: error", e)
                
    return filtered_list