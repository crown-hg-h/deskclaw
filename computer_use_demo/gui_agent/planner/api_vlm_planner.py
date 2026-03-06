import json
import re
import asyncio
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast, Dict, Callable

import os
from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import TextBlock, ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam

from computer_use_demo.tools.screen_capture import get_screenshot
from computer_use_demo.gui_agent.llm_utils.oai import run_oai_interleaved, run_ssh_llm_interleaved, run_ollama_interleaved, run_openai_compatible_interleaved
from computer_use_demo.gui_agent.llm_utils.llm_utils import extract_data, encode_image
from computer_use_demo.tools.colorful_text import colorful_text_showui, colorful_text_vlm
from computer_use_demo.tools.logger import logger


def _parse_plan_json_fallback(raw: str) -> dict:
    """当 json.loads 失败时（如 Thinking 内含未转义引号），用正则提取 action/value/position/summary"""
    out = {"Thinking": "", "action": None, "value": None, "position": None, "summary": ""}
    # action: "action": "CLICK" 等
    m = re.search(r'"action"\s*:\s*"([^"]*)"', raw)
    if m:
        out["action"] = m.group(1).strip() or None
    # value: "value": null 或 "value": "xxx"
    m = re.search(r'"value"\s*:\s*(null|"[^"]*")', raw)
    if m:
        v = m.group(1)
        out["value"] = None if v == "null" else v.strip('"')
    # position: "position": [0.288, 0.259] 或 [[x1,y1],[x2,y2]]（DRAG）或 null
    m = re.search(r'"position"\s*:\s*(\[\[[\d.,\s]+\],\s*\[[\d.,\s]+\]\]|\[[\d.,\s]+\]|null)', raw)
    if m:
        p = m.group(1)
        if p != "null":
            try:
                out["position"] = json.loads(p)
            except json.JSONDecodeError:
                pass
    # Thinking: 取 "Thinking": " 后到 ", "action" 前的内容（贪婪匹配以应对内含引号）
    m = re.search(r'"Thinking"\s*:\s*"(.*)"\s*,\s*"action"', raw, re.DOTALL)
    if m:
        out["Thinking"] = m.group(1).replace('\\"', '"')
    # summary: 可选，action=None 时应有
    m = re.search(r'"summary"\s*:\s*"(.*?)"(?=\s*[}\]])', raw, re.DOTALL)
    if m:
        out["summary"] = m.group(1).replace('\\n', '\n').replace('\\"', '"')
    return out


class APIVLMPlanner:
    def __init__(
        self,
        model: str, 
        provider: str, 
        system_prompt_suffix: str, 
        api_key: str,
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        selected_screen: int = 0,
        print_usage: bool = True,
        target_width: int = 1920,
        target_height: int = 1080,
    ):
        if model == "gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "gpt-4o-mini":
            self.model = "gpt-4o-mini"  # "gpt-4o-mini"
        elif model == "qwen2-vl-max":
            self.model = "qwen2-vl-max"
        elif model == "qwen2-vl-2b (ssh)":
            self.model = "Qwen2-VL-2B-Instruct"
        elif model == "qwen2-vl-7b (ssh)":
            self.model = "Qwen2-VL-7B-Instruct"
        elif model == "qwen2.5-vl-7b (ssh)":
            self.model = "Qwen2.5-VL-7B-Instruct"
        elif model == "qwen2.5-vl (ollama)":
            self.model = "qwen2.5vl:7b"  # Ollama model name
        elif model == "Kimi-K2.5 (Azure)":
            self.model = "Kimi-K2.5"
        elif model == "Custom (OpenAI)":
            self.model = "custom"
        else:
            raise ValueError(f"Model {model} not supported")
        
        self.provider = provider
        self.system_prompt_suffix = system_prompt_suffix
        self.api_key = api_key
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

        if self.only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        # Take a screenshot
        screenshot, screenshot_path = get_screenshot(selected_screen=self.selected_screen, resize=True, target_width=self.target_width, target_height=self.target_height)
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)
        self.output_callback(f'Screenshot for {colorful_text_vlm}:\n<img src="data:image/png;base64,{image_base64}">',
                             sender="bot")
        
        # if isinstance(planner_messages[-1], dict):
        #     if not isinstance(planner_messages[-1]["content"], list):
        #         planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
        #     planner_messages[-1]["content"].append(screenshot_path)
        # elif isinstance(planner_messages[-1], str):
        #     planner_messages[-1] = {"role": "user", "content": [{"type": "text", "text": planner_messages[-1]}]}
        
        # append screenshot
        # planner_messages.append({"role": "user", "content": [{"type": "image", "image": screenshot_path}]})
        
        planner_messages.append(screenshot_path)
        
        print(f"Sending messages to VLMPlanner: {planner_messages}")

        if self.model == "gpt-4o-2024-11-20":
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=self.system_prompt,
                llm=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=0,
            )
            print(f"oai token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.15 / 1000000)  # https://openai.com/api/pricing/
            
        elif self.model == "qwen2-vl-max":
            from computer_use_demo.gui_agent.llm_utils.qwen import run_qwen
            vlm_response, token_usage = run_qwen(
                messages=planner_messages,
                system=self.system_prompt,
                llm=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=0,
            )
            print(f"qwen token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.02 / 7.25 / 1000)  # 1USD=7.25CNY, https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api
        elif self.model == "Kimi-K2.5":
            # Azure/custom OpenAI: api_key stores "api_base|||api_key"
            raw_key = self.api_key or ""
            if "|||" in raw_key:
                api_base, api_key = raw_key.strip().split("|||", 1)
                api_base, api_key = api_base.strip(), api_key.strip()
            else:
                raise ValueError(
                    f"Kimi-K2.5 需要 API Base URL 和 API Key，用 ||| 分隔。\n"
                    f"格式: https://xxx.azure.com/openai/v1|||你的API_KEY\n"
                    f"当前输入长度={len(raw_key)}，缺少 '|||' 分隔符"
                )
            if not api_key or len(api_key) < 20:
                raise ValueError(f"API Key 看起来不完整（长度={len(api_key)}），请检查输入")
            print(f"[Kimi-K2.5] api_base={api_base[:50]}..., key_len={len(api_key)}")
            vlm_response, token_usage = run_openai_compatible_interleaved(
                messages=planner_messages,
                system=self.system_prompt,
                llm=self.model,
                api_base=api_base,
                api_key=api_key,
                max_tokens=self.max_tokens,
                temperature=0,
            )
        elif self.model == "custom":
            # Custom OpenAI-compatible: api_key stores "api_base|||api_key|||model_name"
            raw_key = self.api_key or ""
            if "|||" not in raw_key:
                raise ValueError(
                    f"Custom 需要 API Base URL 和 API Key，用 ||| 分隔。\n"
                    f"格式: https://api.xxx.com/v1|||sk-xxx|||gpt-4o (模型名可选，默认 gpt-4o)\n"
                    f"当前输入长度={len(raw_key)}，缺少 '|||' 分隔符"
                )
            parts = [p.strip() for p in raw_key.strip().split("|||")]
            api_base = parts[0]
            api_key = parts[1] if len(parts) > 1 else ""
            model_name = parts[2] if len(parts) > 2 else "gpt-4o"
            if not api_base or not api_key:
                raise ValueError("Custom 需要有效的 API Base URL 和 API Key")
            print(f"[Custom] api_base={api_base[:50]}..., model={model_name}, key_len={len(api_key)}")
            vlm_response, token_usage = run_openai_compatible_interleaved(
                messages=planner_messages,
                system=self.system_prompt,
                llm=model_name,
                api_base=api_base,
                api_key=api_key,
                max_tokens=self.max_tokens,
                temperature=0,
            )
        elif self.model == "qwen2.5vl:7b":
            # Ollama: api_key stores the api_base URL
            api_base = self.api_key.strip() or "http://localhost:11434"
            if not api_base.startswith("http"):
                api_base = "http://" + api_base
            vlm_response, token_usage = run_ollama_interleaved(
                messages=planner_messages,
                system=self.system_prompt,
                llm=self.model,
                api_base=api_base,
                max_tokens=self.max_tokens,
                temperature=0.01,
            )
        elif "Qwen" in self.model:
            # 从api_key中解析host和port
            try:
                ssh_host, ssh_port = self.api_key.split(":")
                ssh_port = int(ssh_port)
            except ValueError:
                raise ValueError("Invalid SSH connection string. Expected format: host:port")
                
            vlm_response, token_usage = run_ssh_llm_interleaved(
                messages=planner_messages,
                system=self.system_prompt,
                llm=self.model,
                ssh_host=ssh_host,
                ssh_port=ssh_port,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Model {self.model} not supported")
            
        print(f"VLMPlanner response: {vlm_response}")
        
        if self.print_usage:
            print(f"VLMPlanner total token usage so far: {self.total_token_usage}. Total cost so far: $USD{self.total_cost:.5f}")
        
        vlm_response_json = extract_data(vlm_response, "json")

        try:
            plan_data = json.loads(vlm_response_json)
        except json.JSONDecodeError as e:
            logger.warning("Planner JSON 解析失败，使用正则后备解析: %s", e)
            plan_data = _parse_plan_json_fallback(vlm_response_json)

        vlm_plan_str = ""
        for key, value in plan_data.items():
            if key == "Thinking":
                vlm_plan_str += f"{value}"
            else:
                vlm_plan_str += f"\n{key}: {value}"

        self.output_callback(f"{colorful_text_vlm}:\n{vlm_plan_str}", sender="bot")

        return json.dumps(plan_data, ensure_ascii=False)


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
- Example: [0.0, 0.0] = top-left corner
- Example: [1.0, 1.0] = bottom-right corner
To estimate: find the element's pixel position in the screenshot, then divide by screenshot width/height.

=== AVAILABLE ACTIONS (one per response) ===

1. CLICK - Single click at a position (left or right button)
   Required: position [x, y]
   Optional: value "left" (default) or "right" - which mouse button to use
   Use for: buttons, links, checkboxes, context menus (right), focusing input fields
   Example: {{"action": "CLICK", "value": null, "position": [0.5, 0.08]}}
   Example: {{"action": "CLICK", "value": "right", "position": [0.5, 0.08]}}

2. DOUBLE_CLICK - Double click at a position (left or right button)
   Required: position [x, y]
   Optional: value "left" (default) or "right"
   Use for: selecting words, opening files/folders, editing in place
   Example: {{"action": "DOUBLE_CLICK", "value": null, "position": [0.5, 0.3]}}
   Example: {{"action": "DOUBLE_CLICK", "value": "right", "position": [0.5, 0.3]}}

3. INPUT - Type text (the text is typed at current cursor position)
   Required: value (the text string to type)
   Optional: position [x, y] (ignored, use CLICK first to focus the field)
   Use for: typing in text fields, search bars, address bars, etc.
   NOTE: You MUST CLICK the input field FIRST in a previous step before using INPUT.
   NOTE: INPUT only supports ASCII characters (a-z, A-Z, 0-9, symbols). For non-ASCII (Chinese, etc.), use the clipboard method: first CLICK the field, then use KEY with "ctrl+a" to select all, then INPUT the text.
   Example: {{"action": "INPUT", "value": "hello world", "position": null}}

4. HOVER - Move mouse to a position without clicking
   Required: position [x, y]
   Use for: hovering over menus to reveal submenus, tooltips
   Example: {{"action": "HOVER", "value": null, "position": [0.3, 0.1]}}

# 5. PRESS - 已注释，用途有限
#    Example: {{"action": "PRESS", "value": null, "position": [0.5, 0.5]}}

6. DRAG - Drag from start position to end position (left mouse button)
   Required: position [[x1, y1], [x2, y2]] - start and end coordinates (0-1 range)
   Use for: dragging files, reordering items, selecting text by dragging, scrollbars
   Example: {{"action": "DRAG", "value": null, "position": [[0.2, 0.3], [0.6, 0.5]]}}

7. ENTER - Press the Enter/Return key
   No position or value needed.
   Use for: confirming input, submitting forms, executing commands
   Example: {{"action": "ENTER", "value": null, "position": null}}

8. ESCAPE - Press the Escape key
   No position or value needed.
   Use for: closing dialogs, canceling operations, exiting menus
   Example: {{"action": "ESCAPE", "value": null, "position": null}}

9. KEY - Press a key or key combination
   Required: value (key name or combination with + separator)
   No position needed.
   Supported keys: enter, esc, tab, space, backspace, delete, up, down, left, right, home, end, pageup, pagedown, f1-f12
   Modifiers: ctrl, alt, shift, command (Mac) / win (Windows)
   Key combinations use + separator: "ctrl+c", "ctrl+v", "ctrl+a", "alt+f4", "command+q", "ctrl+shift+t"
   Use for: keyboard shortcuts, navigation, closing windows, copy/paste, select all, undo, etc.
   Example: {{"action": "KEY", "value": "ctrl+c", "position": null}}
   Example: {{"action": "KEY", "value": "alt+f4", "position": null}}
   Example: {{"action": "KEY", "value": "tab", "position": null}}

10. SCROLL - Scroll the page up or down
   Required: value ("up" or "down")
   No position needed. Scrolls one page at a time.
   Use for: scrolling web pages, long documents, lists
   Example: {{"action": "SCROLL", "value": "down", "position": null}}

11. ASK_USER - Pause and ask the user for clarification when uncertain
   Use when: you cannot determine the correct action from the screenshot alone (e.g. multiple similar buttons, ambiguous intent, need user to choose).
   Required: value (the question to ask the user, in Chinese or English)
   Example: {{"Thinking": "屏幕上有两个「确定」按钮，无法判断应点击哪一个。", "action": "ASK_USER", "value": "请告诉我应该点击左边还是右边的「确定」按钮？", "position": null}}
   The user's reply will be appended to the conversation and you will continue in the next step.

12. None - Task is completed, no more actions needed
   When outputting action "None", you MUST also output a "summary" field: a brief Chinese description of the steps you took (每步一行，如 1. 点击xxx 2. 输入xxx). This will be saved for future reference.
   Example: {{"Thinking": "任务已完成。", "action": "None", "value": null, "position": null, "summary": "1. 点击 Dock 栏的 WPS 图标\\n2. 点击文件菜单新建表格\\n3. 在第一列输入今天日期"}}

13. FAIL - Task failed, terminate immediately
   Use when: the SAME step has been tried 3 times in a row without completing the task. Check the history of previous plans - if you see the same action (same type, value, position) repeated 3 times, output FAIL to terminate.
   Example: {{"action": "FAIL", "value": "Reason: same click repeated 3 times without progress", "position": null}}

=== OUTPUT FORMAT ===
You MUST output a single JSON object (no markdown, no code fences, no extra text):
{{
    "Thinking": "Brief reasoning about current screen state and what to do next",
    "action": "CLICK|DOUBLE_CLICK|INPUT|HOVER|DRAG|ENTER|ESCAPE|KEY|SCROLL|ASK_USER|None|FAIL",
    "value": "text for INPUT/KEY, up/down for SCROLL, left/right for CLICK/DOUBLE_CLICK, or null",
    "position": [x, y] | [[x1,y1],[x2,y2]] for DRAG | null,
    "summary": "Only when action is None: brief Chinese steps (每步一行，1. xxx 2. xxx)"
}}

=== EXAMPLES ===
{{"Thinking": "I need to click the browser's address bar to type a URL. The address bar is at the top center.", "action": "CLICK", "value": null, "position": [0.5, 0.05]}}
{{"Thinking": "Double-click to select the filename for editing.", "action": "DOUBLE_CLICK", "value": null, "position": [0.4, 0.2]}}
{{"Thinking": "Now I'll type the URL.", "action": "INPUT", "value": "amazon.com", "position": null}}
{{"Thinking": "Press Enter to navigate.", "action": "ENTER", "value": null, "position": null}}
{{"Thinking": "I need to scroll down to see more content.", "action": "SCROLL", "value": "down", "position": null}}
{{"Thinking": "Close the current browser tab with keyboard shortcut.", "action": "KEY", "value": "ctrl+w", "position": null}}
{{"Thinking": "I need to close this window. I'll click the X button at top-right.", "action": "CLICK", "value": null, "position": [0.99, 0.01]}}
{{"Thinking": "Right-click to open context menu.", "action": "CLICK", "value": "right", "position": [0.5, 0.5]}}
{{"Thinking": "Drag the file icon to the folder.", "action": "DRAG", "value": null, "position": [[0.2, 0.4], [0.5, 0.6]]}}
{{"Thinking": "The task is done.", "action": "None", "value": null, "position": null, "summary": "1. 点击地址栏\\n2. 输入 amazon.com\\n3. 按 Enter 打开"}}

=== CRITICAL RULES ===
1. Output ONE action per response. Never combine multiple actions.
2. Coordinates MUST be in 0-1 range. Carefully estimate from the screenshot.
3. Before typing (INPUT), you MUST first CLICK the target field in a prior step.
4. Use KEY for keyboard shortcuts (closing windows, copy-paste, tab switching, etc.)
5. When a task is fully completed, use action "None".
6. If the SAME step (same action type, value, position) has been tried 3 times in a row without completing the task, use action "FAIL" to terminate and report failure. Do not keep retrying.
7. When uncertain (e.g. multiple similar buttons, ambiguous which to click), use action "ASK_USER" with value=your question. The user will reply and you continue in the next step.
8. Output ONLY valid JSON. No markdown code fences, no explanation outside JSON.
""" 

    

def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


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

            # elif msg.get('role') in ['assistant']:
            #     if isinstance(msg["content"][0], TextBlock):
            #         msg["content"][0] = str(msg["content"][0].text)
            #     elif isinstance(msg["content"][0], BetaTextBlock):
            #         msg["content"][0] = str(msg["content"][0].text)
            #     elif isinstance(msg["content"][0], BetaToolUseBlock):
            #         msg["content"][0] = str(msg['content'][0].input)
            #     elif isinstance(msg["content"][0], Dict) and msg["content"][0]["content"][-1]["type"] == "image":
            #         msg["content"][0] = f'<img src="data:image/png;base64,{msg["content"][0]["content"][-1]["source"]["data"]}">'
            #     else:
            #         print("[_message_filter_callback]: drop message", msg)
            #         continue
            #     filtered_list.append(msg["content"][0])  # User message
                
            else:
                print("[_message_filter_callback]: drop message", msg)
                continue
            
    except Exception as e:
        print("[_message_filter_callback]: error", e)
                
    return filtered_list