import ast
import asyncio
from typing import Any, Dict, cast, List, Union
from collections.abc import Callable
import uuid
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from computer_use_demo.tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult
from computer_use_demo.tools.colorful_text import colorful_text_showui, colorful_text_vlm


class ShowUIExecutor:
    def __init__(
        self, 
        output_callback: Callable[[BetaContentBlockParam], None], 
        tool_output_callback: Callable[[Any, str], None],
        selected_screen: int = 0,
        display_name: str | None = None,
    ):
        self.output_callback = output_callback
        self.tool_output_callback = tool_output_callback
        self.selected_screen = selected_screen
        self.display_name = display_name or colorful_text_showui
        self.screen_bbox = self._get_screen_resolution()
        print("Screen BBox:", self.screen_bbox)
        
        self.tool_collection = ToolCollection(
            ComputerTool(selected_screen=selected_screen, is_scaling=False)
        )
        
        self.supported_action_type={
            "CLICK": 'key',
            "DOUBLE_CLICK": "key",
            "DRAG": "key",
            "INPUT": "key",
            "SCROLL": "scroll",
            "ENTER": "key",
            "ESC": "key",
            "ESCAPE": "key",
            # "PRESS": "key",  # 已注释
            "HOVER": "mouse_move",
            "KEY": "key",
        }

    def __call__(self, response: str, messages: list[BetaMessageParam]):
        # response is expected to be :
        # {'content': "{'action': 'CLICK', 'value': None, 'position': [0.83, 0.15]}, ...", 'role': 'assistant'}, 
        
        action_dict = self._format_actor_output(response)  # str -> dict
        
        actions = action_dict["content"]
        role = action_dict["role"]
        
        # Parse the actions from showui
        action_list = self._parse_showui_output(actions)
        print("Parsed Action List:", action_list)
        
        tool_result_content = None
        
        if action_list is not None and len(action_list) > 0:
                    
            for action in action_list:  # Execute the tool (adapting the code from anthropic_executor.py)
            
                tool_result_content: list[BetaToolResultBlockParam] = []
                
                self.output_callback(f"{self.display_name}:\n{action}", sender="bot")
                print("Converted Action:", action)
                
                sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': action["action"], 'text': action["text"], 'coordinate': action["coordinate"]},
                                        name='computer', type='tool_use')
                
                # update messages
                new_message = {
                    "role": "assistant",
                    "content": cast(list[BetaContentBlockParam], [sim_content_block]),
                }
                if new_message not in messages:
                    messages.append(new_message)

                # Run the asynchronous tool execution in a synchronous context
                result = self.tool_collection.sync_call(
                    name=sim_content_block.name,
                    tool_input=cast(dict[str, Any], sim_content_block.input),
                )
                
                tool_result_content.append(
                    _make_api_tool_result(result, sim_content_block.id)
                )
                # print(f"executor: tool_result_content: {tool_result_content}")
                self.tool_output_callback(result, sim_content_block.id)

                # Craft messages based on the content_block
                # Note: to display the messages in the gradio, you should organize the messages in the following way (user message, bot message)
                display_messages = _message_display_callback(messages)
                # Send the messages to the gradio
                for user_msg, bot_msg in display_messages:
                    yield [user_msg, bot_msg], tool_result_content
        
        return tool_result_content
    
    
    def _format_actor_output(self, action_output: str|dict) -> Dict[str, Any]:
        if type(action_output) == dict:
            return action_output
        else:
            try:
                action_dict = ast.literal_eval(action_output)
                return action_dict
            except Exception as e:
                print(f"Error parsing action output: {e}")
                return None
    

    def _parse_showui_output(self, output_text: str) -> Union[List[Dict[str, Any]], None]:
        try:
            output_text = output_text.strip()
            
            # process single dictionary
            if output_text.startswith("{") and output_text.endswith("}"):
                output_text = f"[{output_text}]"

            # Validate if the output resembles a list of dictionaries
            if not (output_text.startswith("[") and output_text.endswith("]")):
                raise ValueError("Output does not look like a valid list or dictionary.")

            print("Output Text:", output_text)

            parsed_output = ast.literal_eval(output_text)

            print("Parsed Output:", parsed_output)

            if isinstance(parsed_output, dict):
                parsed_output = [parsed_output]
            elif not isinstance(parsed_output, list):
                raise ValueError("Parsed output is neither a dictionary nor a list.")

            if not all(isinstance(item, dict) for item in parsed_output):
                raise ValueError("Not all items in the parsed output are dictionaries.")

            # refine key: value pairs, mapping to the Anthropic's format
            refined_output = []
            
            for action_item in parsed_output:
                
                print("Action Item:", action_item)
                # sometime showui returns lower case action names
                action_item["action"] = action_item["action"].upper()
                
                if action_item["action"] not in self.supported_action_type:
                    raise ValueError(f"Action {action_item['action']} not supported. Check the output from ShowUI: {output_text}")
                    # continue
                
                elif action_item["action"] == "CLICK":  # 1. click -> left_click 或 right_click（value: left/right，默认 left）
                    if not action_item.get("position"):
                        raise ValueError("CLICK requires position")
                    x, y = action_item["position"]
                    btn = (action_item.get("value") or "left").strip().lower()
                    action_type = "right_click" if btn == "right" else "left_click"
                    coord = (int(x * (self.screen_bbox[2] - self.screen_bbox[0])),
                             int(y * (self.screen_bbox[3] - self.screen_bbox[1])))
                    refined_output.append({"action": action_type, "text": None, "coordinate": coord})

                elif action_item["action"] == "DOUBLE_CLICK":  # 2. double_click -> double_click 或 right_double（value: left/right）
                    if not action_item.get("position"):
                        raise ValueError("DOUBLE_CLICK requires position")
                    x, y = action_item["position"]
                    btn = (action_item.get("value") or "left").strip().lower()
                    action_type = "right_double_click" if btn == "right" else "double_click"
                    coord = (int(x * (self.screen_bbox[2] - self.screen_bbox[0])),
                             int(y * (self.screen_bbox[3] - self.screen_bbox[1])))
                    refined_output.append({"action": action_type, "text": None, "coordinate": coord})

                elif action_item["action"] == "DRAG":  # 3. drag -> mouse_move 到起点 + left_click_drag 到终点
                    pos = action_item.get("position")
                    if not pos or not isinstance(pos, (list, tuple)) or len(pos) != 2:
                        raise ValueError("DRAG requires position [[x1,y1], [x2,y2]]")
                    start, end = pos[0], pos[1]
                    if not start or not end or len(start) != 2 or len(end) != 2:
                        raise ValueError("DRAG requires position [[x1,y1], [x2,y2]]")
                    w = self.screen_bbox[2] - self.screen_bbox[0]
                    h = self.screen_bbox[3] - self.screen_bbox[1]
                    start_px = (int(start[0] * w), int(start[1] * h))
                    end_px = (int(end[0] * w), int(end[1] * h))
                    refined_output.append({"action": "mouse_move", "text": None, "coordinate": start_px})
                    refined_output.append({"action": "left_click_drag", "text": None, "coordinate": end_px})
                
                elif action_item["action"] == "INPUT":  # 4. input -> type
                    refined_output.append({"action": "type", "text": action_item.get("value") or "", "coordinate": None})
                
                elif action_item["action"] == "ENTER":  # 3. enter -> key, enter
                    refined_output.append({"action": "key", "text": "Enter", "coordinate": None})
                
                elif action_item["action"] == "ESC" or action_item["action"] == "ESCAPE":  # 4. enter -> key, enter
                    refined_output.append({"action": "key", "text": "Escape", "coordinate": None})
                    
                elif action_item["action"] == "HOVER":  # 5. hover -> mouse_move
                    if not action_item.get("position"):
                        raise ValueError("HOVER requires position")
                    x, y = action_item["position"]
                    action_item["position"] = (int(x * (self.screen_bbox[2] - self.screen_bbox[0])),
                                               int(y * (self.screen_bbox[3] - self.screen_bbox[1])))
                    refined_output.append({"action": "mouse_move", "text": None, "coordinate": tuple(action_item["position"])})
                    
                elif action_item["action"] == "SCROLL":  # 6. scroll -> 滚轮滚动（非 pageup/pagedown）
                    val = (action_item.get("value") or "").strip().lower()
                    if val == "up":
                        refined_output.append({"action": "scroll", "text": "up", "coordinate": None})
                    elif val == "down":
                        refined_output.append({"action": "scroll", "text": "down", "coordinate": None})
                    else:
                        raise ValueError(f"Scroll direction {action_item.get('value')} not supported. Use 'up' or 'down'.")

                # elif action_item["action"] == "PRESS":  # 已注释
                #     if not action_item.get("position"):
                #         raise ValueError("PRESS requires position")
                #     x, y = action_item["position"]
                #     action_item["position"] = (int(x * (self.screen_bbox[2] - self.screen_bbox[0])),
                #                                int(y * (self.screen_bbox[3] - self.screen_bbox[1])))
                #     refined_output.append({"action": "mouse_move", "text": None, "coordinate": tuple(action_item["position"])})
                #     refined_output.append({"action": "left_press", "text": None, "coordinate": None})

                elif action_item["action"] == "KEY":  # 8. key/shortcut
                    key_value = (action_item.get("value") or "").strip()
                    if not key_value:
                        raise ValueError("KEY requires value (e.g. 'ctrl+c', 'tab')")
                    refined_output.append({"action": "key", "text": key_value, "coordinate": None})

            return refined_output

        except Exception as e:
            print(f"Error parsing output: {e}")
            return None
        

    def _get_screen_resolution(self):
        from screeninfo import get_monitors
        import platform
        if platform.system() == "Darwin":
            import Quartz  # uncomment this line if you are on macOS
        import subprocess
            
        # Detect platform
        system = platform.system()

        if system == "Windows":
            # Windows: Use screeninfo to get monitor details
            screens = get_monitors()

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s.x)

            if self.selected_screen < 0 or self.selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")

            screen = sorted_screens[self.selected_screen]
            bbox = (screen.x, screen.y, screen.x + screen.width, screen.y + screen.height)

        elif system == "Darwin":  # macOS
            # macOS: Use Quartz to get monitor details
            max_displays = 32  # Maximum number of displays to handle
            active_displays = Quartz.CGGetActiveDisplayList(max_displays, None, None)[1]

            # Get the display bounds (resolution) for each active display
            screens = []
            for display_id in active_displays:
                bounds = Quartz.CGDisplayBounds(display_id)
                screens.append({
                    'id': display_id,
                    'x': int(bounds.origin.x),
                    'y': int(bounds.origin.y),
                    'width': int(bounds.size.width),
                    'height': int(bounds.size.height),
                    'is_primary': Quartz.CGDisplayIsMain(display_id)  # Check if this is the primary display
                })

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s['x'])

            if self.selected_screen < 0 or self.selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")

            screen = sorted_screens[self.selected_screen]
            bbox = (screen['x'], screen['y'], screen['x'] + screen['width'], screen['y'] + screen['height'])

        else:  # Linux or other OS
            cmd = "xrandr | grep ' primary' | awk '{print $4}'"
            try:
                output = subprocess.check_output(cmd, shell=True).decode()
                resolution = output.strip()
                # Parse the resolution format like "1920x1080+1920+0"
                # The format is "WIDTHxHEIGHT+X+Y"
                parts = resolution.split('+')[0]  # Get just the "1920x1080" part
                width, height = map(int, parts.split('x'))
                # Get the X, Y offset if needed
                x_offset = int(resolution.split('+')[1]) if len(resolution.split('+')) > 1 else 0
                y_offset = int(resolution.split('+')[2]) if len(resolution.split('+')) > 2 else 0
                bbox = (x_offset, y_offset, x_offset + width, y_offset + height)
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to get screen resolution on Linux.")
        
        return bbox



def _message_display_callback(messages):
    display_messages = []
    for msg in messages:
        try:
            if isinstance(msg["content"][0], TextBlock):
                display_messages.append((msg["content"][0].text, None))  # User message
            elif isinstance(msg["content"][0], BetaTextBlock):
                display_messages.append((None, msg["content"][0].text))  # Bot message
            elif isinstance(msg["content"][0], BetaToolUseBlock):
                display_messages.append((None, f"Tool Use: {msg['content'][0].name}\nInput: {msg['content'][0].input}"))  # Bot message
            elif isinstance(msg["content"][0], Dict) and msg["content"][0]["content"][-1]["type"] == "image":
                display_messages.append((None, f'<img src="data:image/png;base64,{msg["content"][0]["content"][-1]["source"]["data"]}">'))  # Bot message
            else:
                pass
                # print(msg["content"][0])
        except Exception as e:
            print("error", e)
            pass
    return display_messages


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text



# Testing main function
if __name__ == "__main__":
    def output_callback(content_block):
        # print("Output Callback:", content_block)
        pass

    def tool_output_callback(result, action):
        print("[showui_executor] Tool Output Callback:", result, action)
        pass

    # Instantiate the executor
    executor = ShowUIExecutor(
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
        selected_screen=0
    )

    # test inputs
    response_content = "{'content': \"{'action': 'CLICK', 'value': None, 'position': [0.49, 0.18]}\", 'role': 'assistant'}"
    # response_content = {'content': "{'action': 'CLICK', 'value': None, 'position': [0.49, 0.39]}", 'role': 'assistant'}
    # response_content = "{'content': \"{'action': 'CLICK', 'value': None, 'position': [0.49, 0.42]}, {'action': 'INPUT', 'value': 'weather for New York city', 'position': [0.49, 0.42]}, {'action': 'ENTER', 'value': None, 'position': None}\", 'role': 'assistant'}"

    # Initialize messages
    messages = []

    # Call the executor
    print("Testing ShowUIExecutor with response content:", response_content)
    for message, tool_result_content in executor(response_content, messages):
        print("Message:", message)
        print("Tool Result Content:", tool_result_content)

    # Display final messages
    print("\nFinal messages:")
    for msg in messages:
        print(msg)
        
        

[
    {'role': 'user', 'content': ['open a new tab and go to amazon.com', 'tmp/outputs/screenshot_b4a1b7e60a5c47359bedbd8707573966.png']},
    {'role': 'assistant', 'content': ["History Action: {'action': 'mouse_move', 'text': None, 'coordinate': (1216, 88)}"]}, 
    {'role': 'assistant', 'content': ["History Action: {'action': 'left_click', 'text': None, 'coordinate': None}"]},
    {'content': [
        {'type': 'tool_result', 'content': [{'type': 'text', 'text': 'Moved mouse to (1216, 88)'}], 'tool_use_id': 'toolu_ae4f2886-366c-4789-9fa6-ec13461cef12', 'is_error': False},
        {'type': 'tool_result', 'content': [{'type': 'text', 'text': 'Performed left_click'}], 'tool_use_id': 'toolu_a7377954-e1b7-4746-9757-b2eb4dcddc82', 'is_error': False}
                ], 'role': 'user'}
]