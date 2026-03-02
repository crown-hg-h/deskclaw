"""
Entrypoint for Gradio, see https://gradio.app/
"""

import os

# 启动时加载 .env 中的环境变量（若存在 python-dotenv）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

import platform
import asyncio
import base64
import io
import json
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast, Dict
from PIL import Image

import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock

from screeninfo import get_monitors
from computer_use_demo.tools.logger import logger, truncate_string

logger.info("Starting the gradio app")

screens = get_monitors()
logger.info(f"Found {len(screens)} screens")

from computer_use_demo.loop import APIProvider, TASK_COMPLETE, TASK_FAILED, TASK_STOPPED, sampling_loop_sync
from computer_use_demo.memory import MemoryManager

from computer_use_demo.tools import ToolResult
from computer_use_demo.tools.computer import get_screen_details
SCREEN_NAMES, SELECTED_SCREEN_INDEX = get_screen_details()

WARNING_TEXT = "⚠️ Security Alert: Do not provide access to sensitive accounts or data, as malicious web content can hijack Agent's behavior. Keep monitor on the Agent's actions."

# 记忆系统（参考 pc-agent-loop），懒加载
_memory_manager: MemoryManager | None = None


def _get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(base_dir="./memory")
    return _memory_manager


def setup_state(state):

    if "messages" not in state:
        state["messages"] = []
    # -------------------------------
    if "planner_model" not in state:
        state["planner_model"] = "gpt-4o"  # default
    if "actor_model" not in state:
        state["actor_model"] = "Direct"    # default: Planner outputs coords directly, no Actor
    if "planner_provider" not in state:
        state["planner_provider"] = "openai"  # default
    if "actor_provider" not in state:
        state["actor_provider"] = "local"    # default

     # Fetch API keys from environment variables
    if "openai_api_key" not in state: 
        state["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    if "anthropic_api_key" not in state:
        state["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY", "")    
    if "qwen_api_key" not in state:
        state["qwen_api_key"] = os.getenv("QWEN_API_KEY", "")
    if "ui_tars_url" not in state:
        state["ui_tars_url"] = ""

    # Set the initial api_key based on the provider
    if "planner_api_key" not in state:
        if state["planner_provider"] == "openai":
            state["planner_api_key"] = state["openai_api_key"]
        elif state["planner_provider"] == "anthropic":
            state["planner_api_key"] = state["anthropic_api_key"]
        elif state["planner_provider"] == "qwen":
            state["planner_api_key"] = state["qwen_api_key"]
        elif state["planner_provider"] == "azure":
            state["planner_api_key"] = os.getenv("AZURE_OPENAI_CREDENTIALS", "")
        elif state["planner_provider"] == "custom":
            state["planner_api_key"] = os.getenv("CUSTOM_OPENAI_CREDENTIALS", "")
        elif state["planner_provider"] == "ollama":
            state["planner_api_key"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        else:
            state["planner_api_key"] = ""
    # Azure: 若界面未填写，则从环境变量补充
    elif state["planner_provider"] == "azure" and not (state.get("planner_api_key") or "").strip():
        state["planner_api_key"] = os.getenv("AZURE_OPENAI_CREDENTIALS", "")
    # Custom: 若界面未填写，则从环境变量补充
    elif state["planner_provider"] == "custom" and not (state.get("planner_api_key") or "").strip():
        state["planner_api_key"] = os.getenv("CUSTOM_OPENAI_CREDENTIALS", "")

    logger.info("loaded initial api_key for %s (len=%d)", state["planner_provider"], len(state.get("planner_api_key", "") or ""))

    if not state["planner_api_key"]:
        logger.warning("Planner API key not found. Please set it in the environment or paste in textbox.")


    if "selected_screen" not in state:
        state['selected_screen'] = SELECTED_SCREEN_INDEX if SCREEN_NAMES else 0

    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 10 # 10
    if "custom_system_prompt" not in state:
        state["custom_system_prompt"] = ""
        # remove if want to use default system prompt
        device_os_name = "Windows" if platform.system() == "Windows" else "Mac" if platform.system() == "Darwin" else "Linux"
        state["custom_system_prompt"] += f"\n\nNOTE: you are operating a {device_os_name} machine"
    if "hide_images" not in state:
        state["hide_images"] = False
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
    if "task_completed" not in state:
        state["task_completed"] = False
    if "stop_requested" not in state:
        state["stop_requested"] = False
    if "task_running" not in state:
        state["task_running"] = False
    if "pending_messages" not in state:
        state["pending_messages"] = []
        
    if "target_width" not in state:
        state["target_width"] = 1920
    if "target_height" not in state:
        state["target_height"] = 1080
    # 记忆系统与 SOP（参考 pc-agent-loop）
    if "memory_enabled" not in state:
        state["memory_enabled"] = True
    if "auto_save_sop" not in state:
        state["auto_save_sop"] = True


async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "Setup completed"


def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."


def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response


def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output


def _tuples_to_messages(chatbot_state):
    """Convert legacy tuple format to Gradio 6 messages format."""
    messages = []
    for user_msg, bot_msg in chatbot_state:
        if user_msg is not None and user_msg != "":
            messages.append({"role": "user", "content": user_msg})
        if bot_msg is not None and bot_msg != "":
            messages.append({"role": "assistant", "content": bot_msg})
    return messages


def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        logger.info(f"_render_message: {str(message)[:100]}")

        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
            or message.__class__.__name__ == "CLIResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                # somehow can't display via gr.Image
                # image_data = base64.b64decode(message.base64_image)
                # return gr.Image(value=Image.open(io.BytesIO(image_data)))
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return message.text
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            return f"Tool Use: {message.name}\nInput: {message.input}"
        else:  
            return message


    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    if sender == "bot":
        chatbot_state.append((None, message))
    else:
        chatbot_state.append((message, None))

    # Create a concise version of the chatbot state for logging
    concise_state = [(truncate_string(user_msg), truncate_string(bot_msg)) for user_msg, bot_msg in chatbot_state]
    logger.info(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")


def process_input(user_input, state, memory_enabled=None, auto_save_sop=None):
    
    setup_state(state)
    if memory_enabled is not None:
        state["memory_enabled"] = memory_enabled
    if auto_save_sop is not None:
        state["auto_save_sop"] = auto_save_sop

    # 任务已完成后再输入 = 新对话；任务进行中输入 = 追加消息继续执行
    if state.get("task_completed", False):
        state["messages"] = []
        state["chatbot_messages"] = []
        state["pending_messages"] = []
        state["task_completed"] = False
    state["stop_requested"] = False  # 重置停止标志
    state["task_running"] = True

    # Append the user message to state["messages"]
    state["messages"].append(
            {
                "role": "user",
                "content": [TextBlock(type="text", text=user_input)],
            }
        )

    # Append the user's message to chatbot_messages with None for the assistant's reply
    state['chatbot_messages'].append((user_input, None))
    yield _tuples_to_messages(state['chatbot_messages'])  # Yield to update the chatbot UI with the user's message

    # Azure/Kimi-K2.5/Custom: 若 state 中为空则从环境变量补充
    api_key = state["planner_api_key"]
    if (state.get("planner_provider") == "azure" or state.get("planner_model") == "Kimi-K2.5 (Azure)") and not (api_key or "").strip():
        api_key = os.getenv("AZURE_OPENAI_CREDENTIALS", "")
        state["planner_api_key"] = api_key
    elif (state.get("planner_provider") == "custom" or state.get("planner_model") == "Custom (OpenAI)") and not (api_key or "").strip():
        api_key = os.getenv("CUSTOM_OPENAI_CREDENTIALS", "")
        state["planner_api_key"] = api_key

    def _stop_requested() -> bool:
        return state.get("stop_requested", False)

    # 记忆系统：召回相关 SOP 并注入 prompt
    system_prompt_suffix = state["custom_system_prompt"]
    if state.get("memory_enabled", False):
        try:
            mm = _get_memory_manager()
            sop_hint = mm.recall_sops_as_prompt(user_input, top_k=2)
            if sop_hint:
                system_prompt_suffix += "\n\n" + sop_hint
                logger.info("已召回 SOP 并注入 prompt")
        except Exception as e:
            logger.warning(f"记忆召回异常: {e}")

    def _pop_pending_messages(st: dict) -> list:
        """取出并清空待处理消息，供 loop 在任务中追加"""
        pending = st.get("pending_messages") or []
        st["pending_messages"] = []
        return pending

    def _sop_save_callback(task_desc: str, summary: str) -> None:
        """summary 由 Planner 在任务完成时同一轮对话中给出，不再单独调用 summarizer"""
        if not state.get("auto_save_sop", True):
            return
        try:
            steps_desc = summary.strip() or "任务已完成"
            mm = _get_memory_manager()
            sop = mm.save_sop(task_desc, steps_desc)
            chatbot_output_callback(
                f"📝 **已保存 SOP**：{sop.task_description[:50]}... (success_count={sop.success_count})",
                chatbot_state=state["chatbot_messages"],
                hide_images=state["hide_images"],
            )
        except Exception as e:
            logger.warning(f"SOP 保存失败: {e}")

    try:
        # Run sampling_loop_sync with the chatbot_output_callback
        for loop_msg in sampling_loop_sync(
            system_prompt_suffix=system_prompt_suffix,
            planner_model=state["planner_model"],
            planner_provider=state["planner_provider"],
            actor_model="Direct",
            actor_provider=state.get("actor_provider"),
            messages=state["messages"],
            output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=state["hide_images"]),
            tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
            api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
            api_key=api_key,
            only_n_most_recent_images=state["only_n_most_recent_images"],
            selected_screen=state['selected_screen'],
            showui_max_pixels=1344,
            showui_awq_4bit=False,
            target_width=state.get("target_width", 1920),
            target_height=state.get("target_height", 1080),
            stop_requested=_stop_requested,
            sop_save_callback=_sop_save_callback if state.get("memory_enabled") else None,
            pending_messages_callback=lambda: _pop_pending_messages(state),
        ):  
            if loop_msg is TASK_COMPLETE:
                state["task_completed"] = True
                state["task_running"] = False
                chatbot_output_callback("✅ **任务已完成**", chatbot_state=state['chatbot_messages'], hide_images=state["hide_images"])
                yield _tuples_to_messages(state['chatbot_messages'])
                logger.info("任务已完成，收到终止信号，关闭循环。")
                return

            if loop_msg is TASK_FAILED:
                state["task_completed"] = False
                state["task_running"] = False
                chatbot_output_callback("❌ **任务失败**（同一操作重复三次未完成）", chatbot_state=state['chatbot_messages'], hide_images=state["hide_images"])
                yield _tuples_to_messages(state['chatbot_messages'])
                logger.info("任务失败，同一操作重复三次未完成。")
                return

            if loop_msg is TASK_STOPPED:
                state["task_completed"] = False
                state["task_running"] = False
                chatbot_output_callback("⏹️ **任务已停止**", chatbot_state=state['chatbot_messages'], hide_images=state["hide_images"])
                yield _tuples_to_messages(state['chatbot_messages'])
                logger.info("用户请求停止，任务已终止。")
                return

            yield _tuples_to_messages(state['chatbot_messages'])  # Yield the updated chatbot_messages to update the chatbot UI
    except GeneratorExit:
        state["task_completed"] = False
        state["task_running"] = False
        state["stop_requested"] = False
        # 更新 state 中的聊天记录（无法 yield，UI 可能不刷新，但状态保持一致）
        chatbot_output_callback("⏹️ **任务已停止**", chatbot_state=state['chatbot_messages'], hide_images=state["hide_images"])
        logger.info("用户点击停止，任务已终止。")
        raise


with gr.Blocks() as demo:
    
    state = gr.State({})  # Use Gradio's state management
    setup_state(state.value)  # Initialize the state

    # Retrieve screen details
    gr.Markdown("# DeskClaw")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(WARNING_TEXT)

    with gr.Accordion("Settings", open=True): 
        with gr.Row():
            with gr.Column():
                # --------------------------
                # Planner
                planner_model = gr.Dropdown(
                    label="Planner Model",
                    choices=["gpt-4o", 
                             "gpt-4o-mini", 
                             "qwen2-vl-max", 
                             "Kimi-K2.5 (Azure)",
                             "Custom (OpenAI)",
                             "qwen2.5-vl (ollama)",
                             "qwen2-vl-2b (local)", 
                             "qwen2-vl-7b (local)", 
                             "qwen2-vl-2b (ssh)", 
                             "qwen2-vl-7b (ssh)",
                             "qwen2.5-vl-7b (ssh)"],
                    value="gpt-4o",
                    interactive=True,
                )
            with gr.Column():
                planner_api_provider = gr.Dropdown(
                    label="API Provider",
                    choices=[option.value for option in APIProvider],
                    value="openai",
                    interactive=False,
                )
            with gr.Column(visible=True) as planner_api_key_column:
                planner_api_key = gr.Textbox(
                    label="Planner API Key",
                    type="password",
                    value=state.value.get("planner_api_key", ""),
                    placeholder="Paste your planner model API key",
                    interactive=True,
                )

            with gr.Column():
                custom_prompt = gr.Textbox(
                    label="System Prompt Suffix",
                    value="",
                    interactive=True,
                )
            with gr.Column():
                screen_options, primary_index = get_screen_details()
                SCREEN_NAMES = screen_options
                SELECTED_SCREEN_INDEX = primary_index
                screen_selector = gr.Dropdown(
                    label="Select Screen",
                    choices=screen_options,
                    value=screen_options[primary_index] if screen_options else None,
                    interactive=True,
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True,
                )
            with gr.Column():
                memory_enabled = gr.Checkbox(
                    label="启用记忆/SOP",
                    value=True,
                    interactive=True,
                    info="召回类似任务 SOP 并自动保存新 SOP",
                )
            with gr.Column():
                auto_save_sop = gr.Checkbox(
                    label="任务完成后自动保存 SOP",
                    value=True,
                    interactive=True,
                )
            with gr.Column():
                resolution_selector = gr.Dropdown(
                    label="Screenshot Resolution",
                    choices=["1080p (1920×1080)", "2K (2560×1440)", "4K (3840×2160)"],
                    value="1080p (1920×1080)",
                    interactive=True,
                )

        # Custom (OpenAI) 自定义配置：接口地址、API 密钥、模型 ID
        with gr.Row(visible=False) as custom_config_row:
            with gr.Column():
                custom_api_base = gr.Textbox(
                    label="接口地址",
                    placeholder="https://api.xxx.com/v1",
                    value="",
                    interactive=True,
                )
            with gr.Column():
                custom_api_key = gr.Textbox(
                    label="API 密钥",
                    type="password",
                    value="",
                    placeholder="sk-xxx",
                    interactive=True,
                )
            with gr.Column():
                custom_model_id = gr.Textbox(
                    label="模型 ID",
                    placeholder="gpt-4o",
                    value="gpt-4o",
                    interactive=True,
                )
    
    # Define the merged dictionary with task mappings
    merged_dict = json.load(open("assets/examples/ootb_examples.json", "r"))

    def update_only_n_images(only_n_images_value, state):
        state["only_n_most_recent_images"] = only_n_images_value
    
    # Callback to update the second dropdown based on the first selection
    def update_second_menu(selected_category):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).keys()))

    # Callback to update the third dropdown based on the second selection
    def update_third_menu(selected_category, selected_option):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).get(selected_option, {}).keys()))

    # Callback to update the textbox based on the third selection
    def update_textbox(selected_category, selected_option, selected_task):
        task_data = merged_dict.get(selected_category, {}).get(selected_option, {}).get(selected_task, {})
        prompt = task_data.get("prompt", "")
        preview_image = task_data.get("initial_state", "")
        task_hint = "Task Hint: " + task_data.get("hint", "")
        return prompt, preview_image, task_hint
    
    # Function to update the global variable when the dropdown changes
    RESOLUTION_MAP = {
        "1080p (1920×1080)": (1920, 1080),
        "2K (2560×1440)": (2560, 1440),
        "4K (3840×2160)": (3840, 2160),
    }

    def update_resolution(resolution_selection, state):
        w, h = RESOLUTION_MAP.get(resolution_selection, (1920, 1080))
        state["target_width"] = w
        state["target_height"] = h
        logger.info(f"Screenshot resolution updated to: {w}×{h}")
        return state

    def update_selected_screen(selected_screen_name, state):
        global SCREEN_NAMES
        global SELECTED_SCREEN_INDEX
        SELECTED_SCREEN_INDEX = SCREEN_NAMES.index(selected_screen_name)
        logger.info(f"Selected screen updated to: {SELECTED_SCREEN_INDEX}")
        state['selected_screen'] = SELECTED_SCREEN_INDEX


    def update_planner_model(model_selection, state):
        state["model"] = model_selection
        # Update planner_model
        state["planner_model"] = model_selection
        logger.info(f"Model updated to: {state['planner_model']}")
        
        if model_selection == "qwen2-vl-max":
            provider_choices = ["qwen"]
            provider_value = "qwen"
            provider_interactive = False
            api_key_interactive = True
            api_key_placeholder = "qwen API key"
            api_key_type = "password"  # Display API key in password form
        
        elif model_selection == "Kimi-K2.5 (Azure)":
            provider_choices = ["azure"]
            provider_value = "azure"
            provider_interactive = False
            api_key_interactive = True
            api_key_placeholder = "格式: https://xxx.azure.com/openai/v1|||你的API_KEY"
            api_key_type = "text"
            default_azure = os.getenv("AZURE_OPENAI_CREDENTIALS", "")
            if "planner_api_key" in state and state["planner_api_key"] and "|||" in state["planner_api_key"]:
                state["api_key"] = state["planner_api_key"]
            elif default_azure and "|||" in default_azure:
                state["api_key"] = default_azure
            else:
                state["api_key"] = ""
            state["planner_api_key"] = state["api_key"]

        elif model_selection == "Custom (OpenAI)":
            provider_choices = ["custom"]
            provider_value = "custom"
            provider_interactive = False
            api_key_interactive = True
            api_key_placeholder = ""
            api_key_type = "password"
            default_custom = os.getenv("CUSTOM_OPENAI_CREDENTIALS", "")
            raw = state.get("planner_api_key") or default_custom or ""
            custom_base, custom_key, custom_model = "", "", "gpt-4o"
            if raw and "|||" in raw:
                parts = [p.strip() for p in raw.split("|||")]
                custom_base = parts[0] if len(parts) > 0 else ""
                custom_key = parts[1] if len(parts) > 1 else ""
                custom_model = parts[2] if len(parts) > 2 else "gpt-4o"
            state["custom_api_base"] = custom_base
            state["custom_api_key"] = custom_key
            state["custom_model_id"] = custom_model
            state["planner_api_key"] = f"{custom_base}|||{custom_key}|||{custom_model}" if (custom_base and custom_key) else raw

        elif model_selection in ["qwen2-vl-2b (local)", "qwen2-vl-7b (local)"]:
            # Set provider to "openai", make it unchangeable
            provider_choices = ["local"]
            provider_value = "local"
            provider_interactive = False
            api_key_interactive = False
            api_key_placeholder = "not required"
            api_key_type = "password"  # Maintain consistency

        elif model_selection == "qwen2.5-vl (ollama)":
            provider_choices = ["ollama"]
            provider_value = "ollama"
            provider_interactive = False
            api_key_interactive = True
            api_key_placeholder = "API Base URL (无需 API Key，如 http://localhost:11434)"
            api_key_type = "text"
            if "planner_api_key" in state and state["planner_api_key"]:
                state["api_key"] = state["planner_api_key"]
            else:
                state["api_key"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            state["planner_api_key"] = state["api_key"]  # Ollama 用此字段存 API Base URL

        elif "ssh" in model_selection:
            provider_choices = ["ssh"]
            provider_value = "ssh"
            provider_interactive = False
            api_key_interactive = True
            api_key_placeholder = "ssh host and port (e.g. localhost:8000)"
            api_key_type = "text"  # Display SSH connection info in plain text
            # If SSH connection info already exists, keep it
            if "planner_api_key" in state and state["planner_api_key"]:
                state["api_key"] = state["planner_api_key"]
            else:
                state["api_key"] = ""

        elif model_selection == "gpt-4o" or model_selection == "gpt-4o-mini":
            # Set provider to "openai", make it unchangeable
            provider_choices = ["openai"]
            provider_value = "openai"
            provider_interactive = False
            api_key_interactive = True
            api_key_type = "password"  # Display API key in password form

            api_key_placeholder = "openai API key"
        else:
            raise ValueError(f"Model {model_selection} not supported")

        # Update the provider in state
        state["planner_api_provider"] = provider_value
        state["planner_provider"] = provider_value
        
        # Update api_key in state based on the provider
        if provider_value == "openai":
            state["api_key"] = state.get("openai_api_key", "")
        elif provider_value == "anthropic":
            state["api_key"] = state.get("anthropic_api_key", "")
        elif provider_value == "qwen":
            state["api_key"] = state.get("qwen_api_key", "")
        elif provider_value == "local":
            state["api_key"] = ""
        elif provider_value == "azure":
            state["api_key"] = state.get("planner_api_key", "")
            state["planner_api_key"] = state["api_key"]
        elif provider_value == "custom":
            state["api_key"] = state.get("planner_api_key", "")
            state["planner_api_key"] = state["api_key"]
        elif provider_value == "ollama":
            state["api_key"] = state.get("planner_api_key", "") or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            state["planner_api_key"] = state["api_key"]
        # SSH的情况已经在上面处理过了，这里不需要重复处理

        provider_update = gr.update(
            choices=provider_choices,
            value=provider_value,
            interactive=provider_interactive
        )

        # Update the API Key textbox
        api_key_update = gr.update(
            placeholder=api_key_placeholder,
            value=state["api_key"],
            interactive=api_key_interactive,
            type=api_key_type  # 添加 type 参数的更新
        )

        # Custom 配置区域显示/隐藏
        is_custom = model_selection == "Custom (OpenAI)"
        planner_key_visible = gr.update(visible=not is_custom)
        custom_row_visible = gr.update(visible=is_custom)
        custom_base_val = state.get("custom_api_base", "")
        custom_key_val = state.get("custom_api_key", "")
        custom_model_val = state.get("custom_model_id", "gpt-4o")

        logger.info("Updated state: model=%s, provider=%s", state["planner_model"], state["planner_api_provider"])
        return (
            provider_update, api_key_update, state,
            planner_key_visible, custom_row_visible,
            custom_base_val, custom_key_val, custom_model_val,
        )
    
    def update_api_key_placeholder(provider_value, model_selection):
        if model_selection == "Kimi-K2.5 (Azure)":
            return gr.update(placeholder="API Base URL|||API Key")
        elif model_selection == "qwen2.5-vl (ollama)":
            return gr.update(placeholder="API Base URL (e.g. http://localhost:11434)")
        else:
            return gr.update(placeholder="")

    def update_system_prompt_suffix(system_prompt_suffix, state):
        state["custom_system_prompt"] = system_prompt_suffix

    def update_api_key(api_key_value, state):
        """Handle API key updates"""
        state["planner_api_key"] = api_key_value
        if state["planner_provider"] in ("ssh", "ollama", "azure", "custom"):
            state["api_key"] = api_key_value
        logger.info("API key updated: provider=%s", state["planner_provider"])
        return state

    with gr.Accordion("Quick Start Prompt", open=False):  # open=False 表示默认收
        # Initialize Gradio interface with the dropdowns
        with gr.Row():
            # Set initial values
            initial_category = "Game Play"
            initial_second_options = list(merged_dict[initial_category].keys())
            initial_third_options = list(merged_dict[initial_category][initial_second_options[0]].keys())
            initial_text_value = merged_dict[initial_category][initial_second_options[0]][initial_third_options[0]]

            with gr.Column(scale=2):
                # First dropdown for Task Category
                first_menu = gr.Dropdown(
                    choices=list(merged_dict.keys()), label="Task Category", interactive=True, value=initial_category
                )

                # Second dropdown for Software
                second_menu = gr.Dropdown(
                    choices=initial_second_options, label="Software", interactive=True, value=initial_second_options[0]
                )

                # Third dropdown for Task
                third_menu = gr.Dropdown(
                    choices=initial_third_options, label="Task", interactive=True, value=initial_third_options[0]
                    # choices=["Please select a task"]+initial_third_options, label="Task", interactive=True, value="Please select a task"
                )

            with gr.Column(scale=1):
                initial_image_value = "./assets/examples/init_states/honkai_star_rail_showui.png"  # default image path
                image_preview = gr.Image(value=initial_image_value, label="Reference Initial State", height=260-(318.75-280))
                hintbox = gr.Markdown("Task Hint: Selected options will appear here.")

        # Textbox for displaying the mapped value
        # textbox = gr.Textbox(value=initial_text_value, label="Action")


    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="输入任务指令（任务完成后为新对话）...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="Stop", variant="stop")

    with gr.Row(visible=True):
        with gr.Column(scale=8):
            during_task_input = gr.Textbox(
                show_label=False,
                placeholder="任务执行中可在此输入补充指令，Planner 会接收并继续执行...",
                container=False,
            )
        with gr.Column(scale=1, min_width=80):
            during_task_btn = gr.Button(value="追加指令", variant="secondary")

    chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=540, group_consecutive_messages=False)

    def request_stop(state):
        """设置停止标志，sampling_loop 会在下一轮迭代时检测并终止"""
        state["stop_requested"] = True
        return state
    
    def update_custom_credentials(custom_base, custom_key, custom_model, state):
        """将 Custom 三个字段合并到 planner_api_key"""
        if state.get("planner_model") == "Custom (OpenAI)":
            model_id = (custom_model or "").strip() or "gpt-4o"
            state["planner_api_key"] = f"{(custom_base or '').strip()}|||{(custom_key or '').strip()}|||{model_id}"
        return state

    planner_model.change(
        fn=update_planner_model,
        inputs=[planner_model, state],
        outputs=[
            planner_api_provider, planner_api_key, state,
            planner_api_key_column, custom_config_row,
            custom_api_base, custom_api_key, custom_model_id,
        ],
    )
    custom_api_base.change(fn=update_custom_credentials, inputs=[custom_api_base, custom_api_key, custom_model_id, state], outputs=state)
    custom_api_key.change(fn=update_custom_credentials, inputs=[custom_api_base, custom_api_key, custom_model_id, state], outputs=state)
    custom_model_id.change(fn=update_custom_credentials, inputs=[custom_api_base, custom_api_key, custom_model_id, state], outputs=state)
    planner_api_provider.change(fn=update_api_key_placeholder, inputs=[planner_api_provider, planner_model], outputs=planner_api_key)
    screen_selector.change(fn=update_selected_screen, inputs=[screen_selector, state], outputs=None)
    resolution_selector.change(fn=update_resolution, inputs=[resolution_selector, state], outputs=state)
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    
    # Link callbacks to update dropdowns based on selections
    first_menu.change(fn=update_second_menu, inputs=first_menu, outputs=second_menu)
    second_menu.change(fn=update_third_menu, inputs=[first_menu, second_menu], outputs=third_menu)
    third_menu.change(fn=update_textbox, inputs=[first_menu, second_menu, third_menu], outputs=[chat_input, image_preview, hintbox])

    def route_submit(user_input, st, memory_enabled, auto_save_sop):
        """任务运行中：追加到当前对话；任务完成后：新对话"""
        setup_state(st)
        if st.get("task_running", False):
            if user_input and str(user_input).strip():
                st["pending_messages"] = st.get("pending_messages") or []
                st["pending_messages"].append(str(user_input).strip())
                st["chatbot_messages"] = st.get("chatbot_messages") or []
                st["chatbot_messages"].append((f"📩 追加指令: {user_input}", None))
                logger.info(f"任务中通过主输入框追加指令: {user_input[:50]}...")
            yield _tuples_to_messages(st["chatbot_messages"])
            return
        yield from process_input(user_input, st, memory_enabled, auto_save_sop)

    submit_button.click(
        route_submit,
        [chat_input, state, memory_enabled, auto_save_sop],
        chatbot,
    )
    chat_input.submit(
        route_submit,
        [chat_input, state, memory_enabled, auto_save_sop],
        chatbot,
    )
    stop_button.click(fn=request_stop, inputs=state, outputs=state)

    def add_during_task_message(msg, state):
        """任务执行中追加用户消息，Planner 下一轮会接收"""
        if msg and str(msg).strip():
            state["pending_messages"] = state.get("pending_messages") or []
            state["pending_messages"].append(str(msg).strip())
            state["chatbot_messages"] = state.get("chatbot_messages") or []
            state["chatbot_messages"].append((f"📩 追加指令: {msg}", None))
            logger.info(f"任务中追加指令: {msg[:50]}...")
        return state, _tuples_to_messages(state["chatbot_messages"])

    during_task_btn.click(
        fn=add_during_task_message,
        inputs=[during_task_input, state],
        outputs=[state, chatbot],
    ).then(fn=lambda: "", inputs=None, outputs=during_task_input)

    planner_api_key.change(
        fn=update_api_key,
        inputs=[planner_api_key, state],
        outputs=state
    )

demo.launch(share=False,
            theme=gr.themes.Soft(),
            allowed_paths=["./"])