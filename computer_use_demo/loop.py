"""
Agentic sampling loop that calls the Anthropic API and local implementation of computer use tools.
"""
import time
import json
from collections.abc import Callable
from enum import StrEnum

from anthropic import APIResponse
from anthropic.types.beta import BetaContentBlock, BetaMessage, BetaMessageParam
from computer_use_demo.tools import ToolResult

from computer_use_demo.tools.colorful_text import colorful_text_showui, colorful_text_vlm
from computer_use_demo.tools.screen_capture import get_screenshot
from computer_use_demo.gui_agent.llm_utils.oai import encode_image
from computer_use_demo.tools.logger import logger


# 任务完成时的终止信号，用于状态维护
TASK_COMPLETE = object()
# 任务被用户停止时的信号
TASK_STOPPED = object()
# 任务失败时的信号（同一操作重复三次未完成）
TASK_FAILED = object()

# 同一操作重复次数阈值，超过则终止任务
REPEAT_FAIL_THRESHOLD = 3

# Planner JSON 解析失败时的重试次数
PARSE_RETRY_COUNT = 2


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"
    QWEN = "qwen"
    AZURE = "azure"
    OLLAMA = "ollama"
    SSH = "ssh"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.OPENAI: "gpt-4o",
    APIProvider.QWEN: "qwen2vl",
    APIProvider.SSH: "qwen2-vl-2b",
}

PLANNER_MODEL_CHOICES_MAPPING = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini", 
    "qwen2-vl-max": "qwen2-vl-max",
    "qwen2-vl-2b (local)": "qwen2-vl-2b-instruct",
    "qwen2-vl-7b (local)": "qwen2-vl-7b-instruct",
    "qwen2.5-vl-3b (local)": "qwen2.5-vl-3b-instruct",
    "qwen2.5-vl-7b (local)": "qwen2.5-vl-7b-instruct",
    "qwen2-vl-2b (ssh)": "qwen2-vl-2b (ssh)",
    "qwen2-vl-7b (ssh)": "qwen2-vl-7b (ssh)",
    "qwen2.5-vl-7b (ssh)": "qwen2.5-vl-7b (ssh)",
    "qwen2.5-vl (ollama)": "qwen2.5-vl (ollama)",
    "Kimi-K2.5 (Azure)": "Kimi-K2.5 (Azure)",
    "Custom (OpenAI)": "Custom (OpenAI)",
}


def _action_key(plan_data: dict) -> tuple:
    """生成用于比较操作是否重复的 key。position 四舍五入到 2 位小数。"""
    action = plan_data.get("action") or plan_data.get("Next Action")
    value = plan_data.get("value")
    pos = plan_data.get("position")
    if pos is not None and isinstance(pos, (list, tuple)) and len(pos) >= 2:
        # 标准 position [x, y]
        if isinstance(pos[0], (int, float)) and isinstance(pos[1], (int, float)):
            pos = (round(float(pos[0]), 2), round(float(pos[1]), 2))
        # DRAG position [[x1,y1],[x2,y2]]
        elif isinstance(pos[0], (list, tuple)) and isinstance(pos[1], (list, tuple)) and len(pos[0]) >= 2 and len(pos[1]) >= 2:
            pos = (
                (round(float(pos[0][0]), 2), round(float(pos[0][1]), 2)),
                (round(float(pos[1][0]), 2), round(float(pos[1][1]), 2)),
            )
    return (str(action), str(value) if value is not None else None, pos)


def sampling_loop_sync(
    *,
    planner_model: str,
    planner_provider: APIProvider | None,
    actor_model: str,
    actor_provider: APIProvider | None,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    selected_screen: int = 0,
    showui_max_pixels: int = 1344,
    showui_awq_4bit: bool = False,
    ui_tars_url: str = "",
    target_width: int = 1920,
    target_height: int = 1080,
    stop_requested: Callable[[], bool] | None = None,
    sop_save_callback: Callable[[str, str], None] | None = None,
    pending_messages_callback: Callable[[], list] | None = None,
    ask_user_callback: Callable[[str], str] | None = None,
):
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """

    # ---------------------------
    # Initialize Planner
    # ---------------------------
    
    if planner_model in PLANNER_MODEL_CHOICES_MAPPING:
        planner_model = PLANNER_MODEL_CHOICES_MAPPING[planner_model]
    else:
        raise ValueError(f"Planner Model {planner_model} not supported")
    
    if planner_model in ["gpt-4o", "gpt-4o-mini", "qwen2-vl-max", "qwen2.5-vl (ollama)", "Kimi-K2.5 (Azure)", "Custom (OpenAI)"]:
        
        from computer_use_demo.gui_agent.planner.api_vlm_planner import APIVLMPlanner

        planner = APIVLMPlanner(
            model=planner_model,
            provider=planner_provider,
            system_prompt_suffix=system_prompt_suffix,
            api_key=api_key,
            api_response_callback=api_response_callback,
            selected_screen=selected_screen,
            output_callback=output_callback,
            target_width=target_width,
            target_height=target_height,
        )
    elif planner_model in ["qwen2-vl-2b-instruct", "qwen2-vl-7b-instruct"]:
        
        import torch
        from computer_use_demo.gui_agent.planner.local_vlm_planner import LocalVLMPlanner
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu") # support: 'cpu', 'mps', 'cuda'
        logger.info(f"Planner model {planner_model} inited on device: {device}.")
        
        planner = LocalVLMPlanner(
            model=planner_model,
            provider=planner_provider,
            system_prompt_suffix=system_prompt_suffix,
            api_key=api_key,
            api_response_callback=api_response_callback,
            selected_screen=selected_screen,
            output_callback=output_callback,
            device=device,
            target_width=target_width,
            target_height=target_height,
        )
    elif "ssh" in planner_model:
        planner = APIVLMPlanner(
            model=planner_model,
            provider=planner_provider,
            system_prompt_suffix=system_prompt_suffix,
            api_key=api_key,
            api_response_callback=api_response_callback,
            selected_screen=selected_screen,
            output_callback=output_callback,
            target_width=target_width,
            target_height=target_height,
        )
    else:
        logger.error(f"Planner Model {planner_model} not supported")
        raise ValueError(f"Planner Model {planner_model} not supported")
        

    # ---------------------------
    # Initialize Executor (Direct 模式：Planner 直接输出坐标，无 Actor)
    # ---------------------------
    from computer_use_demo.executor.showui_executor import ShowUIExecutor
    executor = ShowUIExecutor(
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
        selected_screen=selected_screen,
        display_name=colorful_text_vlm,
    )
    logger.info("Direct mode: Planner outputs coordinates directly.")


    tool_result_content = None
    showui_loop_count = 0
    recent_actions: list[tuple] = []  # 用于检测同一操作重复三次
    action_history: list[dict] = []  # 用于 SOP 保存（成功执行的步骤）

    def _extract_user_task(msgs: list) -> str:
        """从 messages 中提取用户任务描述（取第一条 user 消息）"""
        for m in msgs:
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            t = (c.get("text") or "").strip()
                            if t and not t.startswith("History plan:"):
                                return t
                        if hasattr(c, "text"):
                            t = (getattr(c, "text", "") or "").strip()
                            if t and not t.startswith("History plan:"):
                                return t
                if isinstance(content, str) and not content.startswith("History plan:"):
                    return content.strip()
        return ""

    user_task = _extract_user_task(messages)
    if not user_task:
        logger.warning("未能从 messages 提取 user_task，SOP 保存将跳过。messages 首条 user content 格式需为 text")

    logger.info("Start the message loop. user_task=%r, messages_count=%d", (user_task[:60] + "..." if user_task and len(user_task) > 60 else user_task) or "(空)", len(messages))

    def _should_stop() -> bool:
        return stop_requested is not None and stop_requested()

    # planner + actor 模式
    # ------------------------------------------------------
    # 1) planner => get action/position JSON
    # 2) If action is None -> end
    # 3) Convert to executor format and execute
    # 4) repeat
    # ------------------------------------------------------
    while True:
        # 任务中接收用户新消息，追加到对话继续执行
        if pending_messages_callback:
            for new_msg in pending_messages_callback():
                if new_msg and isinstance(new_msg, str) and new_msg.strip():
                    messages.append({
                        "role": "user",
                        "content": [new_msg.strip()],
                    })
                    logger.info(f"任务中收到用户消息，已追加: {new_msg[:50]}...")
        if _should_stop():
            logger.info("收到停止命令，终止任务。")
            yield TASK_STOPPED
            return
        # Step 1: Planner (VLM) response - outputs action, value, position directly
        plan_data = None
        for parse_retry in range(PARSE_RETRY_COUNT + 1):
            try:
                vlm_response = planner(messages=messages)
                plan_data = json.loads(vlm_response)
                break
            except json.JSONDecodeError as e:
                if parse_retry < PARSE_RETRY_COUNT:
                    logger.warning("Planner 返回 JSON 格式错误，重试 %d/%d: %s", parse_retry + 1, PARSE_RETRY_COUNT, e)
                    continue
                logger.error("Planner JSON 解析失败，已达最大重试次数: %s", e)
                raise
        action_type = plan_data.get("action") or plan_data.get("Next Action")

        # Backward compat: "Next Action" format (text) -> treat as None to end
        if isinstance(action_type, str) and ("None" in action_type or not action_type.strip()):
            action_type = "None"

        yield action_type

        # Step 2a: 检查是否 Planner 需要向用户提问（不确定时暂停并等待用户回复）
        if str(action_type).upper() == "ASK_USER":
            question = (plan_data.get("value") or plan_data.get("question") or "").strip()
            if not question:
                question = "请提供更多信息以继续执行任务。"
            if ask_user_callback:
                try:
                    user_reply = ask_user_callback(question)  # 回调内部会展示问题并等待
                    if user_reply and user_reply.strip():
                        messages.append({
                            "role": "user",
                            "content": [f"用户回复: {user_reply.strip()}"],
                        })
                        output_callback(f"收到回复: {user_reply.strip()}", sender="bot")
                        logger.info("用户已回复，继续执行: %s", user_reply[:50])
                except Exception as e:
                    logger.warning("ask_user_callback 异常: %s，将用户回复视为空", e)
            else:
                # 无回调时，将问题作为提示追加，让模型在下一轮自行处理
                messages.append({
                    "role": "user",
                    "content": [f"[系统提示：需要用户确认] {question}"],
                })
            continue  # 不执行动作，直接进入下一轮 Planner

        # Step 2b: 检查是否 Planner 主动输出 FAIL
        if str(action_type).upper() == "FAIL":
            fail_reason = plan_data.get("value", "同一操作重复三次未完成")
            final_sc, final_sc_path = get_screenshot(selected_screen=selected_screen, resize=True, target_width=target_width, target_height=target_height)
            final_image_b64 = encode_image(str(final_sc_path))
            output_callback(
                (
                    f"❌ 任务失败: {fail_reason}\n"
                    f"<img src=\"data:image/png;base64,{final_image_b64}\">"
                ),
                sender="bot"
            )
            yield TASK_FAILED
            return

        # Step 2c: 检测同一操作是否重复三次
        action_key = _action_key(plan_data)
        recent_actions.append(action_key)
        if len(recent_actions) > REPEAT_FAIL_THRESHOLD:
            recent_actions.pop(0)
        if len(recent_actions) >= REPEAT_FAIL_THRESHOLD and len(set(recent_actions)) == 1:
            logger.info(f"同一操作重复 {REPEAT_FAIL_THRESHOLD} 次未完成，终止任务: {action_key}")
            final_sc, final_sc_path = get_screenshot(selected_screen=selected_screen, resize=True, target_width=target_width, target_height=target_height)
            final_image_b64 = encode_image(str(final_sc_path))
            output_callback(
                (
                    f"❌ 任务失败: 同一操作重复 {REPEAT_FAIL_THRESHOLD} 次未能完成，已终止。\n"
                    f"<img src=\"data:image/png;base64,{final_image_b64}\">"
                ),
                sender="bot"
            )
            yield TASK_FAILED
            return

        # Step 2d: Check if no further actions
        if not action_type or str(action_type) == "None":
            final_sc, final_sc_path = get_screenshot(selected_screen=selected_screen, resize=True, target_width=target_width, target_height=target_height)
            final_image_b64 = encode_image(str(final_sc_path))
            output_callback(
                (
                    f"No more actions from {colorful_text_vlm}. End of task. Final State:\n"
                    f'<img src="data:image/png;base64,{final_image_b64}">'
                ),
                sender="bot"
            )
            # SOP 保存：任务成功完成且有步骤时回调
            if not sop_save_callback:
                logger.info("SOP 未保存: memory_enabled 未启用")
            elif not action_history:
                logger.info("SOP 未保存: action_history 为空（Planner 首轮即返回 None，无执行步骤）")
            elif not user_task:
                logger.info("SOP 未保存: user_task 为空（无法从 messages 提取任务描述）")
            else:
                # 使用 Planner 在同一轮对话中给出的 summary，不再单独调用 summarizer
                summary = (plan_data.get("summary") or "").strip()
                if not summary:
                    summary = f"共 {len(action_history)} 步完成"
                try:
                    sop_save_callback(user_task, summary)
                    logger.info("SOP 已保存: task=%s", user_task[:40])
                except Exception as e:
                    logger.warning(f"SOP 保存回调异常: {e}")
            yield TASK_COMPLETE
            break

        # Step 3: Convert planner output to executor format (ShowUI-compatible)
        action_item = {
            "action": str(action_type).upper(),
            "value": plan_data.get("value"),
            "position": plan_data.get("position"),
        }
        # 收集成功步骤用于 SOP 保存（仅 action/value/position，不含 Thinking）
        action_history.append({
            "action": action_item["action"],
            "value": plan_data.get("value"),
            "position": plan_data.get("position"),
        })
        executor_input = {"content": str([action_item]), "role": "assistant"}

        output_callback(
            f"{colorful_text_vlm} executing: {action_type} {plan_data.get('value', '')} @ {plan_data.get('position', '')}",
            sender="bot"
        )

        # Step 4: Execute directly (no Actor)
        for message, tool_result_content in executor(executor_input, messages):
            if _should_stop():
                logger.info("收到停止命令，终止任务。")
                yield TASK_STOPPED
                return
            time.sleep(0.1)  # 短暂缓冲，避免阻塞 UI 更新
            yield message

        # Step 5: Update conversation history
        messages.append({
            "role": "user",
            "content": ["History plan:" + str(plan_data)],
        })

        logger.info(f"End of loop. Total cost: $USD{planner.total_cost:.5f}")
        showui_loop_count += 1
