"""
飞书网关 - 参考 OpenClaw 架构，将 DeskClaw 接入飞书机器人

通过飞书 WebSocket 长连接接收消息，转发给本地 Agent 执行，并将结果回复到飞书。
无需公网 IP，使用 lark-oapi SDK 建立长连接。

使用方式:
    python -m computer_use_demo.feishu_gateway

环境变量:
    FEISHU_APP_ID       - 飞书应用 App ID (cli_xxx)
    FEISHU_APP_SECRET   - 飞书应用 App Secret
    OPENAI_API_KEY       - Planner 模型 API Key (默认 gpt-4o)
    FEISHU_DOMAIN       - 可选，默认 https://open.feishu.cn，国际版用 https://open.larksuite.com
"""

from __future__ import annotations

import base64
import queue
import io
import json
import os
import re

# 启动时加载 .env 中的环境变量
try:
    from dotenv import load_dotenv
    # 优先从 cwd 加载（用户通常从项目根运行）
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    # 再从模块所在项目根加载（兼容子目录运行）
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from functools import partial
from typing import Any

from anthropic.types import TextBlock

from computer_use_demo.loop import APIProvider, TASK_COMPLETE, TASK_FAILED, TASK_STOPPED, sampling_loop_sync
from computer_use_demo.memory import MemoryManager
from computer_use_demo.tools.logger import logger

# 延迟导入，避免未安装 lark-oapi 时影响主应用
def _ensure_lark():
    try:
        import lark_oapi
        return lark_oapi
    except ImportError:
        raise ImportError(
            "飞书网关需要 lark-oapi，请运行: pip install lark-oapi"
        ) from None


def _extract_text_from_content(content: str | None, message_type: str | None) -> str:
    """从飞书消息 content 中提取纯文本。content 为 JSON 字符串。"""
    if not content or not content.strip():
        return ""
    try:
        data = json.loads(content)
        if message_type == "text":
            return (data.get("text") or "").strip()
        # 富文本等可扩展
        return (data.get("text") or str(data)).strip()
    except json.JSONDecodeError:
        return content.strip()


def _strip_mentions(text: str) -> str:
    """移除 @mention 格式，如 @_user_1 """
    import re
    return re.sub(r"@_user_\d+\s*", "", text).strip()


def _strip_html_for_feishu(text: str) -> str:
    """移除 HTML 标签，飞书纯文本不支持富文本"""
    text = re.sub(r"<img[^>]*>", "[图片]", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _extract_base64_images(text: str) -> list:
    """从文本中提取 base64 图片，返回 [(type, content), ...]，type 为 "text" 或 "image"。"""
    # 匹配 <img src="data:image/xxx;base64,..." ...> 或不带引号的变体
    pattern = r'<img[^>]*\bsrc=["\']?(data:image/[^;]+;base64,([^"\'>\s]+))["\']?[^>]*/?>'
    parts = []
    last_end = 0
    for m in re.finditer(pattern, text, re.IGNORECASE):
        prefix = text[last_end : m.start()].strip()
        if prefix:
            cleaned = _strip_html_for_feishu(prefix)
            if cleaned:
                parts.append(("text", cleaned))
        # group(2) 是纯 base64 数据（不含 data:image/png;base64, 前缀）
        parts.append(("image", m.group(2)))
        last_end = m.end()
    suffix = text[last_end:].strip()
    if suffix:
        cleaned = _strip_html_for_feishu(suffix)
        if cleaned:
            parts.append(("text", cleaned))
    return parts


def _upload_image_to_feishu(config, base64_data: str):
    """
    上传 base64 图片到飞书，返回 image_key。
    参考 OpenClaw feishu-bridge: FormData image_type=message + image=Blob。
    """
    try:
        import requests
        # 1. 获取 tenant_access_token
        token_resp = requests.post(
            f"{config.domain.rstrip('/')}/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": config.app_id, "app_secret": config.app_secret},
            timeout=10,
        )
        token_resp.raise_for_status()
        token_json = token_resp.json()
        token = token_json.get("tenant_access_token")
        if not token:
            logger.warning("获取 tenant_access_token 失败: %s", token_json)
            return None
        logger.info("获取 tenant_access_token 成功，开始上传图片...")

        # 2. 解码 base64，去掉可能的 data URI 前缀
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_data)
        logger.info("图片大小: %d bytes", len(img_bytes))

        # 3. 上传图片（Authorization: Bearer token，飞书标准鉴权方式）
        files = {"image": ("screenshot.png", io.BytesIO(img_bytes), "image/png")}
        data = {"image_type": "message"}
        upload_resp = requests.post(
            f"{config.domain.rstrip('/')}/open-apis/im/v1/images",
            headers={"Authorization": f"Bearer {token}"},
            data=data,
            files=files,
            timeout=30,
        )
        upload_resp.raise_for_status()
        resp_json = upload_resp.json()
        logger.info("飞书图片上传响应: %s", resp_json)
        if resp_json.get("code") != 0:
            logger.warning("飞书图片上传返回非0: %s", resp_json)
            return None
        image_key = resp_json.get("data", {}).get("image_key")
        logger.info("图片上传成功，image_key: %s", image_key)
        return image_key
    except Exception as e:
        logger.exception("飞书图片上传失败: %s", e)
        return None


def _render_message_for_feishu(message: Any, hide_images: bool = False) -> str | None:
    """
    与 app 的 _render_message 一致：每条 callback 只产生一条消息。
    返回渲染后的字符串（可能含 <img>），或 None。
    """
    if isinstance(message, str):
        return message
    from computer_use_demo.tools import ToolResult
    is_tool_result = (
        isinstance(message, ToolResult)
        or (hasattr(message, "__class__") and message.__class__.__name__ in ("ToolResult", "CLIResult"))
    )
    if not message or (is_tool_result and hide_images and not getattr(message, "error", None) and not getattr(message, "output", None)):
        return None
    if is_tool_result:
        if getattr(message, "output", None):
            return str(message.output)
        if getattr(message, "error", None):
            return f"Error: {message.error}"
        if getattr(message, "base64_image", None) and not hide_images:
            return f'<img src="data:image/png;base64,{message.base64_image}">'
        return None
    if hasattr(message, "text"):
        return getattr(message, "text", None) or str(message)
    return str(message)


# 飞书场景下用于接收停止命令的共享标志（线程安全）
_stop_requested_flag: dict = {"value": False}


# 飞书「向用户提问」时使用：Planner 不确定时暂停，等待用户下一条消息后继续
_feishu_user_reply_queue = queue.Queue()
_feishu_awaiting_reply = False

# 任务互斥：同一时间只允许一个 Agent 任务执行，避免多条消息导致任务交叉（如 Cursor 任务中突然执行微信操作）
_feishu_task_lock = threading.Lock()
_feishu_task_running = False


def _run_agent_task(
    user_input: str,
    send_reply_fn: callable,
    planner_model: str = "gpt-4o",
    planner_provider: str = "openai",
    api_key: str = "",
    system_prompt_suffix: str = "",
    config=None,
    stop_requested: Callable[[], bool] | None = None,
    ask_user_callback: Callable[[str], str] | None = None,
    selected_screen: int = 0,
) -> None:
    """
    与 app 一致：每条 output_callback/tool_output_callback 产生一条消息，立即发送到飞书。
    """

    def _send_piece(content_type: str, content: str):
        if not content or (content_type == "text" and not content.strip()):
            return
        send_reply_fn(content_type, content)

    def _send_one_message(rendered: str):
        """将一条渲染后的消息发送到飞书（可能含文本+图片，拆成多条飞书消息）"""
        if not rendered or not str(rendered).strip():
            return
        parts = _extract_base64_images(rendered)
        if not parts:
            stripped = _strip_html_for_feishu(rendered)
            if stripped:
                _send_piece("text", stripped)
            return
        for ptype, pcontent in parts:
            if ptype == "text" and pcontent.strip():
                _send_piece("text", pcontent)
            elif ptype == "image":
                _send_piece("image", pcontent)

    def output_callback(block: Any, sender=None):
        rendered = _render_message_for_feishu(block, hide_images=False)
        if rendered:
            _send_one_message(rendered)
        if not rendered and hasattr(block, "output") and block.output:
            _send_one_message(str(block.output))

    def tool_output_callback(tool_result: Any, tool_id: str):
        """与 app 一致：每条 ToolResult 只产生一条消息（output > error > base64_image）"""
        rendered = _render_message_for_feishu(tool_result, hide_images=False)
        if rendered:
            _send_one_message(rendered)

    def api_response_callback(_):
        pass

    provider_enum = APIProvider(planner_provider) if planner_provider in [p.value for p in APIProvider] else APIProvider.OPENAI
    device_os = "Windows" if platform.system() == "Windows" else "Mac" if platform.system() == "Darwin" else "Linux"
    suffix = system_prompt_suffix or f"\n\nNOTE: you are operating a {device_os} machine"

    # 记忆系统：召回相关 SOP 并注入 prompt
    try:
        mm = MemoryManager(base_dir="./memory")
        sop_hint = mm.recall_sops_as_prompt(user_input, top_k=2)
        if sop_hint:
            suffix += "\n\n" + sop_hint
            logger.info("飞书任务已召回 SOP 并注入 prompt")
    except Exception as e:
        logger.warning("飞书任务 SOP 召回异常: %s", e)

    messages = [
        {"role": "user", "content": [TextBlock(type="text", text=user_input)]}
    ]

    def _check_stop() -> bool:
        if stop_requested is not None:
            return stop_requested()
        return _stop_requested_flag.get("value", False)

    def _sop_save_callback(task_desc: str, summary: str) -> None:
        """summary 由 Planner 在任务完成时同一轮对话中给出，不再单独调用 summarizer"""
        try:
            steps_desc = summary.strip() or "任务已完成"
            mm = MemoryManager(base_dir="./memory")
            sop = mm.save_sop(task_desc, steps_desc)
            _send_piece("text", f"📝 已保存 SOP：{sop.task_description[:50]}... (success_count={sop.success_count})")
            logger.info("飞书任务 SOP 已保存: %s", sop.task_description[:40])
        except Exception as e:
            logger.warning("飞书任务 SOP 保存失败: %s", e)

    def _ask_user_for_feishu(question: str) -> str:
        """飞书场景：发送问题，等待用户下一条消息作为回复"""
        global _feishu_awaiting_reply
        _send_piece("text", f"❓ {question}\n\n请直接回复此消息。")
        _feishu_awaiting_reply = True
        try:
            return _feishu_user_reply_queue.get(timeout=300)
        except queue.Empty:
            logger.warning("等待飞书用户回复超时")
            return ""
        finally:
            _feishu_awaiting_reply = False

    _ask_user = ask_user_callback or _ask_user_for_feishu

    try:
        for msg in sampling_loop_sync(
            planner_model=planner_model,
            planner_provider=provider_enum,
            actor_model="Direct",
            actor_provider=None,
            system_prompt_suffix=suffix,
            messages=messages,
            output_callback=output_callback,
            tool_output_callback=tool_output_callback,
            api_response_callback=api_response_callback,
            api_key=api_key,
            only_n_most_recent_images=5,
            max_tokens=4096,
            selected_screen=selected_screen,
            showui_max_pixels=1344,
            showui_awq_4bit=False,
            ui_tars_url="",
            target_width=1920,
            target_height=1080,
            stop_requested=_check_stop,
            sop_save_callback=_sop_save_callback,
            ask_user_callback=_ask_user,
        ):
            if msg is TASK_COMPLETE:
                break
            if msg is TASK_FAILED:
                _send_piece("text", "❌ 任务失败（同一操作重复三次未完成）。")
                return
            if msg is TASK_STOPPED:
                _send_piece("text", "⏹️ 任务已停止。")
                return
        _send_piece("text", "✅ 任务已完成。")
    except Exception as e:
        logger.exception("Agent 执行异常: %s", e)
        _send_piece("text", f"执行出错: {e}")


def create_feishu_gateway(
    app_id: str,
    app_secret: str,
    *,
    domain: str = "https://open.feishu.cn",
    planner_model: str = "gpt-4o",
    planner_provider: str = "openai",
    api_key: str = "",
    system_prompt_suffix: str = "",
    selected_screen: int = 0,
    executor: ThreadPoolExecutor | None = None,
    chat_callback: Callable[[str, str], None] | None = None,
):
    """
    创建飞书网关：WebSocket 长连接 + 事件处理 + Agent 桥接。
    """
    lark = _ensure_lark()
    EventDispatcherHandler = lark.EventDispatcherHandler
    Client = lark.ws.Client
    LogLevel = lark.core.enum.LogLevel

    # 用于调用飞书回复 API（需用 resource.Message，非 model.Message）
    from lark_oapi.core.model import Config  # noqa: E402
    from lark_oapi.api.im.v1.resource.message import Message  # noqa: E402
    from lark_oapi.api.im.v1.model.reply_message_request import ReplyMessageRequest  # noqa: E402
    from lark_oapi.api.im.v1.model.reply_message_request_body import ReplyMessageRequestBody  # noqa: E402

    config = Config()
    config.app_id = app_id
    config.app_secret = app_secret
    config.domain = domain
    message_api = Message(config)

    pool = executor or ThreadPoolExecutor(max_workers=4)

    def send_reply_text(message_id: str, text: str):
        try:
            if len(text) > 1900:
                text = text[:1900] + "\n...(已截断)"
            body = ReplyMessageRequestBody.builder().content(
                json.dumps({"text": text}, ensure_ascii=False)
            ).msg_type("text").build()
            req = ReplyMessageRequest.builder().message_id(message_id).request_body(body).build()
            message_api.reply(req)
            logger.info("已回复飞书文本: %s", message_id[:20])
        except Exception as e:
            logger.exception("回复飞书失败: %s", e)

    def send_reply_image(message_id: str, base64_data: str):
        try:
            image_key = _upload_image_to_feishu(config, base64_data)
            if not image_key:
                send_reply_text(message_id, "[图片上传失败]")
                return
            body = ReplyMessageRequestBody.builder().content(
                json.dumps({"image_key": image_key}, ensure_ascii=False)
            ).msg_type("image").build()
            req = ReplyMessageRequest.builder().message_id(message_id).request_body(body).build()
            message_api.reply(req)
            logger.info("已回复飞书图片: %s", message_id[:20])
        except Exception as e:
            logger.exception("回复飞书图片失败: %s", e)

    def on_message_receive(data):
        """处理 im.message.receive_v1 事件"""
        global _feishu_task_running
        try:
            event = getattr(data, "event", None)
            if not event:
                return
            msg = getattr(event, "message", None)
            sender = getattr(event, "sender", None)
            if not msg or not sender:
                return

            # 忽略机器人自己的消息
            sender_type = getattr(sender, "sender_type", None) or ""
            if sender_type == "app":
                return

            message_id = getattr(msg, "message_id", None)
            chat_id = getattr(msg, "chat_id", None)
            message_type = getattr(msg, "message_type", None)
            content = getattr(msg, "content", None) or ""

            text = _extract_text_from_content(content, message_type)
            text = _strip_mentions(text)
            if not text:
                return

            # 忽略空或纯命令（可扩展）
            if not text.strip():
                return

            # 若任务正在等待用户回复（Planner 不确定时提问），将本条消息作为回复传入
            if _feishu_awaiting_reply:
                try:
                    _feishu_user_reply_queue.put_nowait(text)
                    logger.info("收到用户回复，已传入等待中的任务: %s", text[:50])
                except Exception:
                    pass
                if chat_callback:
                    chat_callback("user", text)
                return

            # 停止命令：设置标志，当前运行中的任务会在下一轮迭代时检测并终止
            if text.strip().lower() in ("stop", "停止", "stop!"):
                _stop_requested_flag["value"] = True
                logger.info("收到停止命令，已设置停止标志")
                send_reply_text(message_id, "⏹️ 已发送停止信号，当前任务将在下一步完成后终止。")
                if chat_callback:
                    chat_callback("assistant", "⏹️ 已发送停止信号")
                return

            # 任务互斥：若有任务正在执行，拒绝新任务，避免 Cursor 任务中突然执行微信操作等交叉
            with _feishu_task_lock:
                if _feishu_task_running:
                    send_reply_text(message_id, "⏳ 当前有任务正在执行，请先发送「停止」或等待完成后再发送新指令。")
                    if chat_callback:
                        chat_callback("assistant", "⏳ 当前有任务正在执行，请等待或发送「停止」。")
                    return
                _feishu_task_running = True

            # 新任务开始，清除停止标志
            _stop_requested_flag["value"] = False

            logger.info("收到飞书消息 [%s]: %s", chat_id[:16] if chat_id else "?", text[:80])
            if chat_callback:
                chat_callback("user", text)

            def _reply_and_callback(content_type: str, content: str):
                if content_type == "image":
                    send_reply_image(message_id, content)
                    # 前端用 base64 内嵌图片显示
                    display = f'<img src="data:image/png;base64,{content}">'
                else:
                    send_reply_text(message_id, content)
                    display = content
                if chat_callback:
                    chat_callback("assistant", display)

            def _run_and_clear():
                try:
                    _run_agent_task(
                        user_input=text,
                        send_reply_fn=_reply_and_callback,
                        planner_model=planner_model,
                        planner_provider=planner_provider,
                        api_key=api_key,
                        system_prompt_suffix=system_prompt_suffix,
                        config=config,
                        selected_screen=selected_screen,
                    )
                finally:
                    with _feishu_task_lock:
                        global _feishu_task_running
                        _feishu_task_running = False

            pool.submit(_run_and_clear)
        except Exception as e:
            logger.exception("处理飞书消息异常: %s", e)

    # WebSocket 长连接模式下，encrypt_key 和 verification_token 可为空
    handler = EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(
        on_message_receive
    ).build()

    client = Client(
        app_id=app_id,
        app_secret=app_secret,
        log_level=LogLevel.INFO,
        event_handler=handler,
        domain=domain,
        auto_reconnect=True,
    )
    return client


def main():
    import argparse
    parser = argparse.ArgumentParser(description="飞书网关 - DeskClaw")
    parser.add_argument("--app-id", default=os.getenv("FEISHU_APP_ID"), help="飞书 App ID")
    parser.add_argument("--app-secret", default=os.getenv("FEISHU_APP_SECRET"), help="飞书 App Secret")
    parser.add_argument("--domain", default=os.getenv("FEISHU_DOMAIN", "https://open.feishu.cn"))
    parser.add_argument("--planner-model", default=os.getenv("FEISHU_PLANNER_MODEL", "gpt-4o"))
    parser.add_argument("--planner-provider", default=os.getenv("FEISHU_PLANNER_PROVIDER", "openai"))
    parser.add_argument("--api-key", default="", help="Planner API Key，也可通过环境变量传入")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # 根据 provider/model 解析 api_key
    api_key = args.api_key or ""
    if (args.planner_provider == "azure" or args.planner_model == "Kimi-K2.5 (Azure)") and not api_key.strip():
        api_key = os.getenv("AZURE_OPENAI_CREDENTIALS", "")
    elif args.planner_provider == "openai" and not api_key.strip():
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not args.app_id or not args.app_secret:
        print("错误: 需要 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
        print("可通过环境变量或 --app-id / --app-secret 传入")
        print("若已写在 .env 中，请确保在项目根目录运行，或执行: pip install python-dotenv")
        raise SystemExit(1)

    if not api_key and args.planner_provider == "openai":
        print("警告: 未设置 OPENAI_API_KEY，Planner 可能无法调用")
    elif not api_key and args.planner_provider == "azure":
        print("警告: 未设置 AZURE_OPENAI_CREDENTIALS，Planner 可能无法调用")

    logger.info("启动飞书网关 (参考 OpenClaw)...")
    client = create_feishu_gateway(
        app_id=args.app_id,
        app_secret=args.app_secret,
        domain=args.domain,
        planner_model=args.planner_model,
        planner_provider=args.planner_provider,
        api_key=api_key,
    )
    client.start()


if __name__ == "__main__":
    main()
