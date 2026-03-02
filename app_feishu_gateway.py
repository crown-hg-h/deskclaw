"""
DeskClaw 飞书网关 - Gradio 配置界面

提供前端配置飞书凭证和 Planner 模型，无需手写 .env。
启动后点击「启动网关」即可连接飞书，在飞书中与机器人对话远程控制电脑。

使用方式:
    python app_feishu_gateway.py
"""

import os

# 加载 .env，从项目根目录加载
_project_root = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_project_root, ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_env_path, encoding="utf-8")
except ImportError:
    import warnings
    warnings.warn("python-dotenv 未安装，.env 不会被加载。请运行: pip install python-dotenv")

import queue
import threading
import gradio as gr

from computer_use_demo.tools.logger import logger
from computer_use_demo.feishu_gateway import create_feishu_gateway, _run_agent_task, _stop_requested_flag

# 启动时确认 .env 加载状态
if os.path.exists(_env_path):
    logger.info("已加载 .env: %s (FEISHU_APP_ID=%s)", _env_path, "已设置" if os.getenv("FEISHU_APP_ID") else "未设置")
else:
    logger.warning(".env 不存在: %s，飞书凭证需在界面中手动填写", _env_path)

# 网关线程与状态
_gateway_thread = None  # threading.Thread
_gateway_client = None  # lark-oapi ws Client（用于停止）
_chat_history = []  # list of {"role": ..., "content": ...}，每条消息独立一项
_chat_lock = threading.Lock()

# 任务中向用户提问时使用：Planner 不确定时会暂停，等待用户回复后继续
_awaiting_user_reply = False
_user_reply_queue = queue.Queue()


def _append_chat(role: str, content: str):
    """追加到聊天历史，每条消息独立一项"""
    global _chat_history
    with _chat_lock:
        if role == "_system_":
            _chat_history.append({"role": "assistant", "content": f"📌 {content}"})
        else:
            _chat_history.append({"role": role, "content": content})
        if len(_chat_history) > 200:
            _chat_history.pop(0)


def _clean_cred(v: str) -> str:
    """去除凭证中的不可见字符（.env 的 CRLF、引号等可能导致飞书 API 报错）"""
    if not v:
        return ""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
        v = v[1:-1].strip()
    return "".join(c for c in v if ord(c) >= 32 or c in "\t").strip()


def _read_feishu_from_env_file(key: str) -> str:
    """直接从 .env 文件读取飞书凭证，避免 python-dotenv 解析或 env 被覆盖导致的值错误"""
    if not os.path.exists(_env_path):
        return ""
    try:
        with open(_env_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig 自动去除 BOM
            for line in f:
                line = line.strip("\r\n\t ")
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key:
                    return _clean_cred(v)
    except Exception as e:
        logger.warning("读取 .env 失败: %s", e)
    return ""


def _run_gateway_thread(
    app_id: str,
    app_secret: str,
    domain: str,
    planner_model: str,
    planner_provider: str,
    api_key: str,
    chat_callback,
):
    """在后台线程中运行网关"""
    global _gateway_client
    try:
        app_id = _clean_cred(app_id or "")
        app_secret = _clean_cred(app_secret or "")
        domain = _clean_cred(domain or "") or "https://open.feishu.cn"
        _append_chat("_system_", "正在连接飞书...")
        logger.info("启动网关 app_id=%s*** app_secret=***", app_id[:8] if app_id else "")
        client = create_feishu_gateway(
            app_id=app_id,
            app_secret=app_secret,
            domain=domain,
            planner_model=planner_model,
            planner_provider=planner_provider,
            api_key=api_key,
            chat_callback=chat_callback,
        )
        _gateway_client = client
        _append_chat("_system_", "已连接飞书 WebSocket，等待消息...")
        client.start()
    except Exception as e:
        _append_chat("_system_", f"网关异常: {e}")
        logger.exception("Gateway error")
    finally:
        _gateway_client = None
        _append_chat("_system_", "网关已停止")


def build_api_key(planner_model: str, api_base: str, api_key: str, model_id: str) -> str:
    """根据三栏配置拼接 api_key"""
    base = (api_base or "").strip()
    key = (api_key or "").strip()
    mid = (model_id or "gpt-4o").strip()
    provider = model_to_provider(planner_model)
    if provider == "azure":
        if base and key:
            return f"{base}|||{key}"
        return os.getenv("AZURE_OPENAI_CREDENTIALS", "")
    if provider == "custom":
        if base and key:
            return f"{base}|||{key}|||{mid}"
        return os.getenv("CUSTOM_OPENAI_CREDENTIALS", "")
    if provider == "openai":
        return key or os.getenv("OPENAI_API_KEY", "")
    if provider == "qwen":
        return key or os.getenv("QWEN_API_KEY", "")
    return key


def start_gateway(app_id, app_secret, domain, planner_model, api_base, planner_api_key, model_id):
    """启动/重启网关按钮回调。若网关已在运行，会先停止再以当前配置重启。"""
    global _gateway_thread, _gateway_client, _chat_history
    # 当 App Secret 为空时，从 .env 回退读取（Gradio 密码框可能未正确传递预填值）
    if not (app_secret or "").strip():
        app_secret = _read_feishu_from_env_file("FEISHU_APP_SECRET")
    if not (app_id or "").strip():
        app_id = _read_feishu_from_env_file("FEISHU_APP_ID")
    if not (app_id and app_secret):
        _append_chat("_system_", "请填写飞书 App ID 和 App Secret")
        return _get_chat_display()
    planner_provider = model_to_provider(planner_model)
    api_key = build_api_key(planner_model, api_base, planner_api_key, model_id)
    if not api_key and planner_provider in ("openai", "azure", "custom"):
        _append_chat("_system_", "警告: 未设置 Planner API Key。Custom/Azure 需填写接口地址、API Key 和模型 ID。")
        return _get_chat_display()
    # 若已在运行，先停止（lark-oapi ws client 支持 stop()）
    if _gateway_client is not None:
        try:
            _gateway_client.stop()
        except Exception:
            pass
        _gateway_client = None
    with _chat_lock:
        _chat_history.clear()

    def chat_callback(role: str, content: str):
        _append_chat(role, content)

    t = threading.Thread(
        target=_run_gateway_thread,
        args=(app_id, app_secret, domain, planner_model, planner_provider, api_key, chat_callback),
        daemon=True,
    )
    _gateway_thread = t
    t.start()
    return _get_chat_display()


def _make_ask_user_callback():
    """创建 ask_user 回调：暂停任务，在聊天中显示问题，等待用户回复后继续"""

    def ask_user(question: str) -> str:
        global _awaiting_user_reply
        _append_chat("assistant", f"❓ {question}\n\n请在下方输入框回复后按发送。")
        _awaiting_user_reply = True
        try:
            return _user_reply_queue.get(timeout=300)  # 5 分钟超时
        except queue.Empty:
            logger.warning("等待用户回复超时")
            return ""
        finally:
            _awaiting_user_reply = False

    return ask_user


def _run_local_task(user_input: str, planner_model: str, api_base: str, planner_api_key: str, model_id: str):
    """在后台线程中直接执行 Agent 任务（不经过飞书），结果写入 _chat_history"""
    planner_provider = model_to_provider(planner_model)
    api_key = build_api_key(planner_model, api_base, planner_api_key, model_id)

    def send_fn(content_type: str, content: str):
        if content_type == "image":
            _append_chat("assistant", f'<img src="data:image/png;base64,{content}">')
        else:
            _append_chat("assistant", content)

    threading.Thread(
        target=_run_agent_task,
        args=(user_input, send_fn, planner_model, planner_provider, api_key),
        kwargs={"ask_user_callback": _make_ask_user_callback()},
        daemon=True,
    ).start()


def _get_chat_display():
    """获取用于 Chatbot 显示的聊天历史（messages 格式，每条消息独立一项）"""
    with _chat_lock:
        if not _chat_history:
            return [{"role": "assistant", "content": "填写配置后点击「启动/重启网关」。飞书消息将作为 DeskClaw 指令发送给 Agent。修改配置后再次点击可重启并应用新配置。"}]
        return list(_chat_history)


def model_to_provider(model: str) -> str:
    """模型 -> API Provider"""
    if model == "Kimi-K2.5 (Azure)":
        return "azure"
    if model in ("gpt-4o", "gpt-4o-mini"):
        return "openai"
    if model == "qwen2-vl-max":
        return "qwen"
    if model == "Custom (OpenAI)":
        return "custom"
    return "openai"


def parse_credentials_from_env(env_val: str, model: str):
    """从 env 解析出 (api_base, api_key, model_id)"""
    if not env_val or "|||" not in env_val:
        return "", env_val or "", "gpt-4o"
    parts = [p.strip() for p in env_val.split("|||")]
    base = parts[0] if len(parts) > 0 else ""
    key = parts[1] if len(parts) > 1 else ""
    model_id = parts[2] if len(parts) > 2 else ("gpt-4o" if model == "Custom (OpenAI)" else "")
    return base, key, model_id


def setup_planner_for_model(model: str):
    """根据模型返回 (base_visible, base_ph, key_ph, base_default, key_default, model_default)"""
    if model == "Kimi-K2.5 (Azure)":
        b, k, _ = parse_credentials_from_env(os.getenv("AZURE_OPENAI_CREDENTIALS", ""), model)
        return True, "https://xxx.azure.com/openai/v1", "API Key", b, k, ""
    if model == "Custom (OpenAI)":
        raw = os.getenv("CUSTOM_OPENAI_CREDENTIALS", "")
        b, k, m = parse_credentials_from_env(raw, model)
        return True, "https://api.xxx.com/v1", "sk-xxx", b, k, m or "gpt-4o"
    if model in ("gpt-4o", "gpt-4o-mini"):
        return False, "", "OpenAI API Key", "", os.getenv("OPENAI_API_KEY", ""), ""
    if model == "qwen2-vl-max":
        return False, "", "Qwen API Key", "", os.getenv("QWEN_API_KEY", ""), ""
    return False, "", "API Key", "", "", ""


with gr.Blocks(title="飞书网关 - DeskClaw") as demo:
    gr.Markdown("# 飞书网关 - DeskClaw")
    gr.Markdown("配置飞书机器人和 Planner 模型，启动后在飞书中与机器人对话即可远程控制电脑。")
    gr.Markdown("> **飞书收到的消息** = App 中的「发往 DeskClaw 的指令」，与在 Gradio 界面直接输入等效。")

    def _env_val(key: str) -> str:
        """飞书凭证优先从 .env 文件直接读取，避免 dotenv 解析问题；其他 key 用 os.getenv"""
        if key.startswith("FEISHU_"):
            v = _read_feishu_from_env_file(key)
            if v:
                return v
        return _clean_cred(os.getenv(key) or "")

    with gr.Accordion("飞书配置", open=True):
        feishu_app_id = gr.Textbox(
            label="App ID",
            value=_env_val("FEISHU_APP_ID"),
            placeholder="cli_xxx",
            interactive=True,
        )
        feishu_app_secret = gr.Textbox(
            label="App Secret",
            value=_env_val("FEISHU_APP_SECRET"),
            type="password",
            placeholder="飞书应用密钥",
            interactive=True,
        )
        feishu_domain = gr.Textbox(
            label="API 域名（可选）",
            value=_env_val("FEISHU_DOMAIN") or "https://open.feishu.cn",
            placeholder="https://open.feishu.cn，国际版用 https://open.larksuite.com",
            interactive=True,
        )
        load_env_btn = gr.Button("从 .env 加载飞书凭证", size="sm", variant="secondary")

        def load_from_env():
            """从 .env 重新读取飞书凭证并更新到输入框"""
            return (
                _read_feishu_from_env_file("FEISHU_APP_ID"),
                _read_feishu_from_env_file("FEISHU_APP_SECRET"),
                _read_feishu_from_env_file("FEISHU_DOMAIN") or "https://open.feishu.cn",
            )

        load_env_btn.click(
            fn=load_from_env,
            inputs=None,
            outputs=[feishu_app_id, feishu_app_secret, feishu_domain],
        )

    with gr.Accordion("Planner 模型配置", open=True):
        planner_model = gr.Dropdown(
            label="Planner Model",
            choices=[
                "gpt-4o",
                "gpt-4o-mini",
                "qwen2-vl-max",
                "Kimi-K2.5 (Azure)",
                "Custom (OpenAI)",
            ],
            value=os.getenv("FEISHU_PLANNER_MODEL", "Kimi-K2.5 (Azure)"),
            interactive=True,
        )
        with gr.Row():
            api_base_url = gr.Textbox(
                label="接口地址",
                value="",
                placeholder="https://xxx.azure.com/openai/v1",
                interactive=True,
            )
            planner_api_key = gr.Textbox(
                label="API Key",
                value="",
                type="password",
                placeholder="API Key",
                interactive=True,
            )
            custom_model_id = gr.Textbox(
                label="模型 ID",
                value="gpt-4o",
                placeholder="gpt-4o",
                interactive=True,
            )

    start_btn = gr.Button("启动/重启网关", variant="primary")
    chat_history = gr.Chatbot(
        label="对话记录（飞书消息 / 本地输入 → DeskClaw 指令 → Agent 回复）",
        value=[{"role": "assistant", "content": "填写配置后点击「启动/重启网关」。也可直接在下方输入框发送指令本地执行。"}],
        height=500,
        sanitize_html=False,
    )
    with gr.Row():
        local_input = gr.Textbox(
            label="本地指令",
            placeholder="直接输入指令，按 Enter 或点击发送，不经过飞书直接执行",
            scale=5,
            lines=1,
            interactive=True,
        )
        send_btn = gr.Button("发送", variant="primary", scale=1)

    def on_model_change(model):
        base_visible, base_ph, key_ph, base_default, key_default, model_default = setup_planner_for_model(model)
        return (
            gr.update(placeholder=base_ph, value=base_default, visible=base_visible),
            gr.update(placeholder=key_ph, value=key_default),
            gr.update(value=model_default, visible=model == "Custom (OpenAI)"),
        )

    planner_model.change(
        fn=on_model_change,
        inputs=[planner_model],
        outputs=[api_base_url, planner_api_key, custom_model_id],
    )

    def init_planner():
        model = os.getenv("FEISHU_PLANNER_MODEL", "Kimi-K2.5 (Azure)")
        base_visible, base_ph, key_ph, base_default, key_default, model_default = setup_planner_for_model(model)
        return (
            gr.update(value=base_default, placeholder=base_ph, visible=base_visible),
            gr.update(value=key_default, placeholder=key_ph),
            gr.update(value=model_default or "gpt-4o", visible=model == "Custom (OpenAI)"),
        )

    demo.load(fn=init_planner, inputs=None, outputs=[api_base_url, planner_api_key, custom_model_id])

    def do_start(app_id, app_secret, domain, pm, base, pk, mid):
        return start_gateway(app_id, app_secret, domain, pm, base, pk, mid)

    start_btn.click(
        fn=do_start,
        inputs=[feishu_app_id, feishu_app_secret, feishu_domain,
                planner_model, api_base_url, planner_api_key, custom_model_id],
        outputs=[chat_history],
    )

    def do_send(user_input, pm, base, pk, mid):
        if not user_input or not user_input.strip():
            return _get_chat_display(), ""
        text = user_input.strip()
        # 若任务正在等待用户回复（Planner 不确定时提问），将输入作为回复传入
        if _awaiting_user_reply:
            _append_chat("user", text)
            try:
                _user_reply_queue.put_nowait(text)
            except Exception:
                pass
            return _get_chat_display(), ""
        _append_chat("user", text)
        # 停止命令：设置标志，当前运行中的任务会在下一轮迭代时检测并终止
        if text.lower() in ("stop", "停止", "stop!"):
            _stop_requested_flag["value"] = True
            _append_chat("assistant", "⏹️ 已发送停止信号，当前任务将在下一步完成后终止。")
            return _get_chat_display(), ""
        _run_local_task(text, pm, base, pk, mid)
        return _get_chat_display(), ""

    send_btn.click(
        fn=do_send,
        inputs=[local_input, planner_model, api_base_url, planner_api_key, custom_model_id],
        outputs=[chat_history, local_input],
    )
    local_input.submit(
        fn=do_send,
        inputs=[local_input, planner_model, api_base_url, planner_api_key, custom_model_id],
        outputs=[chat_history, local_input],
    )

    # 每 2 秒刷新对话记录
    gr.Timer(value=2).tick(fn=_get_chat_display, inputs=None, outputs=[chat_history])

demo.launch(share=False, allowed_paths=["./"], theme=gr.themes.Soft())
