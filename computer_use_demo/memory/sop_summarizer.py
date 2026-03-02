"""
SOP 步骤总结：调用 LLM 将原始操作转为一条可读的语义描述。
"""
import os

from computer_use_demo.tools.logger import logger

from computer_use_demo.gui_agent.llm_utils.oai import (
    run_oai_interleaved,
    run_openai_compatible_interleaved,
)


def summarize_steps_for_sop(
    task_description: str,
    raw_steps: list[dict],
    *,
    api_key: str = "",
    planner_provider: str = "openai",
    planner_model: str = "gpt-4o",
) -> str:
    """
    调用 LLM 将原始步骤总结为一条可读的语义描述。
    返回格式: "1. 点击 Dock 栏的 WPS 图标\n2. 点击文件菜单新建表格\n3. ..."
    若总结失败则返回空字符串（调用方可用简单描述兜底）。
    """
    if not raw_steps:
        return ""

    logger.info("SOP 步骤总结开始: provider=%s model=%s steps=%d", planner_provider, planner_model, len(raw_steps))

    system = """你是一个 GUI 操作步骤总结助手。用户完成了一个桌面任务，记录了原始的低级操作（CLICK/INPUT/PRESS 等）。
请用简洁的中文，将完整操作流程总结为一条可读的步骤描述。每步一行，格式如：
1. 点击 Dock 栏的 WPS 图标
2. 点击文件菜单新建表格
3. 在第一列输入今天日期
不要使用坐标，用界面元素和用户意图描述。直接输出步骤文本，不要输出 JSON 或其他格式。"""

    steps_brief = "、".join(
        f"{s.get('action', '')}({s.get('value') or s.get('position') or ''})" for s in raw_steps[:15]
    )
    user = f"""任务：{task_description}

原始操作序列：{steps_brief}

请输出可读的步骤描述（每步一行，1. 2. 3. ...）："""

    logger.info("SOP 总结发送内容: task=%s | 原始序列=%s", task_description[:60], steps_brief[:200])
    messages = [{"role": "user", "content": [user]}]
    try:
        if planner_provider == "openai":
            key = (api_key or "").strip() or os.getenv("OPENAI_API_KEY", "")
            if not key:
                raise ValueError("OpenAI API Key 未设置")
            resp, _ = run_oai_interleaved(
                messages=messages,
                system=system,
                llm="gpt-4o-mini" if "gpt-4o" in planner_model else planner_model,
                api_key=key,
                max_tokens=1024,
                temperature=0,
            )
        elif planner_provider == "qwen":
            key = (api_key or "").strip() or os.getenv("QWEN_API_KEY", "")
            if not key:
                raise ValueError("Qwen API Key 未设置")
            from computer_use_demo.gui_agent.llm_utils.qwen import run_qwen
            resp, _ = run_qwen(
                messages=messages,
                system=system,
                llm="qwen-vl-max",
                api_key=key,
                max_tokens=1024,
                temperature=0,
            )
        elif planner_provider in ("azure", "custom"):
            raw = (api_key or "").strip()
            if "|||" not in raw:
                raw = os.getenv(
                    "AZURE_OPENAI_CREDENTIALS" if planner_provider == "azure" else "CUSTOM_OPENAI_CREDENTIALS",
                    "",
                )
            if "|||" not in raw:
                raise ValueError("Azure/Custom 需要 API Base|||Key 格式")
            parts = [p.strip() for p in raw.split("|||")]
            base, key = parts[0], parts[1] if len(parts) > 1 else ""
            model = parts[2] if len(parts) > 2 else (planner_model if planner_provider == "custom" else "Kimi-K2.5")
            resp, _ = run_openai_compatible_interleaved(
                messages=messages,
                system=system,
                llm=model,
                api_base=base,
                api_key=key,
                max_tokens=1024,
                temperature=0,
            )
        else:
            logger.warning("SOP 总结不支持 provider=%s，跳过总结", planner_provider)
            return ""

        if isinstance(resp, dict):
            logger.warning("SOP 总结 API 返回异常: %s", resp)
            return ""

        summary = (resp or "").strip()
        if summary:
            logger.info("SOP 步骤已由模型总结")
            return summary
        logger.warning("SOP 总结 API 返回空内容，resp_type=%s resp_preview=%s", type(resp).__name__, repr(resp)[:300])
    except Exception as e:
        logger.warning("SOP 步骤总结失败: %s", e)

    return ""
