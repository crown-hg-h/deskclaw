#!/usr/bin/env python3
"""测试 SOP 步骤总结：直接调用 summarizer 并打印 API 响应"""
import os
import sys

# 加载 .env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))
except ImportError:
    pass

from computer_use_demo.gui_agent.llm_utils.oai import run_openai_compatible_interleaved

# 模拟 summarizer 的调用
raw_steps = [
    {"action": "CLICK", "value": None, "position": [0.627, 0.965]},
    {"action": "CLICK", "value": None, "position": [0.288, 0.259]},
    {"action": "INPUT", "value": "2025-03-02", "position": None},
]
task_description = "打开 wps，新建一个表格，在第一列填入今天日期"

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

messages = [{"role": "user", "content": [user]}]

raw = os.getenv("AZURE_OPENAI_CREDENTIALS", "")
if "|||" not in raw:
    print("错误: 请设置 AZURE_OPENAI_CREDENTIALS (格式: base|||key)")
    sys.exit(1)
parts = [p.strip() for p in raw.split("|||")]
base, key = parts[0], parts[1] if len(parts) > 1 else ""
model = parts[2] if len(parts) > 2 else "Kimi-K2.5"

print("调用 API...")
print("base:", base[:50], "...")
print("model:", model)
try:
    resp, tokens = run_openai_compatible_interleaved(
        messages=messages,
        system=system,
        llm=model,
        api_base=base,
        api_key=key,
        max_tokens=1024,
        temperature=0,
    )
    print("\n=== API 返回 ===")
    print("type(resp):", type(resp))
    print("len:", len(str(resp)) if resp else 0)
    print("resp:", repr(resp)[:500] if resp else "None/empty")
    if resp:
        print("\n=== 解析后的步骤描述 ===")
        print(resp)
except Exception as e:
    print("异常:", e)
    import traceback
    traceback.print_exc()
