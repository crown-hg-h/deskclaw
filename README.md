# DeskClaw

> Desktop GUI Agent based on [Computer Use OOTB](https://github.com/showlab/computer_use_ootb), supporting multi-model, Feishu gateway, and memory system.

## Overview
Like GPT5.4 computer-use 
Out-of-the-box (OOTB) solution for Desktop GUI Agent, supporting:

- **API Models**: **Kimi-K2.5 (Recommended)**, Claude 3.5 Computer Use, GPT-4o, Qwen2-VL, Custom OpenAI-compatible
- **Local Models**: ShowUI, Qwen2-VL local/SSH, Qwen2.5-VL via Ollama
- **Feishu Gateway**: Receive messages via Feishu WebSocket long connection; chat with the bot in Feishu to control your computer remotely, no public IP required
- **Memory System**: SOP recall and auto-save, inspired by pc-agent-loop

**No Docker** required. Supports **Windows** and **macOS**. Gradio-based interface.

## Features

| Feature | Description |
|---------|-------------|
| Multi-display | Any resolution, multiple monitors |
| Remote control | Control via Gradio public link or Feishu from phone/tablet |
| Feishu gateway | Feishu messages → DeskClaw commands, no public IP |
| Ask user | Pause when uncertain, ask user for clarification, then continue |
| Memory/SOP | Auto-save SOP after tasks; recall for similar tasks next time |
| Environment vars | Sensitive data (API keys, Feishu credentials) via `.env` |

## Quick Start

### Requirements

- Python ≥ 3.11

### 1. Clone and Install

```bash
git clone https://github.com/crown-hg-h/deskclaw.git
cd deskclaw
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Configure in `.env`:

- **Planner models** (Kimi-K2.5 recommended): `AZURE_OPENAI_CREDENTIALS` for Kimi-K2.5, or `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `QWEN_API_KEY`, `CUSTOM_OPENAI_CREDENTIALS`, `OLLAMA_API_BASE`, etc.
- **Feishu gateway**: `FEISHU_APP_ID`, `FEISHU_APP_SECRET`, `FEISHU_DOMAIN` (optional)

See [.env.example](.env.example) for details.

### 3. Start the Interface

```bash
python app.py
```

On success you see:

```
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxxxxxx.gradio.live
```

Fill in Feishu credentials and Planner config in the web UI, then click "Start Gateway" to chat with the bot in Feishu.

### 4. Usage

- **Local control**: Open `http://localhost:7860/` in browser
- **Remote control**: Open `https://xxxxxxxx.gradio.live` on phone/tablet (or via Feishu)
- Enter task instructions; the AI will perform desktop actions

## Feishu Gateway

Based on [OpenClaw](https://docs.openclaw.ai/channels/feishu) architecture, via Feishu WebSocket long connection. No public IP needed.

1. Create an enterprise app in [Feishu Open Platform](https://open.feishu.cn/app) and get App ID and App Secret
2. Enable permissions: `im:message`, `im:message:send_as_bot`, etc.
3. Event subscription: add `im.message.receive_v1`, choose "Use long connection for events"
4. Start in one of two ways:
   - **Gradio config UI (recommended)**: `python app.py`, fill in web form and click "Start Gateway"
   - **CLI**: Configure `.env` and run `python -m computer_use_demo.feishu_gateway`

## Supported Models

| Type | Planner | Actor |
|------|---------|-------|
| API | **Kimi-K2.5 (Recommended)** ⭐, GPT-4o, Qwen2-VL-Max, Custom OpenAI | ShowUI |
| Local | Qwen2-VL-2B/7B | ShowUI |
| SSH | Qwen2-VL-2B/7B, Qwen2.5-VL-7B | ShowUI |
| Ollama | Qwen2.5-VL | ShowUI |

> **Tip**: Kimi-K2.5 offers the best balance of accuracy, speed, and cost for desktop automation tasks.

## Workflow

See [WORKFLOW.md](docs/WORKFLOW.md) for:

- End-to-end flow (Gradio / Feishu → Planner + Actor loop)
- Memory system and SOP recall/save flow

## Supported Systems

- **Windows**: Claude ✅, ShowUI ✅
- **macOS**: Claude ✅, ShowUI ✅

## Notes

- **Security**: Models may perform unexpected actions; supervise AI behavior
- **Cost**: Claude 3.5 Computer Use can cost several dollars per task; **Kimi-K2.5** offers the best value with excellent performance at lower cost
- **Privacy**: Do not share public links; others could control your computer
