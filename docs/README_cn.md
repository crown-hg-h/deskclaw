# DeskClaw 中文文档

> 基于 [Computer Use OOTB](https://github.com/showlab/computer_use_ootb) 的桌面 GUI Agent 开源方案，支持多模型、飞书网关与记忆系统。本项目命名为 DeskClaw。

## 概览

本项目是桌面 GUI Agent 的开箱即用（OOTB）解决方案，支持：

- **API 模型**：Claude 3.5 Computer Use、GPT-4o、Qwen2-VL、Kimi-K2.5 (Azure)、Custom OpenAI 兼容
- **本地模型**：ShowUI、Qwen2-VL 本地/SSH、Ollama 部署的 Qwen2.5-VL
- **飞书网关**：通过飞书 WebSocket 长连接接收消息，在飞书中与机器人对话即可远程控制电脑，无需公网 IP
- **记忆系统**：支持 SOP 召回与自动保存，参考 pc-agent-loop

**无需 Docker**，支持 **Windows** 和 **macOS**，基于 Gradio 界面。

## 功能特性

| 功能 | 说明 |
|------|------|
| 多显示器 | 支持任意分辨率、多显示器 |
| 远程控制 | 手机/平板通过 Gradio 公网链接或飞书控制电脑 |
| 飞书网关 | 飞书消息 → DeskClaw 指令，无需公网 IP |
| 记忆/SOP | 任务完成后自动保存 SOP，下次类似任务可召回 |
| 环境变量 | 敏感信息（API Key、飞书凭证等）通过 `.env` 配置 |

## 快速开始

### 环境要求

- Python ≥ 3.11
- 可选（本地 ShowUI）：Windows 需 CUDA GPU ≥6GB；macOS 需 M1 及以上、16GB RAM

### 1. 克隆与安装

```bash
git clone https://github.com/你的用户名/deskclaw.git
cd deskclaw
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入真实凭证：

```bash
cp .env.example .env
```

在 `.env` 中配置：

- **Planner 模型**：`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`QWEN_API_KEY`、`AZURE_OPENAI_CREDENTIALS`、`CUSTOM_OPENAI_CREDENTIALS`、`OLLAMA_API_BASE` 等
- **飞书网关**：`FEISHU_APP_ID`、`FEISHU_APP_SECRET`、`FEISHU_DOMAIN`（可选）

详见 [.env.example](../.env.example)。

### 3. 启动界面

**主界面（Gradio）：**

```bash
python app.py
```

启动成功后终端会显示：

```
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxxxxxx.gradio.live
```

**飞书网关（可选）：**

```bash
python app_feishu_gateway.py
```

在网页中填写飞书凭证和 Planner 配置，点击「启动网关」即可在飞书中与机器人对话。

### 4. 使用方式

- **本机控制**：浏览器打开 `http://localhost:7860/`
- **远程控制**：手机/平板打开 `https://xxxxxxxx.gradio.live`（或通过飞书）
- 在界面输入任务指令，AI 将执行桌面操作

## 飞书网关

参考 [OpenClaw](https://docs.openclaw.ai/channels/feishu) 架构，通过飞书 WebSocket 长连接接入，无需公网 IP。

1. 在 [飞书开放平台](https://open.feishu.cn/app) 创建企业自建应用，获取 App ID 和 App Secret
2. 配置权限：`im:message`、`im:message:send_as_bot` 等，启用机器人能力
3. 事件订阅：添加 `im.message.receive_v1`，选择「使用长连接接收事件」
4. 启动方式二选一：
   - **Gradio 配置界面（推荐）**：`python app_feishu_gateway.py`，在网页中填写飞书凭证和 Planner 配置后点击「启动网关」
   - **命令行**：配置 `.env` 后运行 `python -m computer_use_demo.feishu_gateway`

## 支持模型

| 类型 | Planner | Actor |
|------|---------|-------|
| API | GPT-4o、Qwen2-VL-Max、Kimi-K2.5 (Azure)、Custom OpenAI | ShowUI |
| 本地 | Qwen2-VL-2B/7B | ShowUI |
| SSH | Qwen2-VL-2B/7B、Qwen2.5-VL-7B | ShowUI |
| Ollama | Qwen2.5-VL | ShowUI |

## 工作流程

详见 [WORKFLOW.md](./WORKFLOW.md)，包含：

- 整体流程（Gradio / 飞书入口 → Planner + Actor 循环）
- 记忆系统与 SOP 召回/保存流程

## 支持系统

- **Windows**：Claude ✅、ShowUI ✅
- **macOS**：Claude ✅、ShowUI ✅

## 注意事项

- **安全**：模型可能执行非预期操作，建议持续监督 AI 行为
- **成本**：Claude 3.5 Computer Use 单任务可能花费数美元；使用 GPT-4o + ShowUI 或 Qwen + ShowUI 可显著降低成本
- **隐私**：公网链接请勿分享，否则他人可控制你的电脑
