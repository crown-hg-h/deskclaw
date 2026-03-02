# DeskClaw 工作流程图

## 整体流程

```mermaid
flowchart TB
    subgraph Entry["入口"]
        A[用户输入任务] --> B{入口类型}
        B -->|Gradio UI| C[app.py process_input]
        B -->|飞书消息| D[feishu_gateway _run_agent_task]
    end

    C --> E[sampling_loop_sync]
    D --> E

    subgraph Init["初始化 (loop.py)"]
        E --> F{Planner Model?}
        F -->|gpt-4o / qwen2-vl / Kimi 等| H[Planner + Direct 模式]
        H --> H2[APIVLMPlanner + ShowUIExecutor]
    end

    subgraph PlannerActorLoop["Planner + Direct 循环 (VLM)"]
        H2 --> P1[Planner 调用]
        
        P1 --> P2[截图 get_screenshot]
        P2 --> P3[VLM API 推理]
        P3 --> P4[解析 JSON plan_data]
        P4 --> P5{action 类型?}
        
        P5 -->|FAIL| P6[输出失败信息 + TASK_FAILED]
        P5 -->|同一操作重复3次| P6
        P5 -->|None| P7[输出完成信息 + TASK_COMPLETE]
        P5 -->|有效 action| P8[转换为 executor 格式]
        
        P8 --> P9[Executor 执行]
        P9 --> P10[pyautogui 等操作屏幕]
        P10 --> P11[追加 History plan 到 messages]
        P11 --> P12{用户停止?}
        P12 -->|是| P13[TASK_STOPPED]
        P12 -->|否| P1
    end

    P6 --> End
    P7 --> End
    P13 --> End
```

## Planner + Actor 模式详细流程

```mermaid
flowchart TB
    subgraph Loop["单次循环"]
        A[开始] --> B[Planner 调用]
        B --> B1[过滤 messages]
        B1 --> B2[get_screenshot 截图]
        B2 --> B3[output_callback 展示截图]
        B3 --> B4[VLM API: 任务 + 历史 + 截图]
        B4 --> B5[解析 JSON]
        
        B5 --> C{检查}
        C -->|action=FAIL| D1[输出失败 + TASK_FAILED]
        C -->|最近3次操作相同| D1
        C -->|action=None| D2[输出完成 + TASK_COMPLETE]
        C -->|有效 action| E[记录到 recent_actions]
        
        E --> F[转换为 action_item]
        F --> G[Executor 执行]
        G --> G1[解析 action 列表]
        G1 --> G2[CLICK→mouse_move+left_click]
        G2 --> G3[ComputerTool.sync_call]
        G3 --> G4[pyautogui 操作]
        G4 --> G5[yield 更新 UI]
        G5 --> H[追加 History plan 到 messages]
        H --> A
    end

    D1 --> Z[结束]
    D2 --> Z
```

## 组件关系图

```mermaid
flowchart LR
    subgraph 入口层
        App[app.py Gradio]
        Feishu[feishu_gateway]
    end

    subgraph 循环层
        Loop[loop.py sampling_loop_sync]
    end

    subgraph Planner层
        APIVLM[APIVLMPlanner]
        LocalVLM[LocalVLMPlanner]
    end

    subgraph Executor层
        ShowUIExec[ShowUIExecutor]
    end

    subgraph 工具层
        Computer[ComputerTool]
        Screen[get_screenshot]
    end

    App --> Loop
    Feishu --> Loop
    Loop --> APIVLM
    Loop --> LocalVLM
    Loop --> ShowUIExec
    APIVLM --> Screen
    LocalVLM --> Screen
    ShowUIExec --> Computer
```

## 终止条件汇总

| 条件 | 信号 | 触发位置 |
|------|------|----------|
| 任务完成 | TASK_COMPLETE | action=None 或 executor 无 tool_result |
| 任务失败 | TASK_FAILED | action=FAIL 或 同一操作重复3次 |
| 用户停止 | TASK_STOPPED | stop_requested() 返回 True |

## 记忆系统与 SOP（参考 pc-agent-loop）

```mermaid
flowchart TB
    subgraph Memory["记忆系统"]
        L0[L0 memory_management_sop]
        L2[L2 global_facts.json]
        L3[L3 sops/*.json]
    end

    subgraph TaskFlow["任务流程"]
        A[用户输入任务] --> B{启用记忆?}
        B -->|是| C[召回相关 SOP]
        C --> D[注入 prompt]
        B -->|否| D
        D --> E[Planner 执行]
        E --> F[收集 action_history]
        F --> G{任务完成?}
        G -->|是| H{自动保存 SOP?}
        H -->|是| I[保存到 memory/sops/]
        H -->|否| J[结束]
        I --> J
    end

    L3 --> C
    I --> L3
```

- **召回**：根据任务关键词匹配已有 SOP，注入到 system prompt 作为参考
- **保存**：任务成功完成后，将步骤序列保存为 SOP，下次类似任务可召回
