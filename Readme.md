

# LangGraph Sample API (Human-In-The-Loop)

本项目是一个基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的后端服务 demo，重点展示了如何在 API 环境中实现 **Human-In-The-Loop (HITL)** 交互机制。通过中断（Interrupt）和恢复（Resume）流程，实现了 AI 代理与人类用户之间的协同工作。

## 🌟 项目特性

* **状态管理**：利用 LangGraph 的 `StateGraph` 管理复杂的 agent 工作流。
* **人机协同 (HITL)**：集成 `interrupt` 和 `Command(resume=...)` 模式，支持 AI 在关键节点停下等待人工审批或输入。
* **解耦设计**：遵循功能独立原则，将复杂的业务逻辑拆分为独立的 Node 函数，便于维护和扩展。
* **REST API 集成**：演示了如何将异步的图执行过程封装进 Web 框架（如 FastAPI/Flask），实现线程持久化与状态恢复。

## 🏗️ 核心架构

项目采用模块化设计，确保每个功能块的独立性：

* **State 定义**：独立定义 `TypedDict`，明确节点间传递的数据结构。
* **Nodes (独立功能块)**：
* `call_model`: 负责 LLM 推理逻辑。
* `human_review `: 专门的“中断点”节点，负责挂起任务并等待外部指令。
* `tool_executor`: 独立封装的工具执行层。


* **Graph 编排**：统一的构建文件，负责串联节点并设置 Checkpointer（持久化层）。

## 启动命令
```shell
langgraph dev --host 0.0.0.0

gradio gradio_app.py
```
## Sample

### Sample 1 

### Sample 2 
测试对话和工具调用功能。


### Sample 3
测试对话及工具调用，完成基础功能。修改页面。
#### 项目文件夹结构

| 路径 | 职责描述 | 核心逻辑 |
| :--- | :--- | :--- |
| `agent.py`| **项目总入口** | 定义 LangGraph 状态机、节点循环及条件路由逻辑 |
| `gradio_app.py` | **UI 界面入口** | 负责前端交互、流式输出渲染及 Gradio 布局定义 |
| **`schema/`** | **数据契约层** | 独立存储所有结构化输出的 Pydantic 模型 |
| └── `summary_schema.py` | - | 定义 `GeneralReportSchema` 等，约束 AI 提取字段 |
| **`tools/`** | **工具实现层** | 独立存储 AI 可以调用的原子化函数 |
| └── `summary_tools.py` | - | 包含 `@tool` 装饰的总结函数，与业务逻辑解耦 |
| **`utils/`** | **独立逻辑层** | 存放与 UI 无关的通用处理函数 |
| ├── `graph_manager.py` | - | 封装线程监控、特定线程清理及全量重置逻辑 |
| ├── `message_parser.py` | - | 兼容多种消息格式（AI/Human/Tool）的解析器 |
| └── `formatter.py` | - | 负责将工具调用转化为 Markdown 引用块等特殊样式 |
| **`LLM/`** | **模型层** | 存放与模型调用相关的内容 |
| **`graph/`** | **langgraph manager** | langgraph Manager class |
| **`pages/`** | **pages manager** | langgraph Manager class |
| `.env` | **配置管理** | 存放 API 密钥、模型地址及端口配置 |


### sample 4
增加管理页面，检查服务情况及历史数据信息

#### 项目文件夹结构

| 路径 | 职责描述 | 核心逻辑 |
| :--- | :--- | :--- |
| `agent.py`| **项目总入口** | 定义 LangGraph 状态机、节点循环及条件路由逻辑 |
| `gradio_app.py` | **UI 界面入口** | 负责前端交互、流式输出渲染及 Gradio 布局定义 |
| **`schema/`** | **数据契约层** | 独立存储所有结构化输出的 Pydantic 模型 |
| └── `summary_schema.py` | - | 定义 `GeneralReportSchema` 等，约束 AI 提取字段 |
| **`tools/`** | **工具实现层** | 独立存储 AI 可以调用的原子化函数 |
| └── `summary_tools.py` | - | 包含 `@tool` 装饰的总结函数，与业务逻辑解耦 |
| **`utils/`** | **独立逻辑层** | 存放与 UI 无关的通用处理函数 |
| ├── `graph_manager.py` | - | 封装线程监控、特定线程清理及全量重置逻辑 |
| ├── `message_parser.py` | - | 兼容多种消息格式（AI/Human/Tool）的解析器 |
| └── `formatter.py` | - | 负责将工具调用转化为 Markdown 引用块等特殊样式 |
| **`LLM/`** | **模型层** | 存放与模型调用相关的内容 |
| **`graph/`** | **langgraph manager** | langgraph Manager class |
| **`pages/`** | **pages manager** | langgraph Manager class |
| `.env` | **配置管理** | 存放 API 密钥、模型地址及端口配置 |


### sample 5
ToDo:
增加大纲撰写，通过subagent完成


#### 项目文件夹结构

| 路径 | 职责描述 | 核心逻辑 |
| :--- | :--- | :--- |
| `agent.py`| **项目总入口** | 定义 LangGraph 状态机、节点循环及条件路由逻辑 |
| `gradio_app.py` | **UI 界面入口** | 负责前端交互、流式输出渲染及 Gradio 布局定义 |
| **`schema/`** | **数据契约层** | 独立存储所有结构化输出的 Pydantic 模型 |
| └── `summary_schema.py` | - | 定义 `GeneralReportSchema` 等，约束 AI 提取字段 |
| **`tools/`** | **工具实现层** | 独立存储 AI 可以调用的原子化函数 |
| └── `summary_tools.py` | - | 包含 `@tool` 装饰的总结函数，与业务逻辑解耦 |
| **`utils/`** | **独立逻辑层** | 存放与 UI 无关的通用处理函数 |
| ├── `graph_manager.py` | - | 封装线程监控、特定线程清理及全量重置逻辑 |
| ├── `message_parser.py` | - | 兼容多种消息格式（AI/Human/Tool）的解析器 |
| └── `formatter.py` | - | 负责将工具调用转化为 Markdown 引用块等特殊样式 |
| **`LLM/`** | **模型层** | 存放与模型调用相关的内容 |
| **`graph/`** | **langgraph manager** | langgraph Manager class |
| **`pages/`** | **pages manager** | langgraph Manager class |
| `.env` | **配置管理** | 存放 API 密钥、模型地址及端口配置 |
