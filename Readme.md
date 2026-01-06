

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

### Sample 2 
测试对话和工具调用功能。


### Sample 3
