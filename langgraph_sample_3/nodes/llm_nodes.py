from langchain_core.runnables import RunnableConfig
from LLM.llm import get_llm
from tools.client_tool import tools
from .states import MessageState, WritingState
Default_model_name = "local_qwen"

def call_model_vanilla(state, config: RunnableConfig):
    # 1. 提取 configurable 部分（如果不存在则返回空字典）
    configurable = config.get("configurable", {})
    m_name = configurable.get("model_name", Default_model_name) 
    llm = get_llm(model_name=m_name)
    
    return {"messages": [llm.invoke(state["messages"])]} 

# 这里没有问题？ 三个月？可以的其他的问题

def call_model_tools(state, config: RunnableConfig):
    """动态获取 LLM 并支持工具调用"""
    # 从 config 中动态获取模型名称
    configurable = config.get("configurable", {})
    m_name = configurable.get("model_name", Default_model_name)
    # import pdb; pdb.set_trace()
    # 获取 LLM 实例并绑定工具
    llm = get_llm(model_name=m_name)
    llm_with_tools = llm.bind_tools(tools)
    
    # 执行
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


async def summary_agent_node(state: WritingState, config: RunnableConfig):
    """
    一个节点搞定：意图识别 + 领域聚焦 + 工具分发
    """
    logging.info("--- Agent 正在决策总结策略 ---")
    
    # 获取配置中的模型（通常在 workflow 配置中传入）
    configurable = config.get("configurable", {})
    m_name = configurable.get("model_name", Default_model_name)
    # import pdb; pdb.set_trace()
    # 获取 LLM 实例并绑定工具
    llm = get_llm(model_name=m_name)
    
    # 关键：将所有总结工具绑定到模型上
    from Workflow.tools import all_summary_tools
    llm_with_tools = llm.bind_tools(all_summary_tools)
    
    # 提示词引导：告诉模型不同工具的适用范围
    system_message = (
        "你是一个全能文档分析专家。请阅读内容并调用最合适的工具：\n"
        "- 论文/学术文章 -> summarize_science\n"
        "- 架构/设计文档 -> summarize_architecture\n"
        "- 需求/PRD文档 -> summarize_prd\n"
        "- 新闻/报道类 -> summarize_news\n"
        "- 其他一般报告 -> summarize_general"
    )
    
    # 直接调用
    response = await llm_with_tools.ainvoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": state["task"]}
    ])
    
    # 返回消息。LangGraph 的 ToolNode 会根据这里的 response.tool_calls 自动执行后续逻辑。
    return {"messages": [response]}