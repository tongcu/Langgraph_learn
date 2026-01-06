import os
import sys
# 注册当前目录，确保节点和工具能互相找到
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from nodes.states import WritingState as State

# 导入你独立出来的功能
from nodes.llm_nodes import call_model_vanilla, summary_agent_node

#  State
from tools.client_tool import tools
from tools.summary_tools import summary_tools_ex1
from langgraph.graph.message import add_messages

def custom_router(state: State):
    last_message = state["messages"][-1]
    
    # 逻辑 1: 如果有工具调用，去 "tools"
    if last_message.tool_calls:
        return "tools"
    
    # 逻辑 2: 如果没有工具调用，去 "refining_node" (而不是 END)
    return "call_model_vanilla"


# def vanilla_router(state: State):
#     """
#     判断 vanilla 节点后的去向
#     """
#     last_message = state["messages"][-1]
#     content = last_message.content.lower()
    
#     # 逻辑：如果用户在回复中包含了“谢谢”、“再见”或者你定义的结束词
#     # 或者由 LLM 判断用户是否已经满意
#     if "再见" in content or "完成" in content:
#         return END
    
#     # 否则，流程结束本次运行，等待用户下一次输入
#     # 在 LangGraph 中，如果你想让用户再次输入，通常直接指向 END 
#     # 下次调用 app.invoke 时，State 会保留，从而实现“对话”
#     return END

# class State(TypedDict):
#     messages: Annotated[list, add_messages]

# 2. 定义一个起始路由器
def start_router(state):
    # 根据 state 里的标记决定去向
    return state.get("next_action", "summary_agent")

# 构建图的逻辑
workflow = StateGraph(State)

workflow.add_conditional_edges(START, 
    start_router, 
    {
        "summary_agent": "summary_agent",
        "call_model_vanilla": "call_model_vanilla"
    }
)

workflow.add_node(
    "summary_agent", 
    summary_agent_node
)
workflow.add_node("call_model_vanilla", call_model_vanilla)
workflow.add_node("tools", ToolNode(summary_tools_ex1))

# workflow.add_edge(START, "summary_agent")
# workflow.add_conditional_edges("summary_agent", tools_condition)
workflow.add_conditional_edges(
    "summary_agent", 
    custom_router,
    {
        "tools": "tools", 
        "call_model_vanilla": "call_model_vanilla"
    }
)

workflow.add_edge("tools", "summary_agent")
workflow.add_conditional_edges("call_model_vanilla", END)

# 编译 app
memory = MemorySaver()
app = workflow.compile(
    #checkpointer=memory,
    interrupt_before=["tools"]
)