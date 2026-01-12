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
    # import pdb; pdb.set_trace()
    # 逻辑 1: 如果有工具调用，去 "tools"
    if last_message.tool_calls:
        # 在这里打印 LLM 生成的原始工具调用参数
        for call in last_message.tool_calls:
            print(f" LLM 尝试调用工具: {call['name']}")
            print(f" 传入原始参数: \n {call['args']}")
        return "tools"
    
    # 逻辑 2: 如果没有工具调用，去 "refining_node" (而不是 END)
    return "call_model_vanilla"


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

workflow.add_edge("tools", "call_model_vanilla")
# workflow.add_conditional_edges("call_model_vanilla", END)
workflow.add_edge("call_model_vanilla", END)

# 编译 app
memory = MemorySaver()
app = workflow.compile(
    #checkpointer=memory,
    # interrupt_before=["tools"]
)