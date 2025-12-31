import os
import sys
# 注册当前目录，确保节点和工具能互相找到
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# 导入你独立出来的功能
from nodes.llm_nodes import call_model_dynamic
#  State
from tools.client_tool import tools
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
# 构建图的逻辑
workflow = StateGraph(State)
workflow.add_node("agent", call_model_dynamic)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# 编译 app
memory = MemorySaver()
app = workflow.compile(
    #checkpointer=memory,
    interrupt_before=["tools"]
)