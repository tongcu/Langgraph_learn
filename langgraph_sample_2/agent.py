from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

def assistant_node(state: State):
    last_msg = state["messages"][-1].content
    if "删除" in last_msg:
        return {"messages": [("assistant", "⚠️ 检测到敏感操作，请确认是否继续执行？")]}
    return {"messages": [("assistant", f"已收到消息：{last_msg}")]}

def execute_node(state: State):
    return {"messages": [("assistant", "✅ 操作已成功执行。")]}

def create_graph():
    workflow = StateGraph(State)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("execute", execute_node)
    
    workflow.add_edge(START, "assistant")
    # 此处不需要写复杂的判断，直接定义流向，具体的“停止”由 interrupt 控制
    workflow.add_edge("assistant", "execute")
    workflow.add_edge("execute", END)
    
    # 编译时指定在 execute 节点前中断
    return workflow.compile(interrupt_before=["execute"])

graph = create_graph()