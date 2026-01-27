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
from graph.utils import export_agent_graph

def custom_router(state: State):
    last_message = state["messages"][-1]
    # import pdb; pdb.set_trace()
    print(f"custom_router next_action {state.get("next_action", "")}")
    # 逻辑 1: 如果有工具调用，去 "tools"
    if last_message.tool_calls:
        # 在这里打印 LLM 生成的原始工具调用参数
        for call in last_message.tool_calls:
            print(f" LLM 尝试调用工具: {call['name']}")
            print(f" 传入原始参数: \n {call['args']}")
        return "tools"
    
    # 逻辑 2: 如果没有工具调用，去 "refining_node" (而不是 END)
    return "call_model_vanilla"


def task_coordinator(state: dict):
    """任务协调器：判断当前该去哪个节点"""
    logging.info("--- [Task Coordinator] 正在分配任务 ---")
    
    topic = state.get("topic")
    outline = state.get("outline")
    outline_generated = state.get("outline_generated", False)
    
    # 1. 如果连 Topic 都没有，或者正在对话获取 Topic 阶段
    if not topic:
        logging.info("状态：未确定主题或大纲，进入规划环节")
        return Command(goto="plan_node")
    
    if not outline_generated:
        logging.info("状态：未确定主题或大纲，进入规划环节")
        return Command(goto="plan_node")

    # 2. 如果已经有大纲了，但还没有开始写章节
    if outline and not state.get("chapters"):
        logging.info("状态：大纲已就绪，进入正文写作环节")
        return Command(goto="__end__") # 假设你后续有写作节点

    # 3. 兜底：如果任务都完成了
    return Command(goto="__end__")

# 2. 定义一个起始路由器
def start_router(state):
    # 根据 state 里的标记决定去向
    print(f"start_router summary_text {state.get("summary_text", "unknown")}")
    summary_text = state.get("summary_text", None)
    # import pdb; pdb.set_trace()
    if summary_text == None:
        print("No next action, going to summary_agent")
        return "summary_agent"
    else:
        print("No next action, going to call_model_vanilla")
        return "call_model_vanilla"

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
graph_filename="agent_graph.png"
export_agent_graph(app, graph_filename)
