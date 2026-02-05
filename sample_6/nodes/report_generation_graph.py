"""报告生成子图"""
from langgraph.graph import StateGraph, START, END
from nodes.states import WritingState
from nodes.report_generation_nodes import report_generation_node, report_refinement_node


# 构建报告生成子图
report_generation_builder = StateGraph(WritingState)

# 添加节点
report_generation_builder.add_node("report_generation_node", report_generation_node)
report_generation_builder.add_node("report_refinement_node", report_refinement_node)

# 设置边
report_generation_builder.add_edge(START, "report_generation_node")
report_generation_builder.add_edge("report_generation_node", "report_refinement_node")
report_generation_builder.add_edge("report_refinement_node", END)

# 编译子图
report_generation_subgraph = report_generation_builder.compile()