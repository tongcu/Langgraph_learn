from langgraph.graph import StateGraph, END, START
from nodes.states import WritingState 
from nodes.writings.writing_nodes import outline_node, plan_node, retrieval_node, generate_chapter_node, merge_article_node
import logging
# def generate_outline(state: WritingState):
#     # 根据 topic 生成大纲的逻辑
#     return {"outline": "Generated Outline..."}


def writing_router(state: WritingState):
    """
    控制循环逻辑：
    判断当前完成的章节索引是否小于总章节数
    """
    curr_idx = state.get("current_chapter", 0)
    total_count = state.get("chapter_count", 3)
    
    if curr_idx < total_count:
        # 如果还没写完，返回执行检索的节点名
        logging.info(f"Continue writing chapter {curr_idx + 1} of {total_count}...")
        return "continue_writing"
    else:
        # 如果写完了，流向结束或保存节点
        logging.info(f"Finished {curr_idx} of {total_count}...")
        return "finish"


# 构建写作子图
writing_builder = StateGraph(WritingState)

writing_builder.add_node("outline_node", outline_node)
writing_builder.add_node("plan_node", plan_node)
writing_builder.add_node("retrieval_node", retrieval_node)
writing_builder.add_node("generate_chapter_node", generate_chapter_node)
writing_builder.add_node("merge_article_node", merge_article_node)  # 添加合并节点
# writing_builder.set_entry_point("plan_node")
# 3. 设置逻辑连线
writing_builder.add_edge(START, "plan_node")      # 从检索开始

writing_builder.add_edge("plan_node", "outline_node")
writing_builder.add_edge("outline_node", "retrieval_node")
writing_builder.add_edge("retrieval_node", "generate_chapter_node")

# 4. 关键：设置条件循环
writing_builder.add_conditional_edges(
    "generate_chapter_node",           # 从生成节点出来后进行判断
    writing_router,       # 调用上面的路由函数
    {
        "continue_writing": "retrieval_node", # 如果路由说继续，回到 retrieval 开启下一章
        "finish": "merge_article_node"                   # 如果路由说结束，去合并节点
    }
)

# 合并完成后结束
writing_builder.add_edge("merge_article_node", END)
# writing_builder.add_edge("write_article", END)

writing_subgraph = writing_builder.compile()