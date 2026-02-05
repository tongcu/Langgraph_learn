import gradio as gr
from typing import List, Dict, Any
from functools import partial
from pages.format import extract_message_info, format_ai_response

async def refresh_results_threads(manager):
    """刷新并获取生成结果页面的会话列表"""
    try:
        threads = await manager.client.threads.search(limit=50)
        choices = []
        for t in threads:
            u_id = t['thread_id']
            # 优先使用 task_id 作为展示名称
            display_name = t.get('values', {}).get('task_id') or f"会话({u_id[:8]})"
            choices.append((display_name, u_id))
        return gr.update(choices=choices)
    except Exception:
        return gr.update(choices=[], label="获取列表失败")

def render_results_page(graphmanager, session_id_comp):
    """
    渲染生成结果展示页面
    支持选择历史会话并展示其 final_content 和 chapter_details
    """
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 结果获取")
            refresh_list_btn = gr.Button("刷新会话列表", variant="secondary")
            thread_selector = gr.Dropdown(label="选择历史会话")
            gr.Markdown("从列表中选择一个会话以查看生成结果。")
            
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("完整文章", id="full_article"):
                    final_content_display = gr.Markdown("暂无内容，请从左侧选择会话。", line_breaks=True)
                
                with gr.TabItem("章节分段", id="chapter_segments"):
                    chapter_details_display = gr.Markdown("暂无章节详情。")
                
                with gr.TabItem("对话记录", id="conversation_history"):
                    history_chatbot = gr.Chatbot(label="对话历史回溯（人机交替）", height=600)
                
                # 新增：逐条AI消息标签页
                with gr.TabItem("AI消息逐条", id="ai_messages_individual"):
                    detailed_ai_chatbot = gr.Chatbot(
                        label="AI 消息逐条展示", 
                        height=600
                    )
                
                with gr.TabItem("原始数据", id="raw_data"):
                    raw_json_display = gr.JSON(label="State 原始数据")

    async def fetch_results_by_id(thread_id):
        if not thread_id:
            return "请选择一个有效的会话", "请选择会话", [], [], {}
        
        try:
            # 直接通过 thread_id 获取状态
            state = await graphmanager.client.threads.get_state(thread_id)
            if not state or "values" not in state:
                return "未找到该会话的状态数据。", "无数据", [], [], {}
            
            data = state["values"]
            
            # 1. 处理完整文章内容
            final_text = data.get("final_content") or data.get("merged_article") or "文章正文尚未生成。"
            
            # 2. 处理章节详情
            chapter_details = data.get("chapter_details", [])
            chapters_md = ""
            if chapter_details:
                for idx, detail in enumerate(chapter_details, 1):
                    title = detail.get("title", f"第 {idx} 章节")
                    content = detail.get("content", "内容正在生成中...")
                    chapters_md += f"## {title}\n\n{content}\n\n---\n\n"
            else:
                outline = data.get("outline", [])
                if outline:
                    chapters_md = "### 当前大纲已生成，正文正在写作中...\n\n"
                    for i, item in enumerate(outline, 1):
                        chapters_md += f"{i}. **{item.get('title')}**\n"
                else:
                    chapters_md = "尚未开始写作或未获取到章节详情。"
            
            # 3. 处理对话历史
            messages = data.get("messages", [])
            
            # 标准格式化聊天历史（人机交替）
            formatted_history_standard = []
            for msg in messages:
                role, content, tool_calls = extract_message_info(msg)
                
                gradio_role = "user" if role in ["human", "user"] else "assistant"
                
                # 处理工具调用显示
                display_content = content
                if tool_calls:
                    display_content = f"[工具调用]: {tool_calls[0]['name']}\n{content or ''}"
                    
                formatted_history_standard.append({
                    "role": gradio_role,
                    "content": display_content
                })
                
                # 应用 AI 回复格式化（如思考过程折叠）
                if gradio_role == "assistant":
                    display_content = format_ai_response(display_content)
            
            # 4. 逐条显示 AI 消息的格式化
            formatted_history_detailed = []
            ai_message_counter = 0  # 用于区分连续的AI消息
            for msg in messages:
                role, content, tool_calls = extract_message_info(msg)
                
                gradio_role = "user" if role in ["human", "user"] else "assistant"
                
                if content and content.strip():
                    display_content = content
                    if gradio_role == "assistant":
                        # 为每条AI消息添加唯一标识以防止合并
                        ai_message_counter += 1
                        display_content = f"<!-- AI_MSG_{ai_message_counter} -->{format_ai_response(content)}"
                    
                    formatted_history_detailed.append({
                        "role": gradio_role,
                        "content": display_content
                    })
                
                # 工具调用也单独显示
                if tool_calls and gradio_role == "assistant":
                    for call in tool_calls:
                        ai_message_counter += 1
                        tool_display = f"<!-- AI_MSG_{ai_message_counter} -->[工具调用]: {call['name']}\n参数: {call.get('args', {})}"
                        formatted_history_detailed.append({
                            "role": "assistant",
                            "content": tool_display
                        })
            
            return final_text, chapters_md, formatted_history_standard, formatted_history_detailed, data
        except Exception as e:
            return f"获取失败: {str(e)}", "错误", [], [], {}

    # 绑定事件
    refresh_list_btn.click(
        fn=partial(refresh_results_threads, manager=graphmanager),
        outputs=[thread_selector]
    )
    
    thread_selector.change(
        fn=fetch_results_by_id,
        inputs=[thread_selector],
        outputs=[final_content_display, chapter_details_display, history_chatbot, detailed_ai_chatbot, raw_json_display]
    )
