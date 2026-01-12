import gradio as gr
from graph.graph_manager import GraphManager
# from utils.message_parser import extract_message_info
from pages.format import extract_message_info
from functools import partial
from Utils.id import uuid_to_name_reversible
# from gradio_app import API_URL
# 可以在这里初始化，也可以由外部传入
# manager = GraphManager(api_url=API_URL)
# def render_admin_page(graphmanager: GraphManager)

def format_list_to_lines(items):
    """
    将列表项转换为每行一条的字符串（Markdown 格式）
    """
    if not items:
        return "无数据"
    # 使用 \n\n 确保在 Markdown 中产生清晰的分段
    return "\n\n".join([f"{i+1}. type:{item['type']}: {item}" for i, item in enumerate(items)])

async def refresh_threads_list(manager: GraphManager):
    try:
        threads = await manager.client.threads.search(limit=50)
        choices = []
        for t in threads:
            u_id = t['thread_id']
            try: 
                display_name = t['values']['task_id']
            
            # try:
            #     # --- 关键修改：调用你的可逆还原函数 ---
            #     # display_name = uuid_to_name_reversible(u_id)
            #     display_name = threads[0]['values']['task_id']
            except Exception:
                # 如果还原失败（比如不是按可逆算法生成的），则显示缩略 UUID
                display_name = f"未知会话({u_id[:8]})"
            
            # 这里的格式是 (展示给用户看的, 传给后端的实际值)
            choices.append((display_name, u_id))
        return gr.update(choices=choices)
    except:
        return gr.update(choices=[], label="连接失败")


async def load_thread_detail(thread_id, manager: GraphManager):
    if not thread_id: return "", []
    try:
        state = await manager.client.threads.get_state(thread_id)
        values = state.get("values", {})
        raw_text = values.get("task", "（未找到原文）")
        
        # 转换对话格式
        messages = values.get("messages", [])
        
        chat_history = []
        formatted_history = []
        for msg in messages:
            role, content, tool_calls = extract_message_info(msg)
            
            # 映射角色名
            gradio_role = "user" if role in ["human", "user"] else "assistant"
            
            # 处理工具调用显示
            display_content = content
            if tool_calls:
                display_content = f"🛠️ [工具调用]: {tool_calls[0]['name']}\n{content or ''}"
                
            formatted_history.append({
                "role": gradio_role,
                "content": display_content
            })
        # for msg in messages:
        #     role, content, tool_calls = extract_message_info(msg)
        #     if role in ["human", "user"]:
        #         chat_history.append([content, None])
        #     else:
        #         display = content
        #         if tool_calls:
        #             display = f"🛠️ [工具]: {tool_calls[0]['name']}\n{content}"
        #         if chat_history and chat_history[-1][1] is None:
        #             chat_history[-1][1] = display
        #         else:
        #             chat_history.append([None, display])
        
        return raw_text, formatted_history, format_list_to_lines(messages)
    except Exception as e:
        return f"加载失败: {str(e)}", []

# --- 核心：定义独立组件渲染函数 ---
def render_admin_page(graphmanager: GraphManager):
    """该函数会被 gradio_app.py 引用"""
    with gr.Row():
        with gr.Column(scale=1):
            refresh_btn = gr.Button("🔄 刷新线程列表", variant="primary")
            thread_selector = gr.Dropdown(label="选择历史线程")
            status_box = gr.Markdown("🟢 系统就绪")
            
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("💬 对话回溯"):
                    history_chatbot = gr.Chatbot(label="历史消息流", height=600)
                with gr.TabItem("📄 原始任务"):
                    history_raw = gr.TextArea(label="原文内容", lines=20, interactive=False)
                with gr.TabItem("📄 对话信息"):
                    # history_messages = gr.TextArea(label="messages", lines=20, interactive=False)
                    history_messages =gr.Markdown()

        with gr.Column(scale=1):
            session_id = gr.Textbox(label="会话 ID", value="user_session_01")
            clear_this_btn = gr.Button("🗑️ 删除线程", variant="stop")

    # 绑定事件
    # 2. 清理当前线程
    
    clear_this_btn.click(
                fn=graphmanager.clear_specific_thread,
                inputs=[session_id],
                outputs=[status_box]
            )

    refresh_threads_partial_fn = partial(refresh_threads_list, manager=graphmanager)
    refresh_btn.click(fn=refresh_threads_partial_fn,inputs=[],  outputs=[thread_selector])
    
    load_thread_detail_partial_fn = partial(load_thread_detail, manager=graphmanager)
    thread_selector.change(
        fn=load_thread_detail_partial_fn, 
        inputs=[thread_selector], 
        outputs=[history_raw, history_chatbot, history_messages]
    )
    # 你可以继续绑定 delete_btn 等逻辑