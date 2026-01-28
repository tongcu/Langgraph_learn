import re 
import gradio as gr
import asyncio
import hashlib
from uuid import UUID
from langgraph_sdk import get_client
from langchain_core.messages import AIMessage, HumanMessage
from graph.graph_manager import GraphManager # 引用独立功能
from Utils.id import name_to_uuid_nr as name_to_uuid

# hostname = "http://langgraph-api-learn-2026-pre1231:2024"
# # GRAPH_ID = "my_agent"
# if hostname is None:
#     API_URL = "http://127.0.0.1:2024"
# else:
#     API_URL = hostname



from pages.format import format_tool_call_simple
from pages import render_admin_page
from config import settings
API_URL = settings.API_URL
# 初始化管理器
graphmanager = GraphManager(api_url=API_URL)


GRAPH_ID = "my_agent"

def format_ai_response(text: str) -> str:
    """
    如果存在 <think> 标签，将其包装在折叠框内；
    如果没有，则直接输出结果。
    """
    if not text: return ""
    
    pattern = r'<(?:think|thought)>(.*?)</(?:think|thought)>'
    match = re.search(pattern, text, flags=re.DOTALL)
    
    if match:
        thought_content = match.group(1).strip()
        # 移除正文中的 think 部分
        answer_content = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        return f"<details><summary><b> 思考过程 (点击展开)</b></summary>\n\n{thought_content}\n\n</details>\n\n{answer_content}"
    
    return text


async def get_thread_status(session_id):
    """
    独立功能：探测指定 thread 的实时运行节点
    """
    client = get_client(url=API_URL)
    thread_id = name_to_uuid(session_id)
    
    try:
        # 获取当前 thread 的最新状态
        state = await client.threads.get_state(thread_id)
        
        if not state or not state.get("next"):
            return "✅ 当前没有正在运行或等待的任务。"
        
        # next 字段包含了即将执行或正在执行的节点名称
        current_nodes = state["next"]
        values = state.get("values", {})
        
        status_report = f" **当前停滞位置**: {current_nodes}\n"
        status_report += f" **消息总数**: {len(values.get('messages', []))} 条\n"
        
        if "task" in values:
            status_report += f" **上下文状态**: 已加载 (长度: {len(values['task'])})\n"
            
        return status_report
    except Exception as e:
        return f"❌ 无法获取状态: {str(e)}"

    
def extract_message_info(msg):
    """
    极致兼容版：从各种消息格式中提取角色、内容和工具调用。
    支持：
    1. LangChain 原始对象 (.type, .content)
    2. LangGraph 序列化字典 (['type'], ['content'])
    3. 标准 OpenAI 字典 (['role'], ['content'])
    """
    if not msg:
        return "", "", []

    # 1. 提取角色 (Role/Type)
    # 优先级：字典的 type > 字典的 role > 对象的 type属性
    if isinstance(msg, dict):
        role = msg.get("type") or msg.get("role") or ""
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        
        # 兼容性补丁：有些模型会把 tool_calls 塞在 additional_kwargs 里
        if not tool_calls and "additional_kwargs" in msg:
            tool_calls = msg["additional_kwargs"].get("tool_calls", [])
    else:
        role = getattr(msg, "type", "")
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])

    # 2. 修正：如果 content 为空字符串，但在 tool_calls 里有东西，
    # 我们认为这也是一种有效的“回复”
    return role, content, tool_calls

# --- 2. 重构后的核心预测逻辑 ---
async def predict(message, history, model_selector, task_context, session_id, file_obj):
    client = get_client(url=API_URL)
    thread_id = name_to_uuid(session_id)
    # import pdb 
    # pdb.set_trace()
    # 确保线程存在
    try:
        await client.threads.get(thread_id)
    except:
        await client.threads.create(thread_id=thread_id)
        print(f"INFO: Created new thread: {thread_id}")

    # 构造输入状态
    input_state = {
        "task": task_context,
        "messages": [{"role": "user", "content": message}],
        "task_id" : session_id
    }
    
    if file_obj is not None:
        input_state["files"] = [file_obj.name]

    status_prefix = ""  # 用于存储工具调用的中间状态
    last_yielded_content = ""
    run_config = {
        "configurable": {
            "model_name": model_selector  # 对应你 node 里的 key
        },
        "recursion_limit": 50
    }
    yielded_at_least_once = False # 状态标记
    try:
        async for event in client.runs.stream(
            thread_id,
            GRAPH_ID,
            input=input_state,
            # 同时监听 values(状态全量) 和 updates(节点运行轨迹)
            stream_mode=["values", "updates"],
            config=run_config
            # config={
            #     "configurable": {},
            #     "recursion_limit": 50,    # 递归深度限制
            #     "concurrency_limit": 1    # 单个 Run 内部并行的分支数限制
            #     }
        ):
            # print(f"DEBUG FRONTEND: 收到节点 {event.data} 的更新")
            print(f"DEBUG FRONTEND: 收到event节点 {event.event} 的更新")
            if event.event == "metadata" or not event.data:
                # import pdb; pdb.set_trace()
                continue
            
           
            data = event.data
            # import pdb; pdb.set_trace()
            # stream_mode="values" 返回的是全量消息列表
            messages = data.get("messages", []) if isinstance(data, dict) else data
            if not messages:
                continue
            # import pdb; pdb.set_trace()

            # --- 核心修改：累加所有 AI 相关的行为 ---
            current_bubble_text = ""
            
            # 顺序遍历，把这次任务中产生的所有工具调用和回复拼接起来
            # 注意：只拼接最后一次用户输入之后的 AI 消息
            found_last_user = False
            for msg in reversed(messages):
                role, content, tool_calls = extract_message_info(msg)
                
                # 如果碰到用户刚才的消息，说明往前的 AI 消息是上一轮的，停止拼接
                if role == "human" or role == "user":
                    break
                
                # 处理 AI 消息
                if role in ["assistant", "ai"]:
                    # 1. 如果有工具调用，先拼上工具提示
                    if tool_calls:
                        for call in tool_calls:
                            # 调用上面定义的简单格式化函数
                            tool_text = format_tool_call_simple(call['name'], call['args'])
                            # 拼接到整体输出的前面
                            if tool_text not in current_bubble_text:
                                current_bubble_text = tool_text + "\n" + current_bubble_text
                    
                    # 2. 如果有内容回复，拼上内容
                    if content and content.strip():
                        current_bubble_text += format_ai_response(content)
            
            # 只有内容发生变化才 yield
            if current_bubble_text and current_bubble_text != last_yielded_content:
                print(f"DEBUG FRONTEND: role:{role} current_bubble_text\n: ** {current_bubble_text} 的更新")
                last_yielded_content = current_bubble_text
                yielded_at_least_once = True
                yield current_bubble_text
        
        if not yielded_at_least_once:
            yield "..."
    except Exception as e:
        yield f"❌ 运行异常: {str(e)}"

def main_page():
        
    with gr.Row():
        with gr.Column(scale=1):
            # with gr.Row():
            # 模型选择下拉框

            session_id = gr.Textbox(label="会话 ID", value="user_session_01")
            
            # --- UI 中显示管理功能 ---
            with gr.Accordion("🛠️ 线程高级管理", open=True):
                with gr.Row():
                    monitor_btn = gr.Button("🔍 监控状态", size="sm")
                    clear_this_btn = gr.Button("🗑️ 清理当前", size="sm")
                
                status_box = gr.Markdown("🟢 等待指令")
                
                with gr.Accordion("🚨 危险操作", open=False):
                    clear_all_btn = gr.Button("🔥 清空全库线程", variant="stop")
            
            # --- 🚀 正确插入位置：实时状态字段内容 ---
            with gr.Accordion("📊 实时状态字段内容", open=False):
                field_display_box = gr.Markdown("等待查询...")
                refresh_fields_btn = gr.Button("🔄 刷新字段内容", size="sm")

            file_upload = gr.File(label="参考文档")
            task_context = gr.Textbox(label="分析背景", lines=10)

        with gr.Column(scale=2):
            # 模型参数提取
            model_selector = gr.Dropdown(
                choices=["local_qwen_small", "local_qwen"], 
                value="local_qwen_small", 
                label="选择模型",
                interactive=True  # 显式声明可交互
            )
            chat = gr.ChatInterface(
                fn=predict,
                additional_inputs=[model_selector,task_context, session_id, file_upload],
                chatbot=gr.Chatbot(height=700, label="分析对话流"), 
                fill_height=False # 设置为 False 后，height 才会生效
            )

            

    # --- 绑定独立出来的功能 ---
    monitor_btn.click(
        fn=graphmanager.monitor_thread_state,
        inputs=[session_id],
        outputs=[status_box]
    )
    
    clear_this_btn.click(
        fn=graphmanager.clear_specific_thread,
        inputs=[session_id],
        outputs=[status_box]
    )
    
    clear_all_btn.click(
        fn=graphmanager.clear_all_threads,
        outputs=[status_box]
    )
    
    refresh_fields_btn.click(
        fn=graphmanager.monitor_specific_fields,
        inputs=[session_id],
        outputs=[field_display_box]
    )
                

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="LangGraph 分析专家") as demo:
        gr.Markdown("# 📑 AI 深度报告分析助手")
        
        with gr.Tabs() as tabs:
            # --- Tab 1: 用户对话区 ---
            with gr.TabItem("“总结对话窗口", id=0):
                main_page()
            # --- Tab 2: 后端管理区 ---
            with gr.TabItem(" 库管理与监视", id=1):
                # 调用独立函数，传入管理器实例
                render_admin_page(graphmanager)

    return demo

if __name__ == "__main__":
    # 启动 Gradio
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )