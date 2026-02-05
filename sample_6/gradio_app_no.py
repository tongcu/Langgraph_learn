import re
import gradio as gr
import asyncio
import json
import requests
from uuid import UUID
from pages import render_admin_page
from config import settings
from Utils.id import name_to_uuid_nr as name_to_uuid
from pages.format import format_tool_call_simple
import time
# import sseclient  # 如果需要可以安装: pip install sseclient-py

API_URL = settings.API_URL
# 使用 app_no.py 的端口，通常是 8000
BASE_API_URL = API_URL.replace('/liveshare', '').replace(':2024', ':8000')

def format_ai_response(text: str) -> str:
    """
    如果存在 <think> 或 <thought> 标签，将其包装在折叠框内；
    如果没有，则直接输出结果。
    """
    if not text: 
        return ""
    
    pattern = r'<(?:think|thought)>(.*?)</(?:think|thought)>'
    match = re.search(pattern, text, flags=re.DOTALL)
    
    if match:
        thought_content = match.group(1).strip()
        # 移除正文中的 think 部分
        answer_content = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        return f"<details><summary><b> 思考过程 (点击展开)</b></summary>\n\n{thought_content}\n\n</details>\n\n{answer_content}"
    
    return text


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
    # 我们认为这也是一种有效的"回复"
    return role, content, tool_calls


# 管理功能函数
def get_thread_status(session_id):
    """
    获取线程状态
    """
    try:
        request_data = {"session_id": session_id}
        response = requests.post(f"{BASE_API_URL}/thread/status", json=request_data)
        if response.status_code == 200:
            result = response.json()
            return result.get("details", "获取状态成功")
        else:
            return f"❌ 请求失败: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ 运行异常: {str(e)}"


def clear_specific_thread(session_id):
    """
    清理特定线程
    """
    try:
        request_data = {"session_id": session_id}
        response = requests.post(f"{BASE_API_URL}/thread/clear", json=request_data)
        if response.status_code == 200:
            result = response.json()
            return result.get("message", "清理成功")
        else:
            return f"❌ 请求失败: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ 运行异常: {str(e)}"


def clear_all_threads():
    """
    清理所有线程
    """
    try:
        response = requests.post(f"{BASE_API_URL}/thread/clear-all")
        if response.status_code == 200:
            result = response.json()
            return result.get("message", "全部清理成功")
        else:
            return f"❌ 请求失败: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ 运行异常: {str(e)}"


# --- 重构后的核心预测逻辑 ---
async def predict(message, history, model_selector, task_context, session_id, file_obj):
    """
    预测函数，通过调用 app_no.py 的接口获取响应
    """
    try:
        # 准备请求数据
        request_data = {
            "message": message,
            "history": history if history else [],
            "model_selector": model_selector,
            "task_context": task_context,
            "session_id": session_id
        }
        
        if file_obj is not None:
            request_data["file_obj"] = {"name": file_obj.name}
        
        # 调用 app_no.py 的流式接口
        stream_url = f"{BASE_API_URL}/chat_stream"
        
        # 使用 requests 发送流式请求
        with requests.post(stream_url, json=request_data, stream=True) as response:
            if response.status_code != 200:
                yield f"❌ 请求失败: {response.status_code} - {response.text}"
                return
            
            # 手动解析 SSE 数据
            last_yielded_content = ""
            yielded_at_least_once = False
            
            buffer = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    data_str = line[len("data: "):].strip()
                    if data_str:
                        try:
                            # 解析 SSE 数据
                            data = json.loads(data_str)
                            response_content = data.get('response', '')
                            
                            if response_content and response_content != last_yielded_content:
                                last_yielded_content = response_content
                                yielded_at_least_once = True
                                yield response_content
                        except json.JSONDecodeError:
                            continue
        
        if not yielded_at_least_once:
            yield "..."
    
    except Exception as e:
        yield f"❌ 运行异常: {str(e)}"


def main_page():
    with gr.Row():
        with gr.Column(scale=1):
            # 会话 ID 输入
            session_id = gr.Textbox(label="会话 ID", value="user_session_01")
            
            # --- UI 中显示管理功能 ---
            with gr.Accordion("线程高级管理", open=True):
                with gr.Row():
                    monitor_btn = gr.Button("监控状态", size="sm")
                    clear_this_btn = gr.Button("清理当前", size="sm")
                
                status_box = gr.Markdown("等待指令")
                
                with gr.Accordion("危险操作", open=False):
                    clear_all_btn = gr.Button("清空全库线程", variant="stop")
            
            # --- 实时状态字段内容 ---
            with gr.Accordion("实时状态字段内容", open=False):
                field_display_box = gr.Markdown("等待查询...")
                refresh_fields_btn = gr.Button("刷新字段内容", size="sm")

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
                additional_inputs=[model_selector, task_context, session_id, file_upload],
                chatbot=gr.Chatbot(height=700, label="分析对话流"), 
                fill_height=False # 设置为 False 后，height 才会生效
            )
            
            # 绑定管理功能
            monitor_btn.click(
                fn=get_thread_status,
                inputs=[session_id],
                outputs=[status_box]
            )
            
            clear_this_btn.click(
                fn=clear_specific_thread,
                inputs=[session_id],
                outputs=[status_box]
            )
            
            clear_all_btn.click(
                fn=clear_all_threads,
                outputs=[status_box]
            )


def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="LangGraph 分析专家 (Direct API)") as demo:
        gr.Markdown("# AI 深度报告分析助手 (Direct API)")
        
        with gr.Tabs() as tabs:
            # --- Tab 1: 用户对话区 ---
            with gr.TabItem("总结对话窗口", id=0):
                main_page()
            # --- Tab 2: 后端管理区 ---
            with gr.TabItem("库管理与监视", id=1):
                # 创建一个简单的管理界面
                with gr.Group():
                    gr.Markdown("### 线程管理")
                    session_input = gr.Textbox(label="会话ID", value="user_session_01")
                    with gr.Row():
                        get_status_btn = gr.Button("获取状态")
                        clear_thread_btn = gr.Button("清理线程")
                        clear_all_btn_admin = gr.Button("清理全部")
                    status_output = gr.Textbox(label="状态信息", interactive=False)
                    
                    get_status_btn.click(
                        fn=get_thread_status,
                        inputs=[session_input],
                        outputs=[status_output]
                    )
                    
                    clear_thread_btn.click(
                        fn=clear_specific_thread,
                        inputs=[session_input],
                        outputs=[status_output]
                    )
                    
                    clear_all_btn_admin.click(
                        fn=clear_all_threads,
                        outputs=[status_output]
                    )

    return demo


if __name__ == "__main__":
    # 启动 Gradio
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7861,  # 使用不同端口避免冲突
    )