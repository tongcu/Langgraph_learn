import re 
import gradio as gr
import asyncio
import hashlib
from uuid import UUID
from langgraph_sdk import get_client
hostname = "http://langgraph-api-learn-2026-pre1231:2024"
# GRAPH_ID = "my_agent"
if hostname is None:
    API_URL = "http://127.0.0.1:2024"
else:
    API_URL = hostname

GRAPH_ID = "my_agent"

def name_to_uuid(name: str) -> str:
    """将普通字符串转为 0.5.39 版本强制要求的 UUID 格式"""
    hash_obj = hashlib.md5(name.encode('utf-8'))
    return str(UUID(hash_obj.hexdigest()))

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
        return f"<details><summary><b>🔍 思考过程 (点击展开)</b></summary>\n\n{thought_content}\n\n</details>\n\n{answer_content}"
    
    return text


async def ensure_thread_exists(client, thread_id):
    """独立功能：确保线程存在，不存在则创建"""
    try:
        await client.threads.get(thread_id)
    except Exception:
        # 如果获取失败（404），则手动创建
        # 2026-01-04T05:38:12.152238Z [warning  ] POST /threads/40b9ce26-4c12-cd34-5f55-a803f9cdcfae/runs/stream 404 2ms [langgraph_api.server] api_revision=212ad47 api_variant=local_dev langgraph_api_version=0.5.39 latency_ms=2 method=POST path=/threads/{thread_id}/runs/stream path_params={'thread_id': '40b9ce26-4c12-cd34-5f55-a803f9cdcfae'} proto=1.1 query_string= req_header={} request_id=dc850bd1-3dff-45da-9b50-e150b25509f3 res_header={} route=/threads/{thread_id}/runs/stream status=404 thread_name=MainThread
        await client.threads.create(thread_id=thread_id)
        print(f"DEBUG: Created new thread: {thread_id}")


def extract_content_from_event(data):
    """
    独立功能：从不同的数据结构中提取文本内容
    """
    content = ""
    if isinstance(data, list):
        # 如果是列表，通常包含多条消息或消息片段
        for m in data:
            if isinstance(m, dict) and "content" in m:
                content += m["content"]
            elif hasattr(m, "content"):
                content += m.content
    elif isinstance(data, dict):
        content = data.get("content", "")
    elif hasattr(data, "content"):
        content = data.content
    return content
    
# --- 2. 核心预测逻辑 ---
async def predict(message, history, task_input, file_path):
    client = get_client(url=API_URL)
    thread_id = get_thread_id()
    
    # 确保线程存在
    await ensure_thread_exists(client, thread_id)

    input_state = {
        "task": task_input, # 满足你 state["task"] 的需求
        "messages": [{"role": "user", "content": message}]
    }
    
    msg_cache = {}
    try:
        async for event in client.runs.stream(
            thread_id,
            GRAPH_ID,
            input=input_state,
            stream_mode="values", # 实时同步 State
        ):
            if event.event == "metadata" or not event.data:
                continue
            
            # 获取最新的消息列表
            data = event.data
            messages = data.get("messages", []) if isinstance(data, dict) else data
            
            if not messages:
                continue
                
            current_msg = messages[-1]
            msg_id = getattr(current_msg, "id", "default")
            
            # 使用独立提取函数
            content = extract_content_from_event(current_msg)
            
            # 更新缓存并输出
            msg_cache[msg_id] = content
            full_display = "".join(msg_cache.values())
            
            yield full_display
            
    except Exception as e:
        yield f"❌ 运行异常: {str(e)}"

def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# LangGraph Agent 报告助手")
        
        with gr.Row():
            with gr.Column():
                # 增加会话 ID 字段，默认为你的 gradio_user_session
                session_id = gr.Textbox(label="会话 ID (用于记忆)", value="gradio_user_session")
                task_input = gr.Textbox(label="任务内容", lines=3)
                file_upload = gr.File(label="上传附件")
                submit_btn = gr.Button("发送请求")
            
            with gr.Column():
                output_text = gr.Textbox(label="Agent 回复", lines=10)
                uuid_display = gr.Label(label="当前 Thread UUID")

        # 处理逻辑：先计算 UUID 展示给用户，再调用 API
        def update_uuid(name):
            return name_to_uuid(name)

        session_id.change(update_uuid, inputs=[session_id], outputs=[uuid_display])

        submit_btn.click(
            fn=predict,
            inputs=[task_input, file_upload, session_id],
            outputs=[output_text]
        )
    return demo
if __name__ == "__main__":
    # 启动 Gradio
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )