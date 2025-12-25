import gradio as gr
import asyncio
import hashlib
from uuid import UUID
from langgraph_sdk import get_client

API_URL = "http://127.0.0.1:2024"

def name_to_uuid(name: str) -> str:
    """将普通字符串转为 0.5.39 版本强制要求的 UUID 格式"""
    hash_obj = hashlib.md5(name.encode('utf-8'))
    return str(UUID(hash_obj.hexdigest()))

async def ensure_thread_exists(client, thread_id):
    """独立功能：确保线程存在，不存在则创建"""
    try:
        await client.threads.get(thread_id)
    except Exception:
        # 如果获取失败（404），则手动创建
        await client.threads.create(thread_id=thread_id)
        print(f"DEBUG: Created new thread: {thread_id}")

async def get_agent_response(user_input, raw_tid):
    """主逻辑：处理消息流"""
    client = get_client(url=API_URL)
    thread_id = name_to_uuid(raw_tid)
    
    try:
        # 1. 先确保线程资源在后端已初始化
        await ensure_thread_exists(client, thread_id)
        
        final_msg = ""
        # 2. 发起流式请求
        async for event in client.runs.stream(
            thread_id,
            "my_agent",  # 对应你的 json 键名
            input={"messages": [{"role": "user", "content": user_input}]},
            stream_mode="messages"
        ):
            # 提取消息内容 (兼容 0.5.x 数据结构)
            if event.data and isinstance(event.data, dict):
                # 尝试从不同字段获取文本
                content = event.data.get("content", "")
                if content:
                    final_msg = content
            elif isinstance(event.data, list) and len(event.data) > 0:
                final_msg = event.data[-1].get("content", final_msg)

        return final_msg or "✅ 已发送并处理", "Success"

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return f"运行异常: {str(e)}", "Error"

def gradio_wrapper(msg, tid):
    """Gradio 调用的同步包装器"""
    return asyncio.run(get_agent_response(msg, tid))

# --- UI 构造 ---
with gr.Blocks() as demo:
    gr.Markdown("### LangGraph 联调终端 (Fixed 404)")
    with gr.Row():
        tid = gr.Textbox(value="user_123", label="Thread ID")
        status = gr.Label(label="状态")
    out = gr.Textbox(label="AI 输出")
    inp = gr.Textbox(label="输入消息")
    btn = gr.Button("发送请求")
    
    btn.click(gradio_wrapper, inputs=[inp, tid], outputs=[out, status])

if __name__ == "__main__":
    # 同容器运行，需监听 0.0.0.0
    demo.launch(server_name="0.0.0.0", server_port=7860)