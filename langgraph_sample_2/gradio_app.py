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

# --- 2. 核心预测逻辑 ---

async def predict(message, history):
    """
    使用 SDK Client 连接远程服务
    """
    # 1. 初始化客户端
    client = get_client(url=API_URL)
    
    # 2. 准备 Thread ID (UUID 格式)
    thread_id = name_to_uuid("gradio_user_session")
    
    # 3. 构造输入消息 (LangGraph API 接收字典格式)
    input_data = {
        "messages": [
            {"role": "user", "content": message}
        ]
    }
    
    full_response = ""
    
    try:
        
        # 4. 调用远程流式接口
        # 使用 SDK 提供的 stream 方法
        async for event in client.runs.stream(
            thread_id,
            GRAPH_ID,
            input=input_data,
            stream_mode="messages",
        ):
            # 获取消息内容块
            if event.event == "metadata": continue
            
            # 处理消息流 (不同版本的 SDK 返回格式略有不同，通常 data 是消息对象)
            # data 为消息片断
            data = event.data
            if isinstance(data, list):
                # 某些模式下返回列表
                for m in data:
                    if "content" in m: full_response += m["content"]
            elif isinstance(data, dict) and "content" in data:
                full_response += data["content"]
            elif hasattr(data, "content"):
                full_response += data.content

            # 实时格式化并返回给前端
            yield format_ai_response(full_response)
            
    except Exception as e:
        yield f"❌ 连接 API 失败: {str(e)}\n请检查 API 地址 {API_URL} 是否正确且服务已启动。"

# --- 3. UI 界面 ---

def create_ui():
    # 移除引起报错的 theme 等不确定参数，使用最基础的配置
    # 如果你想换肤，可以在 launch 之前定义主题变量
    demo = gr.ChatInterface(
        fn=predict,
        title="LangGraph Client",
        description=f"Connecting to {API_URL}",
        examples=["帮我写个测试报告大纲"],
    )
    return demo

if __name__ == "__main__":
    # 启动 Gradio
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )