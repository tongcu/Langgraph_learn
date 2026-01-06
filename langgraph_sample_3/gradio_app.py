import re 
import gradio as gr
import asyncio
import hashlib
from uuid import UUID
from langgraph_sdk import get_client
from langchain_core.messages import AIMessage, HumanMessage


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
        
        status_report = f"📍 **当前停滞位置**: {current_nodes}\n"
        status_report += f"📝 **消息总数**: {len(values.get('messages', []))} 条\n"
        
        if "task" in values:
            status_report += f"📄 **上下文状态**: 已加载 (长度: {len(values['task'])})\n"
            
        return status_report
    except Exception as e:
        return f"❌ 无法获取状态: {str(e)}"

# async def ensure_thread_exists(client, thread_id):
#     """独立功能：确保线程存在，不存在则创建"""
#     try:
#         await client.threads.get(thread_id)
#     except Exception:
#         # 如果获取失败（404），则手动创建
#         # 2026-01-04T05:38:12.152238Z [warning  ] POST /threads/40b9ce26-4c12-cd34-5f55-a803f9cdcfae/runs/stream 404 2ms [langgraph_api.server] api_revision=212ad47 api_variant=local_dev langgraph_api_version=0.5.39 latency_ms=2 method=POST path=/threads/{thread_id}/runs/stream path_params={'thread_id': '40b9ce26-4c12-cd34-5f55-a803f9cdcfae'} proto=1.1 query_string= req_header={} request_id=dc850bd1-3dff-45da-9b50-e150b25509f3 res_header={} route=/threads/{thread_id}/runs/stream status=404 thread_name=MainThread
#         await client.threads.create(thread_id=thread_id)
#         print(f"DEBUG: Created new thread: {thread_id}")


# def extract_content_from_event(data):
#     """
#     独立功能：从不同的数据结构中提取文本内容
#     """
#     content = ""
#     if isinstance(data, list):
#         # 如果是列表，通常包含多条消息或消息片段
#         for m in data:
#             if isinstance(m, dict) and "content" in m:
#                 content += m["content"]
#             elif hasattr(m, "content"):
#                 content += m.content
#     elif isinstance(data, dict):
#         content = data.get("content", "")
#     elif hasattr(data, "content"):
#         content = data.content
#     return content
    
def extract_message_info(msg):
    """
    独立功能：从不同格式的消息中提取角色、内容和工具调用。
    支持 LangChain 对象和原始字典格式。
    """
    if isinstance(msg, dict):
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
    else:
        role = getattr(msg, "type", "")  # LangChain 对象通常用 type 标识
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
    
    return role, content, tool_calls

def get_tool_display_text(tool_calls):
    """
    独立功能：将技术性的 tool_calls 转换为用户友好的中文提示。
    """
    if not tool_calls:
        return ""
    
    mapping = {
        "summarize_general": "📝 正在深度分析文章并生成总结...",
        "web_search": "🔍 正在检索互联网实时信息...",
        # 在此添加更多工具名映射
    }
    
    hints = []
    for tool in tool_calls:
        # 兼容不同结构的 tool_call
        name = tool.get("name") if isinstance(tool, dict) else tool.get("function", {}).get("name", "")
        hints.append(mapping.get(name, f"🛠️ 正在调用工具 [{name}] 处理中..."))
    
    return "\n\n".join(hints)

# --- 2. 重构后的核心预测逻辑 ---
async def predict(message, history, task_context, session_id, file_obj):
    client = get_client(url=API_URL)
    thread_id = name_to_uuid(session_id)
    
    # 确保线程存在
    try:
        await client.threads.get(thread_id)
    except:
        await client.threads.create(thread_id=thread_id)
        print(f"INFO: Created new thread: {thread_id}")

    # 构造输入状态
    input_state = {
        "task": task_context,
        "messages": [{"role": "user", "content": message}]
    }
    
    if file_obj is not None:
        input_state["files"] = [file_obj.name]

    status_prefix = ""  # 用于存储工具调用的中间状态
    last_yielded_content = ""

    try:
        async for event in client.runs.stream(
            thread_id,
            GRAPH_ID,
            input=input_state,
            stream_mode="values", 
            config={
                "configurable": {},
                "recursion_limit": 50,    # 递归深度限制
                "concurrency_limit": 1    # 单个 Run 内部并行的分支数限制
                }
        ):
            if event.event == "metadata" or not event.data:
                continue
            
            data = event.data
            # stream_mode="values" 返回的是全量消息列表
            messages = data.get("messages", []) if isinstance(data, dict) else data
            if not messages:
                continue
            
            # 找到最后一条有效的 AI 消息
            # 注意：我们要从后往前找，因为最后一条可能是 ToolMessage 或 UserMessage
            for msg in reversed(messages):
                role, content, tool_calls = extract_message_info(msg)
                
                # 情况 A：模型正在决定调用工具
                if tool_calls:
                    status_prefix = f"> {get_tool_display_text(tool_calls)}\n\n"
                    yield status_prefix
                    break # 找到最新的 tool_call 即可

                # 情况 B：模型给出了正式回复 (assistant)
                elif role in ["assistant", "ai"] and content:
                    # 只有当内容真正更新时才 yield，避免 Gradio 界面抖动
                    full_response = status_prefix + format_ai_response(content)
                    if full_response != last_yielded_content:
                        last_yielded_content = full_response
                        yield full_response
                    break # 找到最新的有效回复即可
                
                # 情况 C：如果是用户消息，我们忽略它（不渲染在回答区），继续向上找
                else:
                    continue
                    
    except Exception as e:
        yield f"❌ 运行异常: {str(e)}"

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📑 AI 深度报告分析助手")
        
        with gr.Row():
            # 左侧配置区
            with gr.Column(scale=1):
                session_id = gr.Textbox(label="会话 ID", value="user_session_01")
                file_upload = gr.File(label="上传参考文档")
                # 这里的 task_context 对应你要求的 state["task"]
                task_context = gr.Textbox(
                    label="待分析的文章/背景内容", 
                    placeholder="在此粘贴长篇文章、数据或背景资料...",
                    lines=15
                )
            
            # 右侧对话区
            with gr.Column(scale=2):
                # 使用 ChatInterface 可以自动处理 history 逻辑
                chat = gr.ChatInterface(
                    fn=predict,
                    additional_inputs=[task_context, session_id, file_upload],
                    #type="messages" # 使用新的 messages 格式
                )
                
    return demo

if __name__ == "__main__":
    # 启动 Gradio
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )