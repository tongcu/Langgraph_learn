import re 
import gradio as gr
import asyncio
import hashlib
from uuid import UUID
from langgraph_sdk import get_client
from langchain_core.messages import AIMessage, HumanMessage
from graph.graph_manager import GraphManager, name_to_uuid # 引用独立功能
from pages.format import format_tool_call_simple

hostname = "http://langgraph-api-learn-2026-pre1231:2024"
# GRAPH_ID = "my_agent"
if hostname is None:
    API_URL = "http://127.0.0.1:2024"
else:
    API_URL = hostname

# 初始化管理器
graphmanager = GraphManager(api_url=API_URL)


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

# 通用 工具内容读取
# def format_tool_args(args):
#     """动态格式化工具参数为易读的字符串"""
#     if not isinstance(args, dict):
#         return str(args)
    
#     parts = []
#     for key, value in args.items():
#         # 将字段名翻译或格式化（例如 summary -> 摘要）
#         label = key.replace("_", " ").title() 
        
#         if isinstance(value, list):
#             # 处理列表（如 key_takeaways）
#             item_str = "\n   · ".join([str(i) for i in value])
#             parts.append(f"🔹 **{label}**:\n   · {item_str}")
#         elif isinstance(value, dict):
#             # 处理嵌套字典
#             parts.append(f"🔹 **{label}**: {list(value.values())[0]}...")
#         else:
#             # 处理普通字符串
#             # 如果内容太长，可以做个截断展示
#             display_val = (str(value)[:100] + "...") if len(str(value)) > 100 else str(value)
#             parts.append(f"🔹 **{label}**: {display_val}")
            
#     return "\n".join(parts)

# def get_tool_display_text(tool_calls):
#     """
#     独立功能：将技术性的 tool_calls 转换为用户友好的中文提示。
#     """
#     if not tool_calls:
#         return ""
    
#     mapping = {
#         "summarize_general": "调用工具 summarize_general 正在深度分析文章并生成总结...",
#         "web_search": "🔍 正在检索互联网实时信息...",
#         # 在此添加更多工具名映射
#     }
    
#     hints = []
#     for tool in tool_calls:
#         # 兼容不同结构的 tool_call
#         name = tool.get("name", "Unknown Tool")
#         args = tool.get("args", {})
    
#         # 1. 获取基本提示语
#         base_hint = mapping.get(name, f"🛠️ 正在执行 {name}...")
#         # 2. 动态获取参数详情
#         detail_hint = format_tool_args(args)
        
#         # 3. 组合
#         full_hint = f"{base_hint}\n\n{detail_hint[:100]}\n"
#         hints.append(full_hint)
    
#     return "\n\n".join(hints)

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
            # 同时监听 values(状态全量) 和 updates(节点运行轨迹)
            stream_mode=["values", "updates"],
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
                yield current_bubble_text
                    
    except Exception as e:
        yield f"❌ 运行异常: {str(e)}"

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📑 AI 深度报告分析助手")
        
        with gr.Row():
            with gr.Column(scale=1):
                session_id = gr.Textbox(label="会话 ID", value="user_session_01")
                
                # --- UI 中显示管理功能 ---
                with gr.Accordion("🛠️ 线程高级管理", open=True):
                    with gr.Row():
                        monitor_btn = gr.Button("🔍 监控状态", size="sm")
                        clear_this_btn = gr.Button("🗑️ 清理当前", size="sm")
                    
                    status_box = gr.Markdown("🟢 等待指令")
                    
                    with gr.Accordion("🚨 危险操作", open=False):
                        clear_all_btn = gr.Button("🔥 清空全库线程", variant="stop")

                file_upload = gr.File(label="参考文档")
                task_context = gr.Textbox(label="分析背景", lines=10)

            with gr.Column(scale=2):
                chat = gr.ChatInterface(
                    fn=predict,
                    additional_inputs=[task_context, session_id, file_upload],
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
                
    return demo

if __name__ == "__main__":
    # 启动 Gradio
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )