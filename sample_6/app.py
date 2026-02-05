from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import json
import re
from uuid import UUID
from langgraph_sdk import get_client
from langchain_core.messages import AIMessage, HumanMessage
from graph.graph_manager import GraphManager
from Utils.id import name_to_uuid_nr as name_to_uuid
from pages.format import format_tool_call_simple
from config import settings

app = FastAPI(title="LangGraph Analysis Assistant API", version="1.0.0")

# 初始化管理器
API_URL = settings.API_URL
GRAPH_ID = "my_agent"
graphmanager = GraphManager(api_url=API_URL)


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


@app.get("/")
async def root():
    return {"message": "Welcome to LangGraph Analysis Assistant API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "api_url": API_URL}


# 请求模型
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    model_selector: str = "local_qwen_small"
    task_context: str = ""
    session_id: str = "user_session_01"
    file_obj: Optional[Dict[str, Any]] = None


class ThreadStatusRequest(BaseModel):
    session_id: str


class ClearThreadRequest(BaseModel):
    session_id: str


class RefreshFieldsRequest(BaseModel):
    session_id: str


# 响应模型
class ChatResponse(BaseModel):
    response: str
    status: str = "success"


class ThreadStatusResponse(BaseModel):
    status: str
    details: str


class ClearThreadResponse(BaseModel):
    status: str
    message: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    聊天接口，对应 gradio_app.py 中的 predict 函数
    """
    try:
        client = get_client(url=API_URL)
        thread_id = name_to_uuid(request.session_id)
        
        # 确保线程存在
        try:
            await client.threads.get(thread_id)
        except:
            await client.threads.create(thread_id=thread_id)
            print(f"INFO: Created new thread: {thread_id}")

        # 构造输入状态
        input_state = {
            "task": request.task_context,
            "messages": [{"role": "user", "content": request.message}],
            "task_id": request.session_id
        }
        
        if request.file_obj is not None:
            input_state["files"] = [request.file_obj.get("name", "unknown")]

        run_config = {
            "configurable": {
                "model_name": request.model_selector  # 对应你 node 里的 key
            },
            "recursion_limit": 50
        }

        # 存储最终响应
        final_response = ""
        
        async for event in client.runs.stream(
            thread_id,
            GRAPH_ID,
            input=input_state,
            stream_mode=["values", "updates"],
            config=run_config
        ):
            print(f"DEBUG FRONTEND: 收到event节点 {event.event} 的更新")
            
            if event.event == "metadata" or not event.data:
                continue
            
            data = event.data
            messages = data.get("messages", []) if isinstance(data, dict) else data
            
            if not messages:
                continue

            # 核心修改：累加所有 AI 相关的行为
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
            
            # 更新最终响应
            if current_bubble_text:
                final_response = current_bubble_text

        if not final_response:
            final_response = "..."

        return ChatResponse(response=final_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ 运行异常: {str(e)}")


@app.post("/chat_stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    流式聊天接口，与 gradio_app.py 中的 predict 函数功能保持一致
    """
    async def event_generator():
        try:
            client = get_client(url=API_URL)
            thread_id = name_to_uuid(request.session_id)
            
            # 确保线程存在
            try:
                await client.threads.get(thread_id)
            except:
                await client.threads.create(thread_id=thread_id)
                print(f"INFO: Created new thread: {thread_id}")

            # 构造输入状态
            input_state = {
                "task": request.task_context,
                "messages": [{"role": "user", "content": request.message}],
                "task_id": request.session_id
            }
            
            if request.file_obj is not None:
                input_state["files"] = [request.file_obj.get("name", "unknown")]

            run_config = {
                "configurable": {
                    "model_name": request.model_selector  # 对应你 node 里的 key
                },
                "recursion_limit": 50
            }
            
            last_yielded_content = ""
            yielded_at_least_once = False
            
            async for event in client.runs.stream(
                thread_id,
                GRAPH_ID,
                input=input_state,
                # 同时监听 values(状态全量) 和 updates(节点运行轨迹)
                stream_mode=["values", "updates"],
                config=run_config
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
                    
                    # 发送JSON格式的响应
                    yield f"data: {json.dumps({'response': current_bubble_text}, ensure_ascii=False)}\n\n"
            
            if not yielded_at_least_once:
                yield f"data: {json.dumps({'response': '...'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'response': f'❌ 运行异常: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/thread/status", response_model=ThreadStatusResponse)
async def get_thread_status_endpoint(request: ThreadStatusRequest):
    """
    获取线程状态，对应 gradio_app.py 中的 get_thread_status 函数
    """
    try:
        client = get_client(url=API_URL)
        thread_id = name_to_uuid(request.session_id)
        
        # 获取当前 thread 的最新状态
        state = await client.threads.get_state(thread_id)
        
        if not state or not state.get("next"):
            status_details = "✅ 当前没有正在运行或等待的任务。"
        else:
            # next 字段包含了即将执行或正在执行的节点名称
            current_nodes = state["next"]
            values = state.get("values", {})
            
            status_details = f" **当前停滞位置**: {current_nodes}\n"
            status_details += f" **消息总数**: {len(values.get('messages', []))} 条\n"
            
            if "task" in values:
                status_details += f" **上下文状态**: 已加载 (长度: {len(values['task'])})\n"
        
        return ThreadStatusResponse(status="success", details=status_details)
    
    except Exception as e:
        return ThreadStatusResponse(status="error", details=f"❌ 无法获取状态: {str(e)}")


@app.post("/thread/clear", response_model=ClearThreadResponse)
async def clear_thread_endpoint(request: ClearThreadRequest):
    """
    清理会话线程
    """
    try:
        result = await graphmanager.clear_specific_thread(request.session_id)
        return ClearThreadResponse(status="success", message=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/thread/clear-all", response_model=ClearThreadResponse)
async def clear_all_threads_endpoint():
    """
    清理所有线程
    """
    try:
        result = await graphmanager.clear_all_threads()
        return ClearThreadResponse(status="success", message=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/thread/refresh-fields", response_model=Dict[str, Any])
async def refresh_fields_endpoint(request: RefreshFieldsRequest):
    """
    刷新字段内容
    """
    try:
        result = await graphmanager.monitor_specific_fields(request.session_id)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket聊天接口，提供实时通信能力
    """
    await websocket.accept()
    
    try:
        while True:
            # 接收来自客户端的消息
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            message = request_data.get("message", "")
            model_selector = request_data.get("model_selector", "local_qwen_small")
            task_context = request_data.get("task_context", "")
            file_obj = request_data.get("file_obj", None)

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
                "messages": [{"role": "user", "content": message}],
                "task_id": session_id
            }
            
            if file_obj is not None:
                input_state["files"] = [file_obj.get("name", "unknown")]

            run_config = {
                "configurable": {
                    "model_name": model_selector  # 对应你 node 里的 key
                },
                "recursion_limit": 50
            }

            # 流式发送响应
            async for event in client.runs.stream(
                thread_id,
                GRAPH_ID,
                input=input_state,
                stream_mode=["values", "updates"],
                config=run_config
            ):
                if event.event == "metadata" or not event.data:
                    continue
                
                data = event.data
                messages = data.get("messages", []) if isinstance(data, dict) else data
                
                if not messages:
                    continue

                # 处理消息
                current_bubble_text = ""
                
                for msg in reversed(messages):
                    role, content, tool_calls = extract_message_info(msg)
                    
                    if role == "human" or role == "user":
                        break
                    
                    if role in ["assistant", "ai"]:
                        if tool_calls:
                            for call in tool_calls:
                                tool_text = format_tool_call_simple(call['name'], call['args'])
                                if tool_text not in current_bubble_text:
                                    current_bubble_text = tool_text + "\n" + current_bubble_text
                        
                        if content and content.strip():
                            current_bubble_text += format_ai_response(content)
                
                if current_bubble_text:
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "content": current_bubble_text
                    }))

    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"❌ 运行异常: {str(e)}"
        }))
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)