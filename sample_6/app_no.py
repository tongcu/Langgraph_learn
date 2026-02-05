from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import json
import re
from uuid import UUID
from langchain_core.messages import AIMessage, HumanMessage
from Utils.id import name_to_uuid_nr as name_to_uuid
from pages.format import format_tool_call_simple
from config import settings
import logging

# 从 agent.py 导入编译好的应用
from agent import app as agent_app

app = FastAPI(title="LangGraph Analysis Assistant API (Direct)", version="1.0.0")

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
    return {"message": "Welcome to LangGraph Analysis Assistant API (Direct)"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent_app_loaded": agent_app is not None}


# 添加管理接口
@app.post("/thread/status")
async def get_thread_status_endpoint(request: ThreadStatusRequest):
    """
    获取线程状态 - 这是一个模拟实现，因为直接的 agent_app 不提供状态管理
    在实际实现中，您可能需要实现自己的状态跟踪机制
    """
    # 由于直接使用的 agent_app 是无状态的，这里返回模拟响应
    return ThreadStatusResponse(
        status="success", 
        details="使用直接模式，状态跟踪功能受限"
    )


@app.post("/thread/clear")
async def clear_thread_endpoint(request: ClearThreadRequest):
    """
    清理会话线程 - 模拟实现
    """
    return ClearThreadResponse(
        status="success", 
        message="使用直接模式，清理功能受限"
    )


@app.post("/thread/clear-all")
async def clear_all_threads_endpoint():
    """
    清理所有线程 - 模拟实现
    """
    return ClearThreadResponse(
        status="success", 
        message="使用直接模式，全局清理功能受限"
    )


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
    聊天接口，直接使用 agent.py 中的编译应用
    """
    try:
        # 构造输入状态
        input_state = {
            "task": request.task_context,
            "messages": [{"role": "user", "content": request.message}],
            "task_id": request.session_id
        }
        
        if request.file_obj is not None:
            input_state["files"] = [request.file_obj.get("name", "unknown")]

        # 直接调用编译的应用
        result = await agent_app.ainvoke(input_state)
        
        # 提取响应内容
        messages = result.get("messages", [])
        if messages:
            # 获取最后一条消息作为响应
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                response_content = str(last_message.content)
            else:
                response_content = str(last_message)
        else:
            response_content = "..."
        
        return ChatResponse(response=response_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ 运行异常: {str(e)}")


@app.post("/chat_stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    流式聊天接口，直接使用 agent.py 中的编译应用
    """
    async def event_generator():
        try:
            # 构造输入状态
            input_state = {
                "task": request.task_context,
                "messages": [{"role": "user", "content": request.message}],
                "task_id": request.session_id
            }
            
            if request.file_obj is not None:
                input_state["files"] = [request.file_obj.get("name", "unknown")]

            # 使用流式调用编译的应用
            last_yielded_content = ""
            yielded_at_least_once = False
            
            async for chunk in agent_app.astream_events(input_state, version="v1"):
                # 检查事件类型和数据
                event = chunk
                if event["event"] == "on_chain_end":
                    # 提取状态变更
                    state = event["data"]["output"]
                    messages = state.get("messages", [])
                    
                    if messages:
                        # 获取最后一条消息
                        last_message = messages[-1]
                        
                        # 提取内容和工具调用
                        role = getattr(last_message, 'type', getattr(last_message, 'role', ''))
                        content = getattr(last_message, 'content', '')
                        tool_calls = getattr(last_message, 'tool_calls', [])
                        
                        # 构建响应文本
                        current_bubble_text = ""
                        
                        # 如果有工具调用，先拼上工具提示
                        if tool_calls:
                            for call in tool_calls:
                                tool_text = format_tool_call_simple(call['name'], call['args'])
                                if tool_text not in current_bubble_text:
                                    current_bubble_text = tool_text + "\n" + current_bubble_text
                        
                        # 如果有内容回复，拼上内容
                        if content and str(content).strip():
                            current_bubble_text += format_ai_response(str(content))
                        
                        # 只有内容发生变化才 yield
                        if current_bubble_text and current_bubble_text != last_yielded_content:
                            last_yielded_content = current_bubble_text
                            yielded_at_least_once = True
                            
                            # 发送JSON格式的响应
                            yield f"data: {json.dumps({'response': current_bubble_text}, ensure_ascii=False)}\n\n"
            
            if not yielded_at_least_once:
                yield f"data: {json.dumps({'response': '...'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'response': f'❌ 运行异常: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket聊天接口，直接使用 agent.py 中的编译应用
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

            # 构造输入状态
            input_state = {
                "task": task_context,
                "messages": [{"role": "user", "content": message}],
                "task_id": session_id
            }
            
            if file_obj is not None:
                input_state["files"] = [file_obj.get("name", "unknown")]

            # 使用流式调用编译的应用
            async for chunk in agent_app.astream_events(input_state, version="v1"):
                event = chunk
                if event["event"] == "on_chain_end":
                    # 提取状态变更
                    state = event["data"]["output"]
                    messages = state.get("messages", [])
                    
                    if messages:
                        # 获取最后一条消息
                        last_message = messages[-1]
                        
                        # 提取内容和工具调用
                        role = getattr(last_message, 'type', getattr(last_message, 'role', ''))
                        content = getattr(last_message, 'content', '')
                        tool_calls = getattr(last_message, 'tool_calls', [])
                        
                        # 构建响应文本
                        current_bubble_text = ""
                        
                        # 如果有工具调用，先拼上工具提示
                        if tool_calls:
                            for call in tool_calls:
                                tool_text = format_tool_call_simple(call['name'], call['args'])
                                if tool_text not in current_bubble_text:
                                    current_bubble_text = tool_text + "\n" + current_bubble_text
                        
                        # 如果有内容回复，拼上内容
                        if content and str(content).strip():
                            current_bubble_text += format_ai_response(str(content))
                        
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