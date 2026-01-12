def format_tool_call_simple(name, args):
    """用简单的 Markdown 引用块区分工具"""
    # 提取参数 key-value
    arg_details = ""
    for k, v in args.items():
        val = str(v)[:80] + "..." if len(str(v)) > 80 else v
        arg_details += f"\n> - **{k}**: {val}"

    return (
        f"#### 🛠️ 正在调用分析工具\n"
        f"> **工具名称**: `{name}`"
        f"{arg_details}\n"
        f"---\n" # 分割线
    )


# extract_message_info
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

def format_to_gradio_messages(messages):
    """
    将原始消息列表转换为 Gradio 5.0+ 要求的消息格式
    格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
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
    return formatted_history