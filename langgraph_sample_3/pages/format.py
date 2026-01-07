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