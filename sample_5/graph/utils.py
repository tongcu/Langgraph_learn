def export_agent_graph(app, filename="agent_graph.png"):
    """
    独立功能：从编译后的 LangGraph 实例中导出图片文件
    """
    try:
        # 获取 mermaid 格式的 PNG 二进制数据
        png_data = app.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"成功保存流程图至: {filename}")
        return filename
    except Exception as e:
        print(f"导出流程图失败: {e}")
        # 如果缺少本地绘图依赖，可以尝试输出 mermaid 源码地址
        return None

