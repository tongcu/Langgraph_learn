from graph import app

def test_workflow():
    # --- 配置区 ---
    # 这里动态传入模型名称和 Thread ID
    config = {
        "configurable": {
            "thread_id": "thread_abc_123",
            "model_name": "local_qwen" # 动态指定模型
        }
    }
    
    # --- 第一步：启动并运行到中断点 ---
    # inputs = {"messages": [("user", "帮我连接到 https://api.github.com，key是 secret123")]}
    inputs = {"messages": [("user", "帮我查询一下")]}
    
    print("\n>>> 正在启动流程...")
    for event in app.stream(inputs, config, stream_mode="values"):
        if event.get("messages"):
            last_msg = event["messages"][-1]
            print(f"当前节点响应: {last_msg.content or '[调用工具中...]'}")
    # import pdb; pdb.set_trace()
    # --- 第二步：检查中断状态 ---
    snapshot = app.get_state(config)
    if snapshot.next:
        print(f"\n[中断] 流程已在 {snapshot.next} 节点前停止。")
        # 打印 LLM 提取的参数供人工审核
        tool_call = last_msg.tool_calls[0]
        print(f"LLM 提取的参数为: {tool_call['args']}")
        
        # 模拟人工输入确认
        confirm = input("\n是否确认执行该配置？(y/n): ")
        if confirm.lower() == 'y':
            # --- 第三步：继续执行 ---
            print(">>> 审批通过，继续执行...")
            for event in app.stream(None, config, stream_mode="values"):
                if event.get("messages"):
                    print(f"最终结果: {event['messages'][-1].content}")
        else:
            print(">>> 流程已人工取消。")

if __name__ == "__main__":
    test_workflow()