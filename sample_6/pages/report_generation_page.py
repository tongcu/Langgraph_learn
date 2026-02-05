import gradio as gr
import asyncio
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path

from KnowledgeManager.FAISSKnowledgeManager import FAISSKnowledgeManager
from graph.graph_manager import GraphManager
from Utils.id import name_to_uuid_nr as name_to_uuid
from agent import app as agent_app


def render_report_generation_page(graphmanager: GraphManager):
    """渲染报告生成页面"""
    
    with gr.Row():
        # 左侧输入区域
        with gr.Column(scale=1):
            gr.Markdown("### 报告生成设置")
            
            # 会话 ID 输入
            session_id = gr.Textbox(label="会话 ID", value="report_session_01")

            # 模型选择
            model_selector = gr.Dropdown(
                choices=["local_qwen_small", "local_qwen"], 
                value="local_qwen_small", 
                label="选择模型",
                interactive=True
            )
            
            # 知识库选择
            kb_list = FAISSKnowledgeManager.list_knowledge_bases()
            knowledge_base_selector = gr.Dropdown(
                label="知识库名称", 
                choices=kb_list,
                value=kb_list[0] if kb_list else None,
                allow_custom_value=True
            )
            
            # 用户要求输入
            user_requirement = gr.Textbox(
                label="用户要求", 
                placeholder="请输入您对报告的具体要求...",
                lines=5
            )
            
            # 资料文件上传
            reference_files = gr.File(
                label="资料文件 (Markdown格式)",
                file_count="multiple",
                file_types=[".md", ".txt", ".pdf", ".docx"]
            )
            
            # 生成控制开关
            enable_auto_generation = gr.Checkbox(
                label="启用自动报告生成",
                value=False,
                info="勾选后将直接触发报告生成流程"
            )
            
            # 生成按钮
            generate_btn = gr.Button("生成报告", variant="primary")
            
            # 状态显示
            status_box = gr.Markdown("状态: **待命**")
            
        # 右侧结果展示区域
        with gr.Column(scale=2):
            with gr.Tabs():
                # 生成结果预览
                with gr.TabItem("报告预览"):
                    report_output = gr.Markdown(
                        label="生成的报告", 
                        elem_classes=["report-output"],
                        line_breaks=True
                    )
                
                # 对话修改辅助
                with gr.TabItem("对话修改辅助"):
                    chat_interface = gr.ChatInterface(
                        fn=chat_modification_handler,
                        additional_inputs=[report_output, model_selector],
                        chatbot=gr.Chatbot(
                            label="对话修改辅助", 
                            height=500
                        ),
                        fill_height=True
                    )
                
                # 原始数据
                with gr.TabItem("原始数据"):
                    raw_output = gr.JSON(label="原始输出数据")


    async def handle_generate_click(sid, model, kb, req, files, auto_gen):
        if auto_gen:
            return await generate_report(sid, model, kb, req, files, graphmanager)
        else:
            return "手动模式已启用，使用对话修改辅助进行调整", {}, "状态: **手动模式**"

    # 绑定事件
    generate_btn.click(
        fn=handle_generate_click,
        inputs=[
            session_id,
            model_selector,
            knowledge_base_selector, 
            user_requirement, 
            reference_files, 
            enable_auto_generation
        ],
        outputs=[report_output, raw_output, status_box]
    )


async def generate_report(
    session_name: str,
    model_name: str,
    knowledge_base: str, 
    user_requirement: str, 
    reference_files, 
    graphmanager: GraphManager
):
    """生成报告的主要逻辑"""
    if not user_requirement.strip():
        return "错误：请提供用户要求", {}, "状态: **缺少必要输入**"
    
    if not session_name.strip():
        return "错误：请提供会话 ID", {}, "状态: **缺少必要输入**"
    
    try:
        # 更新状态
        status_msg = "状态: **正在生成报告...**"
        
        # 准备输入数据
        input_data = {
            "task": user_requirement,
            "messages": [{"role": "user", "content": user_requirement}],
            "knowledge_base": knowledge_base or "default",
            "use_knowledge": bool(knowledge_base),
            "next_step": "report_generation_subgraph",  # 直接触发报告生成子图
            "task_id": session_name
        }
        
        # ... (处理文件的逻辑保持不变)
        if reference_files:
            # 临时存储上传的文件内容
            file_contents = []
            for file_obj in reference_files:
                file_path = file_obj.name
                # 根据文件类型读取内容
                try:
                    if file_path.endswith('.md') or file_path.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    elif file_path.endswith('.pdf'):
                        # 这里应该使用适当的PDF处理库，如PyPDF2
                        content = f"[PDF文件: {os.path.basename(file_path)}, 内容待处理]"
                    elif file_path.endswith('.docx'):
                        # 这里应该使用适当的Word处理库，如python-docx
                        content = f"[DOCX文件: {os.path.basename(file_path)}, 内容待处理]"
                    else:
                        content = f"[不支持的文件类型: {os.path.basename(file_path)}]"
                        
                    file_contents.append({
                        "filename": os.path.basename(file_path),
                        "content": content[:2000] + "..." if len(content) > 2000 else content  # 限制长度
                    })
                except Exception as e:
                    file_contents.append({
                        "filename": os.path.basename(file_path),
                        "content": f"[读取文件出错: {str(e)}]"
                    })
            
            input_data["reference_files"] = file_contents
        
        # 使用输入的 session_name 转换 thread_id
        thread_id = name_to_uuid(session_name)
        
        # --- 满足最低要求：直接引用图从 agent 中进入 ---
        # 构造执行配置
        config = {
            "configurable": {
                "thread_id": thread_id,
                "model_name": model_name
            },
            "recursion_limit": 50
        }
        
        # 直接调用本地编译好的图，绕过 SDK 客户端
        result = await agent_app.ainvoke(input_data, config=config)
        
        # 提取报告内容
        report_content = result.get("final_content", result.get("merged_article", ""))
        if not report_content:
            # 尝试从消息中提取最后的AI回复
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    report_content = last_msg.content
                else:
                    report_content = str(last_msg)
        
        if not report_content:
            report_content = "未能从生成结果中提取到有效内容，请检查输入参数。"
        
        return report_content, result, "状态: **报告生成完成**"
    
    except Exception as e:
        error_msg = f"生成报告时发生错误: {str(e)}"
        return error_msg, {}, f"状态: **错误**: {str(e)}"


async def chat_modification_handler(message, history, current_report, model_name):
    """处理对话修改的辅助功能：支持基于已生成报告的问答和修改建议"""
    from LLM.llm import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    
    if not current_report or current_report.strip() == "暂无内容，请从左侧选择会话。":
        return "请先生成或选择一份报告，然后再进行对话修改。"
    
    try:
        llm = get_llm(model=model_name)
        
        # 构造系统提示词
        system_prompt = f"""你是一个专业的报告修改助手。
当前报告内容如下：
---
{current_report}
---
请基于以上报告内容，回答用户的问题或提供修改建议。"""
        
        # 转换历史记录
        messages = [SystemMessage(content=system_prompt)]
        for turn in history:
            messages.append(HumanMessage(content=turn[0]))
            messages.append(AIMessage(content=turn[1]))
        
        messages.append(HumanMessage(content=message))
        
        # 调用模型
        response = await llm.ainvoke(messages)
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        return f"对话处理出错: {str(e)}"