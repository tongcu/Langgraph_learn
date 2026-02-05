"""报告生成节点"""
import logging
from typing import Dict, Any, Optional, Union
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .states import WritingState
from LLM.llm import get_llm
from KnowledgeManager.KnowledgeManagerFactory import KnowledgeManagerFactory
from nodes.writings.writing_nodes import Default_model_name


async def report_generation_node(state: WritingState, config: RunnableConfig):
    """
    报告生成节点：根据用户要求和知识库内容生成报告
    """
    try:
        logging.info("--- 开始生成报告 ---")
        
        # 获取配置中的模型
        configurable = config.get("configurable", {})
        m_name = configurable.get("model_name", Default_model_name)
        llm = get_llm(model=m_name)
        
        # 获取用户任务要求
        user_task = state.get("task", "")
        knowledge_base = state.get("knowledge_base", "")
        use_knowledge = state.get("use_knowledge", False)
        reference_files = state.get("reference_files", [])
        
        # 构建提示词
        system_prompt = """你是专业的报告撰写专家。请根据用户的要求和提供的参考资料，生成一份结构清晰、内容详实的专业报告。
        
要求：
1. 报告结构完整，包含引言、主体内容和结论
2. 内容要有逻辑性和条理性
3. 如果提供了知识库，要结合知识库中的相关信息
4. 语言专业、客观、准确
5. 根据用户的具体要求调整报告的重点和深度"""
        
        # 如果使用知识库，检索相关信息
        knowledge_content = ""
        if use_knowledge and knowledge_base:
            try:
                km = KnowledgeManagerFactory.create_knowledge_manager("FAISS")
                # 使用用户任务作为查询
                results = km.similarity_search(user_task, knowledge_base_name=knowledge_base)
                if results:
                    knowledge_content = "\n".join([result.page_content for result in results])
                    logging.info(f"从知识库 '{knowledge_base}' 检索到 {len(results)} 条相关信息")
            except Exception as e:
                logging.warning(f"知识库检索失败: {str(e)}")
                knowledge_content = f"[知识库检索失败: {str(e)}]"
        
        # 构建用户输入
        user_content = f"用户要求：\n{user_task}\n\n"
        
        if reference_files:
            user_content += "--- 上传的参考文件内容 ---\n"
            for f in reference_files:
                user_content += f"文件名: {f.get('filename')}\n内容:\n{f.get('content')}\n\n"
            user_content += "--- 参考文件内容结束 ---\n\n"

        if knowledge_content:
            user_content += f"--- 知识库检索内容 ---\n{knowledge_content}\n--- 知识库检索内容结束 ---\n\n"
        
        user_content += "请根据以上信息，按照用户要求生成专业报告。"
        
        # 调用LLM生成报告
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        response = await llm.ainvoke(messages)
        
        # 返回生成的报告
        report_content = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "final_content": report_content,
            "merged_article": report_content,  # 为了兼容性
            "task_completed": True,
            "next_step": "end"
        }
        
    except Exception as e:
        logging.error(f"报告生成失败: {str(e)}")
        return {
            "final_content": f"报告生成过程中出现错误: {str(e)}",
            "next_step": "error_recovery"
        }


async def report_refinement_node(state: WritingState, config: RunnableConfig):
    """
    报告优化节点：对初步生成的报告进行润色和优化
    """
    try:
        logging.info("--- 开始优化报告 ---")
        
        # 获取配置中的模型
        configurable = config.get("configurable", {})
        m_name = configurable.get("model_name", Default_model_name)
        llm = get_llm(model=m_name)
        
        # 获取当前报告内容
        current_report = state.get("final_content", state.get("merged_article", ""))
        
        if not current_report:
            return {"next_step": "error_recovery", "final_content": "没有可优化的报告内容"}
        
        # 构建优化提示词
        system_prompt = """你是专业的文档优化专家。请对提供的报告进行优化，使其更加专业、清晰和易读。
        
优化要求：
1. 保持原意不变
2. 改善语言表达，使句子更流畅
3. 优化结构，增强逻辑性
4. 修正语法错误
5. 保持专业术语的准确性"""

        user_content = f"请优化以下报告：\n\n{current_report}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        response = await llm.ainvoke(messages)
        
        refined_report = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "final_content": refined_report,
            "merged_article": refined_report,
            "next_step": "end"
        }
        
    except Exception as e:
        logging.error(f"报告优化失败: {str(e)}")
        # 即使优化失败，也要返回原始内容
        original_content = state.get("final_content", state.get("merged_article", "报告内容不可用"))
        return {
            "final_content": original_content,
            "next_step": "end"
        }