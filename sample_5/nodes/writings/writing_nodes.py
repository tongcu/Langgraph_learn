from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AssistantMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing import Optional, Union
from LLM.llm import get_llm
# from tools.client_tool import tools
import logging
import re
import json

Default_model_name = "local_qwen"

def _parse_json_from_content(content):
    # 清理前后空格
    content = content.strip()
    
    # 策略 1：尝试匹配 Markdown JSON 代码块
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass # 如果代码块里也是坏的，尝试策略 2

    # 策略 2：尝试提取最外层的 { ... } 或 [ ... ]
    # 这种方法可以过滤掉 LLM 在 JSON 前后加的废话
    structure_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
    if structure_match:
        try:
            return json.loads(structure_match.group(1))
        except json.JSONDecodeError:
            pass

    # 策略 3：最后的挣扎，直接尝试解析全文
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"解析失败。LLM 返回的内容: {content}")
        raise ValueError(f"无法从 LLM 响应中提取有效的 JSON: {e}")
        

async def outline_node(state, config: RunnableConfig):
    # 1. 提取 configurable 部分（如果不存在则返回空字典）
    logging.info("--- call_outline_node 大纲生成节点 ---")
    configurable = config.get("configurable", {})
    m_name = configurable.get("model_name", Default_model_name) 
    llm = get_llm(model_name=m_name)

    """大纲生成节点"""
    
    try:
        logging.info("进入大纲节点")
        
        # 检查是否已有大纲
        if state.get("outline_generated", False) and state.get("outline"):
            logging.info("大纲已存在，跳过生成")
            state["next_step"] = "call_task_coordinator"
            return state
        
        # # 从全局变量获取模型
        # from Workflow.workflow import llm
        # if llm is None:
        #     raise ValueError("LLM模型未初始化")
        
        # 生成大纲
        from Prompts.prompts import outline_prompt
        from Prompts.writing_styles import get_style_prompt_enhancement, normalize_style
        
        # 标准化风格并获取增强信息
        normalized_style = normalize_style(state.get("style", "technical"))
        style_enhancement = get_style_prompt_enhancement(normalized_style)
        chapter_count = state.get("chapter_count",5)


        # 2. 确定 Topic 的优先级逻辑
        # 尝试直接从 state 获取
        topic = state.get("topic")
        
        # 如果 topic 为空，尝试从最后一条消息提取
        if not topic or str(topic).strip() == "":
            messages = state.get("messages", [])
            # 从后往前找第一条用户消息
            for msg in reversed(messages):
                # 兼容字典格式或 LangChain 消息对象
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                role = msg.get("role") if isinstance(msg, dict) else (
                    "user" if "User" in str(type(msg)) else "other"
                )
                
                if content and (role == "user" or "Human" in str(type(msg))):
                    topic = content
                    logging.info(f"Topic 为空，已从历史消息中捕获主题: {topic[:30]}...")
                    break
        # 3. 如果依然为空，则需要中断并请求输入
        if not topic:
            logging.warning("未能获取到任何主题(Topic)")
            return {
                "next_step": "end", # 或者跳转到一个专门的人机交互节点
                "messages": [AssistantMessage(content="抱歉，我没有找到写作主题，请告诉我想写什么。")]
            }

        prompt = outline_prompt.format(
            task=topic,
            chapter_count=chapter_count,
            style_enhancement=style_enhancement
        )
        
        # import pdb; pdb.set_trace()
        
        logging.info("正在生成大纲")
        # response = llm.invoke(prompt)
        response = await llm.ainvoke(prompt)
        
        # 记录LLM预测
        content = response.content.strip()
        
        # 解析大纲
        
        # 提取JSON部分
        outline = _parse_json_from_content(content)
        
        # 保存状态 TBD
        # save_state(state) 
        
        # # 添加消息
        # outline_str = json.dumps(outline, ensure_ascii=False, indent=2)
        # # state["next_step"] = "call_task_coordinator"
        return {
            "outline": outline,
            "outline_generated": True,
            "messages": [AssistantMessage(content=f"大纲生成成功:\n```json\n{json.dumps(outline, ensure_ascii=False, indent=2)}\n```")],
            "last_successful_step": "outline"
        }

    except Exception as e:
        logging.error(f"大纲生成失败: {str(e)}")
        return _handle_outline_error(state, e)


# 1. 定义结构化输出模型
class PlanResponse(BaseModel):
    """规划决策模型"""
    status: str = Field(description="决策状态：'COMPLETE' (信息足够) 或 'INCOMPLETE' (需要追问)")
    topic: Optional[str] = Field(None, description="确定的最终标题/主题")
    chapter_count: Optional[int] = Field(None, description="建议的章节数量", ge=3, le=10)
    ai_response: str = Field(description="如果是INCOMPLETE，这是追问的话术；如果是COMPLETE，这是确认的话术")

async def plan_node(state: dict, config: RunnableConfig):
    logging.info("--- [Plan Node] 开始规划决策 ---")
    
    # 获取 LLM 并绑定结构化输出
    from Utils.llm_config import get_llm # 假设你的 LLM 获取方法
    m_name = config.get("configurable", {}).get("model_name", "gpt-4o")
    base_llm = get_llm(model_name=m_name)
    
    # 核心：使用 with_structured_output 确保输出符合 PlanResponse 类
    structured_llm = base_llm.with_structured_output(PlanResponse)
    
    messages = state.get("messages", [])
    
    # 2. 构造 System Prompt 引导 LLM 进行决策
    system_msg = {
        "role": "system",
        "content": (
            "你是一个专业的写作规划助手。你的目标是确定【写作主题】和【章节数量】。\n"
            "1. 审查对话历史。如果用户没有明确主题，请设法引导他。\n"
            "2. 如果用户给了主题但没给章节数，请根据主题深度建议一个（通常是5章）。\n"
            "3. 只有当你认为【主题】和【章节数】都已明确且合理时，status 才设为 'COMPLETE'。"
        )
    }

    # 3. 调用模型
    # 注意：这里直接 await 得到的是一个 PlanResponse 对象
    try:
        plan_result: PlanResponse = await structured_llm.ainvoke([system_msg] + messages)
    except Exception as e:
        logging.error(f"结构化模型调用失败: {e}")
        # 极端情况下的手动解析兜底（可选）
        return {"messages": [AssistantMessage(content="抱歉，我现在规划系统有点忙，请再试一次。")]}

    # 4. 根据模型决策使用 Command 进行路由
    if plan_result.status == "COMPLETE":
        logging.info(f"规划达成一致: {plan_result.topic}")
        
        # 使用 Command 直接跳转到 outline_node
        return Command(
            update={
                "topic": plan_result.topic,
                "chapter_count": plan_result.chapter_count,
                "messages": [AssistantMessage(content=plan_result.ai_response)]
            },
            goto="outline_node"
        )
    
    else:
        # 信息不足，留在当前节点，等待用户在下一轮对话中输入
        logging.info("信息不足，继续对话...")
        return {
            "messages": [AssistantMessage(content=plan_result.ai_response)]
        }