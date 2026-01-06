import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from Workflow.state import WritingState

# --- 1. 定义多层级总结的结构化模型 (Schema) ---

class ScienceSummary(BaseModel):
    title: str = Field(description="论文/研究标题")
    methodology: str = Field(description="研究方法与实验设计")
    key_findings: List[str] = Field(description="核心发现与数据结论")
    limitations: str = Field(description="研究局限性或未来方向")

class ReportSummary(BaseModel):
    title: str = Field(description="报告名称")
    background: str = Field(description="报告背景与初衷")
    key_metrics: List[str] = Field(description="关键指标与统计数据")
    suggestions: List[str] = Field(description="结论与对策建议")

class NewsSummary(BaseModel):
    title: str = Field(description="新闻标题")
    event_5w1h: str = Field(description="5W1H分析(何时何地何人何事何故如何)")
    social_impact: str = Field(description="该事件的社会影响或舆论反馈")

class ArchSpecSummary(BaseModel):
    title: str = Field(description="架构名称")
    tech_stack: List[str] = Field(description="核心技术栈(语言、数据库、中间件)")
    logic_components: List[str] = Field(description="核心逻辑模块功能说明")
    deployment: str = Field(description="部署环境与拓扑结构")

class CodeSummary(BaseModel):
    title: str = Field(description="代码标题")
    tech_stack: List[str] = Field(description="核心技术栈(语言、数据库、中间件)")
    logic_components: List[str] = Field(description="实现的主要功能")
    deployment: str = Field(description="部署环境及依赖包")

# --- 2. 节点功能实现 ---

async def summary_intent_focus_node(state: WritingState, config):
    """
    二级意图聚焦节点：识别总结的具体领域分类
    """
    try:
        logging.info("--- 正在进行总结领域识别 ---")
        llm = config["configurable"].get("model")
        user_input = state.get("task", "")
        
        focus_prompt = f"""
        请分析用户提供的文档内容，将其归类为以下四类之一：
        - science: 科研论文、学术文章、实验报告
        - report: 商业月报、财务分析、工作总结、行业研究
        - news: 时事新闻、快讯、媒体报道
        - architecture: 技术架构图说明、软件设计文档、API规范
        
        请仅返回分类关键字（science/report/news/architecture）。
        内容摘要: {user_input[:300]}
        """
        
        response = await llm.ainvoke(focus_prompt)
        category = response.content.lower().strip()
        
        # 简单清洗，防止模型返回多余文字
        valid_categories = ["science", "report", "news", "architecture"]
        final_category = next((c for c in valid_categories if c in category), "report")
        
        logging.info(f"识别到总结类别: {final_category}")
        return {
            "summary_category": final_category,
            "next_step": "call_summary_execute"
        }
    except Exception as e:
        logging.error(f"领域聚焦失败: {str(e)}")
        return {"next_step": "error_recovery"}


async def summary_execute_node(state: WritingState, config):
    """
    总结执行节点：根据聚焦的领域应用不同的结构化策略
    """
    try:
        category = state.get("summary_category", "report")
        logging.info(f"--- 开始执行 [{category}] 领域的结构化总结 ---")
        
        llm = config["configurable"].get("model")
        document = state.get("task", "")

        # 策略映射：Schema 和 专用提示词
        strategy_map = {
            "science": (ScienceSummary, "侧重研究方法和数据发现"),
            "report": (ReportSummary, "侧重背景、指标和建议"),
            "news": (NewsSummary, "侧重5W1H事实提取"),
            "architecture": (ArchSpecSummary, "侧重技术栈和模块逻辑")
        }
        
        schema, detail_hint = strategy_map.get(category)
        
        # 使用 Structured Output 确保结果是纯 JSON 对象
        structured_llm = llm.with_structured_output(schema)
        
        summary_result = await structured_llm.ainvoke([
            {"role": "system", "content": f"你是一个专业的总结专家。{detail_hint}"},
            {"role": "user", "content": document}
        ])

        # 将 Pydantic 对象转为字典并格式化为前端展示内容
        res_dict = summary_result.dict()
        formatted_content = f"## {res_dict.get('title', '文档总结')}\n\n"
        for key, value in res_dict.items():
            if key == "title": continue
            formatted_content += f"### {key.capitalize()}\n{value}\n\n"

        return {
            "final_content": formatted_content,
            "task_completed": True,
            "next_step": "end"
        }
    except Exception as e:
        logging.error(f"总结执行失败: {str(e)}")
        return {"next_step": "error_recovery"}