from langchain_core.tools import tool
from schema.summary_schema import ScienceSchema, ArchSchema, PRDSchema, NewsSchema, GeneralReportSchema

@tool
def summarize_science(data: ScienceSchema):
    """用于科研论文、学术期刊、实验报告的深度总结。"""
    return {"category": "science", "result": data.dict()}

@tool
def summarize_architecture(data: ArchSchema):
    """用于系统架构说明书、技术设计方案、基础设施文档的总结。"""
    return {"category": "architecture", "result": data.dict()}

@tool
def summarize_prd(data: PRDSchema):
    """用于产品需求文档 (PRD)、业务规格说明书、功能需求列表的总结。"""
    return {"category": "prd", "result": data.dict()}

@tool
def summarize_news(data: NewsSchema):
    """用于新闻资讯、媒体快讯、时事评论、行业动态的总结。"""
    return {"category": "news", "result": data.dict()}

@tool
def summarize_general(data: GeneralReportSchema):
    """用于普通商业报告、会议纪要、工作周报等非特定领域的文档总结。"""
    return {"category": "general", "result": data.dict()}

# 工具列表，供 Agent 绑定
summary_tools_ex1 = [
    summarize_science, 
    summarize_architecture, 
    summarize_prd, 
    summarize_news, 
    summarize_general
]