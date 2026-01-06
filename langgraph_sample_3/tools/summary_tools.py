from langchain_core.tools import tool
from schema.summary_schema import ScienceSchema, ArchSchema, PRDSchema, NewsSchema, GeneralReportSchema, CodeSummary
# class CodeSummary(BaseModel):
#     title: str = Field(description="代码标题")
#     tech_stack: List[str] = Field(description="核心技术栈(语言、数据库、中间件)")
#     logic_components: List[str] = Field(description="实现的主要功能")
#     deployment: str = Field(description="部署环境及依赖包")

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

@tool
def summarize_code(data: CodeSummary):
    """用于代码片段和功能总结。"""
    return {"category": "code", "result": data.dict()}


# 工具列表，供 Agent 绑定
summary_tools_ex1 = [
    summarize_science, 
    # summarize_architecture, 
    # summarize_prd, 
    # summarize_news, 
    summarize_general
]