from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional

# 1. 科研类 (Science)
class ScienceSchema(BaseModel):
    title: str = Field(description="论文标题")
    methodology: str = Field(description="研究方法与实验设计")
    findings: List[str] = Field(description="核心发现与结论")
    limitations: Optional[str] = Field(description="研究局限性(如有)")

# 2. 架构说明书类 (Architecture)
class ArchSchema(BaseModel):
    system_name: str = Field(description="系统名称")
    tech_stack: List[str] = Field(description="核心技术栈(如Python, Redis, Docker)")
    components: List[str] = Field(description="核心功能组件及其职责")
    data_flow: Optional[str] = Field(description="关键数据流向描述")

# 3. 需求文档类 (PRD)
class PRDSchema(BaseModel):
    project_name: str = Field(description="项目/产品名称")
    target_users: List[str] = Field(description="目标用户画像")
    core_features: List[str] = Field(description="核心功能特性列表")
    user_pain_points: List[str] = Field(description="该文档解决的主要用户痛点")

# 4. 新闻类 (News)
class NewsSchema(BaseModel):
    headline: str = Field(description="新闻标题")
    occurred_at: str = Field(description="事件发生的时间和地点")
    key_entities: List[str] = Field(description="涉及的关键人物、组织或国家")
    event_summary: str = Field(description="事件经过摘要 (5W1H分析)")

# 5. 默认/通用报告类 (General)
class GeneralReportSchema(BaseModel):
    subject: str = Field(description="报告主题")
    summary: str = Field(description="摘要内容")
    key_takeaways: List[str] = Field(description="主要启示或关键结论")
    suggestions: List[str] = Field(description="针对性的对策或建议")

class CodeSummary(BaseModel):
    title: str = Field(description="代码标题")
    tech_stack: List[str] = Field(description="核心技术栈(语言、数据库、中间件)")
    logic_components: List[str] = Field(description="实现的主要功能")
    deployment: str = Field(description="部署环境及依赖包")
# # --- Tool 定义 ---
# @tool
# def science_summary_tool(data: ScienceSchema):
#     """科研论文总结工具"""
#     return {"result": data.dict()}

# @tool
# def architecture_summary_tool(data: ArchSchema):
#     """技术架构文档总结工具"""
#     return {"result": data.dict()}

# # 工具列表供 Agent 绑定
# summary_tools = [science_summary_tool, architecture_summary_tool]