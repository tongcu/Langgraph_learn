from typing import TypedDict, Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import operator

class MessageState(TypedDict):
    messages: Annotated[list, add_messages]
    task_id: str


class WritingState(TypedDict, total=False):
    # 核心消息流，使用 Annotated 保证消息会自动追加
    messages: Annotated[List[dict], add_messages]
    task: str
    quary: str
    task_id: str
    
    outline_generated: bool
    current_chapter: int
    chapter_count: int
    # chapters: List[str]
    
    # tools_results: List[str]
    final_content: str
    topic: str

    # 业务字段
    use_knowledge: bool
    knowledge_content: str
    chapter_knowledge: List[str]
    knowledge_base: str
    search_mode: str
    search_k: int
    score_threshold: float
    vector_weight: float
    keyword_weight: float

    chapters: List[str] 
    chapter_details: List[Dict[str, str]]  # 存储章节详细信息：title 和 content
    search_results: List[Dict[str, Any]]
    outline: List[Dict[str, str]]
    # search_results: List[Dict[str, Any]]
    summary_text: str
    merged_article: str  # 合并后的完整文章（Markdown格式）
    writing_template: str
    reference_files: List[Dict[str, str]]

    # 状态字段
    next_step: str
    last_successful_step: str 


class Main_state(TypedDict, total=False):
    # 核心消息流，使用 Annotated 保证消息会自动追加
    messages: Annotated[List[dict], add_messages]
    task: str
    task_id: str
    
    outline_generated: bool
    current_chapter: int
    chapter_count: int
    chapter_details: List[Dict[str, str]]  # 存储章节详细信息：title 和 content
    tools_results: List[str]
    final_content: str
    topic: str

    # 业务字段
    use_knowledge: bool
    knowledge_content: str
    chapter_knowledge: List[str]
    knowledge_base: str
    search_mode: str
    search_k: int
    score_threshold: float
    vector_weight: float
    keyword_weight: float
    search_results: List[Dict[str, Any]]
    summary_text: str
    merged_article: str  # 合并后的完整文章（Markdown格式）
    reference_files: List[Dict[str, str]]


    chapters: List[str] 
    search_results: List[Dict[str, Any]]
    outline: List[Dict[str, str]]
    writing_template: str


    # 状态字段
    next_step: str
    last_successful_step: str 
