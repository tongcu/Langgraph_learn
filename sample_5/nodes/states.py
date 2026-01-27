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

    chapters: List[str] 
    search_results: List[Dict[str, Any]]
    outline: List[Dict[str, str]]
    # search_results: List[Dict[str, Any]]
    summary_text: str


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
    chapters: List[str]
    tools_results: List[str]
    final_content: str
    topic: str

    # 业务字段
    use_knowledge: bool
    search_results: List[Dict[str, Any]]
    summary_text: str

    chapters: List[str] 
    search_results: List[Dict[str, Any]]
    outline: List[Dict[str, str]]


    # 状态字段
    next_step: str
    last_successful_step: str 