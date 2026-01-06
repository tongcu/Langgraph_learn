from typing import TypedDict, Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class MessageState(TypedDict):
    messages: Annotated[list, add_messages]


class WritingState(TypedDict, total=False):
    # 核心消息流，使用 Annotated 保证消息会自动追加
    messages: Annotated[List[dict], add_messages]
    task: str
    task_id: str
    outline: List[Dict[str, str]]
    outline_generated: bool
    current_chapter: int
    chapter_count: int
    chapters: List[str]
    final_content: str
    next_step: str
    # 业务字段
    use_knowledge: bool
    search_results: List[Dict[str, Any]]