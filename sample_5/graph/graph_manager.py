# utils/graph_manager.py
import hashlib
from uuid import UUID
from langgraph_sdk import get_client
from Utils.id import name_to_uuid_nr as name_to_uuid

class GraphManager:
    def __init__(self, api_url: str):
        self.client = get_client(url=api_url)

    async def clear_all_threads(self):
        """功能 1：彻底清除后端所有线程"""
        try:
            threads = await self.client.threads.search(limit=1000)
            if not threads:
                return "💡 后端很干净，没有发现任何线程记录。"
            
            count = 0
            for t in threads:
                await self.client.threads.delete(t["thread_id"])
                count += 1
            return f"✅ 成功清理 {count} 个历史线程记录！"
        except Exception as e:
            return f"❌ 全量清理失败: {str(e)}"

    async def clear_specific_thread(self, session_id: str):
        """功能 2：清理指定的特定线程"""
        if not session_id:
            return "⚠️ 请输入有效的会话 ID"
        thread_id = name_to_uuid(session_id)
        try:
            await self.client.threads.delete(thread_id)
            return f"✅ 线程 `{session_id}` 已被物理删除。"
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"

    async def monitor_thread_state(self, session_id: str):
        """功能 3：实时监控线程状态"""
        if not session_id:
            return "⚠️ 请输入会话 ID 进行监控"
        thread_id = name_to_uuid(session_id)
        try:
            state = await self.client.threads.get_state(thread_id)
            if not state or "next" not in state:
                return "🔍 该线程尚未启动或没有活动记录。"
            
            next_nodes = state.get("next", [])
            messages = state.get("values", {}).get("messages", [])
            msg_count = len(messages)
            
            report = [
                f"### 📊 线程实时监控",
                f"- **会话 ID**: `{session_id}`",
                f"- **当前活跃节点**: `{next_nodes if next_nodes else '已进入 END'}`",
                f"- **累计消息总数**: `{msg_count}` 条"
            ]
            
            if msg_count > 0:
                last_msg = messages[-1]
                content = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
                report.append(f"- **最后回复预览**: \n> {content[:60]}...")
            
            return "\n".join(report)
        except Exception as e:
            return f"⚠️ 监控获取失败: {str(e)}"