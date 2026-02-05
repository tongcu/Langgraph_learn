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
            return f" 成功清理 {count} 个历史线程记录！"
        except Exception as e:
            return f" 全量清理失败: {str(e)}"

    async def clear_specific_thread(self, session_id: str):
        """功能 2：清理指定的特定线程"""
        if not session_id:
            return " 请输入有效的会话 ID"
        thread_id = name_to_uuid(session_id)
        try:
            await self.client.threads.delete(thread_id)
            return f" 线程 `{session_id}` 已被物理删除。"
        except Exception as e:
            return f" 删除失败: {str(e)}"

    async def monitor_thread_state(self, session_id: str):
        """功能 3：实时监控线程状态"""
        if not session_id:
            return " 请输入会话 ID 进行监控"
        thread_id = name_to_uuid(session_id)
        try:
            state = await self.client.threads.get_state(thread_id)
            if not state or "next" not in state:
                return " 该线程尚未启动或没有活动记录。"
            
            next_nodes = state.get("next", [])
            messages = state.get("values", {}).get("messages", [])
            msg_count = len(messages)
            
            report = [
                f"###  线程实时监控",
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
            return f" 监控获取失败: {str(e)}"

    async def get_thread_values(self, session_id: str, keys: list = None):
        """
        独立功能：获取指定 thread 的 State values 中的特定字段
        :param session_id: 会话ID
        :param keys: 想要获取的字段列表，如 ['task', 'files']。如果为 None 则返回全部。
        """
        # client = get_client(url=self.api_url)
        thread_id = name_to_uuid(session_id)
        
        try:
            state = await self.client.threads.get_state(thread_id)
            if not state or "values" not in state:
                return None
            
            values = state["values"]
            if keys:
                # 只保留用户指定的 key
                return {k: values.get(k) for k in keys if k in values}
            return values
        except Exception as e:
            print(f"Error fetching state values: {e}")
            return None

    async def monitor_specific_fields(self, session_id: str):
        """
        UI 适配功能：获取指定字段并格式化为 Markdown 展示
        """
        # 假设你想监控 'task' 和 'files' 字段
        target_keys = ["task", "chapters"]
        # data = await self.get_thread_values(session_id, keys=target_keys)
        data = await self.get_thread_values(session_id)
        if not data:
            return " 未找到相关状态数据。"
        
        field_display_box = "###  当前 State 关键字段\n"
        for key, value in data.items():
            if key == "files":
                field_display_box += f"** 文件列表**: {value if value else '无'}\n\n"
            elif key == "task":
                # 限制显示长度防止撑破 UI
                display_task = (value[:200] + '...') if isinstance(value, str) and len(value) > 200 else value
                field_display_box += f"** 分析任务**: \n> {display_task}\n\n"
            else:
                field_display_box += f"**🔹 {key}**:\n\n {value}\n\n"
        
        # return field_display_box
        return data # 前端通过json格式展示
    
    async def run_graph(self, inputs: dict, config: dict, graph_id: str = "my_agent"):
        """
        运行图的通用方法
        :param inputs: 图的输入数据
        :param config: 配置信息，其中 configurable 必须包含 thread_id (UUID 字符串)
        :param graph_id: 要运行的图的ID
        :return: 图执行的结果
        """
        try:
            # 从配置中直接获取 thread_id
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                # 兜底：如果没提供，生成一个默认的
                thread_id = name_to_uuid("default_report_thread")
            
            # 运行图
            result = self.client.runs.stream(
                thread_id,
                graph_id,
                input=inputs,
                config=config,
            )
            
            # 收集结果
            final_result = {}
            async for event in result:
                if event.event == "values":
                    final_result = event.data
                elif event.event == "end":
                    # 如果 end 事件中有 output，优先使用
                    if event.data and "output" in event.data:
                        final_result = event.data["output"]
                    break
            
            return final_result
        except Exception as e:
            print(f"Error running graph: {e}")
            return {"error": str(e)}

