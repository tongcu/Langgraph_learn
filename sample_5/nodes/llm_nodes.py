from langchain_core.runnables import RunnableConfig
from LLM.llm import get_llm
from tools.client_tool import tools
import logging
from .states import MessageState, WritingState
Default_model_name = "local_qwen"

async def call_model_vanilla(state, config: RunnableConfig):
    # 1. 提取 configurable 部分（如果不存在则返回空字典）
    logging.info("--- call_model_vanilla 正在决策总结策略 ---")
    configurable = config.get("configurable", {})
    m_name = configurable.get("model_name", Default_model_name) 
    llm = get_llm(model=m_name)

    # print(f"DEBUG: Messages count: {len(state['messages'])}")
    # for i, msg in enumerate(state['messages']):
    #     # 只打印前 100 字符，避免刷屏
    #     print(f"Msg {i} ({msg.type}): {msg.content[:100]}...")
        
    # print(state["messages"])
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]} 

# 这里没有问题？ 三个月？可以的其他的问题

def call_model_tools(state, config: RunnableConfig):
    """动态获取 LLM 并支持工具调用"""
    # 从 config 中动态获取模型名称
    logging.info("--- call_model_tools 正在决策总结策略 ---")
    configurable = config.get("configurable", {})
    m_name = configurable.get("model_name", Default_model_name)
    # import pdb; pdb.set_trace()
    # 获取 LLM 实例并绑定工具
    llm = get_llm(model=m_name)
    llm_with_tools = llm.bind_tools(tools)

    # 执行
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


async def summary_agent_node(state: WritingState, config: RunnableConfig):
    """
    一个节点搞定：意图识别 + 领域聚焦 + 工具分发
    """
    logging.info("--- Agent 正在决策总结策略 ---")
    
    # 获取配置中的模型（通常在 workflow 配置中传入）
    configurable = config.get("configurable", {})
    m_name = configurable.get("model_name", Default_model_name)
    logging.info(f"model name :{m_name}")
    # import pdb; pdb.set_trace()
    # 获取 LLM 实例并绑定工具
    llm = get_llm(model=m_name)

    
    # 关键：将所有总结工具绑定到模型上
    from tools.summary_tools import summary_tools_ex1 as all_summary_tools
    llm_with_tools = llm.bind_tools(all_summary_tools)
    
    # 提示词引导：告诉模型不同工具的适用范围
    system_message = (
        "你是一个全能文档分析专家。请阅读内容并调用最合适的工具：\n"
        # "- 论文/学术文章 -> summarize_science\n"
        # "- 架构/设计文档 -> summarize_architecture\n"
        # "- 需求/PRD文档 -> summarize_prd\n"
        # "- 新闻/报道类 -> summarize_news\n"
        # "- 其他一般报告 -> summarize_general"
    )
    
    # 直接调用
    response = await llm_with_tools.ainvoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": state["task"]}
    ])
    # import pdb; pdb.set_trace()
    # response
    # AIMessage(content='', additional_kwargs={}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'Qwen3-30B-A3B-Thinking-2507-AWQ-4bit', 'model_provider': 'openai'}, id='lc_run--019b9243-ec1b-7f93-ba5c-d4c06b6158ce', tool_calls=[{'name': 'summarize_general', 'args': {'data': '长江经济带十年来航运发展变化', 'summary': '过去十年，长江经济带依托长江黄金水道，实现了航运基础设施、智慧化升级与多式联运的全面突破。长江干线亿吨大港由11个增至18个，高等级航道达1.1万公里，5万吨级海轮可直达南京，万吨级船舶通达武汉，5000吨级船舶直达重庆。重庆果园港等枢纽港实现智慧化改造，场桥远控、铁水公联运无缝衔接，年吞吐量超2600万吨，成为西部出海新通道。三峡船闸通过技术创新，大修停航时间从100多天缩短至30天，提升通航效率。智慧海事系统与电子航道图广泛应用，实现多维感知、智能管控，事故险情下降70%。长江航运货物吞吐量突破42亿吨，稳居世界内河第一。航运成为推动产业梯度转移、区域协调发展和绿色低碳转型的关键支撑，助力长江经济带高质量发展。', 'key_takeaways': ['长江干线亿吨大港增加至18个，航道等级和通航能力显著提升', '重庆果园港实现智慧化升级，成为西部陆海新通道重要节点', '三峡船闸大修周期大幅缩短，通航效率显著提高', '智慧海事系统与电子航道图实现精细化、智能化管理', '长江航运吞吐量超42亿吨，为全球内河运输最繁忙航道'], 'suggestions': ['进一步推广长江电子航道图在智慧航运中的应用场景', '加强沿江港口与中欧班列、西部陆海新通道的协同联动', '深化长江航运绿色低碳转型，推广岸电、清洁能源船舶应用', '持续优化多式联运体制机制，降低综合物流成本']}, 'id': 'chatcmpl-tool-94d1274bcecc4a66b45891fd2b8f3d91', 'type': 'tool_call'}])
    
    # 返回消息。LangGraph 的 ToolNode 会根据这里的 response.tool_calls 自动执行后续逻辑。
    return {"messages": [response]}