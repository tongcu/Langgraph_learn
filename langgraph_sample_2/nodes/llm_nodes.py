from langchain_core.runnables import RunnableConfig
from LLM.llm import get_llm
from tools.client_tool import tools
from .states import MessageState as State
Default_model_name = "local_qwen"

def call_model(state: State, config: RunnableConfig):
    # 1. 提取 configurable 部分（如果不存在则返回空字典）
    configurable = config.get("configurable", {})
    
    # 2. 提取 model_name，并在后面设置【默认值】
    # 如果 config 里没写，它就会拿 "gpt-4o" 去调你的 get_llm()
    m_name = configurable.get("model_name", Default_model_name) 
    # import pdb; pdb.set_trace()
    # 3. 调用你仓库里的 get_llm
    llm = get_llm(model_name=m_name)
    
    return {"messages": [llm.invoke(state["messages"])]} 

# 这里没有问题？ 三个月？可以的其他的问题

def call_model_dynamic(state, config: RunnableConfig):
    """动态获取 LLM 并支持工具调用"""
    # 从 config 中动态获取模型名称
    configurable = config.get("configurable", {})
    m_name = configurable.get("model_name", Default_model_name)
    # import pdb; pdb.set_trace()
    # 获取 LLM 实例并绑定工具
    llm = get_llm(model_name=m_name)
    llm_with_tools = llm.bind_tools(tools)
    
    # 执行
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}