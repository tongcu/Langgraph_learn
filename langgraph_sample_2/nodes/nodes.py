from langchain_core.runnables import RunnableConfig
from LLM.llm import get_llm

def call_model(state: State, config: RunnableConfig):
    # 1. 提取 configurable 部分（如果不存在则返回空字典）
    configurable = config.get("configurable", {})
    
    # 2. 提取 model_name，并在后面设置【默认值】
    # 如果 config 里没写，它就会拿 "gpt-4o" 去调你的 get_llm()
    m_name = configurable.get("model_name", "local_qwen") 
    
    # 3. 调用你仓库里的 get_llm
    llm = get_llm(model_name=m_name)
    
    return {"messages": [llm.invoke(state["messages"])]}