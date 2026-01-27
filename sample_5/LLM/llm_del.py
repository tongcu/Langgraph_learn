import logging
# from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
from config import MODEL_CONFIGS
# _llm = None
def get_llm(
    model: str = "local_qwen",
    *,
    temp: float = None,
    stream: bool = True,
    **kwargs
) -> ChatOpenAI:
    """
    EXample：
        llm()                    → 默认本地小Qwen
        llm("deepseek")          → DeepSeek官方
        llm("local_qwen", 0.1)   → 小Qwen低温精确模式
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"模型不存在: {model}\n可选: {list(MODEL_CONFIGS.keys())}")

    # 1. 拿到原始配置并复制
    cfg = MODEL_CONFIGS[model].copy()

    # 2. temp 优先级：显式传入 > 配置里默认值 > 0.7兜底
    final_temp = 0.7
    if temp is not None:
        final_temp = temp
    elif "temperature" in cfg:
        final_temp = cfg["temperature"]

    # 3. 合并所有额外参数（max_tokens / top_p / stop 等随便传）
    if kwargs:
        cfg.update(kwargs)

    # 4. 强制写入最终值
    cfg["temperature"] = final_temp
    cfg["streaming"] = stream

    logging.info(f"LLM启动 → {model} | temp={final_temp} | stream={stream}")

    return ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        streaming=cfg["streaming"],
        max_tokens=cfg.get("max_tokens"),
        top_p=cfg.get("top_p"),
        **{k: v for k, v in cfg.items() if k not in {"base_url", "api_key", "model", "temperature", "streaming"}}
    )