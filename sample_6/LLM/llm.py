import logging
from langchain_openai import ChatOpenAI
from Config.model_config import MODEL_CONFIGS
from typing import Optional

# 全局缓存：key = (model_name, temperature, streaming, 其他关键参数的元组)
_llm_cache: dict[tuple, ChatOpenAI] = {}


def get_llm(
    model: str = "local_qwen",
    *,
    temp: Optional[float] = None,
    stream: bool = True,
    **kwargs
) -> ChatOpenAI:
    """
    同一个配置只创建一次 LLM 实例
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"模型不存在: {model}\n可选: {list(MODEL_CONFIGS.keys())}")

    # 1. 确定最终 temperature
    cfg = MODEL_CONFIGS[model].copy()
    final_temp = 0.7
    if temp is not None:
        final_temp = temp
    elif "temperature" in cfg:
        final_temp = cfg["temperature"]

    # 2. 合并 kwargs
    if kwargs:
        cfg.update(kwargs)

    # 3. 强制设置核心参数
    cfg["temperature"] = final_temp
    cfg["streaming"] = stream

    # ─── 关键：生成缓存 key ───
    # 使用稳定的、可哈希的 tuple 作为 key
    cache_key = (
        model,
        final_temp,
        stream,
        cfg.get("max_tokens"),
        cfg.get("top_p"),
        # 如果你经常改其他重要参数，也可以加进来，例如：
        # cfg.get("presence_penalty"),
        # cfg.get("frequency_penalty"),
    )

    # 4. 命中缓存直接返回
    if cache_key in _llm_cache:
        logging.debug(f"LLM 缓存命中 → {model} | temp={final_temp} | stream={stream}")
        return _llm_cache[cache_key]

    # 5. 未命中 → 创建新实例
    logging.info(f"创建新的 LLM 实例 → {model} | temp={final_temp} | stream={stream}")

    llm = ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        streaming=cfg["streaming"],
        max_tokens=cfg.get("max_tokens"),
        top_p=cfg.get("top_p"),
        **{k: v for k, v in cfg.items() if k not in {
            "base_url", "api_key", "model", "temperature", "streaming"
        }}
    )

    # 存入缓存
    _llm_cache[cache_key] = llm
    return llm

    ## import openai 不支持bind_tools