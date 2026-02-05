import logging

def get_langchain_text_splitter(class_name):
    """
    兼容性函数，用于动态导入 LangChain 分割器类
    """
    try:
        if class_name == 'RecursiveCharacterTextSplitter':
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter
        elif class_name == 'MarkdownHeaderTextSplitter':
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            return MarkdownHeaderTextSplitter
        else:
            # 兜底尝试
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter
    except ImportError:
        logging.warning(f"无法导入 {class_name}，将尝试替代方案")
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(f"未找到可用的 LangChain 分割器类: {class_name}")
