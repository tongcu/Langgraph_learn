import os
import logging
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from KnowledgeManager.Dependencies.compat import get_langchain_text_splitter
RecursiveCharacterTextSplitter = get_langchain_text_splitter('RecursiveCharacterTextSplitter')

from Config.model_config import RAG_CONFIG
from KnowledgeManager.Dependencies.Embeddings import LocalEmbeddings

# 尝试导入混合文本分割器
try:
    from KnowledgeManager.markdown_hybrid_splitter import MarkdownHybridSplitter
    HYBRID_SPLITTER_AVAILABLE = True
except ImportError:
    HYBRID_SPLITTER_AVAILABLE = False
    logging.warning("混合文本分割器不可用，将使用默认的递归字符分割器")


class BaseKnowledgeManager(ABC):
    """知识管理器抽象基类 (迁移自 report-26v0)"""
    
    def __init__(self, knowledge_base_name: str, embedding_model: Optional[str] = None, 
                 chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None,
                 use_hybrid_splitter: bool = True):
        self.knowledge_base_name = knowledge_base_name
        self.embedding_model = embedding_model or RAG_CONFIG["embeddings"]["default_model"]
        
        # 初始化embedding服务
        self.embeddings = LocalEmbeddings(self.embedding_model)
        
        # 获取embedding维度
        model_config = RAG_CONFIG["embeddings"]["models"].get(self.embedding_model, {})
        self.dimension = model_config.get("dimension", 1024)
        
        # 文本分割器
        if chunk_size is not None and chunk_overlap is not None:
            actual_chunk_size = chunk_size
            actual_chunk_overlap = chunk_overlap
        else:
            actual_chunk_size = model_config.get("chunk_size", RAG_CONFIG["embeddings"]["chunk_size"])
            actual_chunk_overlap = RAG_CONFIG["embeddings"]["chunk_overlap"]
        
        if use_hybrid_splitter and HYBRID_SPLITTER_AVAILABLE:
            try:
                self.text_splitter = MarkdownHybridSplitter(
                    chunk_size=actual_chunk_size,
                    chunk_overlap=actual_chunk_overlap
                )
                logging.info("使用混合文本分割器")
            except Exception as e:
                logging.warning(f"初始化混合文本分割器失败，回退到递归字符分割器: {e}")
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=actual_chunk_size,
                    chunk_overlap=actual_chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""]
                )
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=actual_chunk_size,
                chunk_overlap=actual_chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""]
            )
        
        self.actual_chunk_size = actual_chunk_size
        self.actual_chunk_overlap = actual_chunk_overlap
        
        logging.info(f"初始化知识库管理器: {knowledge_base_name}")
    
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def load_from_folder(self, folder_path: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None, score_threshold: float = 0.3) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def search_keywords(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None, score_threshold: float = 0.3) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def search_bm25(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None, score_threshold: float = 0.3) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def search_hybrid(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None, 
                      vector_weight: float = 0.7, keyword_weight: float = 0.3, score_threshold: float = 0.3) -> Dict[str, Any]:
        pass
    
    def search_with_rerank(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None, 
                          use_rerank: bool = True, score_threshold: float = 0.3) -> Dict[str, Any]:
        search_results = self.search(query, k=k, filters=filters, score_threshold=score_threshold)
        if use_rerank and search_results.get("success", False):
            from KnowledgeManager.KnowledgeManagerFactory import KnowledgeManagerFactory
            reranked_results = KnowledgeManagerFactory.apply_rerank(query, search_results, k)
            return reranked_results
        return search_results
    
    @abstractmethod
    def search_with_details(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None, score_threshold: float = 0.3) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def add_text(self, content: str, source: str = "user_input") -> Dict[str, Any]:
        pass
    
    @staticmethod
    @abstractmethod
    def list_knowledge_bases() -> List[str]:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def delete_knowledge_base(self) -> Dict[str, Any]:
        pass
    
    @staticmethod
    @abstractmethod
    def delete_knowledge_base_by_name(kb_name: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def clear_knowledge_base(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def remove_by_source(self, source_pattern: str) -> Dict[str, Any]:
        pass
