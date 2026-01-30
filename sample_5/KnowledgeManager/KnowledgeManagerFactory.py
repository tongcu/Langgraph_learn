import logging
from typing import Dict, Any, Optional
from Config.model_config import RAG_CONFIG
from KnowledgeManager.FAISSKnowledgeManager import FAISSKnowledgeManager
from KnowledgeManager.reranker import apply_rerank_to_search_results

class KnowledgeManagerFactory:
    """知识管理器工厂类 (迁移自 report-26v0)"""
    
    @staticmethod
    def create_knowledge_manager(knowledge_base_name: str, embedding_model: str = None, 
                                vector_store_type: str = None, use_hybrid_splitter: bool = True,
                                chunk_size: int = None, chunk_overlap: int = None, **kwargs) -> Any:
        if vector_store_type is None:
            vector_store_type = RAG_CONFIG.get("vector_store", {}).get("type", "faiss")
        
        vector_store_type = vector_store_type.lower()
        logging.info(f"创建知识管理器: {knowledge_base_name}, 类型: {vector_store_type}")
        
        if vector_store_type == "faiss":
            return FAISSKnowledgeManager(
                knowledge_base_name=knowledge_base_name,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_hybrid_splitter=use_hybrid_splitter
            )
        else:
            # 目前只支持 FAISS 迁移，其他返回 FAISS 作为兜底
            logging.warning(f"目前迁移版只支持 FAISS，使用默认 FAISS")
            return FAISSKnowledgeManager(
                knowledge_base_name=knowledge_base_name,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_hybrid_splitter=use_hybrid_splitter
            )
    
    @staticmethod
    def apply_rerank(query: str, search_results: Dict[str, Any], top_k: int = None) -> Dict[str, Any]:
        return apply_rerank_to_search_results(query, search_results, top_k)
