import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import faiss  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

from Config.model_config import RAG_CONFIG
from KnowledgeManager.BaseKnowledgeManager import BaseKnowledgeManager
from KnowledgeManager.knowledge_extractor import knowledge_extractor

# 尝试导入混合文本分割器
try:
    from KnowledgeManager.markdown_hybrid_splitter import MarkdownHybridSplitter
    HYBRID_SPLITTER_AVAILABLE = True
except ImportError:
    HYBRID_SPLITTER_AVAILABLE = False
    logging.warning("混合文本分割器不可用，将使用默认的递归字符分割器")

class FAISSKnowledgeManager(BaseKnowledgeManager):
    """FAISS向量数据库知识管理器 (迁移自 report-26v0)"""
    
    def __init__(self, knowledge_base_name: str, embedding_model: Optional[str] = None,
                 chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None,
                 use_hybrid_splitter: bool = True):
        super().__init__(knowledge_base_name, embedding_model, chunk_size, chunk_overlap, use_hybrid_splitter)
        
        vector_config = RAG_CONFIG["vector_store"]
        self.base_directory = Path(vector_config["faiss"]["base_directory"])
        self.kb_directory = self.base_directory / knowledge_base_name
        
        self.index_file = self.kb_directory / f"{vector_config['faiss']['index_prefix']}{knowledge_base_name}.faiss"
        self.metadata_file = self.kb_directory / f"{vector_config['faiss']['metadata_prefix']}{knowledge_base_name}.json"
        
        self.index = None
        self.metadata = []
        self.texts = []
        
        logging.info(f"初始化FAISS知识库管理器: {knowledge_base_name}")
    
    def initialize(self):
        try:
            self.kb_directory.mkdir(parents=True, exist_ok=True)
            if self.index_file.exists() and self.metadata_file.exists():
                self._load_index()
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []
                self.texts = []
        except Exception as e:
            logging.error(f"初始化知识库失败: {str(e)}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.texts = []
    
    def _load_index(self):
        try:
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get('metadata', [])
                self.texts = data.get('texts', [])
        except Exception as e:
            logging.error(f"加载索引失败: {str(e)}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.texts = []
    
    def _save_index(self):
        try:
            self.kb_directory.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'texts': self.texts
                }, f)
        except Exception as e:
            logging.error(f"保存索引失败: {str(e)}")
    
    def load_from_folder(self, folder_path: str) -> Dict[str, Any]:
        if self.index is None:
            self.initialize()
        
        try:
            documents = knowledge_extractor.extract_from_folder(folder_path)
            if not documents:
                return {"success": False, "message": f"未找到文档"}
            
            all_chunks = []
            all_metadata = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc["content"])
                for chunk in chunks:
                    if len(chunk) >= 3:
                        all_chunks.append(chunk)
                        all_metadata.append({
                            "source": doc["source"],
                            "filename": doc["filename"],
                            "format": doc["format"],
                            "knowledge_base": self.knowledge_base_name,
                            "embedding_model": self.embedding_model
                        })
            
            embeddings = self.embeddings.embed_documents(all_chunks)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            if self.index.ntotal == 0 and embeddings_array.shape[1] != self.dimension:
                self.dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)

            self.index.add(embeddings_array)
            self.texts.extend(all_chunks)
            self.metadata.extend(all_metadata)
            self._save_index()
            
            return {"success": True, "message": f"已加载 {len(all_chunks)} 个片段"}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None, score_threshold: float = 0.3) -> Dict[str, Any]:
        if self.index is None or self.index.ntotal == 0:
            self.initialize()
            if self.index.ntotal == 0:
                return {"success": True, "context": "", "context_list": []}
            
        try:
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            search_k = min(k, self.index.ntotal)
            scores, indices = self.index.search(query_vector, search_k)
            
            context_parts = []
            context_list = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.texts):
                    text = self.texts[idx]
                    metadata = self.metadata[idx]
                    score = float(scores[0][i])
                    if score >= score_threshold:
                        context_list.append({
                            "source": metadata.get("filename", "未知"),
                            "metadata": metadata,
                            "score": score,
                            "content": text
                        })
                        context_parts.append(f"[来源: {metadata.get('filename')}, 相似度: {score:.3f}]\n{text}")
            
            return {
                "success": True,
                "context": "\n\n".join(context_parts),
                "context_list": context_list,
                "docs_count": len(context_list)
            }
        except Exception as e:
            return {"success": False, "message": str(e)}

    def search_with_details(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None, score_threshold: float = 0.3) -> Dict[str, Any]:
        return self.search(query, k, filters, score_threshold)

    def search_bm25(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None, score_threshold: float = 0.3) -> Dict[str, Any]:
        # 简化版 BM25 模拟
        return self.search(query, k, filters, score_threshold)

    def search_keywords(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None, score_threshold: float = 0.3) -> Dict[str, Any]:
        return self.search_bm25(query, k, filters, score_threshold)

    def search_hybrid(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None, 
                      vector_weight: float = 0.7, keyword_weight: float = 0.3, score_threshold: float = 0.3) -> Dict[str, Any]:
        return self.search(query, k, filters, score_threshold)

    def add_text(self, content: str, source: str = "user_input") -> Dict[str, Any]:
        if self.index is None: self.initialize()
        try:
            chunks = self.text_splitter.split_text(content)
            embeddings = self.embeddings.embed_documents(chunks)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            self.texts.extend(chunks)
            for _ in chunks:
                self.metadata.append({"source": source, "knowledge_base": self.knowledge_base_name})
            self._save_index()
            return {"success": True, "chunks_count": len(chunks)}
        except Exception as e:
            return {"success": False, "message": str(e)}

    @staticmethod
    def list_knowledge_bases() -> List[str]:
        base_dir = Path(RAG_CONFIG["vector_store"]["faiss"]["base_directory"])
        if not base_dir.exists(): return []
        return [d.name for d in base_dir.iterdir() if d.is_dir()]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "knowledge_base": self.knowledge_base_name,
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_texts": len(self.texts)
        }

    def delete_knowledge_base(self) -> Dict[str, Any]:
        import shutil
        if self.kb_directory.exists():
            shutil.rmtree(self.kb_directory)
            return {"success": True}
        return {"success": False}

    @staticmethod
    def delete_knowledge_base_by_name(kb_name: str) -> Dict[str, Any]:
        base_dir = Path(RAG_CONFIG["vector_store"]["faiss"]["base_directory"])
        kb_dir = base_dir / kb_name
        if kb_dir.exists():
            import shutil
            shutil.rmtree(kb_dir)
            return {"success": True}
        return {"success": False}

    def clear_knowledge_base(self) -> Dict[str, Any]:
        if self.index_file.exists(): self.index_file.unlink()
        if self.metadata_file.exists(): self.metadata_file.unlink()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.texts = []
        return {"success": True}

    def remove_by_source(self, source_pattern: str) -> Dict[str, Any]:
        return {"success": False, "message": "未实现"}
