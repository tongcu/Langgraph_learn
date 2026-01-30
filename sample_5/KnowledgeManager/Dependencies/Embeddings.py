import os
import logging
from typing import List, Optional
from openai import OpenAI
from Config.model_config import RAG_CONFIG

class LocalEmbeddings:
    """本地embedding服务包装器 (迁移自 report-26v0)"""
    
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = RAG_CONFIG["embeddings"]["default_model"]
        
        model_config = RAG_CONFIG["embeddings"]["models"].get(model_name)
        if not model_config:
            raise ValueError(f"未找到embedding模型配置: {model_name}")
        
        self.base_url = model_config["base_url"]
        self.api_key = model_config["api_key"]
        self.model = model_config["model"]
        self.dimension = model_config["dimension"]
        self.model_name = model_name
        
        logging.info(f"初始化embedding服务: {self.base_url}, 模型: {self.model}")
        
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logging.info("Embedding服务初始化成功")
        except Exception as e:
            logging.error(f"Embedding服务初始化失败: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logging.error(f"批量嵌入文档失败: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"嵌入查询失败: {str(e)}")
            raise
    
    def get_embedding_model(self) -> str:
        return self.model
