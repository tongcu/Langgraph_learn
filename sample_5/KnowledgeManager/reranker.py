#!/usr/bin/env python3
"""
独立的rerank模块，用于对检索结果进行重排序
"""

import logging
import requests
import json
from typing import List, Dict, Any, Optional
from Config.model_config import RAG_CONFIG

class Reranker:
    """Reranker类，用于对检索结果进行重排序"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化Reranker
        
        Args:
            model_name: rerank模型名称，默认从配置中获取
        """
        self.model_name = model_name or RAG_CONFIG["rerank"]["default_model"]
        self.rerank_config = RAG_CONFIG["rerank"]["models"].get(self.model_name, {})
        # 修正API URL路径，添加正确的端点
        base_url = self.rerank_config.get("api_url", "http://localhost:8000/v1")
        self.api_url = f"{base_url}/rerank" if not base_url.endswith("/rerank") else base_url
        self.api_key = self.rerank_config.get("api_key", "")
        self.enabled = RAG_CONFIG["rerank"].get("enabled", False)
        
        logging.info(f"初始化Reranker: {self.model_name}, API URL: {self.api_url}")
    
    def is_enabled(self) -> bool:
        """检查rerank功能是否启用"""
        return self.enabled
    
    def rerank(self, query: str, context_list: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询语句
            context_list: 检索结果列表，每个元素包含"context"字段
            top_k: 返回前k个结果，默认返回所有结果
            
        Returns:
            重排序后的结果列表
        """
        if not self.enabled:
            logging.info("Rerank功能未启用，返回原始结果")
            return context_list[:top_k] if top_k else context_list
        
        if not context_list:
            return []
        
        try:
            # 准备rerank请求数据
            documents = [item.get("content", "") for item in context_list]
            
            # 构建请求体
            payload = {
                "model": self.rerank_config.get("model", self.model_name),
                "query": query,
                "documents": documents,
                "top_n": top_k or len(documents)
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # 发送rerank请求
            logging.info(f"发送rerank请求到: {self.api_url}")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30  # 30秒超时
            )
            
            # 检查响应状态
            if response.status_code != 200:
                logging.error(f"Rerank API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                # 返回原始结果
                return context_list[:top_k] if top_k else context_list
            
            # 解析响应
            result = response.json()
            
            # 重新排序结果
            reranked_results = []
            for item in result.get("results", []):
                index = item.get("index")
                if 0 <= index < len(context_list):
                    # 复制原始结果并添加rerank分数
                    reranked_item = context_list[index].copy()
                    reranked_item["rerank_score"] = item.get("relevance_score", 0)
                    reranked_results.append(reranked_item)
            
            # 如果API返回的结果数量少于预期，补充原始结果
            if len(reranked_results) < (top_k or len(context_list)):
                # 获取未被rerank的原始结果
                reranked_indices = {item.get("index") for item in result.get("results", [])}
                for i, original_item in enumerate(context_list):
                    if i not in reranked_indices and len(reranked_results) < (top_k or len(context_list)):
                        reranked_item = original_item.copy()
                        reranked_item["rerank_score"] = 0  # 未rerank的项分数为0
                        reranked_results.append(reranked_item)
            
            logging.info(f"Rerank完成，返回{len(reranked_results)}个结果")
            return reranked_results
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Rerank API请求异常: {str(e)}")
            # 返回原始结果
            return context_list[:top_k] if top_k else context_list
        except Exception as e:
            logging.error(f"Rerank处理异常: {str(e)}")
            # 返回原始结果
            return context_list[:top_k] if top_k else context_list
    
    def rerank_with_context(self, query: str, context_list: List[Dict[str, Any]], top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        对检索结果进行重排序并返回格式化的上下文
        
        Args:
            query: 查询语句
            context_list: 检索结果列表
            top_k: 返回前k个结果
            
        Returns:
            包含重排序结果和格式化上下文的字典
        """
        # 执行rerank
        reranked_results = self.rerank(query, context_list, top_k)
        
        # 构建格式化的上下文字符串
        context_parts = []
        for i, result in enumerate(reranked_results):
            text = result.get("content", "")
            source = result.get("source", "未知来源")
            score = result.get("rerank_score", result.get("score", 0))
            
            context_parts.append(f"[来源: {source}, 相关性: {score:.4f}]\n{text}")
        
        context = "\n\n".join(context_parts)
        
        return {
            "success": True,
            "context": context,
            "context_list": reranked_results,
            "docs_count": len(reranked_results)
        }

def apply_rerank_to_search_results(query: str, search_results: Dict[str, Any], top_k: Optional[int] = None, score_threshold: float = 0.3) -> Dict[str, Any]:
    """
    对搜索结果应用rerank
    
    Args:
        query: 查询语句
        search_results: 搜索结果字典，应包含"context_list"字段
        top_k: 返回前k个结果
        
    Returns:
        应用rerank后的结果
    """
    try:
        # 检查输入是否有效
        if not isinstance(search_results, dict) or "context_list" not in search_results:
            logging.warning("无效的搜索结果格式，跳过rerank")
            return search_results
        
        # 创建Reranker实例
        reranker = Reranker()
        
        # 检查是否启用rerank
        if not reranker.is_enabled():
            logging.info("Rerank功能未启用，返回原始结果")
            return search_results
        
        # 获取原始结果
        context_list = search_results.get("context_list", [])
        
        # 过滤掉低于分数阈值的结果
        filtered_context_list = [item for item in context_list if item.get("score", 0) >= score_threshold]
        
        # 如果过滤后没有结果，返回空结果
        if not filtered_context_list:
            return {
                "success": True,
                "context": "",
                "context_list": [],
                "docs_count": 0
            }
        
        # 应用rerank
        reranked_results = reranker.rerank_with_context(query, filtered_context_list, top_k)
        
        return reranked_results
        
    except Exception as e:
        logging.error(f"应用rerank时发生错误: {str(e)}")
        return search_results
