"""
混合Markdown文本分割器
结合Markdown标题分割和递归字符分割的优势
"""

from typing import List, Tuple
import logging
from KnowledgeManager.Dependencies.compat import get_langchain_text_splitter

RecursiveCharacterTextSplitter = get_langchain_text_splitter('RecursiveCharacterTextSplitter')
MarkdownHeaderTextSplitter = get_langchain_text_splitter('MarkdownHeaderTextSplitter')


class MarkdownHybridSplitter:
    """混合Markdown文本分割器"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
        min_chunk_size: int = 200,
        headers_to_split_on: List[Tuple[str, str]] = None,
    ):
        """
        初始化混合文本分割器
        
        Args:
            chunk_size: 每个文本块的目标大小
            chunk_overlap: 文本块之间的重叠大小
            min_chunk_size: 最小文本块大小，小于该大小的块将被合并
            headers_to_split_on: 标题分割规则，格式为[("#", "Header 1"), ("##", "Header 2"), ...]
        """
        # 默认标题分割规则
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if min_chunk_size > chunk_size:
            min_chunk_size = chunk_size // 5
        self.min_chunk_size = min_chunk_size
        
        # 初始化分割器，设置strip_headers=False以保留标题在内容中
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""]
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        使用混合策略分割文本
        
        Args:
            text: 待分割的Markdown文本
            
        Returns:
            分割后的文本块列表
        """
        # 检查输入是否为空
        if not text or not text.strip():
            return []
        
        # 检查是否看起来像Markdown格式（包含Markdown标题标记）
        if not self._looks_like_markdown(text):
            # 对于非Markdown文本，直接使用递归字符分割器
            return self.recursive_splitter.split_text(text)
        
        # 检查是否包含连续的标题（没有内容的标题）
        if self._has_consecutive_headers_without_content(text):
            # 对于这种特殊情况，使用递归字符分割器以避免丢失标题
            return self.recursive_splitter.split_text(text)
        
        # 第一步：使用Markdown标题分割器按标题分割
        try:
            markdown_splits = self.markdown_splitter.split_text(text)
        except Exception as e:
            logging.warning(f"Markdown分割器处理失败，回退到递归字符分割器: {e}")
            return self.recursive_splitter.split_text(text)
        
        # 第二步：处理每个标题分割块
        processed_chunks = []
        i = 0
        while i < len(markdown_splits):
            # 获取分割块的内容（标题已经包含在内容中）
            split_doc = markdown_splits[i]
            split_content = split_doc.page_content if hasattr(split_doc, 'page_content') else str(split_doc)
            # 注意：由于strip_headers=False，标题已经包含在split_content中，不需要额外处理
            
            # 特殊处理：如果内容为空但有标题元数据（理论上不应该发生，因为strip_headers=False）
            split_metadata = getattr(split_doc, 'metadata', {})
            if not split_content.strip() and split_metadata:
                # 即使内容为空，也要添加标题块
                if split_content.strip():  # 确保标题块不为空
                    processed_chunks.append(split_content)
                i += 1
                continue
            
            # 如果块太小，需要合并处理
            if len(split_content) < self.min_chunk_size:
                # 检查是否可以向后合并（与下一个块合并）
                if i + 1 < len(markdown_splits):
                    next_doc = markdown_splits[i + 1]
                    next_content = next_doc.page_content if hasattr(next_doc, 'page_content') else str(next_doc)
                    
                    if len(split_content) + len(next_content) <= self.chunk_size:
                        # 向后合并
                        merged_chunk = split_content + "\n\n" + next_content
                        processed_chunks.append(merged_chunk)
                        i += 2  # 跳过已合并的下一个块
                        continue
                # 检查是否可以向前合并（与前一个块合并）
                if processed_chunks and len(split_content) + len(processed_chunks[-1]) <= self.chunk_size:
                    # 向前合并
                    processed_chunks[-1] += "\n\n" + split_content
                    i += 1
                    continue
                else:
                    # 无法合并，单独添加
                    processed_chunks.append(split_content)
                    i += 1
            # 如果块大小合适，直接添加
            elif len(split_content) <= self.chunk_size:
                processed_chunks.append(split_content)
                i += 1
            # 如果块太大，使用递归字符分割器进一步分割
            else:
                recursive_splits = self.recursive_splitter.split_text(split_content)
                processed_chunks.extend(recursive_splits)
                i += 1
        
        # 第三步：后处理，确保没有过小的块（除了最后一个）
        final_chunks = self._merge_small_chunks(processed_chunks)
        return final_chunks
    
    def _has_consecutive_headers_without_content(self, text: str) -> bool:
        """
        检查文本是否包含连续的标题而中间没有足够的内容
        
        Args:
            text: 要检查的文本
            
        Returns:
            如果包含连续标题则返回True，否则返回False
        """
        lines = text.split('\n')
        header_count = 0
        content_lines = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('##') or line.startswith('#'):
                header_count += 1
            elif line:  # 非空行且不是标题
                content_lines += 1
                
            # 如果有多个标题但内容很少，可能是连续标题的情况
            if header_count >= 2 and content_lines < header_count:
                return True
                
        return False
    
    def _looks_like_markdown(self, text: str) -> bool:
        """
        简单检查文本是否看起来像Markdown格式
        
        Args:
            text: 要检查的文本
            
        Returns:
            如果文本看起来像Markdown格式则返回True，否则返回False
        """
        # 检查是否包含常见的Markdown标记
        markdown_indicators = ['# ', '## ', '### ', '**', '*', '`', '```', '>', '[', '](', '![', '![](']
        text_lower = text.lower()
        
        # 如果包含至少两个Markdown指示符，则认为是Markdown格式
        indicator_count = sum(1 for indicator in markdown_indicators if indicator in text_lower)
        return indicator_count >= 2
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        合并过小的文本块
        
        Args:
            chunks: 待处理的文本块列表
            
        Returns:
            处理后的文本块列表
        """
        if len(chunks) <= 1:
            return chunks
            
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # 如果当前块太小，尝试合并
            if len(current_chunk) < self.min_chunk_size:
                # 优先向后合并（与下一个块合并）
                if i + 1 < len(chunks) and len(current_chunk) + len(chunks[i + 1]) <= self.chunk_size:
                    merged_chunk = current_chunk + "\n\n" + chunks[i + 1]
                    merged_chunks.append(merged_chunk)
                    i += 2  # 跳过已合并的下一个块
                    continue
                # 向前合并（与前一个块合并）
                elif merged_chunks and len(current_chunk) + len(merged_chunks[-1]) <= self.chunk_size:
                    merged_chunks[-1] += "\n\n" + current_chunk
                    i += 1
                    continue
                else:
                    # 无法合并，单独添加
                    merged_chunks.append(current_chunk)
                    i += 1
            else:
                # 当前块大小合适，直接添加
                merged_chunks.append(current_chunk)
                i += 1
        
        # 检查最后一个块是否太小，如果是且可以向前合并，则合并
        if len(merged_chunks) > 1 and len(merged_chunks[-1]) < self.min_chunk_size:
            if len(merged_chunks[-1]) + len(merged_chunks[-2]) <= self.chunk_size:
                # 合并最后两个块
                last_chunk = merged_chunks.pop()
                merged_chunks[-1] += "\n\n" + last_chunk
            
        return merged_chunks
