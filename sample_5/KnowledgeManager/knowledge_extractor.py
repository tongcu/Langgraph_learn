import logging
from pathlib import Path
from typing import List, Dict, Optional
import docx
import pypdf
import re
from bs4 import BeautifulSoup

class KnowledgeExtractor:
    """文档知识提取器 (迁移自 report-26v0)"""
    
    def __init__(self):
        self.supported_formats = {
            '.txt': self._extract_txt,
            '.md': self._extract_markdown,
            '.pdf': self._extract_pdf,
            '.docx': self._extract_docx,
            '.doc': self._extract_docx,
            '.html': self._extract_html,
            '.htm': self._extract_html
        }
    
    def extract_from_folder(self, folder_path: str) -> List[Dict[str, str]]:
        documents = []
        folder_path = Path(folder_path)
        if not folder_path.exists():
            return documents
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                doc = self.extract_from_file(str(file_path))
                if doc: documents.append(doc)
        return documents
    
    def extract_from_file(self, file_path: str) -> Optional[Dict[str, str]]:
        file_path = Path(file_path)
        if not file_path.exists(): return None
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats: return None
        try:
            content = self.supported_formats[file_ext](file_path)
            content = re.sub(r'\n{2,}', '\n\n', content)
            if content:
                return {
                    "content": content,
                    "source": str(file_path),
                    "filename": file_path.name,
                    "format": file_ext
                }
        except Exception as e:
            logging.error(f"提取文件 {file_path} 时出错: {e}")
        return None
    
    def _extract_txt(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f: return f.read()
    
    def _extract_markdown(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()
    
    def _extract_pdf(self, file_path: Path) -> str:
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            for page in pdf_reader.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    
    def _extract_docx(self, file_path: Path) -> str:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_html(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()

knowledge_extractor = KnowledgeExtractor()
