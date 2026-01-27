import os
import requests
from langchain_core.tools import tool
from typing import Optional

@tool
def convert_pdf_to_markdown(file_path: str) -> str:
    """
    将本地 PDF 文件上传至本地 API 服务，并将其内容转换为 Markdown 格式。
    适用于需要提取 PDF 文本、表格或结构化信息的情况。
    
    Args:
        file_path: 本地 PDF 文件的绝对或相对路径。
    """
    if not os.path.exists(file_path):
        return f"错误：文件 {file_path} 不存在。"

    api_url = "http://localhost:8490/process_file"
    
    try:
        with open(file_path, 'rb') as file:
            # 准备 multipart/form-data 请求
            files = {'file': (os.path.basename(file_path), file)}
            
            # 发送 POST 请求
            response = requests.post(api_url, files=files)
            response.raise_for_status()  # 检查 HTTP 错误
            
            # 解析响应
            result = response.json()
            content = result.get('markdown')
            
            if not content:
                return "错误：API 未能返回有效的 markdown 内容。"
                
            return content

    except requests.exceptions.RequestException as e:
        return f"连接转换服务失败: {str(e)}"
    except Exception as e:
        return f"处理文件时发生未知错误: {str(e)}"

file_tools=[convert_pdf_to_markdown]