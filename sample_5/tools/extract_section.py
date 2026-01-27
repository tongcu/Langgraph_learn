# 文件名建议：tools/extract_section.py

from langchain_core.tools import tool
from typing import Optional
import re
from pathlib import Path


@tool
def extract_section_from_md_file(
    file_path: str,
    section_title: str,
    file_encoding: str = "utf-8",
    max_lines: Optional[int] = None,
    include_title: bool = True
) -> str:
    """
    从单个markdown文本文件（通常是 Markdown）中提取指定标题对应的内容段落。

    参数说明：
    - file_path: 文件的路径（绝对路径或相对于当前工作目录的相对路径）
    - section_title: 要查找的标题名称，例如 "nn"、"安装"、"使用方法"（不需要带 #）
    - file_encoding: 文件编码，默认 utf-8
    - max_lines: 最多提取多少行（防止提取过长内容），为空则提取全部
    - include_title: 返回结果是否包含标题本身，默认 True

    返回：
    - 找到则返回该标题下的内容（字符串）
    - 未找到或出错则返回带说明的错误信息字符串

    支持的标题格式（优先级从高到低）：
    1. Markdown 标题：# Title、## Title、### Title 等
    2. 带等号/破折号的下划线风格：Title\n========== 或 Title\n----------
    3. 纯文本标题后跟空行的情况
    """
    path = Path(file_path)
    if not path.is_file():
        return f"文件不存在或不是文件：{file_path}"

    try:
        content = path.read_text(encoding=file_encoding)
    except Exception as e:
        return f"读取文件失败：{str(e)}（尝试使用 encoding={file_encoding}）"

    # --------------------- 核心提取逻辑 ---------------------

    # 规范化标题（去除首尾空白，支持忽略大小写匹配时可改）
    target = section_title.strip()

    lines = content.splitlines()
    found_start = -1
    found_level = 0

    # 匹配 Markdown # 风格标题
    md_pattern = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$", re.IGNORECASE)

    for i, line in enumerate(lines):
        line = line.rstrip()

        # 匹配 # 风格
        md_match = md_pattern.match(line)
        if md_match:
            level, title = md_match.groups()
            title_clean = title.strip()

            if title_clean == target or title_clean.lower() == target.lower():
                found_start = i
                found_level = len(level)
                break

        # 匹配下划线风格标题（=== 或 ---）
        if i + 1 < len(lines) and line.strip():
            next_line = lines[i + 1].rstrip()
            if (next_line.startswith("==") or next_line.startswith("--")) and len(next_line.strip("=-")) >= 3:
                title_clean = line.strip()
                if title_clean == target or title_clean.lower() == target.lower():
                    found_start = i
                    found_level = 1  # 当作一级标题处理
                    break

    if found_start == -1:
        # 尝试更宽松的全文搜索（当标题没用标准格式时）
        for i, line in enumerate(lines):
            if target in line or target.lower() in line.lower():
                # 找到包含关键词的行，尝试从这里开始取内容
                found_start = i
                break

        if found_start == -1:
            return f"文件 {file_path} 中未找到标题或包含 '{section_title}' 的内容。"

    # --------------------- 收集内容 ---------------------

    result_lines = []
    if include_title and found_start < len(lines):
        result_lines.append(lines[found_start])

    current_level = found_level

    for j in range(found_start + 1, len(lines)):
        line = lines[j].rstrip()

        # 遇到同级或更高级标题 → 停止
        md_match = md_pattern.match(line)
        if md_match:
            new_level = len(md_match.group(1))
            if new_level <= current_level:
                break

        # 下划线风格也作为停止条件
        if j + 1 < len(lines):
            next_l = lines[j + 1].rstrip()
            if (next_l.startswith("==") or next_l.startswith("--")) and len(next_l.strip("=-")) >= 3:
                break

        result_lines.append(line)

        if max_lines is not None and len(result_lines) >= max_lines:
            result_lines.append("...（内容已截断，超过最大行数限制）")
            break

    extracted = "\n".join(result_lines).strip()

    if not extracted:
        return f"找到标题 '{section_title}' 但其下没有内容。"

    return extracted


# ------------------- 测试用例（可删除） -------------------
if __name__ == "__main__":
    # 示例调用
    print(extract_section_from_file.invoke({
        "file_path": "README.md",
        "section_title": "nn",
        "max_lines": 20
    }))