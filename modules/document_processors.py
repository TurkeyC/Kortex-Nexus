import re
import os
from typing import List, Dict, Any, Optional, Tuple
import markdown
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MarkdownProcessor:
    """高级Markdown文档处理器"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ".", " ", ""]
        )
    
    def extract_metadata(self, md_content: str) -> Dict[str, Any]:
        """提取Markdown文档的元数据"""
        metadata = {}
        
        # 提取标题
        title_match = re.search(r'^# (.+)$', md_content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # 提取所有标题
        headers = re.findall(r'^(#+) (.+)$', md_content, re.MULTILINE)
        if headers:
            metadata["headers"] = [
                {"level": len(h[0]), "text": h[1].strip()} 
                for h in headers
            ]
        
        # 提取代码块及其语言
        code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', md_content, re.DOTALL)
        if code_blocks:
            metadata["code_blocks"] = [
                {"language": lang or "text", "code": code.strip()} 
                for lang, code in code_blocks
            ]
        
        # 提取YAML前置元数据 (Front Matter)
        front_matter_match = re.match(r'^---\n(.*?)\n---', md_content, re.DOTALL)
        if front_matter_match:
            front_matter = front_matter_match.group(1)
            # 简单解析YAML
            for line in front_matter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
        
        return metadata
    
    def structure_content(self, md_content: str) -> List[Dict[str, Any]]:
        """将Markdown内容结构化为层次化数据"""
        # 转换为HTML
        html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 提取结构化内容
        structured_content = []
        current_section = {"type": "intro", "content": "", "level": 0}
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'ul', 'ol', 'blockquote']):
            tag_name = element.name
            
            # 处理标题，开始新的章节
            if tag_name.startswith('h'):
                level = int(tag_name[1])
                
                # 保存上一个章节
                if current_section["content"]:
                    structured_content.append(current_section)
                
                # 创建新章节
                current_section = {
                    "type": "section",
                    "level": level,
                    "title": element.get_text(),
                    "content": ""
                }
            # 其他内容都添加到当前章节
            else:
                content_type = tag_name
                content = element.get_text()
                
                if content_type == 'pre':
                    current_section["content"] += f"\n```\n{content}\n```\n"
                else:
                    current_section["content"] += f"\n{content}\n"
        
        # 添加最后一个章节
        if current_section["content"]:
            structured_content.append(current_section)
        
        return structured_content
    
    def semantic_chunking(self, structured_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于语义边界的智能分块"""
        chunks = []
        
        for section in structured_content:
            # 短章节直接作为一个块
            if len(section["content"]) <= self.chunk_size:
                chunks.append({
                    "content": section["content"],
                    "metadata": {
                        "title": section.get("title", ""),
                        "level": section.get("level", 0),
                        "type": section["type"]
                    }
                })
            # 长章节需要进一步分块
            else:
                # 根据段落分隔
                sub_chunks = self.text_splitter.split_text(section["content"])
                
                for i, chunk in enumerate(sub_chunks):
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "title": section.get("title", ""),
                            "level": section.get("level", 0),
                            "type": section["type"],
                            "chunk_id": i
                        }
                    })
        
        return chunks
    
    def process_document(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """处理Markdown文档，返回分块和元数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取文档元数据
        metadata = self.extract_metadata(content)
        metadata["source"] = os.path.basename(file_path)
        metadata["path"] = file_path
        
        # 结构化内容
        structured_content = self.structure_content(content)
        
        # 生成语义分块
        chunks = self.semantic_chunking(structured_content)
        
        return chunks, metadata