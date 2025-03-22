import os
import glob
import torch
from typing import List, Dict, Any, Optional, Tuple
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # 修改这一行，使用langchain_community
import config


class KnowledgeBase:
    """知识库管理"""

    def __init__(self, embedding_model_name=config.EMBEDDING_MODEL):
        self.knowledge_dir = config.KNOWLEDGE_BASE_DIR
        self.embeddings_dir = config.EMBEDDINGS_DIR

        # 使用GPU进行嵌入计算，但使用CPU版本FAISS进行检索
        device = "cuda" if torch.cuda.is_available() and config.ENABLE_GPU else "cpu"
        print(f"使用设备: {device} 进行向量嵌入")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )
        self.vector_store = None

    def _load_markdown_file(self, file_path: str) -> List[Dict[str, Any]]:
        """加载并处理单个Markdown文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 将Markdown转换为纯文本
        text_content = markdown.markdown(content)

        # 提取相对路径作为文档来源标识
        rel_path = os.path.relpath(file_path, self.knowledge_dir)

        # 分割文本
        chunks = self.text_splitter.split_text(text_content)

        # 创建文档对象
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "metadata": {
                    "source": rel_path,
                    "chunk_id": i
                }
            })

        return documents

    def load_knowledge_base(self):
        """加载整个知识库"""
        all_documents = []
        markdown_files = glob.glob(os.path.join(self.knowledge_dir, "**/*.md"), recursive=True)

        if not markdown_files:
            print(f"警告: 知识库目录 {self.knowledge_dir} 中未找到Markdown文件")
            return

        for file_path in markdown_files:
            documents = self._load_markdown_file(file_path)
            all_documents.extend(documents)

        # 创建向量存储
        texts = [doc["content"] for doc in all_documents]
        metadatas = [doc["metadata"] for doc in all_documents]

        if texts:  # 确保有文档
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            # 保存向量存储
            vector_index_path = os.path.join(self.embeddings_dir, "faiss_index")
            self.vector_store.save_local(vector_index_path)
            print(f"向量索引已保存至 {vector_index_path}")

    def load_existing_vectorstore(self):
        """加载已存在的向量存储"""
        vector_index_path = os.path.join(self.embeddings_dir, "faiss_index")
        if os.path.exists(vector_index_path):
            self.vector_store = FAISS.load_local(
                vector_index_path,
                self.embeddings
            )
            return True
        return False

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        if not self.vector_store:
            if not self.load_existing_vectorstore():
                return [{"content": "知识库尚未初始化，请先上传文档。", "metadata": {}}]

        results = self.vector_store.similarity_search_with_score(query, k=top_k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })

        return formatted_results

    # 预留位置: 关键词检索
    def keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """关键词检索 (预开发功能)"""
        # 后续实现关键词提取和模糊匹配
        return []

    # 预留位置: 知识图谱检索
    def knowledge_graph_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """知识图谱检索 (预开发功能)"""
        # 后续实现基于知识图谱的检索
        return []

    # 预留位置: 树状结构知识体系检索
    def tree_structure_search(self, query: str, selected_nodes: List[str] = None) -> List[Dict[str, Any]]:
        """树状结构知识体系检索 (预开发功能)"""
        # 后续实现基于文件目录结构的检索
        return []

    # 预留位置: 混合检索
    def hybrid_search(self, query: str, weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """混合检索 (预开发功能)"""
        # 后续实现多种检索方式的混合
        if weights is None:
            weights = config.RETRIEVAL_CONFIG["weights"]

        # 目前仅返回向量检索结果
        return self.search(query, top_k=config.RETRIEVAL_CONFIG["vector_search_top_k"])