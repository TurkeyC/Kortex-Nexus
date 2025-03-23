import os
import glob
import torch
import json
import time  # Add missing time module import
from typing import List, Dict, Any, Optional, Tuple
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.document_processors import MarkdownProcessor

# 更新导入路径，避免弃用警告
try:
    # 尝试导入新的包路径
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # 如果新包未安装，则使用旧路径但显示警告
    import warnings
    warnings.warn(
        "请安装langchain-huggingface包以获取更新的HuggingFaceEmbeddings: "
        "pip install langchain-huggingface",
        DeprecationWarning
    )
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
import config


class KnowledgeBase:
    """知识库管理"""

    def __init__(self, embedding_model_name=config.EMBEDDING_MODEL):
        self.knowledge_dir = config.KNOWLEDGE_BASE_DIR
        self.embeddings_dir = config.EMBEDDINGS_DIR
        self.embedding_model_name = embedding_model_name

        # 使用GPU进行嵌入计算，但使用CPU版本FAISS进行检索
        device = "cuda" if torch.cuda.is_available() and config.ENABLE_GPU else "cpu"
        print(f"使用设备: {device} 进行向量嵌入")

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": device}
            )
            
            # 立即测试并存储向量维度
            test_text = "维度测试"
            test_vector = self.embeddings.embed_query(test_text)
            self.vector_dimension = len(test_vector)
            print(f"当前嵌入模型维度: {self.vector_dimension}")
            
        except Exception as e:
            print(f"警告: 加载嵌入模型失败，将使用默认模型: {str(e)}")
            # 尝试使用更小的模型作为后备
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={"device": device}
                )
                self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                
                # 获取并存储向量维度
                test_text = "维度测试"
                test_vector = self.embeddings.embed_query(test_text)
                self.vector_dimension = len(test_vector)
                print(f"后备嵌入模型维度: {self.vector_dimension}")
            except:
                print("无法加载嵌入模型，将使用CPU模式")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                
                # 获取并存储向量维度
                test_text = "维度测试"
                test_vector = self.embeddings.embed_query(test_text)
                self.vector_dimension = len(test_vector)
                print(f"CPU后备嵌入模型维度: {self.vector_dimension}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )

        # 添加Markdown处理器
        self.markdown_processor = MarkdownProcessor(
            chunk_size=800,
            chunk_overlap=100
        )

        # 文档元数据存储
        self.document_metadata = {}
        
        # 初始化vector_store为None，避免未定义错误
        self.vector_store = None
        
        # 存储索引维度
        self.index_dimension = None

    def _process_markdown_file(self, file_path: str) -> List[Dict[str, Any]]:
        """使用增强的Markdown处理器处理文档"""
        chunks, metadata = self.markdown_processor.process_document(file_path)

        # 存储文档元数据
        rel_path = os.path.relpath(file_path, self.knowledge_dir)
        self.document_metadata[rel_path] = metadata

        # 为每个块添加文件路径信息
        for chunk in chunks:
            chunk["metadata"]["source"] = rel_path

        return chunks

    def load_knowledge_base(self):
        """加载整个知识库"""
        all_chunks = []
        markdown_files = glob.glob(os.path.join(self.knowledge_dir, "**/*.md"), recursive=True)

        if not markdown_files:
            print(f"警告: 知识库目录 {self.knowledge_dir} 中未找到Markdown文件")
            return

        for file_path in markdown_files:
            chunks = self._process_markdown_file(file_path)
            all_chunks.extend(chunks)

        # 创建向量存储
        texts = [chunk["content"] for chunk in all_chunks]
        metadatas = [chunk["metadata"] for chunk in all_chunks]

        if texts:  # 确保有文档
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # 获取和存储向量维度
            if self.vector_store and hasattr(self.vector_store, 'index'):
                self.vector_dimension = self.vector_store.index.d
            
            # 保存向量存储
            vector_index_path = os.path.join(self.embeddings_dir, "faiss_index")
            self.vector_store.save_local(vector_index_path)

            # 保存索引元数据
            self.save_index_metadata()

            # 保存文档元数据
            metadata_path = os.path.join(self.embeddings_dir, "document_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_metadata, f, ensure_ascii=False, indent=2)

            print(f"向量索引已保存至 {vector_index_path}")
            print(f"文档元数据已保存至 {metadata_path}")

    def load_existing_vectorstore(self):
        """加载已存在的向量存储"""
        vector_index_path = os.path.join(self.embeddings_dir, "faiss_index")
        metadata_path = os.path.join(self.embeddings_dir, "index_metadata.json")
        
        if os.path.exists(vector_index_path):
            try:
                # 检查索引元数据，如果存在
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        index_metadata = json.load(f)
                        stored_model = index_metadata.get("model_name", "")
                        self.index_dimension = index_metadata.get("dimensions")
                        
                        # 如果模型不同或维度不同，打印更详细的警告
                        if stored_model and stored_model != self.embedding_model_name:
                            print(f"警告: 索引使用模型({stored_model})与当前模型({self.embedding_model_name})不同，可能导致维度不匹配")
                        
                        # 检查维度是否匹配
                        if self.index_dimension and self.vector_dimension and self.index_dimension != self.vector_dimension:
                            print(f"警告: 索引维度({self.index_dimension})与当前模型维度({self.vector_dimension})不匹配")
                            print("建议: 请重建索引以解决维度不匹配问题")
                
                self.vector_store = FAISS.load_local(
                    vector_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True  # 添加此参数以允许反序列化
                )
                print(f"成功加载向量索引: {vector_index_path}")
                
                # 获取索引维度
                if self.vector_store and hasattr(self.vector_store, 'index') and hasattr(self.vector_store.index, 'd'):
                    self.index_dimension = self.vector_store.index.d
                    print(f"索引维度: {self.index_dimension}")
                
                # 测试维度兼容性
                if not self._check_dimension_compatibility():
                    print("警告: 维度不匹配。建议重建索引。")
                    return False
                
                return True
            except Exception as e:
                print(f"加载向量索引出错: {str(e)}")
                return False
        print(f"向量索引不存在: {vector_index_path}")
        return False
        
    def _check_dimension_compatibility(self):
        """检查嵌入模型与索引的维度兼容性"""
        if not self.vector_store:
            return False
            
        try:
            # 创建一个测试查询向量
            test_query = "测试查询"
            query_vector = self.embeddings.embed_query(test_query)
            
            # 获取向量维度
            query_dim = len(query_vector)
            
            # 如果索引尚未初始化，这里可能会失败
            index_dim = self.vector_store.index.d
            
            # 存储维度信息
            self.vector_dimension = query_dim
            self.index_dimension = index_dim
            
            # 检查维度是否匹配
            if query_dim != index_dim:
                print(f"警告: 查询向量维度({query_dim})与索引维度({index_dim})不匹配")
                return False
                
            return True
        except Exception as e:
            print(f"检查维度兼容性时出错: {str(e)}")
            return False
    
    def clear_index(self):
        """清除索引并释放资源"""
        if self.vector_store:
            try:
                # 释放资源
                del self.vector_store
                self.vector_store = None
                
                # 标记为需要重建
                return True
            except Exception as e:
                print(f"清除索引出错: {str(e)}")
                return False
        return True
    
    def save_index_metadata(self):
        """保存索引元数据"""
        if not self.vector_store:
            return
            
        try:
            index_metadata = {
                "model_name": self.embedding_model_name,
                "dimensions": self.vector_dimension or 
                              (self.vector_store.index.d if self.vector_store and hasattr(self.vector_store, 'index') else None),
                "created_at": str(time.time())
            }
            
            metadata_path = os.path.join(self.embeddings_dir, "index_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(index_metadata, f, ensure_ascii=False, indent=2)
                
            print(f"索引元数据已保存至 {metadata_path}")
        except Exception as e:
            print(f"保存索引元数据失败: {str(e)}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        if not self.vector_store:
            if not self.load_existing_vectorstore():
                return [{"content": "知识库尚未初始化，请先上传文档。", "metadata": {}}]
        
        # 检查维度兼容性
        if not self._check_dimension_compatibility():
            return [{"content": f"向量维度不匹配。当前模型维度({self.vector_dimension})与索引维度({self.index_dimension})不符。请重建索引。", "metadata": {}}]
        
        try:
            # 安全搜索
            return self._safe_search(query, top_k)
        except Exception as e:
            error_msg = str(e)
            print(f"搜索出错: {error_msg}")
            
            # 如果是维度不匹配错误，给出更明确的提示
            if "assert d == self.d" in error_msg:
                print(f"维度不匹配错误: 查询维度({self.vector_dimension}) != 索引维度({self.index_dimension})")
                return [{"content": f"检测到向量维度不匹配。当前模型维度: {self.vector_dimension}, 索引维度: {self.index_dimension}。请在侧边栏点击'重新加载知识库'重建索引。", "metadata": {}}]
            
            return [{"content": f"搜索时发生错误: {error_msg}", "metadata": {}}]
    
    def _safe_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """安全的搜索方法，处理可能的错误"""
        try:
            # 首先尝试正常搜索
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return formatted_results
        except AssertionError as e:
            # 捕获维度不匹配错误
            error_str = str(e)
            if "assert d == self.d" in error_str:
                index_dim = self.vector_store.index.d if self.vector_store and hasattr(self.vector_store, 'index') else 'unknown'
                query_dim = len(self.embeddings.embed_query(query))
                self.vector_dimension = query_dim
                self.index_dimension = index_dim
                raise ValueError(f"向量维度不匹配。索引维度: {index_dim}, 查询维度: {query_dim}")
            else:
                raise e
        except Exception as e:
            # 其他错误
            raise ValueError(f"搜索时出错: {str(e)}")

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


