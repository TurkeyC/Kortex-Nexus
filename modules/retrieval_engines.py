import re
from typing import List, Dict, Any, Optional
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

class KeywordRetriever:
    """基于关键词的检索引擎"""
    
    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.use_chinese = True
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档到检索器"""
        self.documents = documents
        
        # 预处理文本
        processed_docs = []
        for doc in documents:
            text = doc["content"]
            if self.use_chinese:
                # 使用结巴分词
                words = " ".join(jieba.cut(text))
                processed_docs.append(words)
            else:
                # 简单分词
                words = re.findall(r'\w+', text.lower())
                processed_docs.append(" ".join(words))
        
        # 创建BM25检索器
        tokenized_corpus = [doc.split(" ") for doc in processed_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """根据查询检索文档"""
        if not self.bm25 or not self.documents:
            return []
        
        # 处理查询
        if self.use_chinese:
            tokenized_query = " ".join(jieba.cut(query)).split(" ")
        else:
            tokenized_query = re.findall(r'\w+', query.lower())
        
        # 获取分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取前K个结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0:  # 只返回相关文档
                results.append({
                    "content": self.documents[i]["content"],
                    "metadata": self.documents[i]["metadata"],
                    "score": float(scores[i])
                })
        
        return results

class TreeStructureRetriever:
    """基于树状结构的检索引擎"""
    
    def __init__(self, knowledge_base):
        """初始化树状结构检索器"""
        self.knowledge_base = knowledge_base
        self.document_tree = None
    
    def load_tree_structure(self):
        """加载文档树状结构"""
        self.document_tree = self.knowledge_base.get_document_tree()
        return self.document_tree
    
    def retrieve(self, query: str, selected_nodes: List[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """基于树状结构进行检索"""
        # 如果还没有加载树结构，先加载
        if not self.document_tree:
            self.load_tree_structure()
            
        # 使用知识库的树状结构检索
        results = self.knowledge_base.tree_structure_search(
            query=query,
            selected_nodes=selected_nodes,
            top_k=top_k
        )
        
        return results
    
    def get_tree(self) -> Dict[str, Any]:
        """获取文档树状结构"""
        if not self.document_tree:
            self.load_tree_structure()
        return self.document_tree
    
    def get_node_path(self, node_path: str) -> Dict[str, Any]:
        """获取指定路径的节点"""
        if not self.document_tree:
            self.load_tree_structure()
            
        # 分割路径
        if not node_path:
            return self.document_tree
            
        parts = node_path.split('/')
        
        # 从根节点开始查找
        node = self.document_tree
        
        for part in parts:
            if not part:
                continue
                
            found = False
            for child in node.get("children", []):
                if child["name"] == part:
                    node = child
                    found = True
                    break
                    
            if not found:
                return None
                
        return node

class HybridRetriever:
    """混合检索引擎，结合向量检索、关键词检索和树状结构检索"""
    
    def __init__(self, vector_retriever, keyword_retriever=None, tree_retriever=None, weights: Dict[str, float] = None):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.tree_retriever = tree_retriever
        self.weights = weights or {"vector": 0.7, "keyword": 0.2, "tree": 0.1}
        # 添加归一化方法配置
        self.normalization_method = "minmax"  # 支持 "minmax", "zscore", "rank"
        # 添加多样性参数
        self.diversity_weight = 0.1  # 多样性权重，值越大结果越多样
        # 内容相似度阈值
        self.similarity_threshold = 0.85  # 高于此阈值的内容被视为相似
        # 历史反馈记录
        self.feedback_history = {}
        
    def normalize_scores(self, results: List[Dict[str, Any]], method: str = None) -> List[Dict[str, Any]]:
        """归一化检索分数，支持多种归一化方法"""
        if not results:
            return results
            
        method = method or self.normalization_method
        scores = [r["score"] for r in results]
        
        if method == "minmax":
            # MinMax归一化
            min_score = min(scores)
            max_score = max(scores)
            
            # 防止除以零
            if max_score == min_score:
                for r in results:
                    r["score"] = 1.0
                return results
            
            # 归一化到0-1区间
            for r in results:
                r["score"] = (r["score"] - min_score) / (max_score - min_score)
                
        elif method == "zscore":
            # Z-score归一化
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            if std_score == 0:
                for r in results:
                    r["score"] = 1.0
                return results
                
            for r in results:
                # 将Z分数转换为0-1区间
                z_score = (r["score"] - mean_score) / std_score
                # 使用sigmoid函数将z-score映射到0-1
                r["score"] = 1 / (1 + np.exp(-z_score))
                
        elif method == "rank":
            # 基于排名的归一化
            sorted_indices = np.argsort([-s for s in scores])  # 降序排序
            for i, idx in enumerate(sorted_indices):
                # 排名归一化: 1.0 到 0.0 (线性衰减)
                results[idx]["score"] = 1.0 - (i / len(results)) if len(results) > 1 else 1.0
        
        return results
    
    def dynamic_weight_adjustment(self, query: str) -> Dict[str, float]:
        """根据查询动态调整检索权重"""
        # 基础权重
        weights = self.weights.copy()
        
        # 关键词检测逻辑
        query_lower = query.lower()
        
        # 如果查询很短，可能是关键词搜索，增加关键词检索权重
        if len(query.split()) <= 3:
            weights["keyword"] = min(weights.get("keyword", 0.2) * 1.5, 0.5)
            weights["vector"] = max(weights.get("vector", 0.7) - 0.1, 0.4)
            
        # 如果查询很长，更可能是语义搜索，增加向量检索权重
        elif len(query.split()) >= 8:
            weights["vector"] = min(weights.get("vector", 0.7) * 1.2, 0.8)
            weights["keyword"] = max(weights.get("keyword", 0.2) - 0.05, 0.1)
            
        # 如果查询包含路径相关词语，增加树状结构权重
        path_related_terms = ["目录", "章节", "部分", "分类", "类别", "标题"]
        if any(term in query_lower for term in path_related_terms):
            weights["tree"] = min(weights.get("tree", 0.1) * 2.0, 0.3)
            
        # 重新归一化权重总和为1
        weight_sum = sum(weights.values())
        if weight_sum != 0:
            for k in weights:
                weights[k] = weights[k] / weight_sum
                
        return weights
    
    def filter_similar_content(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤相似内容，提高结果多样性"""
        if not results or len(results) <= 1:
            return results
            
        filtered_results = []
        content_signatures = set()
        
        for result in results:
            # 创建内容签名 (简化版，实际可使用更复杂的指纹算法)
            content = result["content"]
            # 取内容前100个字符作为简单指纹
            content_preview = content[:100].strip().lower() if content else ""
            
            # 检查是否与已有结果过于相似
            is_similar = False
            for sig in content_signatures:
                # 使用简单的重叠率判断相似度
                overlap_ratio = self._calculate_overlap(content_preview, sig)
                if overlap_ratio > self.similarity_threshold:
                    is_similar = True
                    break
                    
            # 如果不相似，添加到结果中
            if not is_similar:
                content_signatures.add(content_preview)
                filtered_results.append(result)
                
        return filtered_results
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """计算两个文本的重叠率"""
        if not text1 or not text2:
            return 0.0
            
        # 将文本分割成词，用于计算重叠
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # 交集除以并集
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def apply_diversity_boost(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """增加结果多样性的加权"""
        if not results or len(results) <= 1:
            return results
            
        # 收集所有文档来源
        sources = {}
        for i, result in enumerate(results):
            source = result["metadata"].get("source", "")
            if source not in sources:
                sources[source] = []
            sources[source].append(i)
            
        # 对每个结果应用多样性加权
        for i, result in enumerate(results):
            source = result["metadata"].get("source", "")
            # 同源文档越多，得分越低
            source_count = len(sources.get(source, []))
            diversity_penalty = self.diversity_weight * (source_count / len(results))
            # 应用多样性调整，保持分数在有效范围内
            results[i]["score"] = max(0, min(1, result["score"] * (1 - diversity_penalty)))
            
        return results
    
    def retrieve(self, query: str, top_k: int = 3, selected_nodes: List[str] = None) -> List[Dict[str, Any]]:
        """混合检索"""
        # 动态调整权重
        adjusted_weights = self.dynamic_weight_adjustment(query)
        
        # 增加检索数量，以便后续过滤
        expanded_top_k = min(top_k * 3, 15)  # 至少检索3倍结果，但不超过15个
        
        # 获取向量检索结果
        vector_results = self.vector_retriever.search(query, top_k=expanded_top_k)
        vector_results = self.normalize_scores(vector_results)
        
        # 初始化其他检索结果
        keyword_results = []
        tree_results = []
        
        # 获取关键词检索结果
        if self.keyword_retriever:
            keyword_results = self.keyword_retriever.retrieve(query, top_k=expanded_top_k)
            keyword_results = self.normalize_scores(keyword_results)
        
        # 获取树状结构检索结果
        if self.tree_retriever and selected_nodes:
            tree_results = self.tree_retriever.retrieve(query, selected_nodes=selected_nodes, top_k=expanded_top_k)
            tree_results = self.normalize_scores(tree_results)
        
        # 创建结果字典，用于合并
        all_results = {}
        
        # 添加向量检索结果
        for item in vector_results:
            doc_id = f"{item['metadata']['source']}_{item['metadata'].get('chunk_id', 0)}"
            all_results[doc_id] = {
                "content": item["content"],
                "metadata": item["metadata"],
                "vector_score": item["score"],
                "keyword_score": 0.0,
                "tree_score": 0.0
            }
        
        # 添加关键词检索结果
        for item in keyword_results:
            doc_id = f"{item['metadata']['source']}_{item['metadata'].get('chunk_id', 0)}"
            if doc_id in all_results:
                all_results[doc_id]["keyword_score"] = item["score"]
            else:
                all_results[doc_id] = {
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "vector_score": 0.0,
                    "keyword_score": item["score"],
                    "tree_score": 0.0
                }
        
        # 添加树状结构检索结果
        for item in tree_results:
            doc_id = f"{item['metadata']['source']}_{item['metadata'].get('chunk_id', 0)}"
            if doc_id in all_results:
                all_results[doc_id]["tree_score"] = item["score"]
            else:
                all_results[doc_id] = {
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "vector_score": 0.0,
                    "keyword_score": 0.0,
                    "tree_score": item["score"]
                }
        
        # 计算混合分数
        results = []
        for doc_id, item in all_results.items():
            score = (adjusted_weights.get("vector", 0.7) * item["vector_score"] + 
                     adjusted_weights.get("keyword", 0.2) * item["keyword_score"] +
                     adjusted_weights.get("tree", 0.1) * item["tree_score"])
                     
            results.append({
                "content": item["content"],
                "metadata": item["metadata"],
                "score": score,
                "vector_score": item["vector_score"],
                "keyword_score": item["keyword_score"],
                "tree_score": item["tree_score"],
                "weights_used": adjusted_weights.copy()  # 记录使用的权重
            })
        
        # 按综合分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 应用多样性boost
        results = self.apply_diversity_boost(results)
        
        # 重新排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 过滤相似内容
        results = self.filter_similar_content(results)
        
        # 返回前k个结果
        return results[:top_k]
    
    def record_feedback(self, query: str, doc_id: str, is_relevant: bool):
        """记录用户反馈，用于未来优化检索权重"""
        if query not in self.feedback_history:
            self.feedback_history[query] = {}
        
        self.feedback_history[query][doc_id] = is_relevant
        
        # 注意：这里只记录反馈，未来可以基于反馈历史优化检索策略

