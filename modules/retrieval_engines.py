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

class HybridRetriever:
    """混合检索引擎，结合向量检索和关键词检索"""
    
    def __init__(self, vector_retriever, keyword_retriever, weights: Dict[str, float] = None):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.weights = weights or {"vector": 0.7, "keyword": 0.3}
        
    def normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """归一化检索分数"""
        if not results:
            return results
            
        scores = [r["score"] for r in results]
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
            
        return results
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """混合检索"""
        # 获取向量检索结果
        vector_results = self.vector_retriever.search(query, top_k=top_k)
        vector_results = self.normalize_scores(vector_results)
        
        # 获取关键词检索结果
        keyword_results = self.keyword_retriever.retrieve(query, top_k=top_k)
        keyword_results = self.normalize_scores(keyword_results)
        
        # 创建结果字典，用于合并
        all_results = {}
        
        # 添加向量检索结果
        for item in vector_results:
            doc_id = f"{item['metadata']['source']}_{item['metadata'].get('chunk_id', 0)}"
            all_results[doc_id] = {
                "content": item["content"],
                "metadata": item["metadata"],
                "vector_score": item["score"],
                "keyword_score": 0.0
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
                    "keyword_score": item["score"]
                }
        
        # 计算混合分数
        results = []
        for doc_id, item in all_results.items():
            score = (self.weights["vector"] * item["vector_score"] + 
                     self.weights["keyword"] * item["keyword_score"])
                     
            results.append({
                "content": item["content"],
                "metadata": item["metadata"],
                "score": score,
                "vector_score": item["vector_score"],
                "keyword_score": item["keyword_score"]
            })
        
        # 按综合分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]