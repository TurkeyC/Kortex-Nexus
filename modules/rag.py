from typing import List, Dict, Any, Optional
import re

class QueryProcessor:
    """查询处理器，负责查询理解与重写"""
    
    def __init__(self, llm):
        """初始化查询处理器"""
        self.llm = llm
        
    def extract_keywords(self, query: str) -> List[str]:
        """提取查询中的关键词"""
        # 使用大模型提取关键词
        prompt = f"""
        请从以下问题中提取关键词，以逗号分隔返回：
        
        问题：{query}
        
        关键词：
        """
        
        response = self.llm.generate_response(prompt)
        # 清理结果
        keywords = [k.strip() for k in response.split(',')]
        return keywords
        
    def rewrite_query(self, query: str) -> str:
        """重写查询以提高检索效果"""
        # 使用大模型重写查询
        prompt = f"""
        请将以下问题重写为更利于搜索的形式，保留所有关键概念和术语，但可以扩展相关概念。直接返回重写后的问题，不要有额外解释：
        
        原始问题：{query}
        """
        
        response = self.llm.generate_response(prompt)
        # 清理结果，取第一段
        rewritten = response.split('\n')[0].strip()
        
        return rewritten

class ContextBuilder:
    """上下文构建器，整合检索结果并生成增强上下文"""
    
    def format_retrieved_context(self, results: List[Dict[str, Any]]) -> str:
        """格式化检索结果为上下文"""
        if not results:
            return "无相关知识库内容。"
        
        context_parts = []
        
        for i, result in enumerate(results):
            source = result["metadata"]["source"]
            title = result["metadata"].get("title", "无标题")
            
            context_parts.append(f"[{i+1}] 来源: {source}")
            if title:
                context_parts.append(f"标题: {title}")
            context_parts.append(f"内容: {result['content']}\n")
        
        return "\n".join(context_parts)
    
    def build_enhanced_prompt(self, query: str, context: str) -> str:
        """构建增强的提示"""
        prompt = f"""
        你是一个基于知识库的问答助手。请根据以下知识库内容回答用户的问题。
        如果知识库内容不足以回答问题，请明确表示，不要编造信息。
        如果引用知识库内容，请注明来源编号，如[1]、[2]等。
        
        知识库内容:
        {context}
        
        用户问题: {query}
        
        请根据知识库内容回答问题：
        """
        
        return prompt

class RAGPipeline:
    """完整的RAG处理流程"""
    
    def __init__(self, llm, retriever):
        """初始化RAG流程"""
        self.llm = llm
        self.retriever = retriever
        self.query_processor = QueryProcessor(llm)
        self.context_builder = ContextBuilder()
        
    def process(self, query: str, chat_history: List[Dict[str, str]] = None, selected_nodes: List[str] = None) -> Dict[str, Any]:
        """处理查询并返回结果"""
        # 查询重写
        rewritten_query = self.query_processor.rewrite_query(query)
        
        # 混合检索，包含树状结构过滤
        results = self.retriever.retrieve(rewritten_query, top_k=5, selected_nodes=selected_nodes)
        
        # 构建上下文
        context = self.context_builder.format_retrieved_context(results)
        
        # 生成增强提示
        prompt = self.context_builder.build_enhanced_prompt(query, context)
        
        # 生成回答
        response = self.llm.generate_response(prompt, chat_history)
        
        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "results": results,
            "context": context,
            "response": response
        }
