from typing import List, Dict, Any, Optional
import time
import json

class ConversationNode:
    """对话树节点"""
    
    def __init__(self, message: Dict[str, str], parent=None):
        self.message = message
        self.parent = parent
        self.children = []
        self.id = str(int(time.time() * 1000))
        self.timestamp = time.time()
        
    def add_child(self, message: Dict[str, str]):
        """添加子节点"""
        child = ConversationNode(message, self)
        self.children.append(child)
        return child
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "id": self.id,
            "message": self.message,
            "timestamp": self.timestamp,
            "children": [child.to_dict() for child in self.children]
        }
        
class ConversationTree:
    """对话树结构"""
    
    def __init__(self):
        self.root = None  # 根节点
        self.current_node = None  # 当前节点
        self.node_map = {}  # 节点ID到节点的映射
        
    def start_conversation(self, system_message: str):
        """开始新对话"""
        self.root = ConversationNode({"role": "system", "content": system_message})
        self.current_node = self.root
        self.node_map[self.root.id] = self.root
        
    def add_message(self, message: Dict[str, str]) -> str:
        """添加消息并返回节点ID"""
        if not self.root:
            if message["role"] == "system":
                self.start_conversation(message["content"])
                return self.root.id
            else:
                # 如果没有根节点且不是系统消息，创建默认系统消息
                self.start_conversation("你好，我是一个知识库助手。")
                
        # 添加消息作为当前节点的子节点
        node = self.current_node.add_child(message)
        self.current_node = node
        self.node_map[node.id] = node
        return node.id
        
    def get_conversation_path(self) -> List[Dict[str, str]]:
        """获取从根节点到当前节点的对话路径"""
        if not self.current_node:
            return []
            
        path = []
        node = self.current_node
        
        # 向上遍历到根节点
        while node:
            path.append(node.message)
            node = node.parent
            
        # 反转以获得正确顺序
        path.reverse()
        return path
        
    def switch_branch(self, node_id: str) -> bool:
        """切换到指定节点所在的分支"""
        if node_id in self.node_map:
            self.current_node = self.node_map[node_id]
            return True
        return False
        
    def get_tree_structure(self) -> Dict[str, Any]:
        """获取整个对话树结构"""
        if not self.root:
            return {"root": None}
        return {"root": self.root.to_dict()}
        
    def to_json(self) -> str:
        """序列化为JSON"""
        return json.dumps(self.get_tree_structure(), ensure_ascii=False)
        
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationTree':
        """从JSON反序列化"""
        tree = cls()
        
        data = json.loads(json_str)
        if not data["root"]:
            return tree
            
        def build_tree(node_data, parent=None):
            message = node_data["message"]
            if parent:
                node = parent.add_child(message)
            else:
                node = ConversationNode(message)
                tree.root = node
                
            node.id = node_data["id"]
            node.timestamp = node_data["timestamp"]
            tree.node_map[node.id] = node
            
            for child_data in node_data["children"]:
                build_tree(child_data, node)
                
            return node
            
        build_tree(data["root"])
        
        # 设置当前节点为根节点
        tree.current_node = tree.root
        
        return tree
        
class ContextManager:
    """上下文管理器，负责压缩和管理上下文窗口"""
    
    def __init__(self, max_tokens: int = 4000, llm=None):
        self.max_tokens = max_tokens
        self.llm = llm
        
    def compress_history(self, history: List[Dict[str, str]], 
                        keep_last: int = 2) -> List[Dict[str, str]]:
        """压缩历史对话，保留最近的keep_last轮对话"""
        if len(history) <= keep_last + 1:  # +1 for system message
            return history
            
        # 保留系统消息
        compressed = [history[0]]
        
        # 如果有LLM可用，使用它来总结中间部分
        if self.llm and len(history) > keep_last + 3:  # 至少有一轮对话可以总结
            middle_messages = history[1:-keep_last*2]
            
            if middle_messages:
                # 转换为文本
                conversation_text = "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in middle_messages
                ])
                
                # 创建总结提示
                summary_prompt = f"""
                请对以下对话进行简洁总结，保留关键信息：
                
                {conversation_text}
                
                总结：
                """
                
                # 获取总结
                summary = self.llm.generate_response(summary_prompt)
                
                # 添加总结为系统消息
                compressed.append({
                    "role": "system",
                    "content": f"前面的对话总结: {summary}"
                })
        
        # 添加最近的对话
        compressed.extend(history[-keep_last*2:])
        
        return compressed