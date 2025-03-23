import os
import shutil
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

class KnowledgeBaseManager:
    """知识库文件管理器"""
    
    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = knowledge_dir
        self.versions_dir = knowledge_dir / "_versions"
        self.versions_dir.mkdir(exist_ok=True, parents=True)
        self.version_info_file = self.versions_dir / "version_info.json"
        self._load_version_info()
        
    def _load_version_info(self):
        """加载版本信息"""
        if self.version_info_file.exists():
            with open(self.version_info_file, 'r', encoding='utf-8') as f:
                self.version_info = json.load(f)
        else:
            self.version_info = {"versions": [], "current_version": 0}
            self._save_version_info()
            
    def _save_version_info(self):
        """保存版本信息"""
        with open(self.version_info_file, 'w', encoding='utf-8') as f:
            json.dump(self.version_info, f, ensure_ascii=False, indent=2)
            
    def save_file(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """保存文件并创建新版本"""
        file_path = self.knowledge_dir / file_name
        
        # 检查文件是否已存在
        is_update = file_path.exists()
        
        # 如果是更新，先备份当前版本
        if is_update:
            self._backup_file(file_name)
        
        # 保存新文件
        with open(file_path, 'wb') as f:
            f.write(file_content)
            
        # 创建新版本信息
        new_version = {
            "id": int(time.time()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "changes": [{"file": file_name, "action": "update" if is_update else "create"}]
        }
        
        # 更新版本信息
        self.version_info["versions"].append(new_version)
        self.version_info["current_version"] = new_version["id"]
        self._save_version_info()
        
        return {
            "file_name": file_name,
            "status": "updated" if is_update else "created",
            "version": new_version["id"]
        }
        
    def _backup_file(self, file_name: str) -> None:
        """备份文件到版本目录"""
        source_file = self.knowledge_dir / file_name
        if not source_file.exists():
            return
            
        # 创建版本子目录
        backup_dir = self.versions_dir / str(int(time.time()))
        backup_dir.mkdir(exist_ok=True)
        
        # 复制文件到版本目录
        shutil.copy2(source_file, backup_dir / file_name)
        
    def delete_file(self, file_name: str) -> Dict[str, Any]:
        """删除文件"""
        file_path = self.knowledge_dir / file_name
        
        if not file_path.exists():
            return {"status": "error", "message": f"文件 {file_name} 不存在"}
            
        # 备份文件
        self._backup_file(file_name)
        
        # 删除文件
        os.remove(file_path)
        
        # 创建新版本信息
        new_version = {
            "id": int(time.time()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "changes": [{"file": file_name, "action": "delete"}]
        }
        
        # 更新版本信息
        self.version_info["versions"].append(new_version)
        self.version_info["current_version"] = new_version["id"]
        self._save_version_info()
        
        return {
            "file_name": file_name,
            "status": "deleted",
            "version": new_version["id"]
        }
        
    def get_file_list(self) -> List[Dict[str, Any]]:
        """获取知识库中的所有文件"""
        files = []
        
        for file_path in self.knowledge_dir.glob("**/*.md"):
            if self.versions_dir in file_path.parents:
                continue  # 跳过版本目录中的文件
                
            rel_path = file_path.relative_to(self.knowledge_dir)
            files.append({
                "name": str(rel_path),
                "path": str(rel_path),
                "size": file_path.stat().st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", 
                                         time.localtime(file_path.stat().st_mtime))
            })
            
        return files
        
    def get_file_content(self, file_path: str) -> Optional[str]:
        """获取文件内容"""
        full_path = self.knowledge_dir / file_path
        
        if not full_path.exists() or not full_path.is_file():
            return None
            
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def get_version_history(self) -> List[Dict[str, Any]]:
        """获取版本历史"""
        return self.version_info["versions"]