import os
from pathlib import Path

# 基本配置
BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"  # 修改为deta/knowledge_base
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"  # 修改为deta/embeddings

# 模型配置 - 修改默认模型为lmstudio
DEFAULT_MODEL = "lmstudio"  # 从"ollama"改为"lmstudio"

# 启用GPU（如果可用）
ENABLE_GPU = True
EMBEDDING_DEVICE = "cuda" if ENABLE_GPU else "cpu"

# 嵌入模型
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 模型配置
MODEL_CONFIG = {
    "ollama": {
        "model_name": "llama3",
        "base_url": "http://localhost:11434"
    },
    "moonshot": {
        "model_name": "moonshot-v1-8k",
        "api_key": os.environ.get("MOONSHOT_API_KEY", "")
    },
    "lmstudio": {
        "model_name": "本地模型",
        "base_url": "http://localhost:1234/v1"
    },
    "deepseek": {
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", "")
    }
}

# UI配置
UI_CONFIG = {
    "title": "Kortex Nexus AI",
    "description": "🧠 本地智能助手 | Local AI Assistant",
    "theme": "light",
    "advanced_features": {
        "enable_swarms": False,
        "enable_storm": False,
        "enable_consciousness": False,
        "enable_hook": False
    }
}

# 检索配置
RETRIEVAL_CONFIG = {
    "vector_search_top_k": 5,
    "keyword_search_top_k": 3,
    "weights": {
        "vector": 0.7,
        "keyword": 0.3
    }
}

# 高级功能配置
ADVANCED_CONFIG = {
    "swarms": {
        "enabled": False,
        "max_agents": 3
    },
    "storm": {
        "enabled": False,
        "iterations": 3
    },
    "consciousness": {
        "enabled": False,
        "depth": 2
    },
    "hook": {
        "enabled": False,
        "steps": 3
    },
    "long_memory": {
        "enabled": False,
        "size": 1000
    }
}

# 确保目录存在
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
