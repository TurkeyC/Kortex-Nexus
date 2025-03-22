import os
from pathlib import Path
from typing import Dict, Any, Optional

# 项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# 数据相关路径
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# 预开发功能数据路径
ADVANCED_FEATURES_DIR = DATA_DIR / "advanced_features"
SWARMS_DIR = ADVANCED_FEATURES_DIR / "swarms"
STORM_DIR = ADVANCED_FEATURES_DIR / "storm"
CONSCIOUSNESS_DIR = ADVANCED_FEATURES_DIR / "consciousness"
HOOK_DIR = ADVANCED_FEATURES_DIR / "hook"
LONG_MEMORY_DIR = ADVANCED_FEATURES_DIR / "long_memory"

# 确保目录存在
for dir_path in [KNOWLEDGE_BASE_DIR, EMBEDDINGS_DIR, SWARMS_DIR, STORM_DIR,
                 CONSCIOUSNESS_DIR, HOOK_DIR, LONG_MEMORY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# GPU配置
ENABLE_GPU = True  # 默认开启GPU支持
CUDA_DEVICE = 0  # 默认使用第一个CUDA设备

# 模型配置
DEFAULT_MODEL = "ollama"
MODEL_CONFIG = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "model_name": "llama3"
    },
    "moonshot": {
        "api_key": os.environ.get("MOONSHOT_API_KEY", ""),
        "model_name": "moonshot-v1-8k"
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "model_name": "本地模型"
    },
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat"
    }
}

# 向量模型配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DEVICE = "cuda" if ENABLE_GPU else "cpu"  # 根据GPU可用性选择设备

# 界面配置
UI_CONFIG = {
    "title": "知识库问答系统",
    "description": "一个基于大模型的知识库问答系统",
    "theme": "light",
    "max_history": 10,
    # 预开发功能UI配置
    "advanced_features": {
        "enable_swarms": False,
        "enable_storm": False,
        "enable_consciousness": False,
        "enable_hook": False,
        "enable_long_memory": False
    }
}

# 检索配置
RETRIEVAL_CONFIG = {
    "vector_search_top_k": 5,
    "keyword_search_top_k": 5,
    "reranker_top_k": 3,
    # 检索权重配置
    "weights": {
        "vector": 0.7,
        "keyword": 0.2,
        "knowledge_tree": 0.1
    }
}

# 预开发功能配置
ADVANCED_CONFIG: Dict[str, Any] = {
    # Swarms配置 - 多智能体协作
    "swarms": {
        "enabled": False,
        "max_agents": 5,
        "consensus_threshold": 0.7,
        "coordinator_model": "gpt-3.5-turbo",
        "agent_models": ["llama3", "moonshot-v1", "deepseek-coder"]
    },

    # Storm配置 - 专家协作系统
    "storm": {
        "enabled": False,
        "max_experts": 3,
        "analysis_model": "gpt-3.5-turbo",
        "expert_selection_strategy": "auto"
    },

    # Consciousness Flow配置 - 创新思考
    "consciousness": {
        "enabled": False,
        "diffusion_steps": 5,
        "thought_temperature": 0.9,
        "evaluation_threshold": 0.6
    },

    # Hook配置 - 引导式对话
    "hook": {
        "enabled": False,
        "analysis_models": ["gpt-3.5-turbo", "llama3"],
        "guidance_strategy": "subtle",
        "max_guidance_steps": 5
    },

    # 长上下文记忆优化
    "long_memory": {
        "enabled": False,
        "strategy": "hippo_rag",  # 'faiss' 或 'hippo_rag'
        "summary_interval": 10,  # 每10轮对话进行一次总结
        "summary_model": "gpt-3.5-turbo-16k"
    }
}