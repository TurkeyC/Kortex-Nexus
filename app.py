import streamlit as st
import os
import torch
import time
from modules.ui import setup_page, apply_custom_styles, display_header, render_sidebar, render_chat_area, \
    display_assistant_response
from modules.models import get_model
from modules.retrieval import KnowledgeBase
import config

# 导入二阶段开发模块
from modules.retrieval_engines import KeywordRetriever, HybridRetriever, TreeStructureRetriever
from modules.rag import RAGPipeline
from modules.kb_manager import KnowledgeBaseManager
from modules.conversation_manager import ConversationTree, ContextManager
# 修正导入, 使用正确的ui_components
from modules.ui_components import render_js_tree_selector, render_search_results


# 预开发功能导入（预留）
from modules.advanced.swarms import Swarms
from modules.advanced.storm import Storm
from modules.advanced.consciousness import ConsciousnessFlow
from modules.advanced.hook import Hook
from modules.advanced.long_memory import LongMemory


import warnings
import logging
# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 设置日志级别
logging.basicConfig(level=logging.ERROR)


# 第一个Streamlit命令必须是set_page_config
def main():
    # 1. 设置页面配置 - 必须是第一个Streamlit命令
    setup_page(
        title=config.UI_CONFIG["title"],
        description=config.UI_CONFIG["description"],
        theme=config.UI_CONFIG["theme"]
    )

    # 2. 应用自定义样式
    apply_custom_styles(theme=config.UI_CONFIG["theme"])

    # 3. 显示标题和描述
    display_header(
        title=config.UI_CONFIG["title"],
        description=config.UI_CONFIG["description"]
    )

    # 4. 检测GPU支持
    if torch.cuda.is_available() and config.ENABLE_GPU:
        st.sidebar.success(f"已检测到GPU: {torch.cuda.get_device_name(0)}")
        device_info = f"CUDA: {torch.version.cuda} | 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        st.sidebar.info(device_info)
    else:
        st.sidebar.warning("未检测到GPU或GPU支持未启用，将使用CPU模式运行")
        config.ENABLE_GPU = False
        config.EMBEDDING_DEVICE = "cpu"

    # 检查模型可用性并提供友好提示
    is_local_model_available = False
    is_api_model_available = False

    try:
        # 检查本地模型服务可用性
        import requests
        ollama_available = False
        lmstudio_available = False

        # 尝试连接Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                ollama_available = True
                st.sidebar.success("✅ Ollama服务已连接")
        except requests.RequestException:
            pass
        # # 显示安装指南链接
        # st.sidebar.markdown("""
        # [安装Ollama](https://ollama.com/download)
        # ```bash
        # # 安装后运行:
        # ollama pull llama3
        # ```
        # """)

        # 尝试连接LM Studio (通常在1234或8000端口)
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            if response.status_code == 200:
                lmstudio_available = True
                st.sidebar.success("✅ LM Studio服务已连接")
        except requests.RequestException:
            pass

        is_local_model_available = ollama_available or lmstudio_available

        if not is_local_model_available:
            st.sidebar.warning("⚠️ 未检测到Ollama或LM Studio服务")
    except Exception as e:
        st.sidebar.warning(f"⚠️ 检测本地模型服务时出错: {str(e)}")


    # 检查API密钥v1
    # 检查各种可能的API密钥
    api_keys = {
        "OpenAI": os.environ.get("OPENAI_API_KEY") or
                  (hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key),
        "Moonshot": os.environ.get("MOONSHOT_API_KEY") or
                    (hasattr(st.session_state, 'moonshot_api_key') and st.session_state.moonshot_api_key),
        "Azure": os.environ.get("AZURE_OPENAI_API_KEY") or
                 (hasattr(st.session_state, 'azure_api_key') and st.session_state.azure_api_key),
        "Anthropic": os.environ.get("ANTHROPIC_API_KEY") or
                     (hasattr(st.session_state, 'anthropic_api_key') and st.session_state.anthropic_api_key)
    }

    # 检查是否有任何API密钥可用
    available_apis = [name for name, key in api_keys.items() if key]
    if available_apis:
        is_api_model_available = True
        st.sidebar.success(f"✅ 已配置API密钥: {', '.join(available_apis)}")
    else:
        st.sidebar.info("ℹ️ 暂未检测到配置任何可用API密钥，需使用本地模型")

    if not is_local_model_available and not is_api_model_available:
        st.warning("⚠️ 警告：未检测到可用的模型服务。请在侧边栏配置API密钥或启动本地模型服务。")

    # 重要：先初始化模型，再初始化依赖模型的组件
    # 如果模型尚未初始化，则初始化
    if "model" not in st.session_state:
        st.session_state.model = get_model(config.DEFAULT_MODEL)
        
    # 定义模型更改回调v1
    def on_model_change(model_name):
        # 检查模型是否已经更改
        current_model_type = getattr(st.session_state.get("model", None), "__model_type__", "")

        if model_name != current_model_type:
            st.session_state.model = get_model(model_name)
            
    # 初始化知识库v1
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = KnowledgeBase()
        # 尝试加载现有向量库
        if st.session_state.knowledge_base.load_existing_vectorstore():
            st.success("成功加载现有知识库")
            # 存储向量维度
            if st.session_state.knowledge_base.vector_store:
                st.session_state.vector_dimension = st.session_state.knowledge_base.vector_dimension
        else:
            st.warning("无法加载知识库，请上传文档或重建索引")
    
    # 知识库重建按钮 - 放在更突出的位置
    if hasattr(st.session_state, "knowledge_base") and "vector_dimension" in st.session_state:
        kb_col1, kb_col2 = st.columns([3, 1])
        with kb_col1:
            st.info(f"知识库状态: {'已加载' if st.session_state.knowledge_base.vector_store else '未加载'}")
        with kb_col2:
            if st.button("重新构建知识库索引", key="rebuild_index"):
                with st.spinner("正在重新构建知识库索引..."):
                    # 清除现有索引
                    st.session_state.knowledge_base.vector_store = None
                    # 重新加载知识库
                    st.session_state.knowledge_base.load_knowledge_base()
                st.success("知识库索引已重建")
                # 重新加载页面
                st.rerun()

    # 初始化知识库管理器v2
    if "kb_manager" not in st.session_state:
        st.session_state.kb_manager = KnowledgeBaseManager(config.KNOWLEDGE_BASE_DIR)

    # 初始化对话树v2
    if "conversation_tree" not in st.session_state:
        st.session_state.conversation_tree = ConversationTree()
        st.session_state.conversation_tree.start_conversation(
            "你是一个知识丰富的智能助手。请根据用户问题提供准确、有用的回答。"
        )

    # 初始化上下文管理器v2 - 现在可以安全地使用model
    if "context_manager" not in st.session_state:
        st.session_state.context_manager = ContextManager(
            max_tokens=4000,
            llm=st.session_state.model
        )

    # 初始化关键词检索器v2 - 同样依赖model
    if "keyword_retriever" not in st.session_state and hasattr(st.session_state, "knowledge_base"):
        # 加载文档数据
        documents = []
        if st.session_state.knowledge_base.vector_store:
            for doc in st.session_state.knowledge_base.vector_store.docstore._dict.values():
                documents.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            # 创建关键词检索器
            st.session_state.keyword_retriever = KeywordRetriever()
            st.session_state.keyword_retriever.add_documents(documents)
            
            # 创建树状结构检索器
            st.session_state.tree_retriever = TreeStructureRetriever(st.session_state.knowledge_base)
            
            # 初始化选中的树节点
            if "selected_tree_nodes" not in st.session_state:
                st.session_state.selected_tree_nodes = []

            # 创建混合检索器
            st.session_state.hybrid_retriever = HybridRetriever(
                vector_retriever=st.session_state.knowledge_base,
                keyword_retriever=st.session_state.keyword_retriever,
                tree_retriever=st.session_state.tree_retriever,
                weights=config.RETRIEVAL_CONFIG["weights"]
            )

            # 创建RAG管道
            st.session_state.rag_pipeline = RAGPipeline(
                llm=st.session_state.model,
                retriever=st.session_state.hybrid_retriever
            )

    # 初始化预开发功能（预留）
    if "advanced_features" not in st.session_state:
        st.session_state.advanced_features = {
            "swarms": Swarms(config.ADVANCED_CONFIG["swarms"]),
            "storm": Storm(config.ADVANCED_CONFIG["storm"]),
            "consciousness": ConsciousnessFlow(config.ADVANCED_CONFIG["consciousness"]),
            "hook": Hook(config.ADVANCED_CONFIG["hook"]),
            "long_memory": LongMemory(config.ADVANCED_CONFIG["long_memory"])
        }

        # 初始化选择的功能模式
        st.session_state.active_feature = "standard"  # 标准模式

    # 定义功能模式更改回调
    def on_feature_mode_change(feature_mode):
        st.session_state.active_feature = feature_mode

    # 渲染侧边栏
    sidebar_config = render_sidebar(
        available_models=list(config.MODEL_CONFIG.keys()),
        on_model_change=on_model_change,
        on_feature_mode_change=on_feature_mode_change,
        available_features=["standard", "swarms", "storm", "consciousness", "hook"],
        advanced_features_enabled=config.UI_CONFIG["advanced_features"]
    )

    # 更新模型配置
    st.session_state.model.set_system_prompt(sidebar_config["system_prompt"])
    st.session_state.model.set_temperature(sidebar_config["temperature"])

    # 渲染树状结构选择器 - 使用修改后的组件
    if "tree_retriever" in st.session_state:
        with st.expander("📚 知识库目录", expanded=False):
            # 获取树结构
            tree = st.session_state.tree_retriever.get_tree()

            # 定义节点选择回调
            def on_node_select(selected_nodes):
                st.session_state.selected_tree_nodes = selected_nodes

            # 使用扁平化组件渲染
            render_js_tree_selector(
                tree,
                st.session_state.selected_tree_nodes,
                on_node_select
            )

    # 定义消息发送回调v1
    def on_send(user_input):
        with st.spinner("思考中..."):
            # 获取聊天历史
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]  # 不包括最新的用户消息
            ]

            # 添加用户消息到对话树
            st.session_state.conversation_tree.add_message({"role": "user", "content": user_input})

            # 获取对话路径
            chat_history = st.session_state.conversation_tree.get_conversation_path()

            # 压缩历史对话
            compressed_history = st.session_state.context_manager.compress_history(chat_history)

            # 使用RAG管道处理查询
            if hasattr(st.session_state, "rag_pipeline"):
                try:
                    # 将选择的树节点传递给检索器
                    selected_nodes = st.session_state.selected_tree_nodes if hasattr(st.session_state, "selected_tree_nodes") else []
                    
                    rag_result = st.session_state.rag_pipeline.process(
                        user_input, 
                        compressed_history,
                        selected_nodes=selected_nodes  # 传递选中的节点
                    )
                    response = rag_result["response"]

                    # 显示检索到的上下文（调试用）
                    with st.expander("查看检索上下文", expanded=False):
                        st.write("重写查询：", rag_result["rewritten_query"])
                        st.write("检索结果：")
                        for i, result in enumerate(rag_result["results"]):
                            st.write(f"[{i+1}] 相关度: {result['score']:.4f}")
                            st.write(f"来源: {result['metadata']['source']}")
                            st.write(result["content"][:200] + "...")
                            st.write("---")
                except Exception as e:
                    # 如果RAG管道失败，回退到基本模型回答
                    st.error(f"RAG处理失败: {str(e)}")
                    response = st.session_state.model.generate_response(user_input, compressed_history)
            else:
                # 回退到基本模型回答
                response = st.session_state.model.generate_response(user_input, compressed_history)

            # 添加助手回复到对话树
            st.session_state.conversation_tree.add_message({"role": "assistant", "content": response})

            # 显示回复
            display_assistant_response(response)

            # 长上下文记忆处理 (预留)
            if config.ADVANCED_CONFIG["long_memory"]["enabled"]:
                # 添加消息到长记忆系统
                st.session_state.advanced_features["long_memory"].add_message({
                    "role": "user",
                    "content": user_input
                })
                st.session_state.advanced_features["long_memory"].add_message({
                    "role": "assistant",
                    "content": response
                })

    # 渲染聊天区域
    render_chat_area(on_send=on_send)


if __name__ == "__main__":
    main()
