import streamlit as st
import os
import torch
from modules.ui import setup_page, apply_custom_styles, display_header, render_sidebar, render_chat_area, \
    display_assistant_response
from modules.models import get_model
from modules.retrieval import KnowledgeBase
import config

# 预开发功能导入（预留）
from modules.advanced.swarms import Swarms
from modules.advanced.storm import Storm
from modules.advanced.consciousness import ConsciousnessFlow
from modules.advanced.hook import Hook
from modules.advanced.long_memory import LongMemory


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

    # 初始化知识库
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = KnowledgeBase()
        # 尝试加载现有向量库
        st.session_state.knowledge_base.load_existing_vectorstore()

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

    # 定义模型更改回调
    def on_model_change(model_name):
        # 检查模型是否已经更改
        current_model_type = getattr(st.session_state.get("model", None), "__model_type__", "")

        if model_name != current_model_type:
            st.session_state.model = get_model(model_name)

    # 如果模型尚未初始化，则初始化
    if "model" not in st.session_state:
        st.session_state.model = get_model(config.DEFAULT_MODEL)

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

    # 定义消息发送回调
    def on_send(user_input):
        with st.spinner("思考中..."):
            # 获取聊天历史
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]  # 不包括最新的用户消息
            ]

            # 尝试检索相关信息
            retrieval_results = []
            try:
                retrieval_results = st.session_state.knowledge_base.search(user_input, top_k=3)
                if retrieval_results and "content" in retrieval_results[0]:
                    context_info = "\n\n知识库参考:\n" + "\n".join(
                        [f"- {res['content'][:200]}..." for res in retrieval_results]
                    )
                    user_input_with_context = user_input + context_info
                else:
                    user_input_with_context = user_input
            except Exception as e:
                st.error(f"检索出错: {str(e)}")
                user_input_with_context = user_input

            # 根据当前活动的功能模式处理查询
            if st.session_state.active_feature == "standard":
                # 标准模式 - 直接使用模型回复
                response = st.session_state.model.generate_response(user_input_with_context, chat_history)
            elif st.session_state.active_feature == "swarms":
                # Swarms模式 (预留)
                response = st.session_state.advanced_features["swarms"].process_query(user_input, chat_history)
            elif st.session_state.active_feature == "storm":
                # Storm模式 (预留)
                response = st.session_state.advanced_features["storm"].process_query(user_input, chat_history)
            elif st.session_state.active_feature == "consciousness":
                # Consciousness Flow模式 (预留)
                response = st.session_state.advanced_features["consciousness"].process_query(user_input, chat_history)
            elif st.session_state.active_feature == "hook":
                # Hook模式 (预留)
                response = st.session_state.advanced_features["hook"].process_query(user_input, chat_history)
            else:
                # 默认模式
                response = st.session_state.model.generate_response(user_input_with_context, chat_history)

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