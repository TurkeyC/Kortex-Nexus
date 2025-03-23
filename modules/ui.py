import streamlit as st
from typing import List, Dict, Any, Callable, Optional


def setup_page(title: str, description: str, theme: str = "light"):
    """设置页面基本配置 - 必须是第一个Streamlit命令"""
    # 这个必须是脚本中执行的第一个st命令
    st.set_page_config(
        page_title=title,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_styles(theme: str = "light"):
    """应用自定义样式 - 在set_page_config之后调用"""
    # 设置主题
    if theme == "dark":
        st.markdown("""
        <style>
        :root {
            --main-bg-color: #0e1117;
            --sidebar-bg-color: #262730;
            --text-color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        :root {
            --main-bg-color: #ffffff;
            --sidebar-bg-color: #f0f2f6;
            --text-color: #31333f;
        }
        </style>
        """, unsafe_allow_html=True)

    # 自定义CSS以提高界面美观度
    st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.bot {
        background-color: #e3f2fd;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
    }

    /* 定制滚动条 */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)


def display_header(title: str, description: str):
    """显示标题和描述"""
    st.title(title)
    st.markdown(description)


def render_sidebar(
        available_models: List[str],
        on_model_change: Callable[[str], None],
        on_feature_mode_change: Optional[Callable[[str], None]] = None,
        available_features: Optional[List[str]] = None,
        advanced_features_enabled: Optional[Dict[str, bool]] = None
):
    """渲染侧边栏"""
    with st.sidebar:
        st.header("设置")

        # 模型选择
        model_display_names = {
            "ollama": "Ollama (本地部署)",
            "moonshot": "Moonshot AI (云API)",
            "lmstudio": "LM Studio (本地部署)",
            "deepseek": "DeepSeek (云API)"
        }

        # 修改为优先选择lmstudio
        selected_model = st.selectbox(
            "选择模型",
            options=available_models,
            format_func=lambda x: model_display_names.get(x, x),
            index=available_models.index("lmstudio") if "lmstudio" in available_models else 0
        )
        on_model_change(selected_model)

        # 根据选择的模型显示额外选项
        if selected_model == "ollama":
            # 对于Ollama，显示模型名称选项
            ollama_models = ["llama3", "mistral", "gemma", "phi3", "llama3:8b", "llama3:70b"]
            ollama_selected = st.selectbox(
                "Ollama模型",
                options=ollama_models,
                index=0
            )
            st.session_state.ollama_model_name = ollama_selected

        elif selected_model == "moonshot":
            # 对于Moonshot，显示上下文窗口选项
            moonshot_models = ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]
            moonshot_selected = st.selectbox(
                "Moonshot模型",
                options=moonshot_models,
                index=0
            )
            st.session_state.moonshot_model_name = moonshot_selected

            # API密钥输入
            moonshot_api_key = st.text_input(
                "Moonshot API密钥",
                value=st.session_state.get("moonshot_api_key", ""),
                type="password"
            )
            st.session_state.moonshot_api_key = moonshot_api_key

        # 系统提示词
        system_prompt = st.text_area(
            "系统提示词",
            value="你是一个知识丰富的智能助手。请根据用户问题提供准确、有用的回答。",
            height=100
        )

        # 温度参数
        temperature = st.slider(
            "温度",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="较高的值会使输出更加随机，较低的值会使输出更加确定。"
        )

        # 知识库管理
        with st.expander("知识库管理", expanded=False):
            # 上传并处理文件
            uploaded_file = st.file_uploader(
                "上传文件到知识库",
                type=["md", "txt", "pdf"],
                help="支持Markdown、txt和PDF格式"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("刷新知识库", use_container_width=True):
                    with st.spinner("刷新中..."):
                        st.session_state.knowledge_base.load_knowledge_base()
                    st.success("知识库已更新")

            with col2:
                if uploaded_file:
                    if st.button("处理上传文件", use_container_width=True):
                        save_path = str(st.session_state.knowledge_base.knowledge_dir / uploaded_file.name)
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        with st.spinner("处理中..."):
                            st.session_state.knowledge_base.load_knowledge_base()
                        st.success(f"已上传并处理文件：{uploaded_file.name}")

        # 预开发功能选择（如果可用）
        if available_features and on_feature_mode_change:
            st.header("功能模式")

            # 创建友好的显示名称映射
            feature_display_names = {
                "standard": "标准模式",
                "swarms": "Swarms - 多智能体协作",
                "storm": "Storm - 专家协作系统",
                "consciousness": "Consciousness Flow - 创新思考",
                "hook": "Hook - 引导式对话"
            }

            # 过滤可用功能
            active_features = ["standard"]  # 标准模式始终可用
            if advanced_features_enabled:
                for feature in available_features:
                    if feature != "standard" and advanced_features_enabled.get(f"enable_{feature}", False):
                        active_features.append(feature)

            # 显示功能选择器
            selected_feature = st.selectbox(
                "选择功能模式",
                options=active_features,
                format_func=lambda x: feature_display_names.get(x, x),
                index=0
            )

            # 更新功能模式
            on_feature_mode_change(selected_feature)

            # 显示功能说明
            feature_descriptions = {
                "standard": "标准对话模式，使用选定的大模型直接回答问题。",
                "swarms": "多智能体协作模式，由多个专业领域模型协同解决问题。",
                "storm": "专家协作系统，分析问题并调用专业模型进行头脑风暴。",
                "consciousness": "创新思考模式，生成无目的思考并筛选有价值内容。",
                "hook": "引导式对话，多模型分析问题，逐步引导用户至结论。"
            }

            st.info(feature_descriptions.get(selected_feature, ""))

        # 版本信息
        st.markdown("---")
        st.markdown("**V 0.1**")

    # 根据选定的模型更新配置
    if selected_model == "ollama" and hasattr(st.session_state, 'ollama_model_name'):
        import config
        config.MODEL_CONFIG["ollama"]["model_name"] = st.session_state.ollama_model_name

    elif selected_model == "moonshot" and hasattr(st.session_state, 'moonshot_model_name'):
        import config
        import os
        config.MODEL_CONFIG["moonshot"]["model_name"] = st.session_state.moonshot_model_name
        if hasattr(st.session_state, 'moonshot_api_key') and st.session_state.moonshot_api_key:
            config.MODEL_CONFIG["moonshot"]["api_key"] = st.session_state.moonshot_api_key
            os.environ["MOONSHOT_API_KEY"] = st.session_state.moonshot_api_key

    return {
        "model": selected_model,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "feature_mode": selected_feature if 'selected_feature' in locals() else "standard"
    }


def render_chat_area(on_send: Callable[[str], None]):
    """渲染聊天区域"""
    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("请输入你的问题..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 处理回复
        on_send(prompt)


def display_assistant_response(response: str):
    """显示助手回复"""
    # 添加助手消息到历史
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 显示助手消息
    with st.chat_message("assistant"):
        st.markdown(response)
