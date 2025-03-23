import streamlit as st
import os
import pandas as pd
from pathlib import Path
import sys
import time

# 添加父目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.kb_manager import KnowledgeBaseManager
import config

def main():
    st.title("知识库管理")
    st.write("在这里管理您的知识库文件")
    
    # 初始化知识库管理器
    kb_manager = KnowledgeBaseManager(config.KNOWLEDGE_BASE_DIR)
    
    # 在使用知识库之前先初始化它
    if "knowledge_base" not in st.session_state:
        from modules.retrieval import KnowledgeBase
        st.session_state.knowledge_base = KnowledgeBase()
    
    # 侧边栏
    with st.sidebar:
        st.header("操作")
        upload_tab, browse_tab, version_tab = st.tabs(["上传文件", "浏览文件", "版本历史"])
        
        # 上传文件选项卡
        with upload_tab:
            uploaded_files = st.file_uploader(
                "上传Markdown文件",
                type=["md"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("处理所有文件"):
                    for file in uploaded_files:
                        with st.spinner(f"处理文件 {file.name}..."):
                            result = kb_manager.save_file(file.getbuffer(), file.name)
                            st.success(f"文件 {result['file_name']} {result['status']}")
                            
                    st.success(f"成功处理 {len(uploaded_files)} 个文件")
                    st.rerun()
        
        # 浏览文件选项卡
        with browse_tab:
            if st.button("刷新文件列表"):
                st.rerun()
        
        # 版本历史选项卡
        with version_tab:
            versions = kb_manager.get_version_history()
            if versions:
                for version in reversed(versions[:10]):  # 显示最新的10个版本
                    st.write(f"版本 {version['id']} - {version['timestamp']}")
                    for change in version["changes"]:
                        st.write(f"- {change['file']} ({change['action']})")
                    st.write("---")
    
    # 主要内容区域 - 文件列表
    files = kb_manager.get_file_list()
    
    if not files:
        st.info("知识库中没有文件。请上传一些Markdown文件。")
    else:
        # 转换为DataFrame以便于显示
        file_df = pd.DataFrame(files)
        
        # 显示文件表格
        st.write(f"共有 {len(files)} 个文件")
        st.dataframe(
            file_df,
            column_config={
                "name": "文件名",
                "path": "路径",
                "size": st.column_config.NumberColumn("大小(字节)"),
                "modified": "修改时间",
                "action": st.column_config.Column("操作", width="medium")
            },
            hide_index=True
        )
        
        # 选择文件查看详情
        selected_file = st.selectbox("选择文件查看详情", [f["path"] for f in files])
        
        if selected_file:
            content = kb_manager.get_file_content(selected_file)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("删除文件"):
                    result = kb_manager.delete_file(selected_file)
                    st.success(f"文件 {result['file_name']} 已删除")
                    time.sleep(1)
                    st.rerun()
                    
            with col2:
                if st.button("重新索引"):
                    st.session_state.knowledge_base.load_knowledge_base()
                    st.success("知识库已重新索引")
            
            # 显示文件内容
            st.subheader(f"文件内容: {selected_file}")
            st.text_area("Markdown源码", content, height=400)
            
            # 预览渲染后的内容
            st.subheader("预览")
            st.markdown(content)

if __name__ == "__main__":
    main()
