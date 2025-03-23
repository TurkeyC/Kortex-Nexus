import streamlit as st
import os
import sys
import time
import json
from pathlib import Path

# 添加父目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.retrieval import KnowledgeBase
import config

def main():
    st.title("知识库维护工具")
    st.write("这个工具可以帮助您修复知识库索引问题，特别是向量维度不匹配的情况")
    
    # 检查知识库目录是否存在
    if not os.path.exists(config.KNOWLEDGE_BASE_DIR):
        st.error(f"知识库目录不存在: {config.KNOWLEDGE_BASE_DIR}")
        if st.button("创建知识库目录"):
            os.makedirs(config.KNOWLEDGE_BASE_DIR, exist_ok=True)
            st.success(f"已创建知识库目录: {config.KNOWLEDGE_BASE_DIR}")
            st.rerun()
        return
        
    # 检查嵌入向量目录是否存在
    if not os.path.exists(config.EMBEDDINGS_DIR):
        st.error(f"嵌入向量目录不存在: {config.EMBEDDINGS_DIR}")
        if st.button("创建嵌入向量目录"):
            os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
            st.success(f"已创建嵌入向量目录: {config.EMBEDDINGS_DIR}")
            st.rerun()
        return
    
    # 初始化知识库对象
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()
        # 尝试加载知识库索引
        st.session_state.kb.load_existing_vectorstore()
    
    # 读取索引元数据，如果存在
    metadata_path = os.path.join(config.EMBEDDINGS_DIR, "index_metadata.json")
    index_path = os.path.join(config.EMBEDDINGS_DIR, "faiss_index")
    
    # 显示维度不匹配警告
    kb = st.session_state.kb
    if hasattr(kb, 'vector_dimension') and hasattr(kb, 'index_dimension') and kb.vector_dimension != kb.index_dimension:
        st.error(f"""
        **维度不匹配错误!** 
        
        当前嵌入模型维度: {kb.vector_dimension}
        索引维度: {kb.index_dimension}
        
        这将导致搜索失败。请点击下方的"重建知识库索引"按钮解决此问题。
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("索引状态")
        if os.path.exists(index_path):
            st.success("✅ FAISS索引文件存在")
            
            # 显示索引大小
            index_size = os.path.getsize(index_path)
            st.info(f"索引大小: {index_size / (1024*1024):.2f} MB")
            
            # 显示当前嵌入模型信息
            st.write("**当前嵌入模型信息:**")
            st.write(f"- 模型名称: {kb.embedding_model_name}")
            st.write(f"- 向量维度: {kb.vector_dimension}")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    st.write("**索引元数据:**")
                    st.write(f"- 嵌入模型: {metadata.get('model_name', '未知')}")
                    st.write(f"- 向量维度: {metadata.get('dimensions', '未知')}")
                    st.write(f"- 创建时间: {metadata.get('created_at', '未知')}")
                    
                    # 检查维度是否匹配
                    stored_dims = metadata.get('dimensions')
                    if stored_dims and kb.vector_dimension and stored_dims != kb.vector_dimension:
                        st.error(f"⚠️ **维度不匹配**: 索引维度({stored_dims}) != 当前模型维度({kb.vector_dimension})")
                    else:
                        st.success("✅ 维度匹配正常")
                except Exception as e:
                    st.error(f"读取元数据失败: {str(e)}")
            else:
                st.warning("⚠️ 索引元数据文件不存在")
        else:
            st.error("❌ FAISS索引文件不存在")
    
    with col2:
        st.subheader("维护操作")
        
        # 测试索引维度兼容性
        if st.button("测试索引维度兼容性"):
            kb = st.session_state.kb
            
            if not kb.vector_store:
                if not kb.load_existing_vectorstore():
                    st.error("无法加载向量索引")
                else:
                    st.success("已加载向量索引")
            
            if kb.vector_store:
                if kb._check_dimension_compatibility():
                    st.success(f"✅ 索引维度兼容性正常 - 当前维度: {kb.vector_dimension}")
                else:
                    st.error(f"❌ 索引维度与当前嵌入模型不兼容: 索引维度({kb.index_dimension}) != 当前模型维度({kb.vector_dimension})")
            else:
                st.error("无法测试，索引未加载")
        
        # 清除索引按钮
        if st.button("清除当前索引"):
            kb = st.session_state.kb
            if kb.clear_index():
                st.success("✅ 索引已清除")
                # 删除文件
                if os.path.exists(index_path):
                    try:
                        os.remove(index_path)
                        st.info(f"已删除索引文件: {index_path}")
                    except Exception as e:
                        st.error(f"删除索引文件失败: {str(e)}")
                        
                if os.path.exists(metadata_path):
                    try:
                        os.remove(metadata_path)
                        st.info(f"已删除元数据文件: {metadata_path}")
                    except Exception as e:
                        st.error(f"删除元数据文件失败: {str(e)}")
            else:
                st.error("清除索引失败")
        
        # 重建索引按钮
        if st.button("重建知识库索引", type="primary"):
            with st.spinner("正在重建知识库索引..."):
                # 备份旧索引
                if os.path.exists(index_path):
                    backup_path = f"{index_path}_backup_{int(time.time())}"
                    try:
                        import shutil
                        shutil.copy2(index_path, backup_path)
                        st.info(f"已备份旧索引到: {backup_path}")
                    except Exception as e:
                        st.warning(f"备份旧索引失败: {str(e)}")
                
                # 清除旧索引
                kb = st.session_state.kb
                kb.clear_index()
                
                # 删除旧索引文件
                try:
                    if os.path.exists(index_path):
                        os.remove(index_path)
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                except Exception as e:
                    st.error(f"删除旧索引失败: {str(e)}")
                
                # 重新初始化知识库
                kb = KnowledgeBase()
                st.session_state.kb = kb
                
                # 创建新索引
                kb.load_knowledge_base()
                
                st.success("✅ 知识库索引已重建")
                
                # 显示新索引信息
                if kb.vector_store:
                    st.info(f"新索引维度: {kb.vector_dimension}")
                else:
                    st.warning("未成功创建索引，请检查知识库中是否有文档")
                
    # 知识库文件检查
    st.subheader("知识库文件检查")
    
    # 显示知识库文件统计
    markdown_files = list(Path(config.KNOWLEDGE_BASE_DIR).glob("**/*.md"))
    st.info(f"知识库中共有 {len(markdown_files)} 个Markdown文件")
    
    # 显示文件列表
    if markdown_files:
        files_df = []
        for md_file in markdown_files:
            rel_path = md_file.relative_to(config.KNOWLEDGE_BASE_DIR)
            size = md_file.stat().st_size / 1024  # KB
            files_df.append({
                "文件名": str(rel_path),
                "大小(KB)": f"{size:.2f}"
            })
        
        st.dataframe(files_df, use_container_width=True)
    else:
        st.warning("知识库中没有Markdown文件，请先上传文档")
        
        # 示例文件上传按钮
        with st.expander("创建示例文档"):
            if st.button("创建示例Markdown文档"):
                example_content = """# 示例文档
                
## 简介
这是一个示例Markdown文档，用于测试知识库功能。

## 主要内容
- 这里是一些示例内容
- 用于测试知识库的索引和检索功能

## 结论
通过这个示例文档，您可以测试知识库的基本功能是否正常工作。
                """
                
                example_path = os.path.join(config.KNOWLEDGE_BASE_DIR, "example.md")
                with open(example_path, 'w', encoding='utf-8') as f:
                    f.write(example_content)
                    
                st.success(f"已创建示例文档: {example_path}")
                st.rerun()

    # 添加嵌入模型测试区域
    st.subheader("嵌入模型测试")
    test_text = st.text_input("输入测试文本", value="这是一个测试文本，用于检查嵌入模型")
    if st.button("测试嵌入"):
        try:
            kb = st.session_state.kb
            vector = kb.embeddings.embed_query(test_text)
            st.success(f"嵌入成功，向量维度: {len(vector)}")
            
            # 显示部分向量值
            if len(vector) > 10:
                preview = vector[:10]
                st.write(f"向量前10个值: {preview}")
        except Exception as e:
            st.error(f"嵌入测试失败: {str(e)}")
    
if __name__ == "__main__":
    main()
