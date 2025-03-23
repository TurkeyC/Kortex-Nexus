import streamlit as st
from typing import List, Dict, Any, Callable, Optional


def render_js_tree_selector(tree: Dict[str, Any], selected_nodes: List[str],
                            on_node_select: Callable[[List[str]], None]):
    """使用JavaScript实现高性能树形选择器，风格简化版"""
    import json
    from streamlit.components.v1 import html

    # 初始化状态
    if "tree_selected" not in st.session_state:
        st.session_state.tree_selected = selected_nodes.copy()

    if "tree_temp_selected" not in st.session_state:
        st.session_state.tree_temp_selected = []

    if "show_tree" not in st.session_state:
        st.session_state.show_tree = True

    # 预处理树结构
    flat_paths = []

    def flatten_tree(node, path=""):
        node_name = node.get("name", "")
        node_type = node.get("type", "folder")
        full_path = f"{path}/{node_name}" if path else node_name

        if node_type == "file":
            flat_paths.append({"path": full_path, "type": "file", "name": node_name})
        else:
            flat_paths.append({"path": full_path, "type": "folder", "name": node_name})

        if "children" in node and node["children"]:
            for child in node["children"]:
                flatten_tree(child, full_path)

    flatten_tree(tree)

    # 显示HTML树
    if st.session_state.show_tree:
        html_code = """
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/jstree@3.3.12/dist/jstree.min.js"></script>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/jstree@3.3.12/dist/themes/default/style.min.css">

        <div style="margin-bottom:10px">
            <input type="text" id="tree_search" placeholder="🔍 搜索文件或文件夹" style="width:100%;padding:8px">
        </div>

        <div id="knowledge_tree" style="max-height:400px;overflow:auto;"></div>

        <div id="selection_count" style="margin-top:10px;padding:8px;background-color:#e8f5e9;display:none;"></div>

        <script>
        // 初始化树
        const treeData = """ + json.dumps(flat_paths) + """;
        const initialSelected = """ + json.dumps(st.session_state.tree_selected) + """;
        let selectedNodes = [...initialSelected];

        function buildTreeData(flatPaths) {
            const root = [];
            const map = {};

            // 创建文件夹节点
            flatPaths.filter(item => item.type === 'folder').forEach(item => {
                const parts = item.path.split('/');
                let currentPath = '';

                parts.forEach((part, i) => {
                    const partPath = currentPath ? `${currentPath}/${part}` : part;
                    currentPath = partPath;

                    if (!map[partPath]) {
                        const newNode = {
                            text: part,
                            path: partPath,
                            type: 'folder',
                            state: {opened: false},
                            children: []
                        };
                        map[partPath] = newNode;

                        if (i === 0) {
                            root.push(newNode);
                        } else {
                            const parentPath = parts.slice(0, i).join('/');
                            if (map[parentPath]) {
                                map[parentPath].children.push(newNode);
                            }
                        }
                    }
                });
            });

            // 添加文件节点
            flatPaths.filter(item => item.type === 'file').forEach(item => {
                const parts = item.path.split('/');
                const fileName = parts.pop();
                const parentPath = parts.join('/');

                const fileNode = {
                    text: fileName,
                    path: item.path,
                    type: 'file',
                    icon: 'jstree-file'
                };

                if (map[parentPath]) {
                    map[parentPath].children.push(fileNode);
                } else if (parts.length === 0) {
                    root.push(fileNode);
                }
            });

            return root;
        }

        const jsTreeData = buildTreeData(treeData);

        $(document).ready(function() {
            $('#knowledge_tree').jstree({
                'core': {
                    'data': jsTreeData,
                    'themes': {'name': 'default'},
                    'check_callback': true,
                    'multiple': true
                },
                'plugins': ['checkbox', 'search', 'types'],
                'checkbox': {
                    'three_state': true, 
                    'cascade': 'up+down'
                },
                'types': {
                    'folder': {'icon': 'jstree-folder'},
                    'file': {'icon': 'jstree-file'}
                },
                'search': {
                    'show_only_matches': true,
                    'show_only_matches_children': true
                }
            });

            const tree = $('#knowledge_tree').jstree(true);

            // 设置初始选择
            setTimeout(() => {
                initialSelected.forEach(path => {
                    tree.get_json('#', {flat:true}).forEach(node => {
                        if(node.original && node.original.path === path) {
                            tree.select_node(node.id);
                        }
                    });
                });
                updateSelectionCount();
            }, 500);

            // 搜索功能
            $('#tree_search').keyup(function() {
                $('#knowledge_tree').jstree('search', $(this).val());
            });

            // 处理选择变化
            $('#knowledge_tree').on('changed.jstree', function (e, data) {
                selectedNodes = [];
                data.selected.forEach(id => {
                    const node = tree.get_node(id);
                    if (node.original && node.original.type === 'file') {
                        selectedNodes.push(node.original.path);
                    }
                });

                // 更新隐藏字段
                document.getElementById('selected_files').value = JSON.stringify(selectedNodes);
                updateSelectionCount();
            });

            function updateSelectionCount() {
                if (selectedNodes.length > 0) {
                    $('#selection_count').text(`已选择 ${selectedNodes.length} 个文件`).show();
                } else {
                    $('#selection_count').hide();
                }
            }
        });
        </script>

        <!-- 隐藏字段存储选择 -->
        <input type="hidden" id="selected_files" value="">
        """

        # 显示HTML
        html(html_code, height=450)

    # 操作按钮 - 提供一个常规的界面来操控树选择
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("确认选择"):
            try:
                # 尝试从JavaScript获取选择结果
                selection_data = []
                if "selected_files" in st.session_state:
                    selection_data = json.loads(st.session_state.selected_files)

                # # 如果获取不到，提供手动输入
                # if not selection_data:
                #     selection_text = st.text_area(
                #         "请从树中选择文件，然后复制选择结果到此处",
                #         placeholder='例如: ["文件路径1", "文件路径2"]',
                #         height=100
                #     )
                #     if selection_text:
                #         selection_data = json.loads(selection_text)

                if isinstance(selection_data, list):
                    st.session_state.tree_selected = selection_data
                    selected_nodes.clear()
                    selected_nodes.extend(selection_data)
                    on_node_select(selection_data)
                    st.success("已应用选择")
            except Exception as e:
                st.error(f"应用选择时出错: {str(e)}")

    with col2:
        if st.button("清除所有选择"):
            st.session_state.tree_selected = []
            selected_nodes.clear()
            on_node_select([])
            st.session_state.show_tree = False  # 触发刷新
            st.session_state.show_tree = True
            st.rerun()

    with col3:
        if st.button("全选文件"):
            all_files = [item["path"] for item in flat_paths if item["type"] == "file"]
            st.session_state.tree_selected = all_files
            selected_nodes.clear()
            selected_nodes.extend(all_files)
            on_node_select(all_files)
            st.session_state.show_tree = False  # 触发刷新
            st.session_state.show_tree = True
            st.rerun()

    # 监听JavaScript消息 - 作为备用通信机制
    st.markdown("""
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.selected) {
            const selectedData = JSON.parse(e.data.selected);
            const key = 'selected_files';
            const value = e.data.selected;

            // 更新Streamlit状态
            if (window.parent.window.streamlitSetComponentValue) {
                window.parent.window.streamlitSetComponentValue(key, value);
            }
        }
    }, false);
    </script>
    """, unsafe_allow_html=True)

    # 显示当前选择状态
    if st.session_state.tree_selected:
        st.success(f"已选择 {len(st.session_state.tree_selected)} 个文件")

def render_search_results(results: List[Dict[str, Any]], query: str = ""):
    """渲染搜索结果
    
    Args:
        results: 搜索结果列表
        query: 搜索查询
    """
    if not results:
        st.info("未找到相关结果")
        return
        
    st.write(f"**搜索结果 ({len(results)})**")
    st.write(f"查询: *{query}*")
    
    # 创建一个选项卡布局，每个结果一个标签页
    tabs = st.tabs([f"结果 {i+1}" for i in range(len(results))])
    
    for i, (tab, result) in enumerate(zip(tabs, results)):
        with tab:
            # 显示元数据
            source = result.get("metadata", {}).get("source", "未知来源")
            title = result.get("metadata", {}).get("title", "无标题")
            score = result.get("score", 0.0)
            
            st.write(f"**来源:** {source}")
            st.write(f"**标题:** {title}")
            st.write(f"**相关度:** {score:.4f}")
            
            # 显示详细信息
            with st.expander("查看内容", expanded=i==0):  # 默认展开第一个结果
                st.markdown(result.get("content", ""))
                
            # 显示多种检索分数（如果有）
            if "vector_score" in result:
                with st.expander("详细分数", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("向量相似度", f"{result.get('vector_score', 0.0):.4f}")
                    col2.metric("关键词分数", f"{result.get('keyword_score', 0.0):.4f}")
                    col3.metric("结构匹配度", f"{result.get('tree_score', 0.0):.4f}")
                    
                    # 如果有使用的权重信息，显示权重
                    if "weights_used" in result:
                        weights = result["weights_used"]
                        st.write("**使用的权重:**")
                        for k, v in weights.items():
                            st.write(f"- {k}: {v:.2f}")
