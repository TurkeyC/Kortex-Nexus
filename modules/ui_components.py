import streamlit as st
from typing import List, Dict, Any, Callable, Optional


def render_js_tree_selector(tree: Dict[str, Any], selected_nodes: List[str],
                           on_node_select: Callable[[List[str]], None]):
    """使用JavaScript实现高性能树形选择器，避免频繁刷新

    Args:
        tree: 文档树结构
        selected_nodes: 当前选中的节点路径列表
        on_node_select: 节点选择回调函数
    """
    import json
    from streamlit.components.v1 import html

    # 创建key来接收组件的返回值
    if "js_tree_selected_nodes" not in st.session_state:
        st.session_state.js_tree_selected_nodes = selected_nodes.copy()

    # 将Python树结构转换为JS友好的格式
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

    # 处理树结构
    flatten_tree(tree)

    # 创建HTML/JS组件 - 使用session_state进行通信
    html_code = f"""
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/jstree@3.3.12/dist/jstree.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/jstree@3.3.12/dist/themes/default/style.min.css">
    
    <div style="margin-bottom:10px">
        <input type="text" id="tree_search" placeholder="🔍 搜索文件或文件夹" style="width:100%;padding:8px;border:1px solid #ccc;border-radius:4px;">
    </div>
    
    <div id="knowledge_tree" style="max-height:400px;overflow:auto;"></div>
    
    <div style="display:flex;justify-content:space-between;margin-top:15px;">
        <button id="confirm_selection" style="background-color:#4CAF50;color:white;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">确认选择</button>
        <button id="clear_selection" style="background-color:#f44336;color:white;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">清除所有选择</button>
        <button id="select_all" style="background-color:#2196F3;color:white;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;">全选</button>
    </div>
    <div id="selection_count" style="margin-top:10px;padding:8px;background-color:#e8f5e9;border-radius:4px;display:none;"></div>
    
    <script>
    // 树结构数据
    const treeData = {{tree_data}};
    const initialSelectedNodes = {selected_nodes};
    let selectedNodes = [...initialSelectedNodes];
    
    // 构建树结构
    function buildTreeData(flatPaths) {{
        const root = [];
        const map = {{}};
        
        // 首先创建所有文件夹节点
        flatPaths.filter(item => item.type === 'folder').forEach(item => {{
            const parts = item.path.split('/');
            let currentPath = '';
            
            parts.forEach((part, i) => {{
                const partPath = currentPath ? `${{currentPath}}/${{part}}` : part;
                currentPath = partPath;
                
                if (!map[partPath]) {{
                    const newNode = {{
                        text: part,
                        path: partPath,
                        type: 'folder',
                        state: {{opened: false}},
                        children: []
                    }};
                    map[partPath] = newNode;
                    
                    if (i === 0) {{
                        root.push(newNode);
                    }} else {{
                        const parentPath = parts.slice(0, i).join('/');
                        if (map[parentPath]) {{
                            map[parentPath].children.push(newNode);
                        }}
                    }}
                }}
            }});
        }});
        
        // 然后添加所有文件节点
        flatPaths.filter(item => item.type === 'file').forEach(item => {{
            const parts = item.path.split('/');
            const fileName = parts.pop();
            const parentPath = parts.join('/');
            
            const fileNode = {{
                text: fileName,
                path: item.path,
                type: 'file',
                icon: 'jstree-file'
            }};
            
            if (map[parentPath]) {{
                map[parentPath].children.push(fileNode);
            }} else if (parts.length === 0) {{
                // 根目录下的文件
                root.push(fileNode);
            }}
        }});
        
        return root;
    }}
    
    // 将扁平路径转换为树结构
    const jsTreeData = buildTreeData(treeData);
    
    // 初始化树
    $(document).ready(function() {{
        // 初始化树组件
        $('#knowledge_tree').jstree({{
            'core': {{
                'data': jsTreeData,
                'themes': {{
                    'name': 'default',
                    'responsive': true
                }},
                'check_callback': true,
                'multiple': true
            }},
            'plugins': ['checkbox', 'search', 'types', 'wholerow'],
            'checkbox': {{
                'three_state': true,
                'cascade': 'up+down'
            }},
            'types': {{
                'folder': {{
                    'icon': 'jstree-folder'
                }},
                'file': {{
                    'icon': 'jstree-file'
                }}
            }},
            'search': {{
                'show_only_matches': true,
                'show_only_matches_children': true
            }}
        }});
        
        // 设置初始选择
        const tree = $('#knowledge_tree').jstree(true);
        initialSelectedNodes.forEach(path => {{
            const node = tree.get_node_by_path(path);
            if (node) {{
                tree.select_node(node);
            }}
        }});
        
        // 添加获取节点路径的方法
        $.jstree.plugins.path = function () {{
            this.get_node_by_path = function (path) {{
                const allNodes = this.get_json('#', {{flat: true}});
                return allNodes.find(node => node.path === path)?.id;
            }};
        }};
        
        // 搜索功能
        $('#tree_search').keyup(function() {{
            $('#knowledge_tree').jstree('search', $(this).val());
        }});
        
        // 处理选择事件
        $('#knowledge_tree').on('changed.jstree', function (e, data) {{
            selectedNodes = data.selected.map(id => {{
                const node = tree.get_node(id);
                if (node.original && node.original.type === 'file') {{
                    return node.original.path;
                }}
                return null;
            }}).filter(Boolean);
            
            updateSelectionCount();
        }});
        
        // 确认选择
        $('#confirm_selection').click(function() {{
            // 使用Streamlit会话状态
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: JSON.stringify(selectedNodes)  // 转换为JSON字符串
            }}, '*');
        }});
        
        // 清除选择
        $('#clear_selection').click(function() {{
            tree.deselect_all();
            selectedNodes = [];
            updateSelectionCount();
        }});
        
        // 全选
        $('#select_all').click(function() {{
            const fileNodes = tree.get_json('#', {{flat: true}})
                .filter(node => node.original && node.original.type === 'file');
            tree.select_node(fileNodes.map(n => n.id));
            updateSelectionCount();
        }});
        
        // 更新选择计数
        function updateSelectionCount() {{
            if (selectedNodes.length > 0) {{
                $('#selection_count').text(`已选择 ${{selectedNodes.length}} 个文件`).show();
            }} else {{
                $('#selection_count').hide();
            }}
        }}
        
        // 初始更新
        updateSelectionCount();
    }});
    </script>
    """.replace("{tree_data}", json.dumps(flat_paths)).replace("{selected_nodes}", json.dumps(st.session_state.js_tree_selected_nodes))

    # 渲染HTML组件
    component_value = html(html_code, height=500, key="js_tree_component")

    # 处理组件返回的值 - 解析JSON字符串
    if component_value:
        try:
            # 尝试解析JSON字符串
            selected_node_data = json.loads(component_value)
            if isinstance(selected_node_data, list):
                st.session_state.js_tree_selected_nodes = selected_node_data
                # 清空原始列表并扩展
                selected_nodes.clear()
                selected_nodes.extend(selected_node_data)
                # 触发回调
                on_node_select(selected_node_data)
        except (json.JSONDecodeError, TypeError):
            # 如果解析失败，保持原样
            pass

    # 显示当前选择状态
    if st.session_state.js_tree_selected_nodes:
        st.success(f"已选择 {len(st.session_state.js_tree_selected_nodes)} 个文件")

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
