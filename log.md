# 第一阶段：完成基础框架搭建
- 前端界面构建
  - 使用Streamlit快速搭建简单界面
  - 实现基本的对话输入输出功能
  - 实现最基本的模型选择功能
- 接入第一个模型
  - 优先选择接入Ollama（本地部署简单）与Moonshot API（国内访问稳定）
  - 实现简单的文本生成功能

# 第二阶段：知识库基础功能

- Markdown文档处理 
  - 实现Markdown解析和预处理
  - 设计文档分块策略(chunk)
- 向量检索实现
  - 实现向量检索的基本功能
  - 实现向量检索的增量更新
- 基本RAG流程 
  - 实现问题分析
  - 文档检索
  - 上下文整合
  - 结果生成

# 第三阶段：增强知识检索

- 关键词检索实现 
  - 使用jieba等工具进行中文分词
  - 实现TF-IDF或BM25算法

- 树状结构知识体系
  - 基于文件目录结构构建知识树
  - 实现树形知识浏览界面

- 混合检索策略
  - 权重配置界面 
  - 检索结果融合算法