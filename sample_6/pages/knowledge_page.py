import gradio as gr
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from KnowledgeManager.KnowledgeManagerFactory import KnowledgeManagerFactory
from KnowledgeManager.FAISSKnowledgeManager import FAISSKnowledgeManager

def render_knowledge_page():
    """渲染知识库管理页面"""
    
    with gr.Row():
        # 左侧控制栏
        with gr.Column(scale=1):
            gr.Markdown("### 知识库选择")
            kb_list = FAISSKnowledgeManager.list_knowledge_bases()
            kb_selector = gr.Dropdown(
                label="当前知识库", 
                choices=kb_list,
                value=kb_list[0] if kb_list else None,
                allow_custom_value=True
            )
            refresh_kb_btn = gr.Button("刷新知识库列表", size="sm")
            
            with gr.Accordion("管理操作", open=False):
                new_kb_name = gr.Textbox(label="新知识库名称", placeholder="输入名称后点击创建")
                create_kb_btn = gr.Button("创建新知识库", variant="secondary", size="sm")
                gr.Markdown("---")
                delete_kb_btn = gr.Button("删除选中知识库", variant="stop", size="sm")
            
            status_box = gr.Markdown("状态: **就绪**")
            
            with gr.Accordion("知识库统计", open=True):
                kb_stats_json = gr.JSON(label="详细统计")
                refresh_stats_btn = gr.Button("刷新统计", size="sm")

        # 右侧内容区
        with gr.Column(scale=3):
            with gr.Tabs():
                # Tab 1: 数据入库
                with gr.TabItem("数据入库"):
                    gr.Markdown("#### 上传文档并解析入库")
                    file_input = gr.File(
                        label="选择文档 (支持 PDF, Docx, MD, TXT, HTML)", 
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".md", ".txt", ".html"]
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            chunk_size_slider = gr.Slider(
                                minimum=100, maximum=2000, value=500, step=100, label="片段大小 (Chunk Size)"
                            )
                        with gr.Column():
                            chunk_overlap_slider = gr.Slider(
                                minimum=0, maximum=500, value=50, step=10, label="重叠大小 (Overlap)"
                            )
                    
                    use_hybrid_splitter = gr.Checkbox(label="使用混合标题分割器", value=True)
                    
                    upload_btn = gr.Button("开始解析并入库", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("#### 危险操作")
                    clear_kb_btn = gr.Button("清空当前知识库内容", variant="stop")

                # Tab 2: 检索测试
                with gr.TabItem("检索测试"):
                    gr.Markdown("#### 检索功能验证")
                    with gr.Row():
                        search_query = gr.Textbox(label="检索关键词", placeholder="输入想要搜索的内容...", scale=4)
                        search_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="返回数量", scale=1)
                    
                    with gr.Row():
                        search_mode = gr.Radio(
                            choices=["vector", "bm25", "hybrid"], 
                            value="vector", 
                            label="检索模式"
                        )
                        score_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.3, step=0.05, label="相似度阈值"
                        )
                    
                    search_btn = gr.Button("执行检索", variant="primary")
                    
                    gr.Markdown("#### 检索结果")
                    search_output = gr.Markdown("等待检索...")
                    search_details = gr.JSON(label="详细结果 (Metadata)")

    # --- 逻辑处理函数 ---

    def handle_refresh_kbs():
        kbs = FAISSKnowledgeManager.list_knowledge_bases()
        return gr.update(choices=kbs)

    def handle_create_kb(name):
        if not name:
            return "状态: <span style='color:red'>请输入知识库名称</span>", gr.update()
        try:
            km = KnowledgeManagerFactory.create_knowledge_manager(knowledge_base_name=name)
            km.initialize()
            return f"状态: <span style='color:green'>知识库 '{name}' 创建成功</span>", gr.update(choices=FAISSKnowledgeManager.list_knowledge_bases(), value=name)
        except Exception as e:
            return f"状态: <span style='color:red'>创建失败: {str(e)}</span>", gr.update()

    def handle_delete_kb(name):
        if not name:
            return "状态: <span style='color:red'>请先选择知识库</span>", gr.update()
        try:
            res = FAISSKnowledgeManager.delete_knowledge_base_by_name(name)
            if res.get("success"):
                kbs = FAISSKnowledgeManager.list_knowledge_bases()
                new_val = kbs[0] if kbs else None
                return f"状态: <span style='color:green'>知识库 '{name}' 已删除</span>", gr.update(choices=kbs, value=new_val)
            else:
                return f"状态: <span style='color:red'>删除失败</span>", gr.update()
        except Exception as e:
            return f"状态: <span style='color:red'>异常: {str(e)}</span>", gr.update()

    def handle_get_stats(name):
        if not name:
            return {"error": "未选择知识库"}
        try:
            km = KnowledgeManagerFactory.create_knowledge_manager(knowledge_base_name=name)
            km.initialize()
            return km.get_stats()
        except Exception as e:
            return {"error": str(e)}

    def handle_clear_kb(name):
        if not name:
            return "状态: <span style='color:red'>请先选择知识库</span>"
        try:
            km = KnowledgeManagerFactory.create_knowledge_manager(knowledge_base_name=name)
            km.initialize()
            km.clear_knowledge_base()
            return f"状态: <span style='color:green'>知识库 '{name}' 已清空</span>"
        except Exception as e:
            return f"状态: <span style='color:red'>清空失败: {str(e)}</span>"

    def handle_upload(files, kb_name, c_size, c_overlap, use_hybrid):
        if not kb_name:
            return "状态: <span style='color:red'>请先选择或创建知识库</span>", {}
        if not files:
            return "状态: <span style='color:red'>未上传文件</span>", {}
        
        try:
            km = KnowledgeManagerFactory.create_knowledge_manager(
                knowledge_base_name=kb_name,
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                use_hybrid_splitter=use_hybrid
            )
            km.initialize()
            
            from KnowledgeManager.knowledge_extractor import knowledge_extractor
            import numpy as np
            import faiss
            
            all_chunks = []
            all_metadata = []
            
            for file_obj in files:
                # Gradio 的 file_obj.name 是临时路径
                doc = knowledge_extractor.extract_from_file(file_obj.name)
                if doc:
                    # 修正 doc 中的文件名，原始 filename 在 file_obj.name 路径最后
                    original_name = Path(file_obj.orig_name if hasattr(file_obj, 'orig_name') else file_obj.name).name
                    doc["filename"] = original_name
                    
                    chunks = km.text_splitter.split_text(doc["content"])
                    for chunk in chunks:
                        if len(chunk) >= 3:
                            all_chunks.append(chunk)
                            all_metadata.append({
                                "source": original_name,
                                "filename": original_name,
                                "format": doc["format"],
                                "knowledge_base": kb_name,
                            })
            
            if not all_chunks:
                return "状态: <span style='color:orange'>未从上传文件中提取到有效内容</span>", {}
                
            embeddings = km.embeddings.embed_documents(all_chunks)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            if km.index.ntotal == 0 and embeddings_array.shape[1] != km.dimension:
                km.dimension = embeddings_array.shape[1]
                km.index = faiss.IndexFlatIP(km.dimension)

            km.index.add(embeddings_array)
            km.texts.extend(all_chunks)
            km.metadata.extend(all_metadata)
            km._save_index()
            
            stats = km.get_stats()
            return f"状态: <span style='color:green'>成功入库 {len(all_chunks)} 个片段</span>", stats
        except Exception as e:
            logging.error(f"入库失败: {str(e)}")
            return f"状态: <span style='color:red'>入库失败: {str(e)}</span>", {}

    def handle_search(kb_name, query, k, mode, threshold):
        if not kb_name:
            return "请选择知识库", {}
        if not query:
            return "请输入关键词", {}
            
        try:
            km = KnowledgeManagerFactory.create_knowledge_manager(knowledge_base_name=kb_name)
            
            if mode == "bm25":
                res = km.search_bm25(query, k=k, score_threshold=threshold)
            elif mode == "hybrid":
                res = km.search_hybrid(query, k=k, score_threshold=threshold)
            else:
                res = km.search_with_details(query, k=k, score_threshold=threshold)
                
            if not res.get("success"):
                return f"搜索失败: {res.get('message', '未知错误')}", res
                
            context = res.get("context", "")
            if not context:
                return "未找到相关结果 (低于阈值或库为空)", res
                
            # 格式化展示
            formatted_res = f"### 找到 {res.get('docs_count', 0)} 条结果\n\n"
            formatted_res += context
            
            return formatted_res, res
        except Exception as e:
            return f"检索异常: {str(e)}", {"error": str(e)}

    # --- 绑定事件 ---
    
    refresh_kb_btn.click(handle_refresh_kbs, outputs=kb_selector)
    
    create_kb_btn.click(
        handle_create_kb, 
        inputs=new_kb_name, 
        outputs=[status_box, kb_selector]
    )
    
    delete_kb_btn.click(
        handle_delete_kb, 
        inputs=kb_selector, 
        outputs=[status_box, kb_selector]
    )
    
    refresh_stats_btn.click(handle_get_stats, inputs=kb_selector, outputs=kb_stats_json)
    kb_selector.change(handle_get_stats, inputs=kb_selector, outputs=kb_stats_json)
    
    clear_kb_btn.click(handle_clear_kb, inputs=kb_selector, outputs=status_box).then(
        handle_get_stats, inputs=kb_selector, outputs=kb_stats_json
    )
    
    upload_btn.click(
        handle_upload, 
        inputs=[file_input, kb_selector, chunk_size_slider, chunk_overlap_slider, use_hybrid_splitter], 
        outputs=[status_box, kb_stats_json]
    )
    
    search_btn.click(
        handle_search, 
        inputs=[kb_selector, search_query, search_k_slider, search_mode, score_threshold], 
        outputs=[search_output, search_details]
    )
