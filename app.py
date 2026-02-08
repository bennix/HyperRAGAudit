"""HyperRAG Audit System — Streamlit Application."""
from __future__ import annotations

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import base64
import json
import logging
import uuid
from pathlib import Path

# Configure terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import streamlit as st

from config.settings import Settings
from hyperrag.agent.audit_agent import AuditAgent
from hyperrag.agent.tools import create_tools
from hyperrag.graph.entity_extractor import EntityRelationExtractor
from hyperrag.graph.kg_store import KnowledgeGraphStore
from hyperrag.graph.vector_store import VectorStore
from hyperrag.highlighter.pdf_highlighter import PDFHighlighter
from hyperrag.llm.client import LLMClientFactory
from hyperrag.models.schemas import BBox, ParsedDocument
from hyperrag.parser.doc_converter import DocConverter
from hyperrag.parser.gemini_parser import GeminiParser


# ======================================================================
# i18n — Translation Dictionaries
# ======================================================================

I18N = {
    "en": {
        # App
        "page_title": "HyperRAG Audit",
        # Sidebar
        "sidebar_title": "HyperRAG Audit",
        "settings_header": "Settings",
        "language": "Language",
        "upload_header": "Upload Documents",
        "upload_label": "PDF, Image, or Word",
        "upload_btn": "Upload & Parse",
        "already_uploaded": "'{name}' already uploaded.",
        "batch_progress": "Overall: file {cur}/{total}",
        "file_step_convert": "Step 1/4 — Converting",
        "file_step_ocr": "Step 2/4 — OCR Parsing",
        "file_step_index": "Step 3/4 — Vectorizing",
        "file_step_kg": "Step 4/4 — Knowledge Graph",
        "file_step_done": "Done",
        "file_step_pending": "Pending",
        "file_step_skipped": "Skipped (duplicate)",
        "converting": "Converting '{name}'...",
        "converting_done": "'{name}' converted: {n} pages",
        "ocr_parsing": "OCR parsing '{name}' ({n} pages)...",
        "ocr_progress": "OCR '{name}': page {cur}/{total}...",
        "ocr_done": "OCR complete: '{name}' ({total} pages)",
        "indexing": "Indexing '{name}'...",
        "indexing_done": "Indexed '{name}': {chunks} chunks",
        "upload_success": "'{name}': {pages} pages, {chunks} chunks indexed.",
        "building_kg": "Building knowledge graph for '{name}'...",
        "kg_progress": "KG '{name}': extracting page {cur}/{total}...",
        "kg_done": "KG complete: {nodes} entities, {edges} relations",
        "kg_updated": "KG updated: {nodes} entities, {edges} relations.",
        "all_done": "'{name}' done: {pages} pages, {chunks} chunks, {nodes} entities, {edges} relations.",
        "batch_done": "All {total} files processed.",
        "docs_header": "Documents",
        "no_docs": "No documents uploaded yet.",
        "rebuild_kg_btn": "Rebuild Knowledge Graph",
        "rebuilding_kg": "Rebuilding knowledge graph...",
        "kg_rebuilt": "KG rebuilt: {nodes} entities, {edges} relations.",
        # Tabs
        "tab_docs": "Document View",
        "tab_audit": "Audit Query",
        "tab_graph": "Knowledge Graph",
        # Document View
        "upload_first": "Upload documents from the sidebar to get started.",
        "select_doc": "Select document",
        "original_doc": "Original Document",
        "parsed_content": "Parsed Content",
        "page_n": "Page {n}",
        # Audit
        "parse_first": "Upload and parse documents first.",
        "query_label": "Enter your audit query",
        "query_placeholder": "e.g. Are there any discrepancies between the contract amounts and invoice totals?",
        "run_audit_btn": "Run Audit",
        "agent_reasoning": "Agent Reasoning",
        "tool_result": "Tool result",
        "audit_findings": "Audit Findings",
        "no_issues": "No issues found.",
        "view_in_pdf": "View in PDF",
        "summary": "Summary",
        "no_pdf_source": "No PDF source available for highlighting.",
        # Knowledge Graph
        "kg_not_built": "Upload and parse documents to build the knowledge graph.",
        "filter_entity_type": "Filter by entity type",
        "search_entity": "Search entity",
        "no_entities": "No entities to display with current filters.",
        "install_agraph": "Install `streamlit-agraph` for interactive graph visualisation.",
        "entities": "Entities",
        "entity_details": "Entity Details",
        "entity_not_found": "Entity '{name}' not found.",
        "enter_entity": "Enter an entity name above to see details.",
    },
    "zh": {
        # App
        "page_title": "HyperRAG 智能审计",
        # Sidebar
        "sidebar_title": "HyperRAG 智能审计",
        "settings_header": "设置",
        "language": "语言",
        "upload_header": "上传文档",
        "upload_label": "PDF、图片或 Word",
        "upload_btn": "上传并解析",
        "already_uploaded": "'{name}' 已上传过。",
        "batch_progress": "总进度：第 {cur}/{total} 个文件",
        "file_step_convert": "步骤 1/4 — 格式转换",
        "file_step_ocr": "步骤 2/4 — OCR 解析",
        "file_step_index": "步骤 3/4 — 向量化",
        "file_step_kg": "步骤 4/4 — 知识图谱",
        "file_step_done": "已完成",
        "file_step_pending": "等待中",
        "file_step_skipped": "已跳过（重复）",
        "converting": "正在转换 '{name}'...",
        "converting_done": "'{name}' 转换完成：{n} 页",
        "ocr_parsing": "OCR 解析 '{name}'（{n} 页）...",
        "ocr_progress": "OCR '{name}'：正在解析第 {cur}/{total} 页...",
        "ocr_done": "OCR 完成：'{name}'（{total} 页）",
        "indexing": "正在索引 '{name}'...",
        "indexing_done": "'{name}' 索引完成：{chunks} 个文本块",
        "upload_success": "'{name}'：{pages} 页，{chunks} 个文本块已索引。",
        "building_kg": "正在为 '{name}' 构建知识图谱...",
        "kg_progress": "知识图谱 '{name}'：正在提取第 {cur}/{total} 页...",
        "kg_done": "知识图谱完成：{nodes} 个实体，{edges} 个关系",
        "kg_updated": "知识图谱已更新：{nodes} 个实体，{edges} 个关系。",
        "all_done": "'{name}' 处理完成：{pages} 页，{chunks} 个文本块，{nodes} 个实体，{edges} 个关系。",
        "batch_done": "全部 {total} 个文件处理完成。",
        "docs_header": "已上传文档",
        "no_docs": "暂无已上传的文档。",
        "rebuild_kg_btn": "重建知识图谱",
        "rebuilding_kg": "正在重建知识图谱...",
        "kg_rebuilt": "知识图谱已重建：{nodes} 个实体，{edges} 个关系。",
        # Tabs
        "tab_docs": "文档预览",
        "tab_audit": "审计查询",
        "tab_graph": "知识图谱",
        # Document View
        "upload_first": "请先从侧边栏上传文档。",
        "select_doc": "选择文档",
        "original_doc": "原始文档",
        "parsed_content": "解析内容",
        "page_n": "第 {n} 页",
        # Audit
        "parse_first": "请先上传并解析文档。",
        "query_label": "输入审计查询",
        "query_placeholder": "例如：合同金额与发票总额是否存在差异？",
        "run_audit_btn": "执行审计",
        "agent_reasoning": "Agent 推理过程",
        "tool_result": "工具返回结果",
        "audit_findings": "审计发现",
        "no_issues": "未发现问题。",
        "view_in_pdf": "在 PDF 中查看",
        "summary": "总结",
        "no_pdf_source": "没有可用的 PDF 来源进行高亮。",
        # Knowledge Graph
        "kg_not_built": "请先上传并解析文档以构建知识图谱。",
        "filter_entity_type": "按实体类型筛选",
        "search_entity": "搜索实体",
        "no_entities": "当前筛选条件下没有实体可显示。",
        "install_agraph": "请安装 `streamlit-agraph` 以使用交互式图谱可视化。",
        "entities": "实体列表",
        "entity_details": "实体详情",
        "entity_not_found": "未找到实体 '{name}'。",
        "enter_entity": "在上方输入实体名称查看详情。",
    },
}


def t(key: str, **kwargs) -> str:
    """Get translated string for current language."""
    lang = st.session_state.get("lang", "zh")
    text = I18N.get(lang, I18N["en"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


# ======================================================================
# Initialisation
# ======================================================================

def _init_session_state() -> None:
    if "initialised" in st.session_state:
        return

    settings = Settings()

    for d in [settings.upload_dir, settings.parsed_dir,
              settings.highlighted_dir, settings.chroma_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    factory = LLMClientFactory(
        api_key=settings.zenmux_api_key,
        base_url=settings.zenmux_base_url,
    )

    st.session_state.settings = settings
    st.session_state.factory = factory
    st.session_state.openai_client = factory.get_openai_client()
    st.session_state.claude_llm = factory.get_langchain_llm(
        model=settings.claude_model
    )

    st.session_state.doc_converter = DocConverter(
        dpi=settings.ocr_dpi,
        jpeg_quality=settings.jpeg_quality,
    )
    st.session_state.gemini_parser = GeminiParser(
        client=st.session_state.openai_client,
        model=settings.gemini_model,
        max_tokens=settings.ocr_max_tokens,
        prompts_dir=settings.prompts_dir,
    )
    st.session_state.vector_store = VectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
    )
    st.session_state.kg_store = KnowledgeGraphStore()
    st.session_state.entity_extractor = EntityRelationExtractor(
        llm=st.session_state.claude_llm,
        prompts_dir=settings.prompts_dir,
    )
    st.session_state.highlighter = PDFHighlighter()

    st.session_state.parsed_docs: dict[str, ParsedDocument] = {}
    st.session_state.uploaded_files: dict[str, str] = {}
    st.session_state.file_paths: dict[str, str] = {}
    st.session_state.kg_built = False

    # Defaults for language and theme
    if "lang" not in st.session_state:
        st.session_state.lang = "zh"
    st.session_state.initialised = True


# ======================================================================
# Sidebar
# ======================================================================

def render_sidebar() -> None:
    st.header(t("sidebar_title"))

    # --- Settings: Language + Theme ---
    st.subheader(t("settings_header"))
    lang_options = {"中文": "zh", "English": "en"}
    current_label = [k for k, v in lang_options.items() if v == st.session_state.lang][0]
    selected_lang = st.selectbox(
        t("language"),
        list(lang_options.keys()),
        index=list(lang_options.keys()).index(current_label),
        key="lang_select",
    )
    if lang_options[selected_lang] != st.session_state.lang:
        st.session_state.lang = lang_options[selected_lang]
        st.rerun()

    st.divider()

    # --- File upload ---
    st.subheader(t("upload_header"))
    uploaded = st.file_uploader(
        t("upload_label"),
        type=["pdf", "png", "jpg", "jpeg", "docx"],
        accept_multiple_files=True,
    )

    if uploaded and st.button(t("upload_btn"), type="primary"):
        settings: Settings = st.session_state.settings

        # Filter out already-uploaded files
        new_files = []
        for f in uploaded:
            if f.name in st.session_state.uploaded_files:
                st.info(t("already_uploaded", name=f.name))
            else:
                new_files.append(f)

        if not new_files:
            pass
        else:
            total_files = len(new_files)

            # Overall batch progress
            batch_bar = st.progress(0, text=t("batch_progress", cur=0, total=total_files))

            # Create a replaceable placeholder for each file
            file_placeholders = {}
            for f in new_files:
                ph = st.empty()
                ph.markdown(f"**{f.name}** — :gray[{t('file_step_pending')}]")
                file_placeholders[f.name] = ph

            for file_idx, f in enumerate(new_files):
                batch_bar.progress(
                    file_idx / total_files,
                    text=t("batch_progress", cur=file_idx + 1, total=total_files),
                )

                save_path = os.path.join(settings.upload_dir, f.name)
                with open(save_path, "wb") as fp:
                    fp.write(f.getbuffer())

                # Replace the pending placeholder with a live status widget
                file_status = file_placeholders[f.name].status(
                    f"**{f.name}** — {t('file_step_convert')}",
                    expanded=True,
                )

                with file_status:
                    # --- Step 1: Convert ---
                    st.write(t("converting", name=f.name))
                    pages = st.session_state.doc_converter.convert(save_path)
                    st.write(t("converting_done", name=f.name, n=len(pages)))

                file_status.update(
                    label=f"**{f.name}** — {t('file_step_ocr')}",
                    state="running",
                )

                with file_status:
                    # --- Step 2: OCR with page-by-page progress ---
                    doc_id = uuid.uuid4().hex[:12]
                    total_pages = len(pages)
                    parsed_pages = []

                    ocr_bar = st.progress(0, text=t("ocr_progress", cur=0, total=total_pages, name=f.name))

                    for i, page in enumerate(pages):
                        ocr_bar.progress(
                            i / total_pages,
                            text=t("ocr_progress", cur=i + 1, total=total_pages, name=f.name),
                        )
                        page_info = st.session_state.gemini_parser.parse_single_page(page, i, total_pages)
                        parsed_pages.append(page_info)

                    ocr_bar.progress(1.0, text=t("ocr_done", name=f.name, total=total_pages))

                parsed = ParsedDocument(
                    doc_id=doc_id,
                    filename=f.name,
                    total_pages=total_pages,
                    pages=parsed_pages,
                )

                # Save parsed JSON
                parsed_path = os.path.join(settings.parsed_dir, f"{doc_id}.json")
                Path(parsed_path).write_text(
                    parsed.model_dump_json(indent=2), encoding="utf-8"
                )

                file_status.update(
                    label=f"**{f.name}** — {t('file_step_index')}",
                    state="running",
                )

                with file_status:
                    # --- Step 3: Vectorize / Index ---
                    st.write(t("indexing", name=f.name))
                    n = st.session_state.vector_store.add_document(parsed)
                    st.write(t("indexing_done", name=f.name, chunks=n))

                st.session_state.parsed_docs[doc_id] = parsed
                st.session_state.uploaded_files[f.name] = doc_id
                st.session_state.file_paths[doc_id] = save_path

                file_status.update(
                    label=f"**{f.name}** — {t('file_step_kg')}",
                    state="running",
                )

                with file_status:
                    # --- Step 4: Build KG ---
                    kg_bar = st.progress(0, text=t("kg_progress", cur=0, total=total_pages, name=f.name))
                    all_entities = []
                    all_relations = []
                    for i, page in enumerate(parsed.pages):
                        kg_bar.progress(
                            i / total_pages,
                            text=t("kg_progress", cur=i + 1, total=total_pages, name=f.name),
                        )
                        page_text = "\n".join(b.text for b in page.content_blocks)
                        if page_text.strip():
                            ents, rels = st.session_state.entity_extractor._extract_from_page(
                                doc_id=doc_id, page_num=page.page_num,
                                page_text=page_text, page=page,
                            )
                            all_entities.extend(ents)
                            all_relations.extend(rels)

                    all_entities = st.session_state.entity_extractor._deduplicate_entities(all_entities)
                    st.session_state.kg_store.add_entities(all_entities)
                    st.session_state.kg_store.add_relations(all_relations)
                    st.session_state.kg_built = True

                    kg_bar.progress(1.0, text=t("kg_done",
                                                 nodes=st.session_state.kg_store.node_count(),
                                                 edges=st.session_state.kg_store.edge_count()))

                # Mark file complete
                file_status.update(
                    label=f"**{f.name}** — {t('file_step_done')} ({total_pages}p, {n} chunks)",
                    state="complete",
                    expanded=False,
                )

            # Batch complete
            batch_bar.progress(1.0, text=t("batch_done", total=total_files))
            st.success(t("batch_done", total=total_files))

    # --- Document list ---
    st.divider()
    st.subheader(t("docs_header"))
    if st.session_state.uploaded_files:
        for fname, doc_id in st.session_state.uploaded_files.items():
            doc = st.session_state.parsed_docs[doc_id]
            st.text(f"{fname} ({doc.total_pages}p) [{doc_id[:6]}]")
    else:
        st.caption(t("no_docs"))

    # --- Rebuild KG ---
    st.divider()
    if st.session_state.parsed_docs:
        if st.button(t("rebuild_kg_btn")):
            with st.spinner(t("rebuilding_kg")):
                st.session_state.kg_store = KnowledgeGraphStore()
                for doc_id, parsed in st.session_state.parsed_docs.items():
                    entities, relations = st.session_state.entity_extractor.extract(parsed)
                    st.session_state.kg_store.add_entities(entities)
                    st.session_state.kg_store.add_relations(relations)
            st.session_state.kg_built = True
            st.success(t("kg_rebuilt",
                         nodes=st.session_state.kg_store.node_count(),
                         edges=st.session_state.kg_store.edge_count()))


# ======================================================================
# Tab 1: Document View
# ======================================================================

def render_document_view() -> None:
    if not st.session_state.parsed_docs:
        st.info(t("upload_first"))
        return

    doc_options = {
        f"{pd.filename} [{did[:6]}]": did
        for did, pd in st.session_state.parsed_docs.items()
    }
    selected_label = st.selectbox(t("select_doc"), list(doc_options.keys()))
    if not selected_label:
        return
    doc_id = doc_options[selected_label]
    parsed = st.session_state.parsed_docs[doc_id]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(t("original_doc"))
        file_path = st.session_state.file_paths.get(doc_id)
        if file_path and file_path.lower().endswith(".pdf"):
            try:
                pdf_bytes = Path(file_path).read_bytes()
                b64 = base64.b64encode(pdf_bytes).decode()
                pdf_html = (
                    f'<iframe src="data:application/pdf;base64,{b64}" '
                    f'width="100%" height="600" type="application/pdf"></iframe>'
                )
                st.markdown(pdf_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))
        elif file_path:
            st.image(file_path, use_container_width=True)

    with col2:
        st.subheader(t("parsed_content"))
        for page in parsed.pages:
            with st.expander(t("page_n", n=page.page_num + 1), expanded=(page.page_num == 0)):
                for block in page.content_blocks:
                    badge = f"**[{block.content_type.value.upper()}]**"
                    st.markdown(badge)
                    st.text(block.text[:1000])
                    st.caption(
                        f"BBox: y[{block.bbox.y_min}-{block.bbox.y_max}] "
                        f"x[{block.bbox.x_min}-{block.bbox.x_max}]"
                    )
                    st.divider()


# ======================================================================
# Tab 2: Audit Query
# ======================================================================

def render_audit_query() -> None:
    if not st.session_state.parsed_docs:
        st.info(t("parse_first"))
        return

    query = st.text_area(
        t("query_label"),
        placeholder=t("query_placeholder"),
        height=100,
    )

    if st.button(t("run_audit_btn"), type="primary") and query.strip():
        tools = create_tools(
            vector_store=st.session_state.vector_store,
            kg_store=st.session_state.kg_store,
            parsed_docs=st.session_state.parsed_docs,
        )
        agent = AuditAgent(
            llm=st.session_state.claude_llm,
            tools=tools,
            prompts_dir=st.session_state.settings.prompts_dir,
            max_iterations=st.session_state.settings.agent_max_iterations,
        )

        reasoning_container = st.container()
        with reasoning_container:
            st.subheader(t("agent_reasoning"))
            final_content = ""

            for step in agent.stream(query):
                if step["type"] == "tool_call":
                    st.info(f"Tool: {step['content']}")
                elif step["type"] == "tool_result":
                    with st.expander(t("tool_result"), expanded=False):
                        st.code(step["content"])
                elif step["type"] == "thinking":
                    final_content = step["content"]

        if final_content:
            report = agent._parse_report(query, final_content)

            st.divider()
            st.subheader(t("audit_findings"))

            if not report.findings:
                st.success(t("no_issues"))
                st.markdown(report.summary)
            else:
                for finding in report.findings:
                    severity_colors = {"high": "red", "medium": "orange", "low": "blue"}
                    color = severity_colors.get(finding.severity, "gray")
                    st.markdown(
                        f"**:{color}[{finding.severity.upper()}]** "
                        f"{finding.description}"
                    )
                    for ev in finding.evidence:
                        st.caption(f"> {ev}")

                    if finding.source_locations:
                        btn_key = f"highlight_{finding.finding_id}"
                        if st.button(t("view_in_pdf"), key=btn_key):
                            _show_highlighted_pdf(finding)

                st.divider()
                st.subheader(t("summary"))
                st.markdown(report.summary)


def _show_highlighted_pdf(finding) -> None:
    settings: Settings = st.session_state.settings
    for loc in finding.source_locations:
        for doc_id, path in st.session_state.file_paths.items():
            if path.lower().endswith(".pdf"):
                output_path = os.path.join(
                    settings.highlighted_dir,
                    f"highlighted_{finding.finding_id}_{doc_id[:6]}.pdf",
                )
                st.session_state.highlighter.highlight(
                    pdf_path=path,
                    locations=finding.source_locations,
                    output_path=output_path,
                )
                pdf_bytes = Path(output_path).read_bytes()
                b64 = base64.b64encode(pdf_bytes).decode()
                pdf_html = (
                    f'<iframe src="data:application/pdf;base64,{b64}" '
                    f'width="100%" height="500" type="application/pdf"></iframe>'
                )
                st.markdown(pdf_html, unsafe_allow_html=True)
                return
    st.warning(t("no_pdf_source"))


# ======================================================================
# Tab 3: Knowledge Graph
# ======================================================================

def render_knowledge_graph() -> None:
    if not st.session_state.kg_built:
        st.info(t("kg_not_built"))
        return

    kg: KnowledgeGraphStore = st.session_state.kg_store

    col1, col2 = st.columns([1, 1])
    with col1:
        all_types = kg.all_entity_types()
        selected_types = st.multiselect(
            t("filter_entity_type"), all_types, default=all_types
        )
    with col2:
        search_entity = st.text_input(t("search_entity"))

    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        nodes_data = kg.all_nodes()
        edges_data = kg.all_edges()

        if selected_types:
            nodes_data = [
                n for n in nodes_data
                if n.get("entity_type", "") in selected_types
            ]
        if search_entity:
            sub = kg.query_neighbors(search_entity, depth=2)
            node_names = {n["name"] for n in sub["nodes"]}
            nodes_data = [n for n in nodes_data if n["name"] in node_names]
            edges_data = [
                e for e in edges_data
                if e["source"] in node_names and e["target"] in node_names
            ]

        type_colors = {
            "Person": "#2e7d32",
            "Company": "#1565c0",
            "Amount": "#e65100",
            "Date": "#6a1b9a",
            "Regulation": "#c62828",
            "Document": "#37474f",
            "Location": "#00838f",
        }
        default_node_color = "#999999"
        edge_color = "#aaaaaa"
        font_color = "#1a1a1a"

        # Map entity names to safe numeric IDs to avoid URL path issues with CJK chars
        visible_names = {n["name"] for n in nodes_data}
        name_to_id = {n["name"]: f"n{i}" for i, n in enumerate(nodes_data)}

        nodes = [
            Node(
                id=name_to_id[n["name"]],
                label=n["name"][:20],
                size=25,
                color=type_colors.get(n.get("entity_type", ""), default_node_color),
                title=f"{n.get('entity_type', '')}: {n['name']}",
                font={"color": font_color, "size": 14},
            )
            for n in nodes_data
        ]
        edges = [
            Edge(
                source=name_to_id[e["source"]],
                target=name_to_id[e["target"]],
                label=e.get("relation_type", ""),
                color=edge_color,
                font={"color": font_color, "size": 11},
            )
            for e in edges_data
            if e["source"] in visible_names and e["target"] in visible_names
        ]

        config = Config(
            width=900,
            height=500,
            directed=True,
            physics=True,
            hierarchical=False,
            backgroundColor="#ffffff",
        )

        if nodes:
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.info(t("no_entities"))

    except ImportError:
        st.warning(t("install_agraph"))
        st.subheader(t("entities"))
        nodes_data = kg.all_nodes()
        if selected_types:
            nodes_data = [
                n for n in nodes_data
                if n.get("entity_type", "") in selected_types
            ]
        for n in nodes_data:
            st.text(f"[{n.get('entity_type', '?')}] {n['name']}")

    st.divider()
    st.subheader(t("entity_details"))
    if search_entity:
        detail = kg.query_entity(search_entity)
        if detail:
            st.json(detail)
        else:
            st.caption(t("entity_not_found", name=search_entity))
    else:
        st.caption(t("enter_entity"))


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    st.set_page_config(
        page_title="HyperRAG Audit",
        page_icon="",
        layout="wide",
    )

    _init_session_state()

    with st.sidebar:
        render_sidebar()

    tab_docs, tab_audit, tab_graph = st.tabs(
        [t("tab_docs"), t("tab_audit"), t("tab_graph")]
    )

    with tab_docs:
        render_document_view()

    with tab_audit:
        render_audit_query()

    with tab_graph:
        render_knowledge_graph()


if __name__ == "__main__":
    main()
