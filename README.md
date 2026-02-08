# HyperRAG Audit

Enterprise-grade RAG (Retrieval-Augmented Generation) audit system that combines Vision LLM for document OCR with bounding-box extraction, and Claude for intelligent audit agent orchestration.

Upload mixed documents (PDF, images, Word), extract structured content with spatial coordinates, build a knowledge graph, and perform audit queries with **source tracing** that highlights evidence directly on the original PDF.

## Features

- **Vision OCR with Bounding Boxes** - Converts each document page to images and sends them to a Vision LLM. Every extracted content block carries `[y_min, x_min, y_max, x_max]` coordinates (normalised 0-1000), enabling precise source tracing back to the original page.
- **Knowledge Graph** - Automatically extracts named entities (Person, Company, Amount, Date, Regulation, Document, Location) and their relationships using Claude, stored in a NetworkX graph.
- **Vector Search** - Content blocks are embedded and stored in ChromaDB with full bbox metadata for semantic retrieval.
- **ReAct Audit Agent** - LangGraph-based agent with 5 tools (document search, graph query, path finding, page content, entity listing) that reasons step-by-step to answer audit queries.
- **PDF Highlighting** - Converts extracted bounding boxes to PDF annotation coordinates, producing highlighted PDFs that pinpoint the exact evidence.
- **Interactive Knowledge Graph Visualisation** - Explore entities and relationships in an interactive graph (streamlit-agraph).
- **Bilingual UI** - Switch between Chinese and English in the Streamlit interface.

## Architecture

```
PDF / Image / DOCX
        |
        v
  DocConverter          (PyMuPDF / PIL / python-docx -> JPEG pages)
        |
        v
  GeminiParser          (Vision LLM via ZenMux -> JSON with bboxes)
        |
        v
  ParsedDocument        (structured pages + content blocks + bboxes)
       / \
      /   \
     v     v
VectorStore    EntityExtractor
(ChromaDB)     (Claude via ZenMux -> entities & relations)
     |              |
     v              v
  AuditAgent     KnowledgeGraphStore
  (LangGraph       (NetworkX)
   ReAct Agent)
     |
     v
  AuditReport -> PDFHighlighter -> Highlighted PDF
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/bennix/HyperRAGAudit.git
cd HyperRAGAudit
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get your API key

All LLM calls go through [**ZenMux**](https://zenmux.ai/invite/GBQMC5), a unified AI gateway with an OpenAI-compatible API that routes to 200+ models from OpenAI, Anthropic, Google, and more.

Register and get your API key: **[https://zenmux.ai/invite/GBQMC5](https://zenmux.ai/invite/GBQMC5)**

### 4. Configure

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` and fill in your ZenMux API key:

```yaml
api:
  base_url: "https://zenmux.ai/api/v1"
  api_key: "sk-your-api-key-here"
```

### 5. Run

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## Configuration

All settings are in `config.yaml`:

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| `api` | `base_url` | ZenMux API endpoint | `https://zenmux.ai/api/v1` |
| `api` | `api_key` | Your [ZenMux](https://zenmux.ai/invite/GBQMC5) API key | - |
| `models` | `ocr_model` | Vision model for OCR | `openai/gpt-5.2-chat` |
| `models` | `agent_model` | Agent model for reasoning | `anthropic/claude-sonnet-4` |
| `parser` | `dpi` | PDF rendering resolution | `150` |
| `parser` | `jpeg_quality` | JPEG compression quality | `75` |
| `parser` | `max_tokens` | Max tokens for OCR response | `4096` |
| `vectordb` | `persist_dir` | ChromaDB storage directory | `data/chroma_db` |
| `agent` | `max_iterations` | Max agent reasoning steps | `10` |

You can switch to any model available on ZenMux by changing the model IDs in the config.

## Project Structure

```
HyperRAGAudit/
├── app.py                          # Streamlit entry point
├── config.yaml.example             # Configuration template
├── requirements.txt                # Python dependencies
├── config/
│   └── settings.py                 # YAML config loader
├── hyperrag/
│   ├── models/
│   │   └── schemas.py              # Pydantic data models (BBox, Entity, AuditReport, etc.)
│   ├── parser/
│   │   ├── doc_converter.py        # PDF/Image/DOCX -> JPEG page images
│   │   └── gemini_parser.py        # Vision LLM OCR with bbox extraction
│   ├── graph/
│   │   ├── entity_extractor.py     # Claude entity & relation extraction
│   │   ├── kg_store.py             # NetworkX knowledge graph
│   │   └── vector_store.py         # ChromaDB vector store
│   ├── agent/
│   │   ├── audit_agent.py          # LangGraph ReAct audit agent
│   │   └── tools.py                # 5 LangChain tools for the agent
│   ├── highlighter/
│   │   └── pdf_highlighter.py      # BBox -> PDF highlight annotations
│   └── llm/
│       └── client.py               # ZenMux OpenAI client factory
├── prompts/
│   ├── ocr_system.txt              # Vision OCR system prompt
│   ├── entity_extraction.txt       # Entity extraction prompt
│   └── audit_system.txt            # Audit agent system prompt
└── data/                           # Runtime data (gitignored)
    ├── uploads/
    ├── parsed/
    ├── highlighted/
    └── chroma_db/
```

## Usage

1. **Upload** - Use the sidebar to upload PDF, image (PNG/JPG), or Word (DOCX) files. The system automatically converts, OCR-parses, indexes, and builds a knowledge graph with real-time progress.

2. **Document View** - Browse parsed content with bounding box coordinates alongside the original document.

3. **Audit Query** - Ask audit questions in natural language. The agent searches documents, cross-references the knowledge graph, and produces structured findings with severity levels and source citations.

4. **Knowledge Graph** - Explore extracted entities and their relationships in an interactive visualisation. Filter by entity type or search for specific entities.

## API - ZenMux

HyperRAG uses [**ZenMux**](https://zenmux.ai/invite/GBQMC5) as its unified LLM gateway. ZenMux provides:

- OpenAI-compatible API (`/v1/chat/completions`)
- Access to 200+ models: GPT, Claude, Gemini, Llama, and more
- Single API key, single endpoint, multiple providers
- Built-in load balancing and failover

Get started: **[https://zenmux.ai/invite/GBQMC5](https://zenmux.ai/invite/GBQMC5)**

## Dependencies

- **LLM**: openai, langchain-openai, langchain-core, langgraph
- **Vector DB**: chromadb
- **Documents**: PyMuPDF, python-docx, Pillow
- **Knowledge Graph**: networkx
- **Web UI**: streamlit, streamlit-agraph
- **Config**: pydantic, PyYAML

## License

MIT
