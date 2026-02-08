from __future__ import annotations

import json
from typing import Callable

from langchain_core.tools import tool

from hyperrag.graph.kg_store import KnowledgeGraphStore
from hyperrag.graph.vector_store import VectorStore
from hyperrag.models.schemas import ParsedDocument


def create_tools(
    vector_store: VectorStore,
    kg_store: KnowledgeGraphStore,
    parsed_docs: dict[str, ParsedDocument],
) -> list:
    """Factory that creates agent tools with injected dependencies."""

    @tool
    def search_documents(query: str, doc_id: str = "") -> str:
        """Search document chunks by semantic similarity.
        Returns top relevant text passages with page numbers and bounding boxes.
        Optionally filter by doc_id.
        """
        results = vector_store.query(
            query_text=query,
            n_results=5,
            filter_doc_id=doc_id if doc_id else None,
        )
        if not results:
            return "No relevant documents found."

        output = []
        for r in results:
            entry = {
                "text": r.text[:500],
                "doc_id": r.doc_id,
                "page_num": r.page_num,
                "content_type": r.content_type,
                "score": round(r.score, 3),
            }
            if r.bbox:
                entry["bbox"] = r.bbox.model_dump()
            output.append(entry)
        return json.dumps(output, ensure_ascii=False, indent=2)

    @tool
    def query_knowledge_graph(entity_name: str) -> str:
        """Look up an entity in the knowledge graph.
        Returns the entity's type, source document, and all related entities/relations.
        """
        result = kg_store.query_entity(entity_name)
        if result is None:
            return f"Entity '{entity_name}' not found in knowledge graph."
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    @tool
    def query_graph_path(source_entity: str, target_entity: str) -> str:
        """Find the relationship path between two entities in the knowledge graph."""
        path = kg_store.get_path(source_entity, target_entity)
        if path is None:
            return f"No path found between '{source_entity}' and '{target_entity}'."
        return " -> ".join(path)

    @tool
    def get_page_content(doc_id: str, page_num: int) -> str:
        """Retrieve the full parsed content of a specific page."""
        doc = parsed_docs.get(doc_id)
        if doc is None:
            return f"Document '{doc_id}' not found."

        for page in doc.pages:
            if page.page_num == page_num:
                blocks = []
                for block in page.content_blocks:
                    blocks.append(
                        f"[{block.content_type.value}] {block.text}"
                    )
                return "\n\n".join(blocks)
        return f"Page {page_num} not found in document '{doc_id}'."

    @tool
    def list_entities_by_type(entity_type: str) -> str:
        """List all entities of a given type (e.g., 'Company', 'Amount', 'Regulation')."""
        entities = kg_store.query_by_type(entity_type)
        if not entities:
            return f"No entities of type '{entity_type}' found."
        return json.dumps(entities, ensure_ascii=False, indent=2, default=str)

    return [
        search_documents,
        query_knowledge_graph,
        query_graph_path,
        get_page_content,
        list_entities_by_type,
    ]
