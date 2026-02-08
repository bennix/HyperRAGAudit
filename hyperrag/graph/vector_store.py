from __future__ import annotations

import json
import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from hyperrag.models.schemas import BBox, ContentBlock, ParsedDocument, RetrievalResult

log = logging.getLogger("hyperrag.vector")


class VectorStore:
    """ChromaDB wrapper for document chunk embedding and retrieval."""

    def __init__(self, persist_dir: str, collection_name: str):
        log.info(f"[VectorStore] Initializing ChromaDB at {persist_dir}")
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"[VectorStore] Collection '{collection_name}' ready, {self._collection.count()} existing chunks")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def add_document(self, parsed_doc: ParsedDocument) -> int:
        """Index all content blocks from a parsed document."""
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for page in parsed_doc.pages:
            for idx, block in enumerate(page.content_blocks):
                chunk_id = f"{parsed_doc.doc_id}_p{page.page_num}_b{idx}"
                ids.append(chunk_id)
                documents.append(block.text)
                metadatas.append(
                    {
                        "doc_id": parsed_doc.doc_id,
                        "filename": parsed_doc.filename,
                        "page_num": page.page_num,
                        "content_type": block.content_type.value,
                        "bbox_json": block.bbox.model_dump_json(),
                    }
                )

        if ids:
            log.info(f"[VectorStore] Indexing {len(ids)} chunks from '{parsed_doc.filename}'...")
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            log.info(f"[VectorStore] Indexed {len(ids)} chunks. Total in DB: {self._collection.count()}")
        return len(ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_doc_id: str | None = None,
    ) -> list[RetrievalResult]:
        log.info(f"[VectorStore] Query: '{query_text[:80]}' (top {n_results})")
        where = {"doc_id": filter_doc_id} if filter_doc_id else None

        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        out: list[RetrievalResult] = []
        if not results["documents"]:
            log.info("[VectorStore] No results found.")
            return out

        for doc_text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            bbox = None
            bbox_json = meta.get("bbox_json")
            if bbox_json:
                bbox = BBox.model_validate_json(bbox_json)

            out.append(
                RetrievalResult(
                    text=doc_text,
                    doc_id=meta["doc_id"],
                    page_num=meta["page_num"],
                    content_type=meta.get("content_type", "text"),
                    bbox=bbox,
                    score=1.0 - dist,
                )
            )

        for r in out:
            preview = r.text[:80].replace("\n", " ")
            log.info(f"  [hit] score={r.score:.3f} page={r.page_num} | {preview}...")
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count(self) -> int:
        return self._collection.count()

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})
