from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

from hyperrag.models.schemas import Entity, PageInfo, ParsedDocument, Relation

log = logging.getLogger("hyperrag.graph")


class EntityRelationExtractor:
    """Uses Claude (via ZenMux) to extract entities and relations from parsed text."""

    def __init__(self, llm: ChatOpenAI, prompts_dir: str = "prompts"):
        self._llm = llm
        self._prompt_template = self._load_prompt(prompts_dir)

    @staticmethod
    def _load_prompt(prompts_dir: str) -> str:
        path = Path(prompts_dir) / "entity_extraction.txt"
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def extract(
        self, parsed_doc: ParsedDocument
    ) -> tuple[list[Entity], list[Relation]]:
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []
        total = len(parsed_doc.pages)

        log.info("=" * 60)
        log.info(f"[KG] Entity extraction for '{parsed_doc.filename}' ({total} pages)")
        log.info("=" * 60)

        for page in parsed_doc.pages:
            page_text = self._get_page_text(page)
            if not page_text.strip():
                log.info(f"[KG] Page {page.page_num + 1}/{total}: empty, skipping")
                continue

            t0 = time.time()
            log.info(f"[KG] Page {page.page_num + 1}/{total}: extracting entities with Claude...")

            entities, relations = self._extract_from_page(
                doc_id=parsed_doc.doc_id,
                page_num=page.page_num,
                page_text=page_text,
                page=page,
            )
            elapsed = time.time() - t0

            log.info(
                f"[KG] Page {page.page_num + 1}/{total} done in {elapsed:.1f}s | "
                f"{len(entities)} entities, {len(relations)} relations"
            )
            for ent in entities:
                log.info(f"  [entity] {ent.entity_type}: {ent.name}")
            for rel in relations:
                log.info(f"  [relation] {rel.source_entity} --{rel.relation_type}--> {rel.target_entity}")

            all_entities.extend(entities)
            all_relations.extend(relations)

        all_entities = self._deduplicate_entities(all_entities)
        log.info(
            f"[KG] Extraction complete: {len(all_entities)} unique entities, "
            f"{len(all_relations)} relations"
        )
        log.info("=" * 60)
        return all_entities, all_relations

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _get_page_text(page: PageInfo) -> str:
        return "\n\n".join(block.text for block in page.content_blocks)

    def _extract_from_page(
        self,
        doc_id: str,
        page_num: int,
        page_text: str,
        page: PageInfo,
    ) -> tuple[list[Entity], list[Relation]]:
        prompt = self._prompt_template.replace("{text}", page_text)

        response = self._llm.invoke(prompt)
        raw = response.content

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning(f"[KG] Page {page_num + 1}: JSON parse failed, raw={raw[:200]}")
            return [], []

        # Build lookup: entity name -> bbox from content blocks
        name_bbox_map = {}
        for block in page.content_blocks:
            for ent_data in data.get("entities", []):
                if ent_data.get("name") and ent_data["name"] in block.text:
                    name_bbox_map[ent_data["name"]] = block.bbox

        entities = [
            Entity(
                name=e["name"],
                entity_type=e["entity_type"],
                source_doc_id=doc_id,
                source_page=page_num,
                source_bbox=name_bbox_map.get(e["name"]),
            )
            for e in data.get("entities", [])
            if "name" in e and "entity_type" in e
        ]

        relations = [
            Relation(
                source_entity=r["source"],
                target_entity=r["target"],
                relation_type=r["relation_type"],
                source_doc_id=doc_id,
                source_page=page_num,
            )
            for r in data.get("relations", [])
            if "source" in r and "target" in r and "relation_type" in r
        ]

        return entities, relations

    @staticmethod
    def _deduplicate_entities(entities: list[Entity]) -> list[Entity]:
        seen: dict[str, Entity] = {}
        for ent in entities:
            key = (ent.name.lower(), ent.entity_type.lower())
            if key not in seen:
                seen[key] = ent
        return list(seen.values())
