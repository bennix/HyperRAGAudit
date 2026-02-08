from __future__ import annotations

import base64
import json
import logging
import re
import time
import uuid
from pathlib import Path

from openai import OpenAI

from hyperrag.models.schemas import (
    BBox,
    ContentBlock,
    ContentType,
    PageImage,
    PageInfo,
    ParsedDocument,
)

log = logging.getLogger("hyperrag.parser")


class GeminiParser:
    """Sends page images to Gemini via ZenMux for OCR + layout analysis."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_tokens: int = 4096,
        prompts_dir: str = "prompts",
    ):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = self._load_prompt(prompts_dir)

    @staticmethod
    def _load_prompt(prompts_dir: str) -> str:
        path = Path(prompts_dir) / "ocr_system.txt"
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def parse_single_page(self, page: PageImage, page_idx: int, total: int) -> PageInfo:
        """Parse a single page image. Returns PageInfo with content blocks."""
        t0 = time.time()
        img_kb = len(page.png_bytes) / 1024
        log.info(
            f"[OCR] Page {page_idx + 1}/{total} | "
            f"size={page.width_px}x{page.height_px} | "
            f"image={img_kb:.0f}KB | sending to {self._model}..."
        )

        page_info = self._parse_page(page)
        elapsed = time.time() - t0

        n_blocks = len(page_info.content_blocks)
        log.info(
            f"[OCR] Page {page_idx + 1}/{total} done in {elapsed:.1f}s | "
            f"{n_blocks} content blocks extracted"
        )

        for block in page_info.content_blocks:
            preview = block.text[:120].replace("\n", " ")
            log.info(
                f"  [{block.content_type.value.upper()}] "
                f"bbox=({block.bbox.x_min},{block.bbox.y_min})-"
                f"({block.bbox.x_max},{block.bbox.y_max}) "
                f"| {preview}..."
            )

        return page_info

    def parse_document(
        self,
        filename: str,
        pages: list[PageImage],
        doc_id: str | None = None,
    ) -> ParsedDocument:
        """Parse all pages (no UI progress). Use parse_single_page for UI progress."""
        doc_id = doc_id or uuid.uuid4().hex[:12]
        total = len(pages)
        log.info("=" * 60)
        log.info(f"[OCR] Start parsing '{filename}' ({total} pages) doc_id={doc_id}")
        log.info("=" * 60)

        parsed_pages = [
            self.parse_single_page(page, i, total)
            for i, page in enumerate(pages)
        ]

        log.info(
            f"[OCR] Completed '{filename}': {total} pages, "
            f"{sum(len(p.content_blocks) for p in parsed_pages)} total blocks"
        )
        log.info("=" * 60)

        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            total_pages=len(parsed_pages),
            pages=parsed_pages,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _parse_page(self, page: PageImage) -> PageInfo:
        b64 = base64.b64encode(page.png_bytes).decode("utf-8")

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all content with bounding boxes from this document page.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        },
                    },
                ],
            },
        ]

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        raw = resp.choices[0].message.content
        return self._parse_response(raw, page)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown fences and surrounding text to isolate JSON."""
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text.strip()

    def _parse_response(self, raw_json: str, page: PageImage) -> PageInfo:
        cleaned = self._extract_json(raw_json)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning(
                f"[OCR] Page {page.page_num + 1}: failed to parse JSON, raw={raw_json[:200]}"
            )
            return PageInfo(
                page_num=page.page_num,
                width_px=page.width_px,
                height_px=page.height_px,
                content_blocks=[],
            )
        blocks: list[ContentBlock] = []

        def _clamp(v: int) -> int:
            return max(0, min(1000, v))

        for item in data.get("content_blocks", []):
            bbox_list = item.get("bbox", [0, 0, 1000, 1000])
            bbox = BBox(
                y_min=_clamp(int(bbox_list[0])),
                x_min=_clamp(int(bbox_list[1])),
                y_max=_clamp(int(bbox_list[2])),
                x_max=_clamp(int(bbox_list[3])),
            )

            ct_raw = item.get("content_type", "text").lower()
            try:
                ct = ContentType(ct_raw)
            except ValueError:
                ct = ContentType.TEXT

            blocks.append(
                ContentBlock(
                    content_type=ct,
                    text=item.get("text", ""),
                    bbox=bbox,
                )
            )

        return PageInfo(
            page_num=page.page_num,
            width_px=page.width_px,
            height_px=page.height_px,
            content_blocks=blocks,
        )
