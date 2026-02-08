from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    """Gemini normalised coordinates [y_min, x_min, y_max, x_max] in 0-1000."""

    y_min: int = Field(ge=0, le=1000)
    x_min: int = Field(ge=0, le=1000)
    y_max: int = Field(ge=0, le=1000)
    x_max: int = Field(ge=0, le=1000)


class PDFRect(BaseModel):
    """Absolute PDF coordinates in points (72 dpi)."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int


# ---------------------------------------------------------------------------
# Parsed content
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class ContentBlock(BaseModel):
    content_type: ContentType
    text: str
    bbox: BBox
    confidence: Optional[float] = None


class PageInfo(BaseModel):
    page_num: int
    width_px: int
    height_px: int
    content_blocks: list[ContentBlock]


class ParsedDocument(BaseModel):
    doc_id: str
    filename: str
    total_pages: int
    pages: list[PageInfo]


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    name: str
    entity_type: str
    source_doc_id: str
    source_page: int
    source_bbox: Optional[BBox] = None


class Relation(BaseModel):
    source_entity: str
    target_entity: str
    relation_type: str
    source_doc_id: str
    source_page: int


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class AuditFinding(BaseModel):
    finding_id: str
    description: str
    severity: str  # "high" | "medium" | "low"
    evidence: list[str]
    source_locations: list[PDFRect]
    related_entities: list[str]


class AuditReport(BaseModel):
    query: str
    findings: list[AuditFinding]
    summary: str


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    text: str
    doc_id: str
    page_num: int
    content_type: str
    bbox: Optional[BBox] = None
    score: float


# ---------------------------------------------------------------------------
# Doc converter helpers
# ---------------------------------------------------------------------------

@dataclass
class PageImage:
    page_num: int
    png_bytes: bytes
    width_px: int
    height_px: int
