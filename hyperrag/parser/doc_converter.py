from __future__ import annotations

import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from hyperrag.models.schemas import PageImage


class DocConverter:
    """Converts PDF / image / DOCX files to a list of page images for OCR."""

    def __init__(self, dpi: int = 150, jpeg_quality: int = 75):
        self._dpi = dpi
        self._jpeg_quality = jpeg_quality

    def convert(self, file_path: str) -> list[PageImage]:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._convert_pdf(file_path)
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
            return self._convert_image(file_path)
        if ext in {".docx", ".doc"}:
            return self._convert_docx(file_path)
        raise ValueError(f"Unsupported file format: {ext}")

    def _to_jpeg_bytes(self, pix) -> tuple[bytes, int, int]:
        """Convert a PyMuPDF Pixmap to compressed JPEG bytes via PIL."""
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._jpeg_quality)
        return buf.getvalue(), pix.width, pix.height

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------
    def _convert_pdf(self, file_path: str) -> list[PageImage]:
        doc = fitz.open(file_path)
        pages: list[PageImage] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=self._dpi)
            jpg_bytes, w, h = self._to_jpeg_bytes(pix)
            pages.append(
                PageImage(
                    page_num=page_num,
                    png_bytes=jpg_bytes,  # field name kept for compat, now JPEG
                    width_px=w,
                    height_px=h,
                )
            )
        doc.close()
        return pages

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------
    def _convert_image(self, file_path: str) -> list[PageImage]:
        img = Image.open(file_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._jpeg_quality)
        return [
            PageImage(
                page_num=0,
                png_bytes=buf.getvalue(),
                width_px=img.width,
                height_px=img.height,
            )
        ]

    # ------------------------------------------------------------------
    # DOCX
    # ------------------------------------------------------------------
    def _convert_docx(self, file_path: str) -> list[PageImage]:
        try:
            doc = fitz.open(file_path)
        except Exception:
            return self._convert_docx_fallback(file_path)

        pages: list[PageImage] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=self._dpi)
            jpg_bytes, w, h = self._to_jpeg_bytes(pix)
            pages.append(
                PageImage(
                    page_num=page_num,
                    png_bytes=jpg_bytes,
                    width_px=w,
                    height_px=h,
                )
            )
        doc.close()
        return pages

    def _convert_docx_fallback(self, file_path: str) -> list[PageImage]:
        """Fallback: render DOCX text onto a blank image via PIL."""
        from docx import Document as DocxDocument

        doc = DocxDocument(file_path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        img = Image.new("RGB", (1240, 1754), "white")  # A4 at 150 dpi
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
        except OSError:
            font = ImageFont.load_default()

        draw.text((50, 50), full_text[:5000], fill="black", font=font)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._jpeg_quality)

        return [
            PageImage(
                page_num=0,
                png_bytes=buf.getvalue(),
                width_px=img.width,
                height_px=img.height,
            )
        ]
