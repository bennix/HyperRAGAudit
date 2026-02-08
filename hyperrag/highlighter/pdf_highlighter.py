from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from hyperrag.models.schemas import BBox, PDFRect


class PDFHighlighter:
    """Converts Gemini normalised bboxes to PDF annotations and generates highlighted PDFs."""

    @staticmethod
    def gemini_bbox_to_pdf_rect(
        bbox: BBox,
        page_num: int,
        pdf_page_width_pt: float,
        pdf_page_height_pt: float,
    ) -> PDFRect:
        """Convert Gemini 0-1000 normalised coords to PDF point coordinates.

        Gemini bbox: [y_min, x_min, y_max, x_max] in 0-1000
        PDF Rect: (x0, y0, x1, y1) in points from top-left
        """
        x0 = (bbox.x_min / 1000.0) * pdf_page_width_pt
        y0 = (bbox.y_min / 1000.0) * pdf_page_height_pt
        x1 = (bbox.x_max / 1000.0) * pdf_page_width_pt
        y1 = (bbox.y_max / 1000.0) * pdf_page_height_pt
        return PDFRect(x0=x0, y0=y0, x1=x1, y1=y1, page_num=page_num)

    def highlight(
        self,
        pdf_path: str,
        locations: list[PDFRect],
        output_path: str,
        color: tuple[float, float, float] = (1.0, 1.0, 0.0),
        opacity: float = 0.35,
    ) -> str:
        """Add highlight annotations to a PDF at specified locations.

        Args:
            pdf_path: Path to the source PDF.
            locations: List of PDFRect with page_num and coordinates.
            output_path: Path to save the highlighted PDF.
            color: RGB colour tuple (0-1 range). Default yellow.
            opacity: Highlight opacity (0-1). Default 0.35.

        Returns:
            Path to the highlighted PDF.
        """
        doc = fitz.open(pdf_path)

        for loc in locations:
            if loc.page_num >= len(doc):
                continue
            page = doc[loc.page_num]
            rect = fitz.Rect(loc.x0, loc.y0, loc.x1, loc.y1)
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=color)
            annot.set_opacity(opacity)
            annot.update()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        doc.save(output_path)
        doc.close()
        return output_path

    def highlight_from_bboxes(
        self,
        pdf_path: str,
        bboxes: list[tuple[int, BBox]],
        output_path: str,
        color: tuple[float, float, float] = (1.0, 1.0, 0.0),
    ) -> str:
        """Convenience method: convert Gemini bboxes and highlight in one step.

        Args:
            pdf_path: Path to the source PDF.
            bboxes: List of (page_num, BBox) tuples.
            output_path: Path to save the highlighted PDF.
            color: RGB colour tuple.

        Returns:
            Path to the highlighted PDF.
        """
        doc = fitz.open(pdf_path)
        locations: list[PDFRect] = []

        for page_num, bbox in bboxes:
            if page_num >= len(doc):
                continue
            page = doc[page_num]
            pdf_rect = self.gemini_bbox_to_pdf_rect(
                bbox=bbox,
                page_num=page_num,
                pdf_page_width_pt=page.rect.width,
                pdf_page_height_pt=page.rect.height,
            )
            locations.append(pdf_rect)

        doc.close()
        return self.highlight(pdf_path, locations, output_path, color)
