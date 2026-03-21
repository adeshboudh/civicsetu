from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import structlog

log = structlog.get_logger(__name__)


@dataclass
class RawPage:
    """Raw text content of a single PDF page."""
    page_number: int
    text: str
    is_empty: bool = False


@dataclass
class ParsedDocument:
    """Full parsed output of a PDF — ready for chunking."""
    source_path: str
    total_pages: int
    pages: list[RawPage] = field(default_factory=list)
    full_text: str = ""


class PDFParser:
    """
    Extracts text from PDFs using PyMuPDF.
    Handles text-based PDFs only (Phase 0).
    Scanned PDFs fall through to is_scanned=True — Surya OCR handles in Phase 2.
    """

    # Minimum characters per page to not be considered scanned/empty
    MIN_CHARS_PER_PAGE = 50

    @staticmethod
    def parse(source: str | Path, max_pages: int | None = None) -> ParsedDocument:
        """
        Parse a PDF from a local file path or URL.
        max_pages: if set, only the first N pages are processed.
        Returns ParsedDocument with per-page text.
        """
        import fitz
        
        source = Path(source)
        doc = fitz.open(str(source))

        log.info("parsing_pdf", source=source)

        all_pages = list(doc)
        if max_pages is not None:
            all_pages = all_pages[:max_pages]

        pages = []
        for page in all_pages:
            text = page.get_text("text") or ""
            pages.append(RawPage(
                page_number=page.number + 1,
                text=text.strip(),
                is_empty=len(text.strip()) < 30,
            ))

        full_text = "\n\n".join(p.text for p in pages if not p.is_empty)
        scanned_pages = sum(1 for p in pages if p.is_empty)

        if scanned_pages > len(pages) * 0.5:
            log.warning(
                "mostly_scanned_pdf",
                source=str(source),
                scanned_pages=scanned_pages,
                total_pages=len(pages),
            )

        log.info(
            "pdf_parsed",
            source=str(source),
            total_pages=len(pages),
            scanned_pages=scanned_pages,
            total_chars=len(full_text),
        )

        return ParsedDocument(
            source_path=str(source),
            full_text=full_text,
            total_pages=len(pages),
            pages=pages,
        )

    @staticmethod
    def parse_from_bytes(pdf_bytes: bytes, name: str = "document") -> ParsedDocument:
        """Parse a PDF from raw bytes — used when downloading from URLs."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        tmp_path = f"/tmp/{name}.pdf"
        doc.save(tmp_path)
        doc.close()
        return PDFParser.parse(tmp_path)
