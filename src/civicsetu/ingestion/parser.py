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
    def parse(source: str | Path) -> ParsedDocument:
        """
        Parse a PDF from a local file path or URL.
        Returns ParsedDocument with per-page text.
        """
        source = str(source)
        log.info("parsing_pdf", source=source)

        doc = pymupdf.open(source)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Normalize whitespace — collapse multiple blank lines to one
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()

            pages.append(RawPage(
                page_number=page_num + 1,
                text=text,
                is_empty=len(text) < PDFParser.MIN_CHARS_PER_PAGE,
            ))

        doc.close()

        full_text = "\n\n".join(p.text for p in pages if not p.is_empty)
        scanned_pages = sum(1 for p in pages if p.is_empty)

        if scanned_pages > len(pages) * 0.5:
            log.warning(
                "likely_scanned_pdf",
                source=source,
                scanned_pages=scanned_pages,
                total_pages=len(pages),
                action="Phase 2 OCR required",
            )

        log.info(
            "pdf_parsed",
            source=source,
            total_pages=len(pages),
            scanned_pages=scanned_pages,
            total_chars=len(full_text),
        )

        return ParsedDocument(
            source_path=source,
            total_pages=len(pages),
            pages=pages,
            full_text=full_text,
        )

    @staticmethod
    def parse_from_bytes(pdf_bytes: bytes, name: str = "document") -> ParsedDocument:
        """Parse a PDF from raw bytes — used when downloading from URLs."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        tmp_path = f"/tmp/{name}.pdf"
        doc.save(tmp_path)
        doc.close()
        return PDFParser.parse(tmp_path)
