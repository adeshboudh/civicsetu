# src/civicsetu/config/document_registry.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from civicsetu.models.enums import DocType, Jurisdiction


@dataclass(frozen=True)
class DocumentSpec:
    """Immutable descriptor for a legal document to be ingested."""
    name: str
    url: str
    jurisdiction: Jurisdiction
    doc_type: DocType
    effective_date: date | None
    filename: str          # local cache filename
    dest_subdir: str       # under data/raw/
    max_pages: int | None = None  # None = all pages; set to cap forms/schedules appendices


# ── Registry ─────────────────────────────────────────────────────────────────

DOCUMENT_REGISTRY: dict[str, DocumentSpec] = {

    "rera_act_2016": DocumentSpec(
        name="RERA Act 2016",
        url="https://www.indiacode.nic.in/bitstream/123456789/2276/1/A2016-16.pdf",
        jurisdiction=Jurisdiction.CENTRAL,
        doc_type=DocType.ACT,
        effective_date=date(2016, 5, 1),
        filename="rera_act_2016.pdf",
        dest_subdir="acts",
    ),

    "mahrera_rules_2017": DocumentSpec(
        name="Maharashtra Real Estate (Regulation and Development) Rules 2017",
        url="https://naredco.in/notification/pdfs/RERA%20Final%20English%20Registration%20Rules21042017.pdf",
        jurisdiction=Jurisdiction.MAHARASHTRA,
        doc_type=DocType.RULES,
        effective_date=date(2017, 4, 21),
        filename="mahrera_rules_2017.pdf",
        dest_subdir="rules",
    ),
    
    "up_rera_rules_2016": DocumentSpec(
        name="Uttar Pradesh Real Estate (Regulation and Development) Rules 2016",
        url="https://www.up-rera.in/pdf/rera.pdf",
        jurisdiction=Jurisdiction.UTTAR_PRADESH,
        doc_type=DocType.RULES,
        effective_date=date(2016, 10, 27),
        filename="up_rera_rules_2016.pdf",
        dest_subdir="rules",
        max_pages=24,   # Pages 1-24 = Rules text; pages 25-52 = prescribed forms/schedules
    ),

    "up_rera_general_regulations_2019": DocumentSpec(
        name="Uttar Pradesh Real Estate Regulatory Authority (General) Regulations 2019",
        url="https://up-rera.in/ViewDocument?Param=8UttarPradeshRealEstateRegulatoryAuthorityGeneralRegulations201927022019.pdf",
        jurisdiction=Jurisdiction.UTTAR_PRADESH,
        doc_type=DocType.CIRCULAR,
        effective_date=date(2019, 2, 27),
        filename="up_rera_general_regulations_2019.pdf",
        dest_subdir="circulars",
    ),

    # ── Karnataka ──────────────────────────────────────────────────────────────
    "karnataka_rera_rules_2017": DocumentSpec(
        name="Karnataka Real Estate (Regulation and Development) Rules 2017",
        url="https://naredco.in/notification/pdfs/Karnataka%20Real%20Estate%20(Regulation%20and%20Development)%20Rules,%202017.pdf",
        jurisdiction=Jurisdiction.KARNATAKA,
        doc_type=DocType.RULES,
        effective_date=date(2017, 7, 10),
        filename="karnataka_rera_rules_2017.pdf",
        dest_subdir="rules",
        max_pages=37
    ),
}
