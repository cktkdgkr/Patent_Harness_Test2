"""
Agent 1 — Patent Retrieval Agent.

Retrieves patent metadata, full text, and sequence listing files
from public patent databases or local PDF files.

Failure at any step → NEEDS_HUMAN_REVIEW with reason logged.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ..core.logging_utils import ProcessingLogger
from ..core.models import (
    AgentName,
    PatentMetadata,
    Provenance,
    ReviewStatus,
)

AGENT = AgentName.PATENT_RETRIEVAL

PATENT_NUMBER_RE = re.compile(
    r"^(US|EP|WO|KR|CN|JP|DE|FR|GB)\s*\d[\d\-/A-Za-z]+$", re.IGNORECASE
)


class PatentRetrievalAgent:
    def __init__(
        self,
        config: dict,
        logger: ProcessingLogger,
        mock_mode: bool = False,
    ) -> None:
        self.config = config
        self.logger = logger
        self.mock_mode = mock_mode
        self.timeout = config.get("patent_retrieval", {}).get("request_timeout", 60)
        self.sources = config.get("patent_retrieval", {}).get(
            "sources", ["google_patents", "epo_ops", "uspto"]
        )

    def retrieve(self, patent_input: str) -> PatentMetadata:
        """
        Main entry point.
        - If input is a file path to a PDF, parse it directly.
        - If input looks like a patent number, search public databases.
        """
        if self._is_pdf_path(patent_input):
            return self._retrieve_from_pdf(patent_input)
        elif PATENT_NUMBER_RE.match(patent_input.strip()):
            return self._retrieve_from_databases(patent_input.strip())
        else:
            self.logger.log_failure(
                AGENT,
                "input_validation",
                f"Unrecognized input format: '{patent_input}'. "
                f"Expected patent number (e.g., US12345678B2) or PDF file path.",
            )
            return PatentMetadata(
                patent_number=patent_input,
                retrieval_status="failed",
                retrieval_error=f"Unrecognized input format: {patent_input}",
            )

    def _is_pdf_path(self, path: str) -> bool:
        return path.lower().endswith(".pdf") and (
            os.path.exists(path) or self.mock_mode
        )

    def _retrieve_from_pdf(self, pdf_path: str) -> PatentMetadata:
        self.logger.log_success(AGENT, "input_type", f"PDF file provided: {pdf_path}")

        if self.mock_mode:
            return self._mock_pdf_retrieval(pdf_path)

        try:
            import pdfplumber

            full_text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text_parts.append(text)

            full_text = "\n".join(full_text_parts)
            if not full_text.strip():
                self.logger.log_failure(
                    AGENT, "pdf_parsing", "PDF parsed but no text extracted (scanned image PDF?)."
                )
                return PatentMetadata(
                    pdf_path=pdf_path,
                    full_text="",
                    retrieval_status="partial",
                    retrieval_error="No text extracted from PDF — may be image-based",
                )

            patent_number = self._extract_patent_number_from_text(full_text)
            has_seq_listing = self._detect_sequence_listing(full_text)

            self.logger.log_success(
                AGENT,
                "pdf_parsing",
                f"Extracted {len(full_text)} chars from {len(full_text_parts)} pages.",
                patent_number=patent_number,
                has_sequence_listing=has_seq_listing,
            )
            return PatentMetadata(
                patent_number=patent_number,
                full_text=full_text,
                pdf_path=pdf_path,
                has_sequence_listing=has_seq_listing,
                retrieval_status="success",
            )

        except Exception as e:
            self.logger.log_failure(
                AGENT, "pdf_parsing", f"Failed to parse PDF: {e}", path=pdf_path
            )
            return PatentMetadata(
                pdf_path=pdf_path,
                retrieval_status="failed",
                retrieval_error=str(e),
            )

    def _retrieve_from_databases(self, patent_number: str) -> PatentMetadata:
        self.logger.log_success(
            AGENT, "input_type", f"Patent number provided: {patent_number}"
        )

        if self.mock_mode:
            return self._mock_db_retrieval(patent_number)

        errors = []
        for source in self.sources:
            try:
                result = self._fetch_from_source(source, patent_number)
                if result and result.retrieval_status == "success":
                    self.logger.log_success(
                        AGENT,
                        "db_retrieval",
                        f"Retrieved from {source}",
                        patent_number=patent_number,
                    )
                    return result
            except Exception as e:
                err_msg = f"{source}: {e}"
                errors.append(err_msg)
                self.logger.log_failure(
                    AGENT, f"db_retrieval_{source}", err_msg, patent_number=patent_number
                )

        all_errors = "; ".join(errors)
        self.logger.log_failure(
            AGENT,
            "db_retrieval",
            f"All sources failed for {patent_number}: {all_errors}",
        )
        return PatentMetadata(
            patent_number=patent_number,
            retrieval_status="failed",
            retrieval_error=f"All sources failed: {all_errors}",
        )

    def _fetch_from_source(self, source: str, patent_number: str) -> PatentMetadata | None:
        """Fetch patent from a specific source. Real implementations would call APIs."""
        import requests

        if source == "google_patents":
            url = f"https://patents.google.com/patent/{patent_number}/en"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                text = resp.text
                has_seq = self._detect_sequence_listing(text)
                return PatentMetadata(
                    patent_number=patent_number,
                    full_text=text,
                    source_url=url,
                    has_sequence_listing=has_seq,
                    retrieval_status="success",
                )
        return None

    def _extract_patent_number_from_text(self, text: str) -> str:
        match = PATENT_NUMBER_RE.search(text[:2000])
        return match.group(0) if match else "UNKNOWN"

    def _detect_sequence_listing(self, text: str) -> bool:
        indicators = [
            "SEQUENCE LISTING",
            "SEQ ID NO",
            "<210>",
            "<400>",
            "sequenceListing",
            "ST.26",
        ]
        text_upper = text.upper()
        return any(ind.upper() in text_upper for ind in indicators)

    def _mock_pdf_retrieval(self, pdf_path: str) -> PatentMetadata:
        self.logger.log_success(AGENT, "mock_pdf", f"Mock PDF retrieval: {pdf_path}")
        return PatentMetadata(
            patent_number="US20200123456A1",
            title="[Mock] Engineered Lipase Variants with Enhanced Thermostability",
            applicant="Mock Biotech Inc.",
            publication_date="2020-04-23",
            full_text=_MOCK_PATENT_TEXT,
            pdf_path=pdf_path,
            has_sequence_listing=True,
            retrieval_status="success",
        )

    def _mock_db_retrieval(self, patent_number: str) -> PatentMetadata:
        self.logger.log_success(
            AGENT, "mock_db", f"Mock DB retrieval: {patent_number}"
        )
        return PatentMetadata(
            patent_number=patent_number,
            title="[Mock] Engineered Lipase Variants with Enhanced Thermostability",
            applicant="Mock Biotech Inc.",
            publication_date="2020-04-23",
            source_url=f"https://patents.google.com/patent/{patent_number}/en",
            full_text=_MOCK_PATENT_TEXT,
            has_sequence_listing=True,
            retrieval_status="success",
        )


_MOCK_PATENT_TEXT = """
ENGINEERED LIPASE VARIANTS WITH ENHANCED THERMOSTABILITY

SEQUENCE LISTING
The present application contains a Sequence Listing in accordance with ST.26.

FIELD OF THE INVENTION
The present invention relates to engineered lipase enzymes (EC 3.1.1.3) derived from
Candida antarctica lipase B (CALB) with enhanced thermostability.

BACKGROUND
Candida antarctica lipase B (UniProt: P41365) is widely used in industrial biocatalysis.

DETAILED DESCRIPTION
SEQ ID NO:1 represents the wild-type CALB sequence:
LPSGSDPAFSQPKSVLDAGLTCQGASPSSVSKPILLVPGTGTTGPQSFDSNWIPLSTQLG
YTPCWISPPPFMLGPFHAGSIAPDTRLFNKYLQMKSFNLPTLQNAFPGSYITLNTTSNTK
DALVRQKLAAGSTSNIGNYALAAAQSLFANTSTPNAKIREVYTDSRSNPAATFYYVNALTS
PSIAATATFKASGLIPQTLNSTFDTAQYASVSNLNSTFANLTPNKLGVSHMAGASGIYGAG
CSSALKNVTPQLKASFALGYGAGVTGTANASANIGNSAATFQAIGTSEATLANRLAQRFAP

SEQ ID NO:2 represents the T103K/D104N variant with enhanced thermostability:
LPSGSDPAFSQPKSVLDAGLTCQGASPSSVSKPILLVPGTGTTGPQSFDSNWIPLSTQLG
YTPCWISPPPFMLGPFHAGSIAPDKNLFNKYLQMKSFNLPTLQNAFPGSYITLNTTSNTK
DALVRQKLAAGSTSNIGNYALAAAQSLFANTSTPNAKIREVYTDSRSNPAATFYYVNALTS
PSIAATATFKASGLIPQTLNSTFDTAQYASVSNLNSTFANLTPNKLGVSHMAGASGIYGAG
CSSALKNVTPQLKASFALGYGAGVTGTANASANIGNSAATFQAIGTSEATLANRLAQRFAP

CLAIMS
1. An isolated polypeptide having at least 90% sequence identity to SEQ ID NO:1
   and having lipase activity (EC 3.1.1.3).
2. The polypeptide of claim 1 comprising substitutions T103K and D104N.
3. A composition comprising the polypeptide of claim 1 and a pharmaceutically
   acceptable carrier.

The enzyme may also be referred to by accession number P41365 (UniProt) or
GenBank: CAA83122.1.
"""
