"""
Agent 2 — Sequence Extraction Agent.

Multi-stage fallback extraction to maximize recall:
  Stage A: Parse standard sequence listing (ST.26 XML / ST.25 txt)
  Stage B: Regex + LLM-assisted extraction from patent body text
  Stage C: Detect image-only sequences, flag for human/OCR
  Stage D: Delegate to Web Sequence Finder (Agent 3)

Every sequence carries provenance. Zero extractions → NEEDS_HUMAN_REVIEW.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..core.logging_utils import ProcessingLogger
from ..core.models import (
    AgentName,
    ExtractionStage,
    HumanReviewItem,
    PatentMetadata,
    Provenance,
    ReviewStatus,
    RiskLevel,
    SequenceConfidence,
    SequenceRecord,
)

if TYPE_CHECKING:
    from ..core.llm_client import LLMClient
    from .web_sequence_finder import WebSequenceFinderAgent

AGENT = AgentName.SEQUENCE_EXTRACTION

AA_SINGLE_LETTER = set("ACDEFGHIKLMNPQRSTVWY")
AA_BLOCK_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY\s]{20,}", re.IGNORECASE)
SEQ_ID_RE = re.compile(r"SEQ\s*ID\s*NO\s*[:\.]?\s*(\d+)", re.IGNORECASE)
ACCESSION_RE = re.compile(
    r"\b([A-Z][0-9][A-Z0-9]{3}[0-9]|[A-Z]{2}[0-9]{6}|[A-Z]{3}[0-9]{5})\b"
)
EC_NUMBER_RE = re.compile(r"EC\s*[\d]+\.[\d]+\.[\d]+\.[\d]+", re.IGNORECASE)


class SequenceExtractionAgent:
    def __init__(
        self,
        config: dict,
        logger: ProcessingLogger,
        llm_client: LLMClient,
        web_finder: WebSequenceFinderAgent | None = None,
        mock_mode: bool = False,
    ) -> None:
        self.config = config
        self.logger = logger
        self.llm = llm_client
        self.web_finder = web_finder
        self.mock_mode = mock_mode
        self.min_seq_length = config.get("sequence_extraction", {}).get(
            "min_sequence_length", 10
        )

    def extract(self, patent: PatentMetadata) -> list[SequenceRecord]:
        """Run all extraction stages in fallback order. Return ALL found sequences."""
        all_sequences: list[SequenceRecord] = []
        stage_results: dict[str, int] = {}

        # Stage A: Standard sequence listing
        stage_a = self._stage_a_sequence_listing(patent)
        all_sequences.extend(stage_a)
        stage_results["A"] = len(stage_a)

        # Stage B: Text extraction with regex + LLM
        stage_b = self._stage_b_text_extraction(patent)
        stage_b_deduped = self._deduplicate(stage_b, all_sequences)
        all_sequences.extend(stage_b_deduped)
        stage_results["B"] = len(stage_b_deduped)

        # Stage C: Image detection
        stage_c = self._stage_c_image_detection(patent)
        all_sequences.extend(stage_c)
        stage_results["C"] = len(stage_c)

        # Stage D: Web search fallback (if previous stages found nothing or found clues)
        if self.web_finder and (not all_sequences or self._has_unresolved_references(patent)):
            stage_d = self._stage_d_web_search(patent, all_sequences)
            all_sequences.extend(stage_d)
            stage_results["D"] = len(stage_d)

        self.logger.log_success(
            AGENT,
            "extraction_summary",
            f"Total sequences extracted: {len(all_sequences)} "
            f"(A={stage_results.get('A', 0)}, B={stage_results.get('B', 0)}, "
            f"C={stage_results.get('C', 0)}, D={stage_results.get('D', 0)})",
        )

        if not all_sequences:
            self.logger.log_failure(
                AGENT,
                "extraction_summary",
                "ZERO sequences extracted from all stages — NEEDS_HUMAN_REVIEW",
            )

        return all_sequences

    def _stage_a_sequence_listing(self, patent: PatentMetadata) -> list[SequenceRecord]:
        """Parse structured sequence listing (ST.26 XML or ST.25 text)."""
        self.logger.log(AGENT, "stage_A", "started", "Checking for standard sequence listing...")

        if not patent.has_sequence_listing:
            self.logger.log_skipped(
                AGENT, "stage_A", "No sequence listing detected in patent."
            )
            return []

        if patent.sequence_listing_path:
            return self._parse_sequence_listing_file(patent.sequence_listing_path, patent)

        sequences = self._parse_sequence_listing_from_text(patent.full_text, patent)
        if sequences:
            self.logger.log_success(
                AGENT, "stage_A", f"Extracted {len(sequences)} sequences from listing in text."
            )
        else:
            self.logger.log_uncertain(
                AGENT, "stage_A",
                "Sequence listing indicators found but no sequences parsed.",
            )
        return sequences

    def _parse_sequence_listing_from_text(
        self, text: str, patent: PatentMetadata
    ) -> list[SequenceRecord]:
        sequences = []
        seq_blocks = re.split(r"SEQ\s*ID\s*NO\s*[:\.]?\s*(\d+)", text, flags=re.IGNORECASE)

        for i in range(1, len(seq_blocks) - 1, 2):
            seq_id = seq_blocks[i]
            block = seq_blocks[i + 1][:2000]
            aa_matches = AA_BLOCK_RE.findall(block)

            for match in aa_matches:
                cleaned = re.sub(r"\s+", "", match).upper()
                if len(cleaned) >= self.min_seq_length and self._is_likely_protein(cleaned):
                    sequences.append(
                        SequenceRecord(
                            amino_acid_sequence=cleaned,
                            confidence=SequenceConfidence.DIRECT_SEQUENCE_LISTING,
                            extraction_stage=ExtractionStage.STAGE_A_SEQUENCE_LISTING,
                            provenance=Provenance(
                                source_type="sequence_listing",
                                patent_number=patent.patent_number,
                                seq_id_no=f"SEQ ID NO:{seq_id}",
                                extraction_stage=ExtractionStage.STAGE_A_SEQUENCE_LISTING,
                            ),
                            status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                            status_reason="Extracted from sequence listing — awaiting human verification",
                            raw_extraction_context=block[:500],
                        )
                    )
        return sequences

    def _parse_sequence_listing_file(
        self, path: str, patent: PatentMetadata
    ) -> list[SequenceRecord]:
        """Parse ST.26 XML or ST.25 text file."""
        self.logger.log(AGENT, "stage_A_file", "started", f"Parsing listing file: {path}")
        try:
            if path.endswith(".xml"):
                return self._parse_st26_xml(path, patent)
            else:
                with open(path, "r") as f:
                    content = f.read()
                return self._parse_sequence_listing_from_text(content, patent)
        except Exception as e:
            self.logger.log_failure(
                AGENT, "stage_A_file", f"Failed to parse listing file: {e}", path=path
            )
            return []

    def _parse_st26_xml(self, path: str, patent: PatentMetadata) -> list[SequenceRecord]:
        """Parse ST.26 XML format sequence listing."""
        try:
            from lxml import etree

            tree = etree.parse(path)
            root = tree.getroot()
            ns = {"st26": root.nsmap.get(None, "")}
            sequences = []

            for seq_elem in root.iter():
                if "SequenceData" in seq_elem.tag:
                    seq_id = seq_elem.get("sequenceIDNumber", "unknown")
                    for feature in seq_elem.iter():
                        if "residues" in feature.tag.lower() or feature.text:
                            text = (feature.text or "").strip()
                            cleaned = re.sub(r"\s+", "", text).upper()
                            if len(cleaned) >= self.min_seq_length and self._is_likely_protein(cleaned):
                                sequences.append(
                                    SequenceRecord(
                                        amino_acid_sequence=cleaned,
                                        confidence=SequenceConfidence.DIRECT_SEQUENCE_LISTING,
                                        extraction_stage=ExtractionStage.STAGE_A_SEQUENCE_LISTING,
                                        provenance=Provenance(
                                            source_type="st26_xml",
                                            patent_number=patent.patent_number,
                                            seq_id_no=f"SEQ ID NO:{seq_id}",
                                            extraction_stage=ExtractionStage.STAGE_A_SEQUENCE_LISTING,
                                        ),
                                        status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                                        status_reason="Extracted from ST.26 XML — awaiting human verification",
                                    )
                                )
            return sequences
        except Exception as e:
            self.logger.log_failure(AGENT, "st26_parse", f"ST.26 XML parse error: {e}")
            return []

    def _stage_b_text_extraction(self, patent: PatentMetadata) -> list[SequenceRecord]:
        """Extract sequences from patent body text using regex + LLM."""
        self.logger.log(AGENT, "stage_B", "started", "Extracting sequences from body text...")

        if not patent.full_text:
            self.logger.log_skipped(AGENT, "stage_B", "No full text available.")
            return []

        sequences = []

        # Regex-based extraction
        regex_seqs = self._regex_extract_sequences(patent)
        sequences.extend(regex_seqs)

        # LLM-assisted extraction
        llm_seqs = self._llm_extract_sequences(patent)
        llm_deduped = self._deduplicate(llm_seqs, sequences)
        sequences.extend(llm_deduped)

        self.logger.log_success(
            AGENT, "stage_B",
            f"Found {len(regex_seqs)} via regex, {len(llm_deduped)} additional via LLM.",
        )
        return sequences

    def _regex_extract_sequences(self, patent: PatentMetadata) -> list[SequenceRecord]:
        """Find amino acid blocks in text using pattern matching."""
        sequences = []
        text = patent.full_text

        blocks = AA_BLOCK_RE.findall(text)
        for block in blocks:
            cleaned = re.sub(r"\s+", "", block).upper()
            if len(cleaned) >= self.min_seq_length and self._is_likely_protein(cleaned):
                context_start = max(0, text.find(block) - 100)
                context_end = min(len(text), text.find(block) + len(block) + 100)
                context = text[context_start:context_end]

                seq_id_match = SEQ_ID_RE.search(context)
                seq_id = f"SEQ ID NO:{seq_id_match.group(1)}" if seq_id_match else None

                sequences.append(
                    SequenceRecord(
                        amino_acid_sequence=cleaned,
                        confidence=SequenceConfidence.EXTRACTED_FROM_TEXT,
                        extraction_stage=ExtractionStage.STAGE_B_TEXT_EXTRACTION,
                        provenance=Provenance(
                            source_type="text_regex",
                            patent_number=patent.patent_number,
                            seq_id_no=seq_id,
                            extraction_stage=ExtractionStage.STAGE_B_TEXT_EXTRACTION,
                        ),
                        status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                        status_reason="Extracted from text via regex — awaiting human verification",
                        raw_extraction_context=context[:500],
                    )
                )
        return sequences

    def _llm_extract_sequences(self, patent: PatentMetadata) -> list[SequenceRecord]:
        """Use LLM to find sequences that regex might miss."""
        system_prompt = (
            "You are a bioinformatics expert extracting amino acid sequences from patent text. "
            "Find ALL protein/peptide sequences mentioned in the text, including fragments. "
            "For each sequence found, provide the amino acid sequence in single-letter code, "
            "the SEQ ID NO if mentioned, and the approximate location in the text. "
            "Also extract any accession numbers (UniProt, GenBank), enzyme names, and EC numbers. "
            "Return results as JSON."
        )
        text_chunk = patent.full_text[:8000]

        try:
            result = self.llm.query_json(system_prompt, text_chunk)
            sequences = []
            for item in result.get("sequences_found", []):
                seq = item.get("sequence", "")
                cleaned = re.sub(r"\s+", "", seq).upper()
                if len(cleaned) >= self.min_seq_length and self._is_likely_protein(cleaned):
                    confidence_score = item.get("confidence", 0.5)
                    sequences.append(
                        SequenceRecord(
                            amino_acid_sequence=cleaned,
                            confidence=SequenceConfidence.EXTRACTED_FROM_TEXT,
                            extraction_stage=ExtractionStage.STAGE_B_TEXT_EXTRACTION,
                            provenance=Provenance(
                                source_type="text_llm",
                                patent_number=patent.patent_number,
                                seq_id_no=item.get("seq_id"),
                                notes=f"LLM confidence: {confidence_score}, location: {item.get('location', 'unknown')}",
                                extraction_stage=ExtractionStage.STAGE_B_TEXT_EXTRACTION,
                            ),
                            status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                            status_reason=f"LLM-assisted extraction (confidence={confidence_score}) — awaiting human verification",
                        )
                    )
            return sequences
        except Exception as e:
            self.logger.log_failure(AGENT, "stage_B_llm", f"LLM extraction failed: {e}")
            return []

    def _stage_c_image_detection(self, patent: PatentMetadata) -> list[SequenceRecord]:
        """Detect if sequences might exist only as images."""
        self.logger.log(AGENT, "stage_C", "started", "Checking for image-only sequences...")

        if not patent.full_text:
            return []

        text = patent.full_text.upper()
        image_indicators = [
            "FIG.", "FIGURE", "SEE DRAWING", "SEQUENCE IS SHOWN IN",
            "DEPICTED IN", "ILLUSTRATED IN",
        ]
        seq_ref_without_data = []

        aa_strict_re = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{20,}")
        seen_ids = set()
        seq_id_matches = SEQ_ID_RE.findall(patent.full_text)
        for seq_id in seq_id_matches:
            if seq_id in seen_ids:
                continue
            seen_ids.add(seq_id)
            context_match = re.search(
                rf"SEQ\s*ID\s*NO\s*[:\.]?\s*{seq_id}.{{0,200}}", patent.full_text, re.IGNORECASE
            )
            if context_match:
                context = context_match.group(0)
                has_nearby_sequence = bool(aa_strict_re.search(context))
                has_image_ref = any(ind in context.upper() for ind in image_indicators)
                if not has_nearby_sequence and has_image_ref:
                    seq_ref_without_data.append(seq_id)

        results = []
        for seq_id in seq_ref_without_data:
            self.logger.log_uncertain(
                AGENT, "stage_C",
                f"SEQ ID NO:{seq_id} may only exist as an image — needs OCR or manual extraction.",
            )
            results.append(
                SequenceRecord(
                    amino_acid_sequence="",
                    confidence=SequenceConfidence.IMAGE_ONLY,
                    extraction_stage=ExtractionStage.STAGE_C_IMAGE_DETECTION,
                    provenance=Provenance(
                        source_type="image_detection",
                        patent_number=patent.patent_number,
                        seq_id_no=f"SEQ ID NO:{seq_id}",
                        notes="Sequence likely in image/figure only",
                        extraction_stage=ExtractionStage.STAGE_C_IMAGE_DETECTION,
                    ),
                    status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                    status_reason="Sequence appears to be image-only — OCR or manual extraction required",
                )
            )

        if results:
            self.logger.log_uncertain(
                AGENT, "stage_C",
                f"Found {len(results)} potential image-only sequences.",
            )
        else:
            self.logger.log_success(AGENT, "stage_C", "No image-only sequences detected.")

        return results

    def _stage_d_web_search(
        self, patent: PatentMetadata, existing: list[SequenceRecord]
    ) -> list[SequenceRecord]:
        """Fallback: use Web Sequence Finder to search external databases."""
        self.logger.log(
            AGENT, "stage_D", "started",
            "Falling back to web search for additional sequences...",
        )

        if not self.web_finder:
            self.logger.log_skipped(
                AGENT, "stage_D", "Web Sequence Finder agent not available."
            )
            return []

        clues = self._extract_search_clues(patent)
        web_sequences = self.web_finder.search(patent, clues)

        web_deduped = self._deduplicate(web_sequences, existing)
        self.logger.log_success(
            AGENT, "stage_D",
            f"Web search returned {len(web_sequences)} candidates, "
            f"{len(web_deduped)} new after dedup.",
        )
        return web_deduped

    def _extract_search_clues(self, patent: PatentMetadata) -> dict:
        """Extract accession numbers, enzyme names, EC numbers from patent text."""
        text = patent.full_text or ""
        clues = {
            "accession_numbers": list(set(ACCESSION_RE.findall(text))),
            "ec_numbers": list(set(EC_NUMBER_RE.findall(text))),
            "patent_number": patent.patent_number,
            "title": patent.title,
            "applicant": patent.applicant,
        }

        try:
            result = self.llm.query_json(
                "Extract search clues from patent text for finding protein sequences. "
                "Return JSON with: enzyme_names, organism_names, gene_names, "
                "accession_numbers, ec_numbers, keywords.",
                text[:6000],
            )
            clues["enzyme_names"] = result.get("enzyme_names", [])
            clues["organism_names"] = result.get("organism_names", [])
            clues["gene_names"] = result.get("gene_names", [])
            clues["keywords"] = result.get("keywords", [])
            for acc in result.get("accession_numbers", []):
                if acc not in clues["accession_numbers"]:
                    clues["accession_numbers"].append(acc)
        except Exception as e:
            self.logger.log_failure(AGENT, "clue_extraction", f"LLM clue extraction failed: {e}")

        return clues

    def _has_unresolved_references(self, patent: PatentMetadata) -> bool:
        """Check if patent text references external sequences not yet extracted."""
        if not patent.full_text:
            return False
        text = patent.full_text
        has_accessions = bool(ACCESSION_RE.search(text))
        has_external_refs = any(
            kw in text.upper()
            for kw in ["UNIPROT", "GENBANK", "NCBI", "ACCESSION", "PDB"]
        )
        return has_accessions or has_external_refs

    def _is_likely_protein(self, seq: str) -> bool:
        """Check if a string is likely an amino acid sequence (not DNA/random)."""
        if not seq:
            return False
        aa_chars = sum(1 for c in seq if c in AA_SINGLE_LETTER)
        dna_only = set("ACGT")
        is_dna = all(c in dna_only for c in seq) and len(seq) > 30
        return (aa_chars / len(seq)) > 0.8 and not is_dna

    def _deduplicate(
        self, new: list[SequenceRecord], existing: list[SequenceRecord]
    ) -> list[SequenceRecord]:
        """Remove sequences from `new` that are identical to those in `existing`."""
        existing_seqs = {r.amino_acid_sequence for r in existing if r.amino_acid_sequence}
        return [s for s in new if s.amino_acid_sequence not in existing_seqs]

    def build_review_items(self, sequences: list[SequenceRecord], patent: PatentMetadata) -> list[HumanReviewItem]:
        """Convert extracted sequences into human review queue items."""
        items = []
        for seq in sequences:
            risk = RiskLevel.UNKNOWN
            if seq.confidence == SequenceConfidence.INFERRED_FROM_WEB:
                risk = RiskLevel.MEDIUM
            elif seq.confidence == SequenceConfidence.IMAGE_ONLY:
                risk = RiskLevel.MEDIUM
            elif seq.confidence == SequenceConfidence.EXTRACTION_FAILED:
                risk = RiskLevel.HIGH

            items.append(
                HumanReviewItem(
                    patent_number=patent.patent_number,
                    item_type="sequence_extraction",
                    status=seq.status,
                    risk_level=risk,
                    summary=f"Sequence ({seq.confidence.value}): {seq.amino_acid_sequence[:50]}..."
                    if seq.amino_acid_sequence
                    else f"Sequence extraction issue: {seq.status_reason}",
                    evidence={
                        "sequence_id": seq.sequence_id,
                        "length": seq.length,
                        "confidence": seq.confidence.value,
                        "extraction_stage": seq.extraction_stage.value if seq.extraction_stage else None,
                    },
                    provenance=seq.provenance,
                    related_sequence_ids=[seq.sequence_id],
                )
            )

        if not sequences:
            items.append(
                HumanReviewItem(
                    patent_number=patent.patent_number,
                    item_type="extraction_failure",
                    status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                    risk_level=RiskLevel.HIGH,
                    summary="No sequences extracted from any stage — manual extraction required",
                    evidence={"stages_attempted": ["A", "B", "C", "D"]},
                )
            )

        return items
