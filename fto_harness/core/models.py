"""
Data models for the FTO Patent Review Harness.

Design principles enforced here:
- Every item defaults to NEEDS_HUMAN_REVIEW (recall-first).
- CLEARED status can only be set through explicit human action, never by automation.
- All data carries provenance (source, extraction method, URLs).
- The review queue blocks final report until 100% human-reviewed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ReviewStatus(Enum):
    """Review status for any queue item. Automation may only set the first two."""
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
    POTENTIAL_RISK = "POTENTIAL_RISK"
    REVIEWED_NO_RISK = "REVIEWED_NO_RISK"
    REVIEWED_RISK_CONFIRMED = "REVIEWED_RISK_CONFIRMED"
    CLEARED = "CLEARED"

    @classmethod
    def automation_allowed(cls) -> set[ReviewStatus]:
        return {cls.NEEDS_HUMAN_REVIEW, cls.POTENTIAL_RISK}

    def is_human_resolved(self) -> bool:
        return self in {
            ReviewStatus.REVIEWED_NO_RISK,
            ReviewStatus.REVIEWED_RISK_CONFIRMED,
            ReviewStatus.CLEARED,
        }


class SequenceConfidence(Enum):
    """How the sequence was obtained — lower confidence = louder flags."""
    DIRECT_SEQUENCE_LISTING = "DIRECT_SEQUENCE_LISTING"
    EXTRACTED_FROM_TEXT = "EXTRACTED_FROM_TEXT"
    INFERRED_FROM_WEB = "INFERRED_FROM_WEB"
    IMAGE_ONLY = "IMAGE_ONLY"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


class ExtractionStage(Enum):
    """Which extraction pipeline stage produced this sequence."""
    STAGE_A_SEQUENCE_LISTING = "A_SEQUENCE_LISTING"
    STAGE_B_TEXT_EXTRACTION = "B_TEXT_EXTRACTION"
    STAGE_C_IMAGE_DETECTION = "C_IMAGE_DETECTION"
    STAGE_D_WEB_SEARCH = "D_WEB_SEARCH"


class RiskLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class AgentName(Enum):
    PATENT_RETRIEVAL = "patent_retrieval"
    SEQUENCE_EXTRACTION = "sequence_extraction"
    WEB_SEQUENCE_FINDER = "web_sequence_finder"
    ALIGNMENT = "alignment"
    CLAIM_ANNOTATOR = "claim_annotator"
    ORCHESTRATOR = "orchestrator"


@dataclass
class Provenance:
    """Tracks where a piece of data came from — enforces verifiable evidence."""
    source_type: str
    patent_number: str | None = None
    url: str | None = None
    page_number: int | None = None
    seq_id_no: str | None = None
    accession: str | None = None
    database: str | None = None
    search_query: str | None = None
    extraction_stage: ExtractionStage | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if v is not None:
                d[k] = v.value if isinstance(v, Enum) else v
        return d


@dataclass
class SequenceRecord:
    """A single amino acid sequence extracted from or linked to a patent."""
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    amino_acid_sequence: str = ""
    length: int = 0
    confidence: SequenceConfidence = SequenceConfidence.EXTRACTION_FAILED
    provenance: Provenance | None = None
    extraction_stage: ExtractionStage | None = None
    # Recall-first: every sequence defaults to needing human review.
    status: ReviewStatus = ReviewStatus.NEEDS_HUMAN_REVIEW
    status_reason: str = "Awaiting human review (default)"
    raw_extraction_context: str = ""

    def __post_init__(self) -> None:
        if self.amino_acid_sequence:
            self.length = len(self.amino_acid_sequence.replace(" ", "").replace("\n", ""))
        if self.confidence == SequenceConfidence.INFERRED_FROM_WEB:
            self.status = ReviewStatus.NEEDS_HUMAN_REVIEW
            self.status_reason = "Sequence inferred from web search — requires human verification"
        if self.confidence == SequenceConfidence.IMAGE_ONLY:
            self.status = ReviewStatus.NEEDS_HUMAN_REVIEW
            self.status_reason = "Sequence exists only as image — OCR or manual extraction needed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "amino_acid_sequence": self.amino_acid_sequence,
            "length": self.length,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "extraction_stage": self.extraction_stage.value if self.extraction_stage else None,
            "status": self.status.value,
            "status_reason": self.status_reason,
        }


@dataclass
class AlignmentResult:
    """Result of aligning a reference sequence against a patent sequence."""
    alignment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    reference_id: str = ""
    patent_sequence_id: str = ""
    percent_identity: float = 0.0
    percent_coverage: float = 0.0
    evalue: float = float("inf")
    alignment_visualization: str = ""
    raw_blast_output: str = ""
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    status: ReviewStatus = ReviewStatus.NEEDS_HUMAN_REVIEW
    status_reason: str = "Awaiting human review (default)"
    is_web_inferred: bool = False
    mmseqs2_agrees: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alignment_id": self.alignment_id,
            "reference_id": self.reference_id,
            "patent_sequence_id": self.patent_sequence_id,
            "percent_identity": self.percent_identity,
            "percent_coverage": self.percent_coverage,
            "evalue": self.evalue,
            "alignment_visualization": self.alignment_visualization,
            "risk_level": self.risk_level.value,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "is_web_inferred": self.is_web_inferred,
            "mmseqs2_agrees": self.mmseqs2_agrees,
        }


@dataclass
class ClaimAnnotation:
    """Extracted claim language relevant to sequences."""
    claim_number: int | str = ""
    claim_text: str = ""
    sequence_references: list[str] = field(default_factory=list)
    identity_thresholds_mentioned: list[str] = field(default_factory=list)
    interpretation_notes: str = ""
    confidence: str = "LOW"
    status: ReviewStatus = ReviewStatus.NEEDS_HUMAN_REVIEW
    status_reason: str = "Claim analysis is LLM-assisted — requires human verification"

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_number": self.claim_number,
            "claim_text": self.claim_text,
            "sequence_references": self.sequence_references,
            "identity_thresholds_mentioned": self.identity_thresholds_mentioned,
            "interpretation_notes": self.interpretation_notes,
            "confidence": self.confidence,
            "status": self.status.value,
            "status_reason": self.status_reason,
        }


@dataclass
class PatentMetadata:
    """Metadata for a retrieved patent."""
    patent_number: str = ""
    title: str = ""
    applicant: str = ""
    publication_date: str = ""
    source_url: str = ""
    full_text: str = ""
    has_sequence_listing: bool = False
    sequence_listing_path: str | None = None
    pdf_path: str | None = None
    retrieval_status: str = "pending"
    retrieval_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "patent_number": self.patent_number,
            "title": self.title,
            "applicant": self.applicant,
            "publication_date": self.publication_date,
            "source_url": self.source_url,
            "has_sequence_listing": self.has_sequence_listing,
            "retrieval_status": self.retrieval_status,
            "retrieval_error": self.retrieval_error,
        }


@dataclass
class ProcessingLogEntry:
    """Single log entry tracking what happened at each pipeline step."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    agent: str = ""
    stage: str = ""
    status: str = ""
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "agent": self.agent,
            "stage": self.stage,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class HumanReviewItem:
    """A single item in the human review queue."""
    item_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    patent_number: str = ""
    item_type: str = ""  # "sequence", "alignment", "claim", "retrieval_failure"
    status: ReviewStatus = ReviewStatus.NEEDS_HUMAN_REVIEW
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    summary: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None
    related_sequence_ids: list[str] = field(default_factory=list)
    related_alignment_ids: list[str] = field(default_factory=list)
    reviewer_notes: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "patent_number": self.patent_number,
            "item_type": self.item_type,
            "status": self.status.value,
            "risk_level": self.risk_level.value,
            "summary": self.summary,
            "evidence": self.evidence,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "related_sequence_ids": self.related_sequence_ids,
            "related_alignment_ids": self.related_alignment_ids,
            "reviewer_notes": self.reviewer_notes,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
        }


@dataclass
class HumanReviewQueue:
    """
    The complete human review queue.

    DESIGN PRINCIPLE: is_complete() returns True ONLY when every single item
    has been resolved by a human. The tool never auto-clears anything.
    """
    items: list[HumanReviewItem] = field(default_factory=list)

    def add_item(self, item: HumanReviewItem) -> None:
        self.items.append(item)

    def pending_items(self) -> list[HumanReviewItem]:
        return [i for i in self.items if not i.status.is_human_resolved()]

    def is_complete(self) -> bool:
        """True only when ALL items have been human-reviewed. Zero items = not complete."""
        if not self.items:
            return False
        return all(item.status.is_human_resolved() for item in self.items)

    def sorted_by_risk(self) -> list[HumanReviewItem]:
        risk_order = {
            RiskLevel.HIGH: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.LOW: 2,
            RiskLevel.UNKNOWN: 3,
        }
        status_order = {
            ReviewStatus.POTENTIAL_RISK: 0,
            ReviewStatus.NEEDS_HUMAN_REVIEW: 1,
            ReviewStatus.REVIEWED_RISK_CONFIRMED: 2,
            ReviewStatus.REVIEWED_NO_RISK: 3,
            ReviewStatus.CLEARED: 4,
        }
        return sorted(
            self.items,
            key=lambda i: (status_order.get(i.status, 99), risk_order.get(i.risk_level, 99)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_items": len(self.items),
            "pending_count": len(self.pending_items()),
            "is_complete": self.is_complete(),
            "items": [i.to_dict() for i in self.sorted_by_risk()],
        }


@dataclass
class FTOAnalysisResult:
    """Top-level container for a complete FTO analysis run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    patent_input: str = ""
    reference_sequences: list[str] = field(default_factory=list)
    patent_metadata: PatentMetadata | None = None
    extracted_sequences: list[SequenceRecord] = field(default_factory=list)
    alignment_results: list[AlignmentResult] = field(default_factory=list)
    claim_annotations: list[ClaimAnnotation] = field(default_factory=list)
    review_queue: HumanReviewQueue = field(default_factory=HumanReviewQueue)
    processing_log: list[ProcessingLogEntry] = field(default_factory=list)

    def can_issue_clearance(self) -> bool:
        """CLEARED report is IMPOSSIBLE until every queue item is human-resolved."""
        return self.review_queue.is_complete()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "patent_input": self.patent_input,
            "can_issue_clearance": self.can_issue_clearance(),
            "patent_metadata": self.patent_metadata.to_dict() if self.patent_metadata else None,
            "extracted_sequences": [s.to_dict() for s in self.extracted_sequences],
            "alignment_results": [a.to_dict() for a in self.alignment_results],
            "claim_annotations": [c.to_dict() for c in self.claim_annotations],
            "review_queue": self.review_queue.to_dict(),
            "processing_log": [e.to_dict() for e in self.processing_log],
        }
