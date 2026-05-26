"""
Agent 5 — Claim Risk Annotator Agent.

Extracts sequence-related limitations from patent claims using LLM analysis.
Provides supplementary information only — NEVER makes definitive legal judgments.
All outputs default to NEEDS_HUMAN_REVIEW.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from ..core.logging_utils import ProcessingLogger
from ..core.models import (
    AgentName,
    ClaimAnnotation,
    HumanReviewItem,
    PatentMetadata,
    Provenance,
    ReviewStatus,
    RiskLevel,
)

if TYPE_CHECKING:
    from ..core.llm_client import LLMClient

AGENT = AgentName.CLAIM_ANNOTATOR

CLAIM_RE = re.compile(
    r"(?:^|\n)\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.\s+|\Z)",
    re.DOTALL,
)


class ClaimAnnotatorAgent:
    def __init__(
        self,
        config: dict,
        logger: ProcessingLogger,
        llm_client: LLMClient,
        mock_mode: bool = False,
    ) -> None:
        self.config = config
        self.logger = logger
        self.llm = llm_client
        self.mock_mode = mock_mode

    def annotate(self, patent: PatentMetadata) -> list[ClaimAnnotation]:
        """Extract and annotate sequence-related claim limitations."""
        self.logger.log(
            AGENT, "annotate_start", "started",
            f"Analyzing claims for patent {patent.patent_number}",
        )

        if not patent.full_text:
            self.logger.log_failure(
                AGENT, "annotate", "No patent text available for claim analysis."
            )
            return []

        claims_text = self._extract_claims_section(patent.full_text)
        if not claims_text:
            self.logger.log_uncertain(
                AGENT, "claims_extraction",
                "Could not identify a distinct CLAIMS section in the patent text.",
            )
            claims_text = patent.full_text

        if self.mock_mode:
            return self._mock_annotate(patent)

        return self._llm_annotate(claims_text, patent)

    def _extract_claims_section(self, text: str) -> str | None:
        """Try to isolate the claims section from the full patent text."""
        patterns = [
            r"(?:CLAIMS|What is claimed is:?|The claims:?)\s*\n(.*?)(?:ABSTRACT|DESCRIPTION OF DRAWINGS|\Z)",
            r"(?:CLAIMS)\s*\n(.*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _llm_annotate(
        self, claims_text: str, patent: PatentMetadata
    ) -> list[ClaimAnnotation]:
        system_prompt = (
            "You are a patent analyst specializing in biotechnology claims. "
            "Analyze the following patent claims and extract ONLY factual information about "
            "sequence-related limitations. For each claim that references a sequence:\n"
            "1. Identify the claim number\n"
            "2. Extract the exact claim text\n"
            "3. List any SEQ ID NO references\n"
            "4. List any identity/similarity thresholds mentioned (e.g., 'at least 90% identity')\n"
            "5. Note any functional limitations tied to the sequence\n\n"
            "IMPORTANT: Do NOT make legal interpretations. Only extract factual content. "
            "If anything is ambiguous, say so explicitly.\n\n"
            "Return JSON with a 'claims' array where each entry has: "
            "claim_number, text, sequence_references, identity_thresholds, "
            "functional_limitations, ambiguities."
        )

        try:
            result = self.llm.query_json(system_prompt, claims_text[:6000])
            annotations = []

            for claim_data in result.get("claims", []):
                ambiguities = claim_data.get("ambiguities", "")
                confidence = "LOW" if ambiguities else "MEDIUM"

                annotation = ClaimAnnotation(
                    claim_number=claim_data.get("claim_number", "unknown"),
                    claim_text=claim_data.get("text", ""),
                    sequence_references=claim_data.get("sequence_references", []),
                    identity_thresholds_mentioned=claim_data.get("identity_thresholds", []),
                    interpretation_notes=(
                        f"Functional limitations: {claim_data.get('functional_limitations', 'none identified')}. "
                        f"Ambiguities: {ambiguities or 'none noted'}. "
                        f"NOTE: This is LLM-assisted extraction, not legal interpretation."
                    ),
                    confidence=confidence,
                    status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                    status_reason="LLM-assisted claim analysis — requires human patent attorney review",
                )
                annotations.append(annotation)

            self.logger.log_success(
                AGENT, "llm_annotate",
                f"Extracted {len(annotations)} claim annotations.",
            )
            return annotations

        except Exception as e:
            self.logger.log_failure(
                AGENT, "llm_annotate", f"LLM claim analysis failed: {e}",
            )
            return [
                ClaimAnnotation(
                    claim_text=claims_text[:500],
                    interpretation_notes=f"Automated analysis failed ({e}). Full manual review required.",
                    confidence="NONE",
                    status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                    status_reason=f"Claim analysis failed: {e}",
                )
            ]

    def _mock_annotate(self, patent: PatentMetadata) -> list[ClaimAnnotation]:
        self.logger.log_success(AGENT, "mock_annotate", "Returning mock claim annotations.")
        return [
            ClaimAnnotation(
                claim_number=1,
                claim_text=(
                    "An isolated polypeptide having at least 90% sequence identity "
                    "to SEQ ID NO:1 and having lipase activity (EC 3.1.1.3)."
                ),
                sequence_references=["SEQ ID NO:1"],
                identity_thresholds_mentioned=["90%"],
                interpretation_notes=(
                    "Functional limitations: lipase activity (EC 3.1.1.3). "
                    "Ambiguities: none noted. "
                    "NOTE: This is LLM-assisted extraction, not legal interpretation."
                ),
                confidence="MEDIUM",
                status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                status_reason="LLM-assisted claim analysis — requires human patent attorney review",
            ),
            ClaimAnnotation(
                claim_number=2,
                claim_text=(
                    "The polypeptide of claim 1 comprising substitutions T103K and D104N."
                ),
                sequence_references=["SEQ ID NO:1"],
                identity_thresholds_mentioned=[],
                interpretation_notes=(
                    "Functional limitations: specific point mutations. "
                    "Ambiguities: none noted. "
                    "NOTE: This is LLM-assisted extraction, not legal interpretation."
                ),
                confidence="MEDIUM",
                status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                status_reason="LLM-assisted claim analysis — requires human patent attorney review",
            ),
        ]

    def build_review_items(
        self, annotations: list[ClaimAnnotation], patent: PatentMetadata
    ) -> list[HumanReviewItem]:
        items = []
        for ann in annotations:
            items.append(
                HumanReviewItem(
                    patent_number=patent.patent_number,
                    item_type="claim_annotation",
                    status=ann.status,
                    risk_level=RiskLevel.MEDIUM if ann.identity_thresholds_mentioned else RiskLevel.LOW,
                    summary=(
                        f"Claim {ann.claim_number}: "
                        f"refs={ann.sequence_references}, "
                        f"thresholds={ann.identity_thresholds_mentioned}"
                    ),
                    evidence={
                        "claim_number": ann.claim_number,
                        "claim_text": ann.claim_text,
                        "sequence_references": ann.sequence_references,
                        "identity_thresholds": ann.identity_thresholds_mentioned,
                        "confidence": ann.confidence,
                    },
                    provenance=Provenance(
                        source_type="claim_analysis",
                        patent_number=patent.patent_number,
                    ),
                )
            )
        return items
