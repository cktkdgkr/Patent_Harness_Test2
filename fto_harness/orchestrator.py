"""
Orchestrator — Central coordinator for the FTO Patent Review Harness.

Calls agents in sequence, aggregates results, and enforces:
- Every item enters the human review queue (no auto-exclusion).
- Any agent failure/uncertainty → NEEDS_HUMAN_REVIEW escalation.
- Complete processing log for auditability.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from .agents.alignment import AlignmentAgent
from .agents.claim_annotator import ClaimAnnotatorAgent
from .agents.patent_retrieval import PatentRetrievalAgent
from .agents.sequence_extraction import SequenceExtractionAgent
from .agents.web_sequence_finder import WebSequenceFinderAgent
from .core.llm_client import LLMClient, load_config
from .core.logging_utils import ProcessingLogger
from .core.models import (
    AgentName,
    FTOAnalysisResult,
    HumanReviewItem,
    HumanReviewQueue,
    ReviewStatus,
    RiskLevel,
)
from .reporting.report_builder import ReportBuilder

AGENT = AgentName.ORCHESTRATOR


class FTOOrchestrator:
    def __init__(
        self,
        config_path: str | None = None,
        mock_mode: bool | None = None,
        output_dir: str = "output",
    ) -> None:
        self.config = load_config(config_path)
        if mock_mode is not None:
            self.config.setdefault("mock_mode", {})["enabled"] = mock_mode
        self.mock_mode = self.config.get("mock_mode", {}).get("enabled", False)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.logger = ProcessingLogger()
        self.llm = LLMClient(self.config, mock_mode=self.mock_mode)

        self.web_finder = WebSequenceFinderAgent(
            self.config, self.logger, self.llm, mock_mode=self.mock_mode
        )
        self.patent_retriever = PatentRetrievalAgent(
            self.config, self.logger, mock_mode=self.mock_mode
        )
        self.sequence_extractor = SequenceExtractionAgent(
            self.config, self.logger, self.llm,
            web_finder=self.web_finder, mock_mode=self.mock_mode,
        )
        self.aligner = AlignmentAgent(
            self.config, self.logger, mock_mode=self.mock_mode
        )
        self.claim_annotator = ClaimAnnotatorAgent(
            self.config, self.logger, self.llm, mock_mode=self.mock_mode
        )
        self.report_builder = ReportBuilder(self.config)

    def run(
        self,
        patent_input: str,
        reference_fasta_path: str | None = None,
        reference_sequences: list[str] | None = None,
    ) -> FTOAnalysisResult:
        """
        Execute the full FTO analysis pipeline.

        Args:
            patent_input: Patent number (e.g., US12345678B2) or PDF path.
            reference_fasta_path: Path to FASTA file with reference sequences.
            reference_sequences: Direct list of amino acid sequences.
        """
        result = FTOAnalysisResult(patent_input=patent_input)
        review_queue = HumanReviewQueue()

        self.logger.log_success(
            AGENT, "pipeline_start",
            f"Starting FTO analysis for: {patent_input}",
            mock_mode=self.mock_mode,
        )

        # --- Load reference sequences ---
        ref_seqs = self._load_references(reference_fasta_path, reference_sequences)
        result.reference_sequences = ref_seqs
        if not ref_seqs:
            self.logger.log_failure(
                AGENT, "reference_loading", "No reference sequences provided or loaded."
            )
            review_queue.add_item(HumanReviewItem(
                patent_number=patent_input,
                item_type="reference_missing",
                status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                risk_level=RiskLevel.HIGH,
                summary="No reference sequences available — cannot perform alignment",
            ))

        # --- Step 1: Patent Retrieval ---
        self.logger.log_success(AGENT, "step_1", "Starting patent retrieval...")
        patent = self.patent_retriever.retrieve(patent_input)
        result.patent_metadata = patent

        if patent.retrieval_status != "success":
            self.logger.log_failure(
                AGENT, "step_1_result",
                f"Patent retrieval failed/partial: {patent.retrieval_error}",
            )
            review_queue.add_item(HumanReviewItem(
                patent_number=patent_input,
                item_type="retrieval_failure",
                status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                risk_level=RiskLevel.HIGH,
                summary=f"Patent retrieval issue: {patent.retrieval_error}",
                evidence={"retrieval_status": patent.retrieval_status},
            ))
        else:
            self.logger.log_success(
                AGENT, "step_1_result",
                f"Patent retrieved: {patent.title or patent.patent_number}",
            )

        # --- Step 2: Sequence Extraction (multi-stage with web fallback) ---
        self.logger.log_success(AGENT, "step_2", "Starting sequence extraction...")
        extracted_sequences = self.sequence_extractor.extract(patent)
        result.extracted_sequences = extracted_sequences

        seq_review_items = self.sequence_extractor.build_review_items(extracted_sequences, patent)
        for item in seq_review_items:
            review_queue.add_item(item)

        self.logger.log_success(
            AGENT, "step_2_result",
            f"Extracted {len(extracted_sequences)} sequences, "
            f"generated {len(seq_review_items)} review items.",
        )

        # --- Step 3: Alignment ---
        if ref_seqs and any(s.amino_acid_sequence for s in extracted_sequences):
            self.logger.log_success(AGENT, "step_3", "Starting sequence alignment...")
            alignment_results = self.aligner.align_all(ref_seqs, extracted_sequences)
            result.alignment_results = alignment_results

            align_review_items = self.aligner.build_review_items(alignment_results, patent)
            for item in align_review_items:
                review_queue.add_item(item)

            self.logger.log_success(
                AGENT, "step_3_result",
                f"Completed {len(alignment_results)} alignments, "
                f"generated {len(align_review_items)} review items.",
            )
        else:
            reason = "no reference sequences" if not ref_seqs else "no extracted sequences with AA data"
            self.logger.log_skipped(
                AGENT, "step_3", f"Skipping alignment: {reason}.",
            )
            review_queue.add_item(HumanReviewItem(
                patent_number=patent.patent_number,
                item_type="alignment_skipped",
                status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                risk_level=RiskLevel.HIGH,
                summary=f"Alignment skipped: {reason}",
            ))

        # --- Step 4: Claim Annotation ---
        self.logger.log_success(AGENT, "step_4", "Starting claim annotation...")
        claim_annotations = self.claim_annotator.annotate(patent)
        result.claim_annotations = claim_annotations

        claim_review_items = self.claim_annotator.build_review_items(claim_annotations, patent)
        for item in claim_review_items:
            review_queue.add_item(item)

        self.logger.log_success(
            AGENT, "step_4_result",
            f"Generated {len(claim_annotations)} claim annotations.",
        )

        # --- Finalize ---
        result.review_queue = review_queue
        result.processing_log = self.logger.entries

        self._escalate_on_issues(review_queue)
        self._save_outputs(result)

        pending = len(review_queue.pending_items())
        total = len(review_queue.items)
        self.logger.log_success(
            AGENT, "pipeline_complete",
            f"FTO analysis complete. Review queue: {pending}/{total} items pending. "
            f"{'CLEARED status BLOCKED until all items reviewed.' if pending > 0 else ''}",
        )

        return result

    def _load_references(
        self,
        fasta_path: str | None,
        direct_sequences: list[str] | None,
    ) -> list[str]:
        """Load reference sequences from FASTA file or direct input."""
        sequences = []

        if direct_sequences:
            sequences.extend(direct_sequences)
            self.logger.log_success(
                AGENT, "ref_loading",
                f"Loaded {len(direct_sequences)} reference sequences from direct input.",
            )

        if fasta_path:
            try:
                from Bio import SeqIO

                for record in SeqIO.parse(fasta_path, "fasta"):
                    sequences.append(str(record.seq))
                self.logger.log_success(
                    AGENT, "ref_loading",
                    f"Loaded {len(sequences)} reference sequences from {fasta_path}.",
                )
            except ImportError:
                with open(fasta_path) as f:
                    current_seq = []
                    for line in f:
                        line = line.strip()
                        if line.startswith(">"):
                            if current_seq:
                                sequences.append("".join(current_seq))
                                current_seq = []
                        elif line:
                            current_seq.append(line)
                    if current_seq:
                        sequences.append("".join(current_seq))
                self.logger.log_success(
                    AGENT, "ref_loading",
                    f"Loaded {len(sequences)} reference sequences from {fasta_path} (manual parser).",
                )
            except Exception as e:
                self.logger.log_failure(
                    AGENT, "ref_loading", f"Failed to load references from {fasta_path}: {e}",
                )

        return sequences

    def _escalate_on_issues(self, queue: HumanReviewQueue) -> None:
        """If any processing failures/uncertainties exist, ensure they're in the queue."""
        for entry in self.logger.get_failures() + self.logger.get_uncertainties():
            already_covered = any(
                entry.agent in (item.summary or "") or entry.message in (item.summary or "")
                for item in queue.items
            )
            if not already_covered:
                queue.add_item(HumanReviewItem(
                    item_type="processing_issue",
                    status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                    risk_level=RiskLevel.MEDIUM,
                    summary=f"[{entry.agent}] {entry.stage}: {entry.message}",
                    evidence=entry.details,
                ))

    def _save_outputs(self, result: FTOAnalysisResult) -> None:
        """Save all output files."""
        queue_path = os.path.join(self.output_dir, "human_review_queue.json")
        with open(queue_path, "w") as f:
            json.dump(result.review_queue.to_dict(), f, indent=2, ensure_ascii=False)

        log_path = os.path.join(self.output_dir, "processing_log.json")
        with open(log_path, "w") as f:
            json.dump(
                [e.to_dict() for e in result.processing_log],
                f, indent=2, ensure_ascii=False,
            )

        report_path = os.path.join(self.output_dir, "report.md")
        report_content = self.report_builder.build(result)
        with open(report_path, "w") as f:
            f.write(report_content)

        full_path = os.path.join(self.output_dir, "full_result.json")
        with open(full_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        self.logger.log_success(
            AGENT, "output_saved",
            f"Outputs saved to {self.output_dir}/",
            files=["human_review_queue.json", "processing_log.json", "report.md", "full_result.json"],
        )
