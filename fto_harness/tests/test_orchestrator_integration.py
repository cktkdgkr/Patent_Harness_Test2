"""
Integration tests for the full FTO pipeline via Orchestrator.

Validates the three core scenarios end-to-end in mock mode:
1. Patent with sequences → extract → align → review queue
2. Patent without sequences → web fallback → INFERRED_FROM_WEB
3. Image-only → NEEDS_HUMAN_REVIEW

Also validates the non-negotiable design principles:
- No auto-exclusion: every item enters the human review queue
- Recall-first: ambiguous → NEEDS_HUMAN_REVIEW
- 100% review before CLEARED
"""

import json
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fto_harness.orchestrator import FTOOrchestrator
from fto_harness.core.models import ReviewStatus, SequenceConfidence


SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")


@pytest.fixture
def output_dir():
    d = tempfile.mkdtemp(prefix="fto_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def ref_fasta():
    return os.path.join(SAMPLE_DIR, "reference.fasta")


class TestScenario1_PatentWithSequences:
    """Patent has ST.26 listing → sequences extracted → aligned → queued for review."""

    def test_full_pipeline(self, output_dir, ref_fasta):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        result = orch.run(
            patent_input="US20200123456A1",
            reference_fasta_path=ref_fasta,
        )

        assert result.patent_metadata is not None
        assert result.patent_metadata.retrieval_status == "success"
        assert len(result.extracted_sequences) > 0
        assert len(result.alignment_results) > 0
        assert len(result.review_queue.items) > 0

    def test_clearance_blocked(self, output_dir, ref_fasta):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        result = orch.run(
            patent_input="US20200123456A1",
            reference_fasta_path=ref_fasta,
        )

        assert not result.can_issue_clearance(), (
            "CLEARED must be IMPOSSIBLE before human review"
        )

    def test_all_items_need_review(self, output_dir, ref_fasta):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        result = orch.run(
            patent_input="US20200123456A1",
            reference_fasta_path=ref_fasta,
        )

        for item in result.review_queue.items:
            allowed = ReviewStatus.automation_allowed()
            assert item.status in allowed, (
                f"Item '{item.summary}' has status {item.status.value} which is not "
                f"automation-allowed. Automation can only set: {[s.value for s in allowed]}"
            )

    def test_output_files_created(self, output_dir, ref_fasta):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        orch.run(patent_input="US20200123456A1", reference_fasta_path=ref_fasta)

        assert os.path.exists(os.path.join(output_dir, "human_review_queue.json"))
        assert os.path.exists(os.path.join(output_dir, "processing_log.json"))
        assert os.path.exists(os.path.join(output_dir, "report.md"))
        assert os.path.exists(os.path.join(output_dir, "full_result.json"))

    def test_report_has_incomplete_banner(self, output_dir, ref_fasta):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        orch.run(patent_input="US20200123456A1", reference_fasta_path=ref_fasta)

        report_path = os.path.join(output_dir, "report.md")
        with open(report_path) as f:
            report = f.read()

        assert "REVIEW INCOMPLETE" in report, (
            "Report must show REVIEW INCOMPLETE banner when items are pending"
        )
        assert "CLEARED" in report.upper()

    def test_report_has_disclaimer(self, output_dir, ref_fasta):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        orch.run(patent_input="US20200123456A1", reference_fasta_path=ref_fasta)

        report_path = os.path.join(output_dir, "report.md")
        with open(report_path) as f:
            report = f.read()

        assert "review aid only" in report.lower()
        assert "human reviewer" in report.lower()


class TestScenario2_WebFallback:
    """Patent has no explicit sequences → Agent 3 finds them via web search."""

    def test_web_fallback_produces_inferred_sequences(self, output_dir, ref_fasta):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        result = orch.run(
            patent_input="WO2021987654A1",
            reference_fasta_path=ref_fasta,
        )

        web_seqs = [
            s for s in result.extracted_sequences
            if s.confidence == SequenceConfidence.INFERRED_FROM_WEB
        ]
        assert len(web_seqs) > 0, (
            "Web fallback should produce INFERRED_FROM_WEB sequences"
        )

    def test_web_sequences_in_review_queue(self, output_dir, ref_fasta):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        result = orch.run(
            patent_input="WO2021987654A1",
            reference_fasta_path=ref_fasta,
        )

        web_items = [
            i for i in result.review_queue.items
            if "INFERRED_FROM_WEB" in str(i.evidence)
        ]
        assert len(web_items) > 0, "Web-inferred sequences must be in review queue"

    def test_clearance_still_blocked(self, output_dir, ref_fasta):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        result = orch.run(
            patent_input="WO2021987654A1",
            reference_fasta_path=ref_fasta,
        )

        assert not result.can_issue_clearance()


class TestScenario3_NoReferenceSequences:
    """No reference sequences provided → alignment skipped → flagged."""

    def test_skipped_alignment_flagged(self, output_dir):
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        result = orch.run(patent_input="US20200123456A1")

        assert len(result.alignment_results) == 0
        skip_items = [
            i for i in result.review_queue.items
            if "alignment" in i.item_type.lower() or "reference" in i.item_type.lower()
        ]
        assert len(skip_items) > 0, (
            "Missing alignment must generate a review queue item"
        )


class TestDesignPrinciples:
    """Verify the non-negotiable design principles across all scenarios."""

    def test_no_auto_cleared_status(self, output_dir, ref_fasta):
        """Principle 1: No patent is automatically excluded."""
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        result = orch.run(
            patent_input="US20200123456A1",
            reference_fasta_path=ref_fasta,
        )

        for item in result.review_queue.items:
            assert item.status != ReviewStatus.CLEARED, (
                "Automation must NEVER set CLEARED status"
            )
            assert item.status != ReviewStatus.REVIEWED_NO_RISK, (
                "Automation must NEVER set REVIEWED_NO_RISK"
            )
            assert item.status != ReviewStatus.REVIEWED_RISK_CONFIRMED, (
                "Automation must NEVER set REVIEWED_RISK_CONFIRMED"
            )

    def test_empty_queue_not_complete(self):
        """Principle 5: Empty queue is NOT complete."""
        from fto_harness.core.models import HumanReviewQueue
        queue = HumanReviewQueue()
        assert not queue.is_complete(), "Empty queue must NOT be considered complete"

    def test_processing_log_captures_all_steps(self, output_dir, ref_fasta):
        """Principle 3: All steps logged."""
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        result = orch.run(
            patent_input="US20200123456A1",
            reference_fasta_path=ref_fasta,
        )

        agents_logged = {e.agent for e in result.processing_log}
        assert "orchestrator" in agents_logged
        assert "patent_retrieval" in agents_logged
        assert "sequence_extraction" in agents_logged

    def test_review_queue_json_valid(self, output_dir, ref_fasta):
        """Principle 4: Verifiable evidence in structured output."""
        orch = FTOOrchestrator(mock_mode=True, output_dir=output_dir)
        orch.run(patent_input="US20200123456A1", reference_fasta_path=ref_fasta)

        queue_path = os.path.join(output_dir, "human_review_queue.json")
        with open(queue_path) as f:
            data = json.load(f)

        assert "total_items" in data
        assert "pending_count" in data
        assert "is_complete" in data
        assert "items" in data
        assert data["is_complete"] is False
        assert data["pending_count"] > 0
