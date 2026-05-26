"""
Tests for the Sequence Extraction Agent.

Validates:
- Multi-stage fallback extraction (A→B→C→D)
- All sequences default to NEEDS_HUMAN_REVIEW
- Zero extraction → NEEDS_HUMAN_REVIEW flag
- Image-only detection
- Provenance tagging on every sequence
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fto_harness.core.models import (
    ExtractionStage,
    PatentMetadata,
    ReviewStatus,
    SequenceConfidence,
)
from fto_harness.core.llm_client import LLMClient, load_config
from fto_harness.core.logging_utils import ProcessingLogger
from fto_harness.agents.sequence_extraction import SequenceExtractionAgent
from fto_harness.agents.web_sequence_finder import WebSequenceFinderAgent


SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")


def _load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    return load_config(config_path)


def _make_patent_with_sequences() -> PatentMetadata:
    text_path = os.path.join(SAMPLE_DIR, "mock_patent_with_sequences.txt")
    with open(text_path) as f:
        text = f.read()
    return PatentMetadata(
        patent_number="US20200123456A1",
        title="Engineered Lipase Variants",
        full_text=text,
        has_sequence_listing=True,
        retrieval_status="success",
    )


def _make_patent_no_sequences() -> PatentMetadata:
    text_path = os.path.join(SAMPLE_DIR, "mock_patent_no_sequences.txt")
    with open(text_path) as f:
        text = f.read()
    return PatentMetadata(
        patent_number="WO2021987654A1",
        title="Thermostable Enzyme",
        full_text=text,
        has_sequence_listing=False,
        retrieval_status="success",
    )


def _make_patent_image_only() -> PatentMetadata:
    text_path = os.path.join(SAMPLE_DIR, "mock_patent_image_only.txt")
    with open(text_path) as f:
        text = f.read()
    return PatentMetadata(
        patent_number="EP3456789B1",
        title="Novel Protease",
        full_text=text,
        has_sequence_listing=False,
        retrieval_status="success",
    )


class TestSequenceExtractionStageA:
    """Scenario 1: Patent with standard sequence listing."""

    def test_extracts_sequences_from_listing(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = SequenceExtractionAgent(config, logger, llm, mock_mode=True)
        patent = _make_patent_with_sequences()

        sequences = agent.extract(patent)

        assert len(sequences) > 0, "Should extract at least one sequence from listing"

    def test_all_sequences_need_human_review(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = SequenceExtractionAgent(config, logger, llm, mock_mode=True)
        patent = _make_patent_with_sequences()

        sequences = agent.extract(patent)

        for seq in sequences:
            assert not seq.status.is_human_resolved(), (
                f"Sequence {seq.sequence_id} must NOT be auto-resolved. "
                f"Status: {seq.status.value}"
            )

    def test_sequences_have_provenance(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = SequenceExtractionAgent(config, logger, llm, mock_mode=True)
        patent = _make_patent_with_sequences()

        sequences = agent.extract(patent)

        for seq in sequences:
            assert seq.provenance is not None, (
                f"Sequence {seq.sequence_id} must have provenance"
            )
            assert seq.provenance.source_type, "Provenance must have source_type"

    def test_stage_a_sequences_have_correct_confidence(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = SequenceExtractionAgent(config, logger, llm, mock_mode=True)
        patent = _make_patent_with_sequences()

        sequences = agent._stage_a_sequence_listing(patent)

        for seq in sequences:
            assert seq.confidence == SequenceConfidence.DIRECT_SEQUENCE_LISTING
            assert seq.extraction_stage == ExtractionStage.STAGE_A_SEQUENCE_LISTING


class TestSequenceExtractionStageC:
    """Scenario 3: Patent with image-only sequences."""

    def test_detects_image_only_sequences(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = SequenceExtractionAgent(config, logger, llm, mock_mode=True)
        patent = _make_patent_image_only()

        image_results = agent._stage_c_image_detection(patent)

        assert len(image_results) > 0, "Should detect image-only sequence references"
        for seq in image_results:
            assert seq.confidence == SequenceConfidence.IMAGE_ONLY
            assert seq.status == ReviewStatus.NEEDS_HUMAN_REVIEW

    def test_image_only_has_empty_sequence(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = SequenceExtractionAgent(config, logger, llm, mock_mode=True)
        patent = _make_patent_image_only()

        image_results = agent._stage_c_image_detection(patent)

        for seq in image_results:
            assert seq.amino_acid_sequence == "", "Image-only should have empty sequence"


class TestSequenceExtractionStageD:
    """Scenario 2: Patent without sequences → web fallback."""

    def test_web_fallback_triggered(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        web_finder = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)
        agent = SequenceExtractionAgent(
            config, logger, llm, web_finder=web_finder, mock_mode=True
        )
        patent = _make_patent_no_sequences()

        sequences = agent.extract(patent)

        web_seqs = [
            s for s in sequences
            if s.confidence == SequenceConfidence.INFERRED_FROM_WEB
        ]
        assert len(web_seqs) > 0, "Web fallback should produce INFERRED_FROM_WEB sequences"

    def test_web_sequences_always_need_review(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        web_finder = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)
        agent = SequenceExtractionAgent(
            config, logger, llm, web_finder=web_finder, mock_mode=True
        )
        patent = _make_patent_no_sequences()

        sequences = agent.extract(patent)

        for seq in sequences:
            if seq.confidence == SequenceConfidence.INFERRED_FROM_WEB:
                assert seq.status == ReviewStatus.NEEDS_HUMAN_REVIEW, (
                    "Web-inferred sequences MUST be NEEDS_HUMAN_REVIEW"
                )
                assert "web" in seq.status_reason.lower() or "inferred" in seq.status_reason.lower()


class TestReviewQueueIntegrity:
    """Ensure the review queue enforces 100% human review."""

    def test_review_items_generated_for_all_sequences(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = SequenceExtractionAgent(config, logger, llm, mock_mode=True)
        patent = _make_patent_with_sequences()

        sequences = agent.extract(patent)
        items = agent.build_review_items(sequences, patent)

        assert len(items) >= len(sequences), (
            "Must have at least one review item per sequence"
        )

    def test_zero_extraction_generates_failure_item(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = SequenceExtractionAgent(config, logger, llm, mock_mode=True)
        patent = PatentMetadata(
            patent_number="EMPTY",
            full_text="This patent has no biological content.",
            has_sequence_listing=False,
            retrieval_status="success",
        )

        sequences = agent.extract(patent)
        items = agent.build_review_items(sequences, patent)

        if not sequences:
            failure_items = [i for i in items if i.item_type == "extraction_failure"]
            assert len(failure_items) > 0, (
                "Zero extraction MUST generate an extraction_failure review item"
            )
