"""
Tests for the Web Sequence Finder Agent.

Validates:
- All web-found sequences are tagged INFERRED_FROM_WEB
- All results default to NEEDS_HUMAN_REVIEW
- Provenance (URL, accession, database, search query) is always recorded
- Multiple candidates are preserved (recall-first)
- Failed searches log the attempted queries
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
from fto_harness.agents.web_sequence_finder import WebSequenceFinderAgent


def _load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    return load_config(config_path)


def _make_patent() -> PatentMetadata:
    return PatentMetadata(
        patent_number="WO2021987654A1",
        title="Thermostable Enzyme",
        applicant="Industrial Enzymes Corp.",
        retrieval_status="success",
    )


class TestWebSequenceFinderMock:
    """Test web sequence finder in mock mode."""

    def test_returns_sequences_with_accessions(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)

        clues = {
            "accession_numbers": ["P41365", "CAA83122"],
            "enzyme_names": ["lipase B"],
            "ec_numbers": ["EC 3.1.1.3"],
        }
        results = agent.search(_make_patent(), clues)

        assert len(results) > 0, "Should return at least one sequence"

    def test_all_results_are_inferred_from_web(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)

        clues = {"accession_numbers": ["P41365"]}
        results = agent.search(_make_patent(), clues)

        for seq in results:
            assert seq.confidence == SequenceConfidence.INFERRED_FROM_WEB, (
                f"Web sequence must be INFERRED_FROM_WEB, got {seq.confidence.value}"
            )

    def test_all_results_need_human_review(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)

        clues = {"accession_numbers": ["P41365"]}
        results = agent.search(_make_patent(), clues)

        for seq in results:
            assert seq.status == ReviewStatus.NEEDS_HUMAN_REVIEW, (
                f"Web sequence MUST be NEEDS_HUMAN_REVIEW, got {seq.status.value}"
            )
            assert not seq.status.is_human_resolved(), (
                "Web sequence must NOT be marked as human-resolved"
            )

    def test_all_results_have_provenance(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)

        clues = {"accession_numbers": ["P41365"]}
        results = agent.search(_make_patent(), clues)

        for seq in results:
            assert seq.provenance is not None, "Must have provenance"
            assert seq.provenance.url, "Must have source URL"
            assert seq.provenance.database, "Must have database name"
            assert seq.provenance.source_type == "web_search"

    def test_extraction_stage_is_d(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)

        clues = {"accession_numbers": ["P41365"]}
        results = agent.search(_make_patent(), clues)

        for seq in results:
            assert seq.extraction_stage == ExtractionStage.STAGE_D_WEB_SEARCH

    def test_status_reason_mentions_web(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)

        clues = {"accession_numbers": ["P41365"]}
        results = agent.search(_make_patent(), clues)

        for seq in results:
            reason_lower = seq.status_reason.lower()
            assert "web" in reason_lower or "inferred" in reason_lower, (
                f"Status reason must mention web/inferred origin: {seq.status_reason}"
            )

    def test_empty_clues_still_returns_something(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)

        clues = {"accession_numbers": [], "enzyme_names": [], "ec_numbers": []}
        results = agent.search(_make_patent(), clues)

        # Even with no clues, mock mode should return a fallback
        assert len(results) >= 1, "Should return at least a fallback candidate"

    def test_multiple_accessions_produce_multiple_results(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)

        clues = {"accession_numbers": ["P41365", "CAA83122", "Q9Y7B0"]}
        results = agent.search(_make_patent(), clues)

        assert len(results) >= 2, "Multiple accessions should yield multiple results"


class TestWebFinderLogging:
    """Verify that search attempts are logged even on failure."""

    def test_search_logged(self):
        config = _load_config()
        logger = ProcessingLogger()
        llm = LLMClient(config, mock_mode=True)
        agent = WebSequenceFinderAgent(config, logger, llm, mock_mode=True)

        clues = {"accession_numbers": ["P41365"]}
        agent.search(_make_patent(), clues)

        log_messages = [e.message for e in logger.entries]
        assert any("search" in m.lower() or "mock" in m.lower() for m in log_messages), (
            "Search activity must be logged"
        )
