"""
Reporting Layer — Generates human-readable Markdown report and structured outputs.

ENFORCED: "REVIEW INCOMPLETE — CLEARED 판정 불가" banner is shown until
ALL human review queue items are resolved. The tool never produces a
CLEARED report automatically.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.models import FTOAnalysisResult


DISCLAIMER = (
    "> **DISCLAIMER**: This tool is a review aid only. Final FTO determination "
    "must be made by a qualified human reviewer (patent attorney / IP specialist). "
    "No patent has been automatically excluded from review. All items require human verification."
)

INCOMPLETE_BANNER = """
---

## !! REVIEW INCOMPLETE — CLEARED DETERMINATION NOT POSSIBLE !!

**{pending} of {total} review items have NOT been resolved by a human reviewer.**

A final FTO clearance report CANNOT be issued until every item in the
human review queue has been examined and resolved by a qualified reviewer.

---
"""

COMPLETE_BANNER = """
---

## All Review Items Resolved

All {total} review items have been examined by a human reviewer.
A qualified patent attorney may now issue a final FTO determination
based on the findings below.

---
"""


class ReportBuilder:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.custom_disclaimer = config.get("reporting", {}).get("disclaimer", "")

    def build(self, result: FTOAnalysisResult) -> str:
        sections = []

        # Header
        sections.append(f"# FTO Analysis Report")
        sections.append(f"**Run ID**: {result.run_id}")
        sections.append(f"**Generated**: {datetime.utcnow().isoformat()} UTC")
        sections.append(f"**Patent Input**: {result.patent_input}")
        sections.append("")

        # Disclaimer — always shown
        sections.append(DISCLAIMER)
        if self.custom_disclaimer:
            sections.append(f"\n> {self.custom_disclaimer}")
        sections.append("")

        # Completion status banner — enforces 100% review rule
        queue = result.review_queue
        pending = len(queue.pending_items())
        total = len(queue.items)
        if not queue.is_complete():
            sections.append(INCOMPLETE_BANNER.format(pending=pending, total=total))
        else:
            sections.append(COMPLETE_BANNER.format(total=total))

        # Patent metadata
        sections.append(self._patent_section(result))

        # Sequence extraction summary
        sections.append(self._sequences_section(result))

        # Alignment results
        sections.append(self._alignment_section(result))

        # Claim annotations
        sections.append(self._claims_section(result))

        # Review queue summary
        sections.append(self._queue_section(result))

        # Processing log summary
        sections.append(self._log_section(result))

        return "\n".join(sections)

    def _patent_section(self, result: FTOAnalysisResult) -> str:
        lines = ["## 1. Patent Information", ""]
        pm = result.patent_metadata
        if pm:
            lines.append(f"| Field | Value |")
            lines.append(f"|-------|-------|")
            lines.append(f"| Patent Number | {pm.patent_number} |")
            lines.append(f"| Title | {pm.title} |")
            lines.append(f"| Applicant | {pm.applicant} |")
            lines.append(f"| Publication Date | {pm.publication_date} |")
            lines.append(f"| Source URL | {pm.source_url} |")
            lines.append(f"| Sequence Listing | {'Yes' if pm.has_sequence_listing else 'No'} |")
            lines.append(f"| Retrieval Status | {pm.retrieval_status} |")
            if pm.retrieval_error:
                lines.append(f"| Error | {pm.retrieval_error} |")
        else:
            lines.append("*No patent metadata available.*")
        lines.append("")
        return "\n".join(lines)

    def _sequences_section(self, result: FTOAnalysisResult) -> str:
        lines = ["## 2. Extracted Sequences", ""]
        seqs = result.extracted_sequences
        if not seqs:
            lines.append("**No sequences extracted.** Manual extraction required.")
            lines.append("")
            return "\n".join(lines)

        lines.append(f"**Total sequences extracted**: {len(seqs)}")
        lines.append("")

        by_stage = {}
        for s in seqs:
            stage = s.extraction_stage.value if s.extraction_stage else "unknown"
            by_stage.setdefault(stage, []).append(s)

        for stage, stage_seqs in sorted(by_stage.items()):
            lines.append(f"### Stage {stage} ({len(stage_seqs)} sequences)")
            lines.append("")
            for s in stage_seqs:
                aa_preview = s.amino_acid_sequence[:60] + "..." if len(s.amino_acid_sequence) > 60 else s.amino_acid_sequence
                lines.append(f"- **{s.sequence_id}** [{s.confidence.value}]")
                lines.append(f"  - Sequence ({s.length} aa): `{aa_preview}`")
                lines.append(f"  - Status: {s.status.value} — {s.status_reason}")
                if s.provenance:
                    prov = s.provenance.to_dict()
                    prov_str = ", ".join(f"{k}={v}" for k, v in prov.items() if v and k != "source_type")
                    lines.append(f"  - Provenance: {prov_str}")
                lines.append("")

        return "\n".join(lines)

    def _alignment_section(self, result: FTOAnalysisResult) -> str:
        lines = ["## 3. Alignment Results", ""]
        alignments = result.alignment_results
        if not alignments:
            lines.append("*No alignments performed.*")
            lines.append("")
            return "\n".join(lines)

        lines.append(f"**Total alignments**: {len(alignments)}")
        identity_threshold = self.config.get("alignment", {}).get("identity_threshold_percent", 70.0)
        lines.append(f"**Risk threshold**: >= {identity_threshold}% identity → POTENTIAL_RISK")
        lines.append("")

        # Sort by risk (high first)
        risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
        sorted_aligns = sorted(alignments, key=lambda a: risk_order.get(a.risk_level.value, 99))

        lines.append("| Ref | Patent Seq | Identity | Coverage | E-value | Risk | Web? | Status |")
        lines.append("|-----|-----------|----------|----------|---------|------|------|--------|")
        for a in sorted_aligns:
            web_flag = "WEB" if a.is_web_inferred else "-"
            lines.append(
                f"| {a.reference_id} | {a.patent_sequence_id} | "
                f"{a.percent_identity:.1f}% | {a.percent_coverage:.1f}% | "
                f"{a.evalue:.2e} | **{a.risk_level.value}** | {web_flag} | {a.status.value} |"
            )
        lines.append("")

        # Detailed alignment visualizations
        high_risk = [a for a in sorted_aligns if a.risk_level.value in ("HIGH", "MEDIUM")]
        if high_risk:
            lines.append("### High/Medium Risk Alignment Details")
            lines.append("")
            for a in high_risk:
                lines.append(f"#### {a.reference_id} vs {a.patent_sequence_id} ({a.risk_level.value})")
                lines.append(f"- {a.status_reason}")
                if a.mmseqs2_agrees is not None:
                    lines.append(f"- MMseqs2 cross-validation: {'agrees' if a.mmseqs2_agrees else 'DISAGREES'}")
                if a.alignment_visualization:
                    lines.append("```")
                    lines.append(a.alignment_visualization[:2000])
                    lines.append("```")
                lines.append("")

        return "\n".join(lines)

    def _claims_section(self, result: FTOAnalysisResult) -> str:
        lines = ["## 4. Claim Annotations", ""]
        annotations = result.claim_annotations
        if not annotations:
            lines.append("*No claim annotations generated.*")
            lines.append("")
            return "\n".join(lines)

        lines.append(
            "> Note: Claim analysis is LLM-assisted and provided as supplementary "
            "information only. A qualified patent attorney must review all claims."
        )
        lines.append("")

        for ann in annotations:
            lines.append(f"### Claim {ann.claim_number}")
            lines.append(f"- **Text**: {ann.claim_text}")
            lines.append(f"- **Sequence refs**: {', '.join(ann.sequence_references) or 'none'}")
            lines.append(f"- **Identity thresholds**: {', '.join(ann.identity_thresholds_mentioned) or 'none'}")
            lines.append(f"- **Notes**: {ann.interpretation_notes}")
            lines.append(f"- **Confidence**: {ann.confidence}")
            lines.append(f"- **Status**: {ann.status.value}")
            lines.append("")

        return "\n".join(lines)

    def _queue_section(self, result: FTOAnalysisResult) -> str:
        lines = ["## 5. Human Review Queue", ""]
        queue = result.review_queue
        pending = queue.pending_items()
        total = len(queue.items)

        lines.append(f"**Total items**: {total}")
        lines.append(f"**Pending review**: {len(pending)}")
        lines.append(f"**Queue complete**: {'Yes' if queue.is_complete() else 'No'}")
        lines.append("")

        if queue.items:
            lines.append("| # | Type | Risk | Status | Summary |")
            lines.append("|---|------|------|--------|---------|")
            for i, item in enumerate(queue.sorted_by_risk(), 1):
                summary_short = (item.summary[:80] + "...") if len(item.summary) > 80 else item.summary
                lines.append(
                    f"| {i} | {item.item_type} | **{item.risk_level.value}** | "
                    f"{item.status.value} | {summary_short} |"
                )
            lines.append("")

        return "\n".join(lines)

    def _log_section(self, result: FTOAnalysisResult) -> str:
        lines = ["## 6. Processing Log Summary", ""]
        log = result.processing_log

        failures = [e for e in log if e.status == "failure"]
        uncertainties = [e for e in log if e.status == "uncertain"]

        if failures:
            lines.append(f"### Failures ({len(failures)})")
            for e in failures:
                lines.append(f"- [{e.agent}] {e.stage}: {e.message}")
            lines.append("")

        if uncertainties:
            lines.append(f"### Uncertainties ({len(uncertainties)})")
            for e in uncertainties:
                lines.append(f"- [{e.agent}] {e.stage}: {e.message}")
            lines.append("")

        if not failures and not uncertainties:
            lines.append("*No failures or uncertainties recorded.*")
            lines.append("")

        lines.append(f"*Full processing log: see processing_log.json ({len(log)} entries)*")
        lines.append("")
        return "\n".join(lines)
