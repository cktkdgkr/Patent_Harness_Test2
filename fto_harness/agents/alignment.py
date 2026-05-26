"""
Agent 4 — Alignment Agent.

Runs local BLASTP (and optionally MMseqs2) to compare reference sequences
against patent-derived sequences.

Recall-first: identity >= threshold (default 70%) → POTENTIAL_RISK.
Web-inferred sequences are flagged separately.
MMseqs2 disagreement → NEEDS_HUMAN_REVIEW escalation.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from ..core.logging_utils import ProcessingLogger
from ..core.models import (
    AgentName,
    AlignmentResult,
    HumanReviewItem,
    PatentMetadata,
    Provenance,
    ReviewStatus,
    RiskLevel,
    SequenceConfidence,
    SequenceRecord,
)

AGENT = AgentName.ALIGNMENT


class AlignmentAgent:
    def __init__(
        self,
        config: dict,
        logger: ProcessingLogger,
        mock_mode: bool = False,
    ) -> None:
        self.config = config
        self.logger = logger
        self.mock_mode = mock_mode
        align_cfg = config.get("alignment", {})
        self.identity_threshold = align_cfg.get("identity_threshold_percent", 70.0)
        self.coverage_threshold = align_cfg.get("coverage_threshold_percent", 50.0)
        self.evalue_cutoff = align_cfg.get("evalue_cutoff", 1e-5)
        self.use_mmseqs2 = align_cfg.get("use_mmseqs2_cross_validation", False)

    def align_all(
        self,
        reference_sequences: list[str],
        patent_sequences: list[SequenceRecord],
    ) -> list[AlignmentResult]:
        """Align every reference sequence against every patent sequence."""
        if not reference_sequences:
            self.logger.log_failure(AGENT, "align", "No reference sequences provided.")
            return []
        if not patent_sequences:
            self.logger.log_failure(AGENT, "align", "No patent sequences to align against.")
            return []

        results = []
        for ref_idx, ref_seq in enumerate(reference_sequences):
            for pat_seq in patent_sequences:
                if not pat_seq.amino_acid_sequence:
                    self.logger.log_skipped(
                        AGENT, "align",
                        f"Skipping empty sequence {pat_seq.sequence_id} (e.g., image-only).",
                    )
                    continue

                result = self._align_pair(
                    ref_seq=ref_seq,
                    ref_id=f"ref_{ref_idx}",
                    patent_seq=pat_seq,
                )
                results.append(result)

        self.logger.log_success(
            AGENT, "align_all",
            f"Completed {len(results)} alignments "
            f"({len(reference_sequences)} refs x {len([s for s in patent_sequences if s.amino_acid_sequence])} seqs).",
        )
        return results

    def _align_pair(
        self,
        ref_seq: str,
        ref_id: str,
        patent_seq: SequenceRecord,
    ) -> AlignmentResult:
        """Align one reference against one patent sequence."""
        is_web = patent_seq.confidence == SequenceConfidence.INFERRED_FROM_WEB

        if self.mock_mode:
            return self._mock_align(ref_seq, ref_id, patent_seq, is_web)

        blast_result = self._run_blastp(ref_seq, ref_id, patent_seq)

        if blast_result is None:
            self.logger.log_failure(
                AGENT, "blastp",
                f"BLAST failed for {ref_id} vs {patent_seq.sequence_id}.",
            )
            return AlignmentResult(
                reference_id=ref_id,
                patent_sequence_id=patent_seq.sequence_id,
                status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                status_reason="BLAST alignment failed — manual alignment required",
                is_web_inferred=is_web,
            )

        # MMseqs2 cross-validation
        mmseqs_agrees = None
        if self.use_mmseqs2:
            mmseqs_agrees = self._run_mmseqs2_check(ref_seq, patent_seq, blast_result)
            if mmseqs_agrees is False:
                blast_result.mmseqs2_agrees = False
                blast_result.status = ReviewStatus.NEEDS_HUMAN_REVIEW
                blast_result.status_reason = (
                    "BLAST and MMseqs2 results disagree — manual verification required"
                )
                self.logger.log_uncertain(
                    AGENT, "cross_validation",
                    f"BLAST/MMseqs2 disagreement for {ref_id} vs {patent_seq.sequence_id}.",
                )
            else:
                blast_result.mmseqs2_agrees = True

        return blast_result

    def _run_blastp(
        self,
        ref_seq: str,
        ref_id: str,
        patent_seq: SequenceRecord,
    ) -> AlignmentResult | None:
        """Run local blastp and parse results."""
        is_web = patent_seq.confidence == SequenceConfidence.INFERRED_FROM_WEB

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".fasta", delete=False
            ) as ref_f:
                ref_f.write(f">{ref_id}\n{ref_seq}\n")
                ref_path = ref_f.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".fasta", delete=False
            ) as subj_f:
                subj_f.write(f">{patent_seq.sequence_id}\n{patent_seq.amino_acid_sequence}\n")
                subj_path = subj_f.name

            db_path = subj_path + "_db"
            subprocess.run(
                ["makeblastdb", "-in", subj_path, "-dbtype", "prot", "-out", db_path],
                capture_output=True, text=True, check=True,
            )

            result = subprocess.run(
                [
                    "blastp",
                    "-query", ref_path,
                    "-db", db_path,
                    "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen",
                    "-evalue", str(self.evalue_cutoff),
                    "-max_target_seqs", "1",
                ],
                capture_output=True, text=True, check=True,
            )

            # Also get alignment visualization
            viz_result = subprocess.run(
                [
                    "blastp",
                    "-query", ref_path,
                    "-db", db_path,
                    "-evalue", str(self.evalue_cutoff),
                    "-max_target_seqs", "1",
                ],
                capture_output=True, text=True,
            )

            return self._parse_blast_output(
                result.stdout, viz_result.stdout, ref_id, patent_seq, is_web
            )

        except FileNotFoundError:
            self.logger.log_failure(
                AGENT, "blastp",
                "BLAST+ not found on PATH. Install NCBI BLAST+ and ensure 'blastp' is available.",
            )
            return None
        except subprocess.CalledProcessError as e:
            self.logger.log_failure(
                AGENT, "blastp", f"BLAST error: {e.stderr}", ref_id=ref_id,
            )
            return None
        finally:
            for p in [ref_path, subj_path]:
                if os.path.exists(p):
                    os.unlink(p)
            for ext in [".phr", ".pin", ".psq", ".pdb", ".pot", ".ptf", ".pto"]:
                p = db_path + ext
                if os.path.exists(p):
                    os.unlink(p)

    def _parse_blast_output(
        self,
        tabular: str,
        visualization: str,
        ref_id: str,
        patent_seq: SequenceRecord,
        is_web: bool,
    ) -> AlignmentResult:
        """Parse BLAST tabular (format 6) output."""
        lines = [l.strip() for l in tabular.strip().split("\n") if l.strip()]

        if not lines:
            return AlignmentResult(
                reference_id=ref_id,
                patent_sequence_id=patent_seq.sequence_id,
                percent_identity=0.0,
                percent_coverage=0.0,
                evalue=float("inf"),
                risk_level=RiskLevel.LOW,
                status=ReviewStatus.NEEDS_HUMAN_REVIEW,
                status_reason="No BLAST hit found — sequences may be unrelated, but human should verify",
                is_web_inferred=is_web,
                raw_blast_output=tabular,
                alignment_visualization=visualization,
            )

        fields = lines[0].split("\t")
        pident = float(fields[2])
        align_len = int(fields[3])
        evalue = float(fields[10])
        qlen = int(fields[12]) if len(fields) > 12 else len("unknown")
        slen = int(fields[13]) if len(fields) > 13 else patent_seq.length

        coverage = (align_len / max(qlen, 1)) * 100.0

        risk_level, status, reason = self._classify_risk(pident, coverage, evalue, is_web)

        self.logger.log_success(
            AGENT, "blastp_result",
            f"{ref_id} vs {patent_seq.sequence_id}: "
            f"identity={pident:.1f}%, coverage={coverage:.1f}%, E={evalue:.2e} → {risk_level.value}",
        )

        return AlignmentResult(
            reference_id=ref_id,
            patent_sequence_id=patent_seq.sequence_id,
            percent_identity=pident,
            percent_coverage=coverage,
            evalue=evalue,
            alignment_visualization=visualization,
            raw_blast_output=tabular,
            risk_level=risk_level,
            status=status,
            status_reason=reason,
            is_web_inferred=is_web,
        )

    def _classify_risk(
        self, pident: float, coverage: float, evalue: float, is_web: bool
    ) -> tuple[RiskLevel, ReviewStatus, str]:
        """
        Classify alignment risk. Recall-first: thresholds are intentionally
        low to catch borderline cases.
        """
        if pident >= self.identity_threshold and coverage >= self.coverage_threshold:
            risk = RiskLevel.HIGH
            status = ReviewStatus.POTENTIAL_RISK
            reason = (
                f"High sequence similarity ({pident:.1f}% identity, {coverage:.1f}% coverage) "
                f"exceeds threshold ({self.identity_threshold}%). POTENTIAL FTO RISK."
            )
        elif pident >= self.identity_threshold * 0.8:
            risk = RiskLevel.MEDIUM
            status = ReviewStatus.NEEDS_HUMAN_REVIEW
            reason = (
                f"Moderate similarity ({pident:.1f}% identity). Below primary threshold "
                f"but within margin — human review recommended."
            )
        elif evalue < self.evalue_cutoff:
            risk = RiskLevel.MEDIUM
            status = ReviewStatus.NEEDS_HUMAN_REVIEW
            reason = (
                f"Statistically significant hit (E={evalue:.2e}) despite lower identity "
                f"({pident:.1f}%). May indicate partial overlap — human review recommended."
            )
        else:
            risk = RiskLevel.LOW
            status = ReviewStatus.NEEDS_HUMAN_REVIEW
            reason = (
                f"Low similarity ({pident:.1f}% identity, E={evalue:.2e}). "
                f"Likely low risk but queued for human verification per policy."
            )

        if is_web:
            reason += " [SOURCE: INFERRED_FROM_WEB — sequence not confirmed from patent text]"
            if status != ReviewStatus.POTENTIAL_RISK:
                status = ReviewStatus.NEEDS_HUMAN_REVIEW

        return risk, status, reason

    def _run_mmseqs2_check(
        self,
        ref_seq: str,
        patent_seq: SequenceRecord,
        blast_result: AlignmentResult,
    ) -> bool | None:
        """Run MMseqs2 and check if it agrees with BLAST."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ref_path = os.path.join(tmpdir, "ref.fasta")
                subj_path = os.path.join(tmpdir, "subj.fasta")
                with open(ref_path, "w") as f:
                    f.write(f">ref\n{ref_seq}\n")
                with open(subj_path, "w") as f:
                    f.write(f">{patent_seq.sequence_id}\n{patent_seq.amino_acid_sequence}\n")

                result = subprocess.run(
                    [
                        "mmseqs", "easy-search",
                        ref_path, subj_path,
                        os.path.join(tmpdir, "result.m8"),
                        os.path.join(tmpdir, "tmp"),
                        "--format-output", "query,target,pident,alnlen,evalue",
                    ],
                    capture_output=True, text=True, timeout=120,
                )

                if result.returncode != 0:
                    self.logger.log_failure(
                        AGENT, "mmseqs2", f"MMseqs2 error: {result.stderr}"
                    )
                    return None

                out_path = os.path.join(tmpdir, "result.m8")
                if not os.path.exists(out_path):
                    return None

                with open(out_path) as f:
                    lines = f.readlines()

                if not lines:
                    return blast_result.percent_identity < 30.0

                mmseqs_pident = float(lines[0].split("\t")[2])
                diff = abs(blast_result.percent_identity - mmseqs_pident)
                agrees = diff < 10.0
                self.logger.log_success(
                    AGENT, "mmseqs2",
                    f"MMseqs2 identity={mmseqs_pident:.1f}%, BLAST={blast_result.percent_identity:.1f}%, "
                    f"diff={diff:.1f}%, agrees={agrees}",
                )
                return agrees

        except FileNotFoundError:
            self.logger.log_skipped(AGENT, "mmseqs2", "MMseqs2 not found on PATH. Skipping.")
            return None
        except Exception as e:
            self.logger.log_failure(AGENT, "mmseqs2", f"MMseqs2 error: {e}")
            return None

    def _mock_align(
        self,
        ref_seq: str,
        ref_id: str,
        patent_seq: SequenceRecord,
        is_web: bool,
    ) -> AlignmentResult:
        """Generate mock alignment results for testing."""
        # Simulate different identity levels based on sequence similarity
        if ref_seq[:20] == patent_seq.amino_acid_sequence[:20]:
            pident = 95.2
            coverage = 98.0
            evalue = 1e-50
        elif any(ref_seq[i:i+10] in patent_seq.amino_acid_sequence for i in range(0, min(len(ref_seq), 50), 10)):
            pident = 75.3
            coverage = 80.0
            evalue = 1e-20
        else:
            pident = 35.0
            coverage = 45.0
            evalue = 1e-3

        risk_level, status, reason = self._classify_risk(pident, coverage, evalue, is_web)

        viz = (
            f"Query  {ref_id}  1    {ref_seq[:50]}  50\n"
            f"                      {'|' * 30}{'.' * 20}\n"
            f"Sbjct  {patent_seq.sequence_id}  1    {patent_seq.amino_acid_sequence[:50]}  50\n"
        )

        self.logger.log_success(
            AGENT, "mock_align",
            f"Mock: {ref_id} vs {patent_seq.sequence_id}: "
            f"identity={pident}%, coverage={coverage}%, risk={risk_level.value}",
        )

        return AlignmentResult(
            reference_id=ref_id,
            patent_sequence_id=patent_seq.sequence_id,
            percent_identity=pident,
            percent_coverage=coverage,
            evalue=evalue,
            alignment_visualization=viz,
            raw_blast_output=f"mock\t{pident}\t{coverage}\t{evalue}",
            risk_level=risk_level,
            status=status,
            status_reason=reason,
            is_web_inferred=is_web,
        )

    def build_review_items(
        self, results: list[AlignmentResult], patent: PatentMetadata
    ) -> list[HumanReviewItem]:
        """Convert alignment results into human review queue items."""
        items = []
        for ar in results:
            items.append(
                HumanReviewItem(
                    patent_number=patent.patent_number,
                    item_type="alignment",
                    status=ar.status,
                    risk_level=ar.risk_level,
                    summary=(
                        f"Alignment: {ar.reference_id} vs {ar.patent_sequence_id} — "
                        f"{ar.percent_identity:.1f}% identity, {ar.percent_coverage:.1f}% coverage"
                        f"{' [WEB-INFERRED]' if ar.is_web_inferred else ''}"
                    ),
                    evidence={
                        "alignment_id": ar.alignment_id,
                        "percent_identity": ar.percent_identity,
                        "percent_coverage": ar.percent_coverage,
                        "evalue": ar.evalue,
                        "risk_level": ar.risk_level.value,
                        "is_web_inferred": ar.is_web_inferred,
                        "mmseqs2_agrees": ar.mmseqs2_agrees,
                    },
                    provenance=Provenance(
                        source_type="blast_alignment",
                        patent_number=patent.patent_number,
                    ),
                    related_alignment_ids=[ar.alignment_id],
                )
            )
        return items
