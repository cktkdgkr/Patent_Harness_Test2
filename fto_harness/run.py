#!/usr/bin/env python3
"""
CLI entry point for the FTO Patent Review Harness.

Usage:
    python -m fto_harness.run --patent US12345678B2 --ref ref.fasta
    python -m fto_harness.run --pdf patent.pdf --ref ref.fasta
    python -m fto_harness.run --patent US12345678B2 --ref ref.fasta --mock
"""

from __future__ import annotations

import argparse
import sys

from .orchestrator import FTOOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FTO Patent Review Harness — Enzyme patent sequence analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m fto_harness.run --patent US20200123456A1 --ref ref.fasta\n"
            "  python -m fto_harness.run --pdf patent.pdf --ref ref.fasta\n"
            "  python -m fto_harness.run --patent US20200123456A1 --ref ref.fasta --mock\n"
            "\n"
            "The --mock flag uses sample data instead of live APIs (no BLAST+ or API key needed).\n"
        ),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--patent", type=str,
        help="Patent number (e.g., US12345678B2, WO2020123456A1, KR10-2020-...)",
    )
    input_group.add_argument(
        "--pdf", type=str,
        help="Path to patent PDF file",
    )

    parser.add_argument(
        "--ref", type=str,
        help="Path to reference FASTA file with sequences to compare against",
    )
    parser.add_argument(
        "--ref-seq", type=str, action="append",
        help="Direct reference amino acid sequence (can be specified multiple times)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: fto_harness/config.yaml)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run in mock mode using sample data (no external APIs or BLAST+ needed)",
    )

    args = parser.parse_args()

    patent_input = args.patent or args.pdf
    if not args.ref and not args.ref_seq:
        print(
            "WARNING: No reference sequences provided (--ref or --ref-seq). "
            "Alignment will be skipped.",
            file=sys.stderr,
        )

    orchestrator = FTOOrchestrator(
        config_path=args.config,
        mock_mode=args.mock,
        output_dir=args.output,
    )

    result = orchestrator.run(
        patent_input=patent_input,
        reference_fasta_path=args.ref,
        reference_sequences=args.ref_seq,
    )

    # Summary
    queue = result.review_queue
    pending = len(queue.pending_items())
    total = len(queue.items)

    print("\n" + "=" * 60, file=sys.stderr)
    print("FTO ANALYSIS COMPLETE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Patent: {result.patent_input}", file=sys.stderr)
    print(f"  Sequences extracted: {len(result.extracted_sequences)}", file=sys.stderr)
    print(f"  Alignments performed: {len(result.alignment_results)}", file=sys.stderr)
    print(f"  Claim annotations: {len(result.claim_annotations)}", file=sys.stderr)
    print(f"  Review queue: {pending}/{total} items pending", file=sys.stderr)
    print(f"  Can issue clearance: {'YES' if result.can_issue_clearance() else 'NO'}", file=sys.stderr)
    print(f"  Output: {args.output}/", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if pending > 0:
        print(
            f"\n  ** {pending} items require human review before FTO clearance. **",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
