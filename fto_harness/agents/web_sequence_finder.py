"""
Agent 3 — Web Sequence Finder Agent.

Searches public sequence databases when patent text lacks explicit sequences.
ALL results are tagged INFERRED_FROM_WEB and default to NEEDS_HUMAN_REVIEW.
The tool NEVER asserts that a web-found sequence IS the patent sequence.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..core.logging_utils import ProcessingLogger
from ..core.models import (
    AgentName,
    ExtractionStage,
    PatentMetadata,
    Provenance,
    ReviewStatus,
    SequenceConfidence,
    SequenceRecord,
)

if TYPE_CHECKING:
    from ..core.llm_client import LLMClient

AGENT = AgentName.WEB_SEQUENCE_FINDER

UNIPROT_ACCESSION_RE = re.compile(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b")
GENBANK_ACCESSION_RE = re.compile(r"\b([A-Z]{3}\d{5}(\.\d+)?)\b")


class WebSequenceFinderAgent:
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
        self.max_candidates = config.get("web_sequence_finder", {}).get("max_candidates", 20)
        self.timeout = config.get("web_sequence_finder", {}).get("request_timeout", 30)
        self.databases = config.get("web_sequence_finder", {}).get(
            "databases", ["uniprot", "ncbi_protein", "genbank", "pdb", "brenda"]
        )

    def search(
        self, patent: PatentMetadata, clues: dict
    ) -> list[SequenceRecord]:
        """
        Search external databases for sequences related to the patent.

        All returned sequences are INFERRED_FROM_WEB / NEEDS_HUMAN_REVIEW.
        Multiple candidates are preserved (recall-first).
        """
        self.logger.log(
            AGENT, "search_start", "started",
            f"Searching for sequences related to patent {patent.patent_number}",
            clues=clues,
        )

        if self.mock_mode:
            return self._mock_search(patent, clues)

        all_sequences: list[SequenceRecord] = []
        queries_tried: list[str] = []

        # Strategy 1: Direct accession lookup
        accessions = clues.get("accession_numbers", [])
        for acc in accessions:
            seqs = self._fetch_by_accession(acc, patent)
            all_sequences.extend(seqs)
            queries_tried.append(f"accession:{acc}")

        # Strategy 2: Search by enzyme name + organism
        if len(all_sequences) < self.max_candidates:
            enzyme_names = clues.get("enzyme_names", [])
            organisms = clues.get("organism_names", [])
            for name in enzyme_names:
                for org in (organisms or [""]):
                    query = f"{name} {org}".strip()
                    seqs = self._search_databases(query, patent)
                    all_sequences.extend(seqs)
                    queries_tried.append(f"name_search:{query}")

        # Strategy 3: Search by EC number
        if len(all_sequences) < self.max_candidates:
            ec_numbers = clues.get("ec_numbers", [])
            for ec in ec_numbers:
                seqs = self._search_databases(ec, patent)
                all_sequences.extend(seqs)
                queries_tried.append(f"ec_search:{ec}")

        # Deduplicate
        seen = set()
        unique = []
        for seq in all_sequences:
            if seq.amino_acid_sequence and seq.amino_acid_sequence not in seen:
                seen.add(seq.amino_acid_sequence)
                unique.append(seq)

        # Cap at max_candidates
        unique = unique[: self.max_candidates]

        if unique:
            self.logger.log_success(
                AGENT, "search_complete",
                f"Found {len(unique)} candidate sequences from web search.",
                queries=queries_tried,
            )
        else:
            self.logger.log_failure(
                AGENT, "search_complete",
                f"No sequences found via web search. Queries tried: {queries_tried}",
                queries=queries_tried,
            )

        return unique

    def _fetch_by_accession(
        self, accession: str, patent: PatentMetadata
    ) -> list[SequenceRecord]:
        """Fetch sequence directly by accession number from public databases."""
        self.logger.log(
            AGENT, "accession_fetch", "started", f"Fetching accession: {accession}"
        )

        try:
            import requests

            # Try UniProt
            if UNIPROT_ACCESSION_RE.match(accession):
                url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
                resp = requests.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    seq = self._parse_fasta(resp.text)
                    if seq:
                        self.logger.log_success(
                            AGENT, "accession_fetch",
                            f"Found sequence for {accession} on UniProt ({len(seq)} aa).",
                        )
                        return [self._make_web_record(
                            seq, accession, "uniprot", url, patent,
                            f"Direct UniProt fetch for {accession}",
                        )]

            # Try NCBI
            url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                f"db=protein&id={accession}&rettype=fasta&retmode=text"
            )
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200 and ">" in resp.text:
                seq = self._parse_fasta(resp.text)
                if seq:
                    self.logger.log_success(
                        AGENT, "accession_fetch",
                        f"Found sequence for {accession} on NCBI ({len(seq)} aa).",
                    )
                    return [self._make_web_record(
                        seq, accession, "ncbi_protein", url, patent,
                        f"Direct NCBI fetch for {accession}",
                    )]

        except Exception as e:
            self.logger.log_failure(
                AGENT, "accession_fetch",
                f"Failed to fetch {accession}: {e}",
            )

        return []

    def _search_databases(
        self, query: str, patent: PatentMetadata
    ) -> list[SequenceRecord]:
        """Search databases by keyword query."""
        self.logger.log(
            AGENT, "keyword_search", "started", f"Searching databases for: {query}"
        )

        results = []
        try:
            import requests

            # UniProt search
            url = (
                f"https://rest.uniprot.org/uniprotkb/search?"
                f"query={query}&format=fasta&size=5"
            )
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200 and resp.text.strip():
                entries = resp.text.strip().split(">")[1:]
                for entry in entries[: self.max_candidates]:
                    full_entry = ">" + entry
                    seq = self._parse_fasta(full_entry)
                    header = entry.split("\n")[0] if "\n" in entry else entry[:100]
                    acc_match = re.search(r"\|(\w+)\|", header)
                    acc = acc_match.group(1) if acc_match else "unknown"
                    if seq:
                        results.append(self._make_web_record(
                            seq, acc, "uniprot", url, patent,
                            f"UniProt search: '{query}', header: {header[:100]}",
                        ))

        except Exception as e:
            self.logger.log_failure(
                AGENT, "keyword_search", f"Database search failed for '{query}': {e}",
            )

        return results

    def _parse_fasta(self, fasta_text: str) -> str:
        """Parse FASTA format and return the amino acid sequence."""
        lines = fasta_text.strip().split("\n")
        seq_lines = [l.strip() for l in lines if not l.startswith(">") and l.strip()]
        return "".join(seq_lines).upper()

    def _make_web_record(
        self,
        sequence: str,
        accession: str,
        database: str,
        url: str,
        patent: PatentMetadata,
        search_notes: str,
    ) -> SequenceRecord:
        return SequenceRecord(
            amino_acid_sequence=sequence,
            confidence=SequenceConfidence.INFERRED_FROM_WEB,
            extraction_stage=ExtractionStage.STAGE_D_WEB_SEARCH,
            provenance=Provenance(
                source_type="web_search",
                patent_number=patent.patent_number,
                url=url,
                accession=accession,
                database=database,
                notes=search_notes,
                extraction_stage=ExtractionStage.STAGE_D_WEB_SEARCH,
            ),
            status=ReviewStatus.NEEDS_HUMAN_REVIEW,
            status_reason=(
                f"Sequence inferred from web ({database}, accession: {accession}) "
                f"— NOT confirmed as the patent sequence. Human verification required."
            ),
        )

    def _mock_search(
        self, patent: PatentMetadata, clues: dict
    ) -> list[SequenceRecord]:
        """Return mock sequences for testing."""
        self.logger.log_success(AGENT, "mock_search", "Returning mock web search results.")

        mock_seqs = []
        accessions = clues.get("accession_numbers", [])
        if accessions:
            for acc in accessions[:3]:
                mock_seqs.append(self._make_web_record(
                    sequence=(
                        "LPSGSDPAFSQPKSVLDAGLTCQGASPSSVSKPILLVPGTGTTGPQSFDS"
                        "NWIPLSTQLGYTPCWISPPPFMLGPFHAGSIAPDTRLFNKYLQMKSFNLP"
                    ),
                    accession=acc,
                    database="uniprot_mock",
                    url=f"https://www.uniprot.org/uniprot/{acc}",
                    patent=patent,
                    search_notes=f"Mock fetch for accession {acc}",
                ))

        if not mock_seqs:
            mock_seqs.append(self._make_web_record(
                sequence=(
                    "MKTLLLTLVVVTLVLSSPCILSQPVLTQPPSVSAAPGQRVTISCSGSSSN"
                    "IGSNTVNWYQQLPGTAPKLLIYSNNQRPSGVPDRFSGSKSGTSA"
                ),
                accession="MOCK001",
                database="mock_db",
                url="https://example.com/mock",
                patent=patent,
                search_notes="Mock fallback sequence — no accessions found in patent",
            ))

        return mock_seqs
