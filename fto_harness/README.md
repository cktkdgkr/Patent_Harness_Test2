# FTO Patent Review Harness

Multi-agent system for accelerating Freedom-to-Operate (FTO) patent reviews
in enzyme development. This tool is a **co-pilot** — it assists human reviewers
but **never** automatically excludes patents from review.

## Design Principles (Non-Negotiable)

1. **Co-pilot, not automation** — Every patent/sequence enters the human review queue.
2. **Recall-first** — Thresholds are set to over-flag (false positives OK, false negatives minimized).
3. **Self-aware of limits** — All failures and uncertainties are flagged `NEEDS_HUMAN_REVIEW`.
4. **Verifiable evidence** — Every result carries source, raw data, and reasoning.
5. **100% review before clearance** — No `CLEARED` report until all queue items are human-resolved.

## Architecture

```
Orchestrator
├── Agent 1: Patent Retrieval       (patent number/PDF → metadata + text)
├── Agent 2: Sequence Extraction    (4-stage fallback: A→B→C→D)
│   ├── Stage A: ST.26/ST.25 sequence listing parser
│   ├── Stage B: Regex + LLM text extraction
│   ├── Stage C: Image-only sequence detection
│   └── Stage D: Web Sequence Finder (Agent 3)
├── Agent 3: Web Sequence Finder    (UniProt/NCBI/GenBank/PDB search)
├── Agent 4: Alignment              (BLASTP + optional MMseqs2)
├── Agent 5: Claim Risk Annotator   (LLM-assisted claim analysis)
└── Reporting Layer
    ├── human_review_queue.json
    ├── processing_log.json
    ├── report.md
    └── full_result.json
```

## Installation

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. NCBI BLAST+ (Required for alignment)

```bash
# Ubuntu/Debian
sudo apt-get install ncbi-blast+

# macOS (Homebrew)
brew install blast

# Conda
conda install -c bioconda blast

# Verify
blastp -version
```

### 3. MMseqs2 (Optional, for cross-validation)

```bash
# Conda (recommended)
conda install -c conda-forge -c bioconda mmseqs2

# Or download from https://github.com/soedinglab/MMseqs2
```

### 4. Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

### Basic: Patent number + reference FASTA

```bash
python -m fto_harness.run --patent US20200123456A1 --ref reference.fasta
```

### From PDF

```bash
python -m fto_harness.run --pdf patent.pdf --ref reference.fasta
```

### Mock mode (no API key or BLAST+ needed)

```bash
python -m fto_harness.run --patent US20200123456A1 --ref tests/sample_data/reference.fasta --mock
```

### Direct reference sequence

```bash
python -m fto_harness.run --patent US20200123456A1 --ref-seq "MKTLLLTLVVVT..." --mock
```

### Custom output directory

```bash
python -m fto_harness.run --patent US20200123456A1 --ref ref.fasta -o results/
```

## Output Files

| File | Description |
|------|-------------|
| `human_review_queue.json` | All items sorted by risk, with status and evidence |
| `processing_log.json` | Step-by-step log of what each agent did |
| `report.md` | Human-readable summary with INCOMPLETE/COMPLETE banner |
| `full_result.json` | Complete structured result for programmatic use |

## Test Scenarios

### Scenario 1: Patent with sequence listing
Patent contains ST.26 sequences → extracted directly → aligned with reference → queued.

```bash
python -m fto_harness.run --patent US20200123456A1 \
  --ref tests/sample_data/reference.fasta --mock
```

### Scenario 2: Patent without explicit sequences
No sequences in text → Web Sequence Finder searches UniProt/NCBI by accession → results tagged `INFERRED_FROM_WEB` → queued for human verification.

```bash
python -m fto_harness.run --patent WO2021987654A1 \
  --ref tests/sample_data/reference.fasta --mock
```

### Scenario 3: Image-only sequences
Sequences exist only as figures → detection flags `IMAGE_ONLY` → `NEEDS_HUMAN_REVIEW`.

## Running Tests

```bash
# All tests
pytest fto_harness/tests/ -v

# Specific test file
pytest fto_harness/tests/test_sequence_extraction.py -v
pytest fto_harness/tests/test_web_sequence_finder.py -v
pytest fto_harness/tests/test_orchestrator_integration.py -v
```

## Configuration

Edit `config.yaml` to adjust:

- **`alignment.identity_threshold_percent`**: Default 70% (recall-first). Sequences above this → `POTENTIAL_RISK`.
- **`alignment.evalue_cutoff`**: BLAST E-value cutoff.
- **`alignment.use_mmseqs2_cross_validation`**: Enable MMseqs2 cross-check.
- **`web_sequence_finder.max_candidates`**: Max sequences to retrieve from web search.
- **`mock_mode.enabled`**: Use sample data instead of live APIs.
- **`llm.model`**: Anthropic model to use for LLM-assisted extraction.

## Disclaimer

This tool is a review aid only. Final FTO determination must be made by a
qualified human reviewer (patent attorney / IP specialist). No patent is
automatically excluded from review. All items require human verification.
