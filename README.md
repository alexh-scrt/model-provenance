# model-provenance

> Audit AI models for supply chain integrity — before they reach production.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`model-provenance` is a command-line tool that audits AI models for supply chain integrity. It computes cryptographic fingerprints of model files, checks them against a known-good hash database, scans for tampered weights and malicious payloads, and generates structured provenance reports with licensing and regulatory compliance notes. Supports both Hugging Face Hub models and local model directories.

---

## Quick Start

```bash
# Install
pip install model-provenance

# Initialize the local hash database
model-provenance db init

# Verify a Hugging Face Hub model
model-provenance verify bert-base-uncased

# Verify a local model directory
model-provenance verify ./models/my-finetuned-model --local

# Generate a full provenance report as JSON
model-provenance report bert-base-uncased --format json
```

A passing audit exits with code `0`. A `WARN` or `FAIL` verdict exits with a non-zero code, making it easy to integrate into CI/CD pipelines.

---

## Features

- **Cryptographic fingerprinting** — Computes SHA-256 hashes for all model weight and config files and compares them against a local SQLite database seeded from a bundled YAML manifest of known-good hashes.
- **Hugging Face Hub integration** — Audits remote models without downloading full weights by using the HF Hub API to fetch file listings and model card metadata.
- **Supply chain attack scanner** — Detects pickle exploits, embedded shell scripts, unexpected ELF/PE executables, suspicious hardcoded URLs, and nested archive bombs inside model directories.
- **License & compliance checker** — Parses model card license fields, flags restricted or non-commercial licenses (e.g. CC-BY-NC, proprietary), and emits notes for EU AI Act and NIST RMF relevance.
- **Structured report output** — Renders results as Rich terminal tables, JSON, or YAML, with a clear `PASS` / `WARN` / `FAIL` verdict, per-file hash status, and actionable remediation guidance.

---

## Usage Examples

### Verify a Hugging Face Hub model

```bash
model-provenance verify bert-base-uncased
```

```
┌─────────────────────────────────────────────┐
│  model-provenance audit: bert-base-uncased  │
└─────────────────────────────────────────────┘
 File                          Status     Hash Match
 ─────────────────────────────────────────────────
 config.json                   ✓ PASS     known-good
 tokenizer_config.json         ✓ PASS     known-good
 pytorch_model.bin             ✓ PASS     known-good
 vocab.txt                     ✓ PASS     known-good

 License : apache-2.0    ✓ No restrictions
 Scanner : No suspicious findings

 Verdict: PASS
```

### Verify a local model directory

```bash
model-provenance verify ./models/my-model --local
```

### Generate a JSON provenance report

```bash
model-provenance report mistralai/Mistral-7B-v0.1 --format json --output report.json
```

```json
{
  "model_id": "mistralai/Mistral-7B-v0.1",
  "revision": "main",
  "verdict": "WARN",
  "fingerprint_coverage": 0.95,
  "files": [
    { "path": "config.json", "status": "match", "sha256": "a3f9..." },
    { "path": "model.safetensors", "status": "unknown", "sha256": "7c2b..." }
  ],
  "scan_findings": [],
  "license": {
    "identifier": "apache-2.0",
    "restriction_level": "permissive",
    "compliance_notes": []
  }
}
```

### Generate a YAML report

```bash
model-provenance report ./models/my-model --local --format yaml
```

### Manage the local hash database

```bash
# Seed / reinitialize the database from the bundled YAML
model-provenance db init

# Add a known-good hash manually
model-provenance db add --model-id owner/my-model --revision main \
  --file-path pytorch_model.bin --sha256 <hex-digest>

# List all audited models in the database
model-provenance db list

# Query hashes for a specific model
model-provenance db query owner/my-model

# Remove a hash record
model-provenance db remove --model-id owner/my-model --file-path pytorch_model.bin
```

---

## Project Structure

```
model-provenance/
├── pyproject.toml              # Project metadata, dependencies, CLI entry point
├── README.md
├── data/
│   └── known_hashes.yaml       # Bundled seed database of known-good fingerprints
├── model_provenance/
│   ├── __init__.py             # Package init, version string
│   ├── cli.py                  # Typer CLI: verify, report, db sub-commands
│   ├── fingerprint.py          # SHA-256 hashing and manifest construction
│   ├── checker.py              # Fingerprint comparison and tamper detection
│   ├── fetcher.py              # HF Hub and local directory file listing
│   ├── scanner.py              # Suspicious file pattern detection
│   ├── license_check.py        # License parsing and compliance flagging
│   ├── reporter.py             # Report assembly (Rich / JSON / YAML)
│   └── db.py                   # SQLite fingerprint database management
└── tests/
    ├── test_fingerprint.py
    ├── test_checker.py
    ├── test_scanner.py
    ├── test_license_check.py
    ├── test_reporter.py
    ├── test_db.py
    └── test_fetcher.py
```

---

## Configuration

`model-provenance` works out of the box with sensible defaults. The following options are available:

| Option / Env Var | Default | Description |
|---|---|---|
| `--db-path` / `MODEL_PROVENANCE_DB` | `~/.model-provenance/hashes.db` | Path to the local SQLite fingerprint database |
| `--format` | `rich` | Report output format: `rich`, `json`, or `yaml` |
| `--output` | stdout | File path to write the report to |
| `--revision` | `main` | Git revision (branch, tag, or commit SHA) for HF Hub models |
| `--local` | `false` | Treat the model argument as a local directory path |
| `--hf-token` / `HF_TOKEN` | *(none)* | Hugging Face API token for private model access |

### Example: audit a private model with a custom DB path

```bash
export HF_TOKEN=hf_xxxx
export MODEL_PROVENANCE_DB=/opt/audit/hashes.db

model-provenance report my-org/private-model --format yaml --output audit.yaml
```

---

## Development

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/your-org/model-provenance
cd model-provenance
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with async support
pytest --asyncio-mode=auto
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
