# model-provenance

> Audit AI models for supply chain integrity — cryptographic fingerprinting,
> tamper detection, suspicious file scanning, and structured provenance reports.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`model-provenance` is a command-line tool that helps developers and MLOps teams
verify AI model origin and integrity before production deployment. It:

- **Computes SHA-256 fingerprints** for all model weight and config files.
- **Compares fingerprints** against a local SQLite database and a bundled YAML
  seed database of known-good hashes.
- **Scans for supply chain attack vectors** — pickle exploits, embedded shell
  scripts, unexpected executables.
- **Checks licenses** and flags restricted or proprietary licenses with EU AI
  Act and NIST RMF compliance notes.
- **Generates structured provenance reports** as Rich terminal tables, JSON, or
  YAML.

## Quick Start

### Installation

```bash
# From PyPI (once published)
pip install model-provenance

# From source
git clone https://github.com/example/model-provenance
cd model-provenance
pip install -e ".[dev]"
```

### Usage

#### Verify a Hugging Face Hub model

```bash
# Basic verification of a Hub model
model-provenance verify bert-base-uncased

# Verify with a specific revision
model-provenance verify bert-base-uncased --revision main

# Verify and output JSON report
model-provenance verify gpt2 --format json

# Verify and save YAML report to file
model-provenance verify gpt2 --format yaml --output report.yaml
```

#### Verify a local model directory

```bash
# Verify files in a local directory
model-provenance verify ./path/to/my-model --local

# Verify local directory, output JSON
model-provenance verify ./my-model --local --format json
```

#### Generate a full provenance report

```bash
# Full provenance report for a Hub model (Rich table output)
model-provenance report bert-base-uncased

# Full provenance report as JSON
model-provenance report gpt2 --format json

# Full provenance report as YAML, saved to disk
model-provenance report facebook/opt-125m --format yaml --output provenance.yaml
```

#### Manage the local known-good hash database

```bash
# Initialize / seed the database from the bundled YAML
model-provenance db init

# Add a known-good hash entry
model-provenance db add bert-base-uncased config.json <sha256-hex>

# List all stored entries
model-provenance db list

# Query a specific model
model-provenance db query bert-base-uncased
```

## Output Format

### Rich console table (default)

```
╭─────────────────────────────────────────────────────────────────╮
│          Model Provenance Report — bert-base-uncased            │
╰─────────────────────────────────────────────────────────────────╯

Verdict: ✅ PASS

┌──────────────────────────┬────────────┬───────────┬────────────┐
│ File                     │ SHA-256    │ DB Status │ Scan       │
├──────────────────────────┼────────────┼───────────┼────────────┤
│ config.json              │ a7f4ab64…  │ ✅ match  │ ✅ clean   │
│ tokenizer_config.json    │ b3c2d1e0…  │ ✅ match  │ ✅ clean   │
│ vocab.txt                │ c4d3e2f1…  │ ✅ match  │ ✅ clean   │
│ pytorch_model.bin        │ d5e4f3a2…  │ ✅ match  │ ✅ clean   │
└──────────────────────────┴────────────┴───────────┴────────────┘

License: apache-2.0  │  Compliance: ✅ No restrictions noted
```

### JSON output

```json
{
  "model_id": "bert-base-uncased",
  "revision": "main",
  "verdict": "pass",
  "timestamp": "2024-01-15T12:00:00Z",
  "files": [
    {
      "path": "config.json",
      "sha256": "a7f4ab64e2a64c7f...",
      "db_status": "match",
      "scan_status": "clean",
      "size_bytes": 512
    }
  ],
  "license": {
    "spdx_id": "apache-2.0",
    "restricted": false,
    "compliance_notes": []
  },
  "scan_findings": [],
  "remediation": []
}
```

### YAML output

```yaml
model_id: bert-base-uncased
revision: main
verdict: pass
timestamp: '2024-01-15T12:00:00Z'
files:
  - path: config.json
    sha256: a7f4ab64e2a64c7f...
    db_status: match
    scan_status: clean
    size_bytes: 512
license:
  spdx_id: apache-2.0
  restricted: false
  compliance_notes: []
scan_findings: []
remediation: []
```

## Verdicts

| Verdict | Meaning |
|---------|----------------------------------------------------------|
| `PASS`  | All hashes match, no suspicious files, no license issues |
| `WARN`  | Minor issues: unknown hashes, non-OSI license, minor scan findings |
| `FAIL`  | Hash mismatch detected, malicious patterns found, or critical license violation |

## Supported Scan Detections

| Category | Detection |
|----------|-----------|
| Pickle exploit | Dangerous opcodes in `.pkl` / `.bin` files (e.g., `REDUCE` calling `os.system`) |
| Embedded scripts | Shell scripts (`.sh`), Python scripts (`.py`) not expected in model repos |
| Unexpected executables | ELF / PE binaries, `.so` / `.dll` files in suspicious locations |
| Archive bombs | Deeply nested ZIP/tar archives |
| Suspicious URLs | Hard-coded remote URLs in config files that may exfiltrate data |

## License Compliance

| License | Restriction Level | EU AI Act Note | NIST RMF Note |
|---------|------------------|----------------|---------------|
| Apache-2.0, MIT, BSD | ✅ Permissive | Low risk | Low risk |
| CC-BY-SA-4.0 | ⚠️ ShareAlike | Review required | Document usage |
| CC-BY-NC-4.0 | ⚠️ Non-commercial | Prohibited for commercial EU AI systems | Document restrictions |
| Proprietary / custom | ❌ Restricted | Legal review required | High risk — document and assess |
| RAIL, OpenRAIL | ⚠️ Conditional | Conditions may conflict with EU AI Act | Review use-case restrictions |

## Known-Good Hash Database

`model-provenance` ships with a seed YAML database (`data/known_hashes.yaml`)
containing known-good fingerprints for popular public models. At runtime these
are loaded into a local SQLite database (`~/.model-provenance/hashes.db`).

To contribute new known-good hashes:

1. Run `model-provenance verify <model-id> --format yaml` on a trusted machine.
2. Copy the file hashes into `data/known_hashes.yaml`.
3. Open a pull request.

## Development

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/example/model-provenance
cd model-provenance
pip install -e ".[dev]"

# Run tests
pytest

# Run a specific test module
pytest tests/test_fingerprint.py -v
```

## Architecture

```
model_provenance/
├── __init__.py      # Version string
├── cli.py           # Typer CLI (verify, report, db commands)
├── fingerprint.py   # SHA-256 hashing & manifest construction
├── fetcher.py       # HF Hub file listing + local directory scan
├── checker.py       # Hash comparison against DB
├── scanner.py       # Suspicious file pattern detection
├── license_check.py # License parsing & compliance flags
├── reporter.py      # Report assembly & rendering (Rich/JSON/YAML)
└── db.py            # SQLite known-good hash database
data/
└── known_hashes.yaml  # Bundled seed fingerprint database
```

## License

MIT — see [LICENSE](LICENSE) for details.
