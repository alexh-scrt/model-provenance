"""scanner.py: Suspicious file pattern detection for model supply chain security.

This module scans model files for indicators of supply chain attacks and
malicious content, including:

- **Pickle exploits**: Dangerous opcodes in ``.pkl`` / ``.bin`` / ``.pt`` files
  that could execute arbitrary code when loaded with ``pickle.load()``.
- **Embedded shell scripts**: Unexpected ``.sh`` files or files containing
  shell shebang lines.
- **Unexpected executables**: ELF binaries, PE binaries, or shared libraries
  (``.so``, ``.dll``) found in unexpected locations.
- **Suspicious URLs**: Hard-coded remote URLs in config files that may
  exfiltrate data or pull in additional malicious payloads.
- **Embedded Python scripts**: Unexpected ``.py`` files that are not part
  of a legitimate model package.
- **Archive bombs**: Nested ZIP or tar archives that may hide malicious
  content.

All findings are returned as :class:`ScanFinding` objects collected into a
:class:`ScanReport`.
"""

from __future__ import annotations

import logging
import os
import re
import struct
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of bytes to read from a file for header/magic detection.
_MAGIC_READ_SIZE: int = 512

#: Maximum number of bytes to scan for suspicious patterns (URLs, shebangs).
_SCAN_READ_SIZE: int = 64 * 1024  # 64 KiB

#: ELF magic bytes (Linux/Unix executable).
_ELF_MAGIC: bytes = b"\x7fELF"

#: PE magic bytes (Windows executable/DLL).
_PE_MAGIC: bytes = b"MZ"

#: Pickle protocol opcodes considered dangerous.
#: These opcodes can trigger arbitrary code execution during unpickling.
#: See: https://docs.python.org/3/library/pickle.html#pickle-inst
_DANGEROUS_PICKLE_OPCODES: frozenset[int] = frozenset(
    [
        ord("R"),  # REDUCE — calls a callable with args
        ord("i"),  # INST — deprecated REDUCE variant
        ord("o"),  # OBJ — create object by applying cls to args
        ord("\x93"),  # STACK_GLOBAL — push self.find_class(module, name)
        ord("c"),   # GLOBAL — push self.find_class(module, name)
        ord("\x81"),  # NEWOBJ — build object by calling cls.__new__(cls, *args)
        ord("\x82"),  # EXT1 — extension registry (one-byte code)
        ord("\x83"),  # EXT2 — extension registry (two-byte code)
        ord("\x84"),  # EXT4 — extension registry (four-byte code)
    ]
)

#: Particularly dangerous module/class combinations in pickle streams.
_DANGEROUS_PICKLE_PATTERNS: list[bytes] = [
    b"os\nsystem",
    b"os\npopen",
    b"subprocess\ncall",
    b"subprocess\nPopen",
    b"subprocess\ncheck_output",
    b"builtins\nexec",
    b"builtins\neval",
    b"__builtin__\nexec",
    b"__builtin__\neval",
    b"commands\ngetoutput",
    b"nt\nsystem",  # Windows os.system equivalent
    b"posix\nsystem",
    b"marshal\nloads",
    b"importlib\nimport_module",
    b"runpy\nrun_module",
]

#: Regex pattern for detecting HTTP/HTTPS URLs in text content.
_URL_PATTERN: re.Pattern[bytes] = re.compile(
    rb"https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]{10,}",
    re.IGNORECASE,
)

#: Suspicious URL patterns that may indicate data exfiltration or C2 channels.
_SUSPICIOUS_URL_PATTERNS: list[re.Pattern[bytes]] = [
    re.compile(rb"https?://(?!huggingface\.co|github\.com|pytorch\.org|tensorflow\.org|\w+\.amazonaws\.com|cdn\.)", re.IGNORECASE),
]

#: File extensions considered model weight files (may contain pickle data).
_PICKLE_EXTENSIONS: frozenset[str] = frozenset(
    {".bin", ".pkl", ".pickle", ".pt", ".pth", ".ckpt"}
)

#: File extensions considered suspicious in a model repository.
_SUSPICIOUS_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".sh",   # Shell scripts
        ".bash", # Bash scripts
        ".zsh",  # Zsh scripts
        ".fish", # Fish shell scripts
        ".ps1",  # PowerShell scripts
        ".bat",  # Windows batch files
        ".cmd",  # Windows command files
        ".exe",  # Windows executables
        ".com",  # DOS executables
        ".scr",  # Windows screensavers (often malware)
        ".vbs",  # VBScript
        ".js",   # JavaScript (unexpected in model repos)
        ".wsf",  # Windows Script File
    }
)

#: Extensions for archive files that may contain nested malicious content.
_ARCHIVE_EXTENSIONS: frozenset[str] = frozenset(
    {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar"}
)

#: Shared library / DLL extensions.
_SHARED_LIB_EXTENSIONS: frozenset[str] = frozenset(
    {".so", ".dll", ".dylib"}
)

#: File extensions where URLs in content should be scanned.
_TEXT_SCAN_EXTENSIONS: frozenset[str] = frozenset(
    {".json", ".yaml", ".yml", ".txt", ".toml", ".cfg", ".ini", ".py", ".md"}
)

#: Directories to skip during scanning.
_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".hf_cache", ".cache", "node_modules"}
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FindingSeverity(str, Enum):
    """Severity level of a scan finding.

    Values:
        CRITICAL: Highly likely to be malicious — immediate action required.
        HIGH: Strong indicator of a supply chain attack.
        MEDIUM: Suspicious but may be legitimate in some contexts.
        LOW: Informational — warrants review but probably benign.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingCategory(str, Enum):
    """Category / type of a scan finding.

    Values:
        PICKLE_EXPLOIT: Dangerous pickle opcode or pattern detected.
        EXECUTABLE: Unexpected ELF or PE binary found.
        SHELL_SCRIPT: Shell script or shebang line detected.
        EMBEDDED_SCRIPT: Unexpected Python or other script file.
        SUSPICIOUS_URL: Hard-coded URL that may exfiltrate data.
        ARCHIVE_BOMB: Nested archive file that may hide malicious content.
        SHARED_LIBRARY: Unexpected shared library / DLL.
        UNEXPECTED_FILE: File that does not belong in a model repository.
    """

    PICKLE_EXPLOIT = "pickle_exploit"
    EXECUTABLE = "executable"
    SHELL_SCRIPT = "shell_script"
    EMBEDDED_SCRIPT = "embedded_script"
    SUSPICIOUS_URL = "suspicious_url"
    ARCHIVE_BOMB = "archive_bomb"
    SHARED_LIBRARY = "shared_library"
    UNEXPECTED_FILE = "unexpected_file"


# ---------------------------------------------------------------------------
# Finding data model
# ---------------------------------------------------------------------------


@dataclass
class ScanFinding:
    """A single suspicious pattern or anomaly detected during a file scan.

    Attributes:
        path: Relative path of the file that triggered this finding.
        category: Broad category of the finding as a :class:`FindingCategory`.
        severity: Estimated severity as a :class:`FindingSeverity`.
        title: Short human-readable title for the finding.
        description: Detailed description including evidence (e.g. matched
            bytes, opcode values, or URLs).
        offset: Byte offset within the file where the pattern was found.
            ``None`` if not applicable.
        remediation: Suggested action to address the finding.
    """

    path: str
    category: FindingCategory
    severity: FindingSeverity
    title: str
    description: str
    offset: int | None = None
    remediation: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary suitable for JSON / YAML output."""
        return {
            "path": self.path,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "offset": self.offset,
            "remediation": self.remediation,
        }

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"[{self.severity.value.upper()}] {self.category.value}: "
            f"{self.path} — {self.title}"
        )


# ---------------------------------------------------------------------------
# Scan report
# ---------------------------------------------------------------------------


@dataclass
class ScanReport:
    """Aggregated results of scanning all files in a model directory or listing.

    Attributes:
        model_id: Identifier of the scanned model.
        findings: List of all :class:`ScanFinding` objects detected.
        scanned_files: List of relative paths of all files that were scanned.
        skipped_files: List of relative paths of files that were skipped
            (e.g. due to read errors or being too large).
        scan_error: If non-``None``, a top-level error occurred during
            scanning.
    """

    model_id: str
    findings: list[ScanFinding] = field(default_factory=list)
    scanned_files: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    scan_error: str | None = None

    # ------------------------------------------------------------------
    # Filtered views
    # ------------------------------------------------------------------

    @property
    def critical_findings(self) -> list[ScanFinding]:
        """Return only findings with ``CRITICAL`` severity."""
        return [f for f in self.findings if f.severity == FindingSeverity.CRITICAL]

    @property
    def high_findings(self) -> list[ScanFinding]:
        """Return only findings with ``HIGH`` severity."""
        return [f for f in self.findings if f.severity == FindingSeverity.HIGH]

    @property
    def medium_findings(self) -> list[ScanFinding]:
        """Return only findings with ``MEDIUM`` severity."""
        return [f for f in self.findings if f.severity == FindingSeverity.MEDIUM]

    @property
    def low_findings(self) -> list[ScanFinding]:
        """Return only findings with ``LOW`` severity."""
        return [f for f in self.findings if f.severity == FindingSeverity.LOW]

    @property
    def is_clean(self) -> bool:
        """Return ``True`` if no findings were detected."""
        return len(self.findings) == 0

    @property
    def has_critical_or_high(self) -> bool:
        """Return ``True`` if any critical or high severity findings exist."""
        return bool(self.critical_findings or self.high_findings)

    @property
    def finding_count(self) -> int:
        """Total number of findings."""
        return len(self.findings)

    @property
    def scanned_count(self) -> int:
        """Total number of files that were scanned."""
        return len(self.scanned_files)

    def findings_for_file(self, path: str) -> list[ScanFinding]:
        """Return all findings for a specific file path.

        Args:
            path: Relative file path to filter on.

        Returns:
            List of :class:`ScanFinding` objects for that file.
        """
        normalised = path.replace("\\", "/")
        return [f for f in self.findings if f.path == normalised]

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary suitable for JSON / YAML output."""
        return {
            "model_id": self.model_id,
            "is_clean": self.is_clean,
            "finding_count": self.finding_count,
            "scanned_count": self.scanned_count,
            "skipped_count": len(self.skipped_files),
            "critical_count": len(self.critical_findings),
            "high_count": len(self.high_findings),
            "medium_count": len(self.medium_findings),
            "low_count": len(self.low_findings),
            "findings": [f.to_dict() for f in self.findings],
            "scanned_files": self.scanned_files,
            "skipped_files": self.skipped_files,
            "scan_error": self.scan_error,
        }


# ---------------------------------------------------------------------------
# Individual file scanners
# ---------------------------------------------------------------------------


def scan_for_pickle_exploits(path: str, data: bytes) -> list[ScanFinding]:
    """Scan binary data for dangerous pickle opcodes and module patterns.

    Checks for opcodes that can trigger arbitrary code execution during
    unpickling (``REDUCE``, ``GLOBAL``, ``STACK_GLOBAL``, etc.) as well as
    known-dangerous module/callable patterns like ``os.system``.

    Args:
        path: Relative path label used in returned findings.
        data: Raw binary content of the file to scan.

    Returns:
        List of :class:`ScanFinding` objects, possibly empty.
    """
    findings: list[ScanFinding] = []

    # Check for well-known dangerous module/callable byte sequences.
    for pattern in _DANGEROUS_PICKLE_PATTERNS:
        idx = data.find(pattern)
        if idx != -1:
            readable = pattern.decode("ascii", errors="replace")
            findings.append(
                ScanFinding(
                    path=path,
                    category=FindingCategory.PICKLE_EXPLOIT,
                    severity=FindingSeverity.CRITICAL,
                    title="Dangerous pickle pattern detected",
                    description=(
                        f"Found dangerous pickle pattern '{readable}' at offset {idx}. "
                        "This pattern can execute arbitrary code when the file is "
                        "loaded with pickle.load()."
                    ),
                    offset=idx,
                    remediation=(
                        "Do NOT load this file with pickle.load(). Investigate the "
                        "file origin and consider using safetensors format instead."
                    ),
                )
            )
            # One finding per pattern is sufficient.

    # Check for high-risk pickle opcodes (only scan first 16 KiB for opcodes
    # to avoid false positives in large weight tensors).
    scan_window = data[:16384]
    found_dangerous_opcodes: set[int] = set()
    for byte_val in scan_window:
        if byte_val in _DANGEROUS_PICKLE_OPCODES and byte_val not in found_dangerous_opcodes:
            found_dangerous_opcodes.add(byte_val)

    # Only report opcode findings if no pattern findings were already emitted
    # (to avoid duplicate noisy findings for the same underlying issue).
    if found_dangerous_opcodes and not findings:
        opcode_strs = ", ".join(f"0x{op:02x}" for op in sorted(found_dangerous_opcodes))
        findings.append(
            ScanFinding(
                path=path,
                category=FindingCategory.PICKLE_EXPLOIT,
                severity=FindingSeverity.HIGH,
                title="Potentially dangerous pickle opcodes found",
                description=(
                    f"Found pickle opcodes that can invoke callables: {opcode_strs}. "
                    "These opcodes (REDUCE, GLOBAL, STACK_GLOBAL, etc.) are present "
                    "in many legitimate PyTorch models but should be reviewed if the "
                    "model origin is untrusted."
                ),
                offset=None,
                remediation=(
                    "Verify the model source. For maximum safety, prefer the "
                    "safetensors format which does not execute code on load."
                ),
            )
        )

    return findings


def scan_for_executable_magic(path: str, data: bytes) -> list[ScanFinding]:
    """Detect ELF or PE binary magic bytes at the start of a file.

    Args:
        path: Relative path label used in returned findings.
        data: File header bytes (first few hundred bytes are sufficient).

    Returns:
        List of :class:`ScanFinding` objects, possibly empty.
    """
    findings: list[ScanFinding] = []

    if data[:4] == _ELF_MAGIC:
        findings.append(
            ScanFinding(
                path=path,
                category=FindingCategory.EXECUTABLE,
                severity=FindingSeverity.CRITICAL,
                title="ELF executable detected",
                description=(
                    f"File '{path}' has an ELF magic header (\\x7fELF). "
                    "Executable binaries should not be present in model repositories."
                ),
                offset=0,
                remediation=(
                    "Remove this file immediately and investigate the model source. "
                    "This is a strong indicator of a supply chain attack."
                ),
            )
        )
    elif data[:2] == _PE_MAGIC:
        # Validate it's a real PE (not just a coincidental 'MZ' start).
        # A valid PE has a PE header offset at bytes 60-63.
        is_pe = False
        if len(data) >= 64:
            pe_offset = struct.unpack_from("<I", data, 60)[0]
            if pe_offset < len(data) - 4:
                pe_sig = data[pe_offset : pe_offset + 4]
                if pe_sig == b"PE\x00\x00":
                    is_pe = True
        # Also flag even without full PE validation if extension matches.
        suffix = Path(path).suffix.lower()
        if is_pe or suffix in (".exe", ".dll", ".com", ".scr"):
            findings.append(
                ScanFinding(
                    path=path,
                    category=FindingCategory.EXECUTABLE,
                    severity=FindingSeverity.CRITICAL,
                    title="Windows PE executable/DLL detected",
                    description=(
                        f"File '{path}' appears to be a Windows PE binary (MZ header). "
                        "Executable files should not be present in model repositories."
                    ),
                    offset=0,
                    remediation=(
                        "Remove this file immediately and investigate the model source. "
                        "This is a strong indicator of a supply chain attack."
                    ),
                )
            )

    return findings


def scan_for_shell_scripts(path: str, data: bytes) -> list[ScanFinding]:
    """Detect shell scripts by extension or shebang line.

    Args:
        path: Relative path label used in returned findings.
        data: File content (first few hundred bytes are sufficient for shebangs).

    Returns:
        List of :class:`ScanFinding` objects, possibly empty.
    """
    findings: list[ScanFinding] = []
    suffix = Path(path).suffix.lower()

    # Check by extension.
    if suffix in _SUSPICIOUS_EXTENSIONS and suffix in (".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd"):
        findings.append(
            ScanFinding(
                path=path,
                category=FindingCategory.SHELL_SCRIPT,
                severity=FindingSeverity.HIGH,
                title=f"Shell script file detected ({suffix})",
                description=(
                    f"File '{path}' is a shell script (extension: {suffix}). "
                    "Shell scripts are unexpected in model repositories and may "
                    "execute malicious commands."
                ),
                offset=None,
                remediation=(
                    "Remove this script and investigate the model source. "
                    "Shell scripts should never be present in model weight repositories."
                ),
            )
        )
        return findings

    # Check for shebang line in binary content.
    if data[:2] == b"#!":
        shebang_end = data.find(b"\n", 0, 128)
        shebang_line = data[: shebang_end if shebang_end > 0 else 128]
        shebang_readable = shebang_line.decode("ascii", errors="replace").strip()

        # Determine if it's a shell shebang.
        shell_indicators = ["/bin/sh", "/bin/bash", "/usr/bin/env sh",
                            "/usr/bin/env bash", "/bin/zsh", "/usr/bin/env python",
                            "/usr/bin/perl", "/usr/bin/env ruby"]
        is_shell = any(ind in shebang_readable for ind in shell_indicators)
        if is_shell:
            findings.append(
                ScanFinding(
                    path=path,
                    category=FindingCategory.SHELL_SCRIPT,
                    severity=FindingSeverity.HIGH,
                    title="Shebang line detected in file",
                    description=(
                        f"File '{path}' starts with a shebang line: "
                        f"'{shebang_readable}'. "
                        "Executable scripts are unexpected in model repositories."
                    ),
                    offset=0,
                    remediation=(
                        "Investigate why an executable script is present. "
                        "Remove if not intentional."
                    ),
                )
            )

    return findings


def scan_for_unexpected_python(path: str, data: bytes) -> list[ScanFinding]:
    """Detect unexpected Python script files in model repositories.

    Python files are only expected in packages (alongside ``__init__.py``);
    a standalone ``.py`` script at the repository root is suspicious.

    Args:
        path: Relative path label used in returned findings.
        data: File content (used to check for Python import/exec patterns).

    Returns:
        List of :class:`ScanFinding` objects, possibly empty.
    """
    findings: list[ScanFinding] = []
    suffix = Path(path).suffix.lower()

    if suffix != ".py":
        return findings

    # Python files are suspicious unless they look like standard packaging.
    basename = Path(path).name
    benign_python_names = {
        "__init__.py",
        "setup.py",
        "tokenization_fast.py",
        "tokenization.py",
        "configuration.py",
        "modeling.py",
    }

    if basename in benign_python_names:
        return findings

    # Check for particularly dangerous patterns in the script.
    danger_patterns = [
        b"os.system(",
        b"subprocess.call(",
        b"subprocess.Popen(",
        b"subprocess.run(",
        b"eval(",
        b"exec(",
        b"__import__(",
        b"importlib.import_module(",
        b"urllib.request.urlopen(",
        b"requests.get(",
        b"httpx.get(",
        b"socket.connect(",
    ]

    found_patterns: list[str] = []
    for dp in danger_patterns:
        if dp in data:
            found_patterns.append(dp.decode("ascii", errors="replace"))

    if found_patterns:
        severity = FindingSeverity.CRITICAL
        desc = (
            f"Python script '{path}' contains dangerous patterns: "
            f"{', '.join(found_patterns)}. This script could execute "
            "malicious code when imported or run."
        )
    else:
        severity = FindingSeverity.MEDIUM
        desc = (
            f"Unexpected Python script file '{path}' found in model repository. "
            "Python scripts are unexpected unless this is a model package."
        )

    findings.append(
        ScanFinding(
            path=path,
            category=FindingCategory.EMBEDDED_SCRIPT,
            severity=severity,
            title="Unexpected Python script file",
            description=desc,
            offset=None,
            remediation=(
                "Review this script carefully before loading the model. "
                "Remove if not intentionally part of the model package."
            ),
        )
    )

    return findings


def scan_for_suspicious_urls(path: str, data: bytes) -> list[ScanFinding]:
    """Scan file content for suspicious hard-coded URLs.

    Looks for HTTP/HTTPS URLs that are not from well-known trusted domains
    (Hugging Face, GitHub, PyTorch, TensorFlow, AWS CDN).

    Args:
        path: Relative path label used in returned findings.
        data: File content to scan (text-oriented; binary files may produce
            false positives but those are filtered by extension).

    Returns:
        List of :class:`ScanFinding` objects, possibly empty.
    """
    findings: list[ScanFinding] = []
    suffix = Path(path).suffix.lower()

    # Only scan text-oriented files for URLs.
    if suffix not in _TEXT_SCAN_EXTENSIONS:
        return findings

    # Find all URLs.
    all_urls = _URL_PATTERN.findall(data)
    if not all_urls:
        return findings

    # Filter for suspicious (non-trusted) URLs.
    trusted_domains = [
        b"huggingface.co",
        b"github.com",
        b"githubusercontent.com",
        b"pytorch.org",
        b"tensorflow.org",
        b"amazonaws.com",
        b"cloudfront.net",
        b"cdn.",
        b"pypi.org",
        b"python.org",
        b"arxiv.org",
        b"openai.com",
        b"googleapis.com",
        b"gstatic.com",
        b"storage.googleapis.com",
        b"example.com",
        b"schemas.xmlsoap.org",
        b"www.w3.org",
        b"json-schema.org",
        b"localhost",
        b"127.0.0.1",
    ]

    suspicious_urls: list[str] = []
    for url in all_urls:
        url_lower = url.lower()
        is_trusted = any(domain in url_lower for domain in trusted_domains)
        if not is_trusted:
            url_str = url.decode("ascii", errors="replace")
            if url_str not in suspicious_urls:
                suspicious_urls.append(url_str)

    if suspicious_urls:
        # Limit to first 5 to avoid report bloat.
        displayed = suspicious_urls[:5]
        extra = len(suspicious_urls) - 5 if len(suspicious_urls) > 5 else 0
        url_list = ", ".join(f"'{u}'" for u in displayed)
        if extra:
            url_list += f" ... and {extra} more"

        findings.append(
            ScanFinding(
                path=path,
                category=FindingCategory.SUSPICIOUS_URL,
                severity=FindingSeverity.MEDIUM,
                title="Suspicious hard-coded URL(s) detected",
                description=(
                    f"File '{path}' contains URL(s) from untrusted domains: "
                    f"{url_list}. These may be used for data exfiltration or "
                    "to download additional malicious payloads."
                ),
                offset=None,
                remediation=(
                    "Review the URLs and ensure they are legitimate. "
                    "Do not deploy this model if the URLs are unexpected."
                ),
            )
        )

    return findings


def scan_for_shared_libraries(path: str, data: bytes) -> list[ScanFinding]:
    """Detect shared library files (``.so``, ``.dll``, ``.dylib``).

    Shared libraries can contain arbitrary native code and are a serious
    supply chain risk if found unexpectedly in a model repository.

    Args:
        path: Relative path label used in returned findings.
        data: File header bytes.

    Returns:
        List of :class:`ScanFinding` objects, possibly empty.
    """
    findings: list[ScanFinding] = []
    suffix = Path(path).suffix.lower()

    if suffix not in _SHARED_LIB_EXTENSIONS:
        return findings

    # ELF shared library (.so, .dylib)
    is_elf_so = data[:4] == _ELF_MAGIC and suffix in (".so", ".dylib")
    # Windows DLL
    is_pe_dll = data[:2] == _PE_MAGIC and suffix == ".dll"

    if is_elf_so or is_pe_dll or suffix in _SHARED_LIB_EXTENSIONS:
        findings.append(
            ScanFinding(
                path=path,
                category=FindingCategory.SHARED_LIBRARY,
                severity=FindingSeverity.HIGH,
                title=f"Shared library file detected ({suffix})",
                description=(
                    f"File '{path}' is a shared library ({suffix}). "
                    "Native shared libraries can execute arbitrary code when "
                    "loaded and are not expected in model weight repositories "
                    "unless this is a compiled extension package."
                ),
                offset=None,
                remediation=(
                    "Verify the library's origin and purpose. If this model "
                    "package intentionally includes native extensions, ensure "
                    "they are from a trusted, audited source."
                ),
            )
        )

    return findings


def scan_for_archive_bombs(path: str, data: bytes) -> list[ScanFinding]:
    """Detect nested archive files that may hide malicious content.

    Args:
        path: Relative path label used in returned findings.
        data: File content.

    Returns:
        List of :class:`ScanFinding` objects, possibly empty.
    """
    findings: list[ScanFinding] = []
    suffix = Path(path).suffix.lower()

    if suffix not in _ARCHIVE_EXTENSIONS:
        return findings

    # Check ZIP magic.
    is_zip = data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
    # Check tar gzip magic.
    is_gzip = data[:2] == b"\x1f\x8b"
    # Check bzip2 magic.
    is_bzip2 = data[:3] == b"BZh"
    # Check XZ magic.
    is_xz = data[:6] == b"\xfd7zXZ\x00"

    if is_zip or is_gzip or is_bzip2 or is_xz or suffix in _ARCHIVE_EXTENSIONS:
        findings.append(
            ScanFinding(
                path=path,
                category=FindingCategory.ARCHIVE_BOMB,
                severity=FindingSeverity.MEDIUM,
                title=f"Nested archive file detected ({suffix})",
                description=(
                    f"File '{path}' is an archive ({suffix}) found inside the "
                    "model repository. Nested archives can be used to hide "
                    "malicious files from shallow scanners."
                ),
                offset=None,
                remediation=(
                    "Inspect the archive contents before deployment. "
                    "Extract and scan the contents for additional threats."
                ),
            )
        )

    return findings


def scan_file_content(
    path: str,
    data: bytes,
    *,
    check_pickle: bool = True,
    check_executables: bool = True,
    check_scripts: bool = True,
    check_urls: bool = True,
    check_shared_libs: bool = True,
    check_archives: bool = True,
) -> list[ScanFinding]:
    """Run all configured scanners against a single file's content.

    Args:
        path: Relative file path label for findings.
        data: Raw file content (or a representative prefix for large files).
        check_pickle: If ``True``, scan for pickle exploits.
        check_executables: If ``True``, detect ELF/PE executables.
        check_scripts: If ``True``, detect shell and Python scripts.
        check_urls: If ``True``, scan for suspicious URLs.
        check_shared_libs: If ``True``, detect shared libraries.
        check_archives: If ``True``, detect nested archives.

    Returns:
        Combined list of all :class:`ScanFinding` objects from all active
        scanners.
    """
    findings: list[ScanFinding] = []
    suffix = Path(path).suffix.lower()

    if check_executables:
        findings.extend(scan_for_executable_magic(path, data))

    if check_shared_libs:
        findings.extend(scan_for_shared_libraries(path, data))

    if check_scripts:
        findings.extend(scan_for_shell_scripts(path, data))
        findings.extend(scan_for_unexpected_python(path, data))

    if check_pickle and suffix in _PICKLE_EXTENSIONS:
        findings.extend(scan_for_pickle_exploits(path, data))

    if check_urls:
        findings.extend(scan_for_suspicious_urls(path, data))

    if check_archives:
        findings.extend(scan_for_archive_bombs(path, data))

    return findings


# ---------------------------------------------------------------------------
# Directory / file-list scanner
# ---------------------------------------------------------------------------


class ModelScanner:
    """Scans all files in a model directory or file list for supply chain threats.

    Args:
        max_file_size_mb: Maximum file size (in MiB) to fully scan for pickle
            exploits and URL patterns.  Files larger than this threshold are
            still inspected for magic bytes and extension-based checks, but
            only the first :attr:`_SCAN_READ_SIZE` bytes are read for content
            patterns.  Defaults to 100 MiB.
        check_pickle: Enable pickle exploit scanning.
        check_executables: Enable executable magic byte detection.
        check_scripts: Enable shell and Python script detection.
        check_urls: Enable suspicious URL scanning.
        check_shared_libs: Enable shared library detection.
        check_archives: Enable nested archive detection.
    """

    def __init__(
        self,
        max_file_size_mb: float = 100.0,
        check_pickle: bool = True,
        check_executables: bool = True,
        check_scripts: bool = True,
        check_urls: bool = True,
        check_shared_libs: bool = True,
        check_archives: bool = True,
    ) -> None:
        """Initialise the ModelScanner with configurable check flags."""
        self._max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self._check_pickle = check_pickle
        self._check_executables = check_executables
        self._check_scripts = check_scripts
        self._check_urls = check_urls
        self._check_shared_libs = check_shared_libs
        self._check_archives = check_archives

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_directory(
        self,
        directory: str | Path,
        model_id: str | None = None,
    ) -> ScanReport:
        """Scan all files in *directory* for supply chain threats.

        Recursively walks the directory, skipping hidden/cache directories,
        and runs all configured checks against each file.

        Args:
            directory: Path to the local model directory.
            model_id: Human-readable model identifier.  Defaults to the
                directory base name.

        Returns:
            A :class:`ScanReport` with all findings.

        Raises:
            NotADirectoryError: If *directory* does not exist or is not a
                directory.
        """
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            raise NotADirectoryError(f"Directory does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {dir_path}")

        effective_model_id = model_id if model_id is not None else dir_path.name
        report = ScanReport(model_id=effective_model_id)

        for file_path in sorted(_iter_files(dir_path)):
            try:
                rel = file_path.resolve().relative_to(dir_path)
                rel_str = str(rel).replace("\\", "/")
            except ValueError:
                rel_str = str(file_path).replace("\\", "/")

            findings = self._scan_single_file(file_path, rel_str)
            if findings is None:
                report.skipped_files.append(rel_str)
            else:
                report.scanned_files.append(rel_str)
                report.findings.extend(findings)

        return report

    def scan_files(
        self,
        file_paths: Sequence[str | Path],
        base_dir: str | Path | None = None,
        model_id: str = "unknown",
    ) -> ScanReport:
        """Scan a specific list of file paths.

        Args:
            file_paths: Sequence of file paths to scan.
            base_dir: If provided, relative paths in findings are computed
                relative to this directory.
            model_id: Human-readable model identifier.

        Returns:
            A :class:`ScanReport` with all findings.
        """
        report = ScanReport(model_id=model_id)
        base = Path(base_dir).resolve() if base_dir is not None else None

        for fp in file_paths:
            file_path = Path(fp).resolve()
            if base is not None:
                try:
                    rel = file_path.relative_to(base)
                    rel_str = str(rel).replace("\\", "/")
                except ValueError:
                    rel_str = str(file_path).replace("\\", "/")
            else:
                rel_str = str(file_path).replace("\\", "/")

            findings = self._scan_single_file(file_path, rel_str)
            if findings is None:
                report.skipped_files.append(rel_str)
            else:
                report.scanned_files.append(rel_str)
                report.findings.extend(findings)

        return report

    def scan_file_bytes(
        self,
        path: str,
        data: bytes,
        model_id: str = "unknown",
    ) -> ScanReport:
        """Scan in-memory file content.

        Useful for scanning files that have been downloaded into memory (e.g.
        from the Hugging Face Hub API) without writing to disk.

        Args:
            path: Relative path label for the file.
            data: Raw file content.
            model_id: Human-readable model identifier.

        Returns:
            A :class:`ScanReport` with all findings for this single file.
        """
        report = ScanReport(model_id=model_id)
        findings = scan_file_content(
            path=path,
            data=data,
            check_pickle=self._check_pickle,
            check_executables=self._check_executables,
            check_scripts=self._check_scripts,
            check_urls=self._check_urls,
            check_shared_libs=self._check_shared_libs,
            check_archives=self._check_archives,
        )
        report.scanned_files.append(path)
        report.findings.extend(findings)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_single_file(
        self,
        file_path: Path,
        rel_str: str,
    ) -> list[ScanFinding] | None:
        """Scan a single file and return its findings.

        Args:
            file_path: Absolute path to the file.
            rel_str: Relative path string for use in findings.

        Returns:
            List of :class:`ScanFinding` objects, or ``None`` if the file
            was skipped due to a read error.
        """
        try:
            size = file_path.stat().st_size
        except OSError as exc:
            logger.warning("Could not stat %s: %s", file_path, exc)
            return None

        try:
            # Always read at least the magic header bytes.
            read_size = min(size, max(_MAGIC_READ_SIZE, _SCAN_READ_SIZE))
            with file_path.open("rb") as fh:
                data = fh.read(read_size)
        except OSError as exc:
            logger.warning("Could not read %s: %s", file_path, exc)
            return None

        return scan_file_content(
            path=rel_str,
            data=data,
            check_pickle=self._check_pickle,
            check_executables=self._check_executables,
            check_scripts=self._check_scripts,
            check_urls=self._check_urls,
            check_shared_libs=self._check_shared_libs,
            check_archives=self._check_archives,
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def scan_directory(
    directory: str | Path,
    model_id: str | None = None,
) -> ScanReport:
    """Convenience wrapper: scan a local model directory with default settings.

    Args:
        directory: Path to the local model directory.
        model_id: Optional model identifier.

    Returns:
        A :class:`ScanReport` with all findings.
    """
    scanner = ModelScanner()
    return scanner.scan_directory(directory=directory, model_id=model_id)


def scan_file_bytes(
    path: str,
    data: bytes,
    model_id: str = "unknown",
) -> ScanReport:
    """Convenience wrapper: scan in-memory file content with default settings.

    Args:
        path: Relative path label for the file.
        data: Raw file content.
        model_id: Optional model identifier.

    Returns:
        A :class:`ScanReport` with all findings for this single file.
    """
    scanner = ModelScanner()
    return scanner.scan_file_bytes(path=path, data=data, model_id=model_id)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _iter_files(directory: Path) -> Iterator[Path]:
    """Yield all regular files under *directory*, skipping hidden/cache dirs.

    Args:
        directory: Root directory to walk.

    Yields:
        Absolute :class:`~pathlib.Path` objects for each regular file found.
    """
    for entry in os.scandir(directory):
        if entry.name in _SKIP_DIRS:
            continue
        if entry.is_dir(follow_symlinks=False):
            yield from _iter_files(Path(entry.path))
        elif entry.is_file(follow_symlinks=False):
            yield Path(entry.path)
