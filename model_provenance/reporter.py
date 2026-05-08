"""reporter.py: Provenance report assembly and multi-format rendering.

This module assembles audit results from the fingerprint checker, file scanner,
and license compliance modules into a unified :class:`ProvenanceReport` data
structure, then renders it in one of three output formats:

- **Rich console tables** (default) — colourful terminal output with per-file
  status rows, a summary panel, and a verdict banner.
- **JSON** — machine-readable compact or pretty-printed JSON.
- **YAML** — human-readable YAML document.

The module is designed so that every component is independently serialisable
(via ``to_dict()``) and the top-level :func:`render_report` function handles
all three output modes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import TextIO

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from model_provenance.checker import CheckResult, FileCheckStatus, Verdict
from model_provenance.fingerprint import FingerprintManifest
from model_provenance.license_check import LicenseReport, LicenseRestrictionLevel
from model_provenance.scanner import ScanReport, FindingSeverity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping from overall verdict to (emoji, Rich style string).
_VERDICT_STYLE: dict[Verdict, tuple[str, str]] = {
    Verdict.PASS: ("✅", "bold green"),
    Verdict.WARN: ("⚠️", "bold yellow"),
    Verdict.FAIL: ("❌", "bold red"),
}

#: Mapping from FileCheckStatus to (symbol, Rich style).
_STATUS_STYLE: dict[FileCheckStatus, tuple[str, str]] = {
    FileCheckStatus.MATCH: ("✅ match", "green"),
    FileCheckStatus.MISMATCH: ("❌ MISMATCH", "bold red"),
    FileCheckStatus.UNKNOWN: ("❓ unknown", "yellow"),
    FileCheckStatus.ERROR: ("⚠️ error", "red"),
    FileCheckStatus.NEW: ("🆕 new", "cyan"),
}

#: Mapping from FindingSeverity to (symbol, Rich style).
_SEVERITY_STYLE: dict[FindingSeverity, tuple[str, str]] = {
    FindingSeverity.CRITICAL: ("🔴 CRITICAL", "bold red"),
    FindingSeverity.HIGH: ("🟠 HIGH", "red"),
    FindingSeverity.MEDIUM: ("🟡 MEDIUM", "yellow"),
    FindingSeverity.LOW: ("🔵 LOW", "cyan"),
}


# ---------------------------------------------------------------------------
# Unified provenance report data structure
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceReport:
    """A unified provenance report aggregating all audit results for a model.

    This is the top-level data structure that combines fingerprint manifest
    data, hash checker results, suspicious file scan results, and license
    compliance information into a single serialisable object.

    Attributes:
        model_id: Hugging Face model ID or local path label.
        revision: Git revision or ``'local'`` for locally audited models.
        source: ``'hub'`` or ``'local'``.
        timestamp: ISO-8601 UTC timestamp of when the report was generated.
        verdict: Overall :class:`~model_provenance.checker.Verdict` —
            ``PASS``, ``WARN``, or ``FAIL``.
        manifest: The :class:`~model_provenance.fingerprint.FingerprintManifest`
            from which hashes were computed.  May be ``None`` if fingerprinting
            was skipped.
        check_result: The :class:`~model_provenance.checker.CheckResult` from
            comparing fingerprints against the known-good database.  May be
            ``None`` if checking was skipped.
        scan_report: The :class:`~model_provenance.scanner.ScanReport` from
            the suspicious file scanner.  May be ``None`` if scanning was
            skipped.
        license_report: The :class:`~model_provenance.license_check.LicenseReport`
            from the license compliance checker.  May be ``None`` if license
            checking was skipped.
        remediation_notes: Aggregated actionable remediation notes from all
            sub-components.
        errors: List of top-level error strings that occurred during the audit.
    """

    model_id: str
    revision: str = "local"
    source: str = "local"
    timestamp: str = field(default_factory=lambda: _utc_now_iso())
    verdict: Verdict = Verdict.WARN
    manifest: FingerprintManifest | None = None
    check_result: CheckResult | None = None
    scan_report: ScanReport | None = None
    license_report: LicenseReport | None = None
    remediation_notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def file_count(self) -> int:
        """Total number of files in the manifest (0 if no manifest)."""
        if self.manifest is not None:
            return self.manifest.file_count
        if self.check_result is not None:
            return self.check_result.file_count
        return 0

    @property
    def has_scan_findings(self) -> bool:
        """Return ``True`` if the scan report contains any findings."""
        if self.scan_report is None:
            return False
        return not self.scan_report.is_clean

    @property
    def has_license_issues(self) -> bool:
        """Return ``True`` if the license report has warnings or critical notes."""
        if self.license_report is None:
            return False
        return self.license_report.has_warnings

    @property
    def aggregate_sha256(self) -> str | None:
        """Return the manifest aggregate SHA-256, if available."""
        if self.manifest is not None:
            return self.manifest.aggregate_sha256
        return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialise the full report to a plain dictionary.

        The resulting dictionary is suitable for JSON or YAML serialisation.
        All sub-components are serialised via their own ``to_dict()`` methods.

        Returns:
            A nested plain-Python dictionary representation of the report.
        """
        result: dict[str, object] = {
            "model_id": self.model_id,
            "revision": self.revision,
            "source": self.source,
            "timestamp": self.timestamp,
            "verdict": self.verdict.value,
            "aggregate_sha256": self.aggregate_sha256,
            "file_count": self.file_count,
            "errors": self.errors,
            "remediation": self.remediation_notes,
        }

        # Fingerprint manifest summary.
        if self.manifest is not None:
            result["manifest"] = {
                "model_id": self.manifest.model_id,
                "revision": self.manifest.revision,
                "source": self.manifest.source,
                "computed_at": self.manifest.computed_at,
                "aggregate_sha256": self.manifest.aggregate_sha256,
                "file_count": self.manifest.file_count,
                "total_size_bytes": self.manifest.total_size_bytes,
                "files": [f.to_dict() for f in self.manifest.files],
            }
        else:
            result["manifest"] = None

        # Hash checker results.
        if self.check_result is not None:
            result["check_result"] = self.check_result.to_dict()
        else:
            result["check_result"] = None

        # Scanner results.
        if self.scan_report is not None:
            result["scan_report"] = self.scan_report.to_dict()
        else:
            result["scan_report"] = None

        # License compliance.
        if self.license_report is not None:
            result["license_report"] = self.license_report.to_dict()
        else:
            result["license_report"] = None

        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string.

        Args:
            indent: Number of spaces for JSON indentation.  Defaults to 2.

        Returns:
            Pretty-printed JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_yaml(self) -> str:
        """Serialise the report to a YAML string.

        Returns:
            YAML document string.
        """
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def assemble_report(
    model_id: str,
    revision: str = "local",
    source: str = "local",
    manifest: FingerprintManifest | None = None,
    check_result: CheckResult | None = None,
    scan_report: ScanReport | None = None,
    license_report: LicenseReport | None = None,
    errors: list[str] | None = None,
) -> ProvenanceReport:
    """Assemble a :class:`ProvenanceReport` from individual audit components.

    Calculates the overall :class:`~model_provenance.checker.Verdict` by
    considering the hash check verdict, scan findings severity, and license
    restriction level.  Collects remediation notes from all sub-components.

    Args:
        model_id: Hugging Face model ID or local path label.
        revision: Git revision string or ``'local'``.
        source: ``'hub'`` or ``'local'``.
        manifest: Optional fingerprint manifest from the fingerprint module.
        check_result: Optional hash check result from the checker module.
        scan_report: Optional scan report from the scanner module.
        license_report: Optional license compliance report from the
            license_check module.
        errors: Optional list of top-level error strings that occurred
            during the audit.

    Returns:
        A fully assembled :class:`ProvenanceReport`.
    """
    effective_errors: list[str] = list(errors or [])
    remediation: list[str] = []

    # ---- Determine overall verdict -------------------------------------
    # Start with PASS and downgrade based on findings.
    verdict = Verdict.PASS

    # Hash check verdict.
    if check_result is not None:
        if check_result.verdict == Verdict.FAIL:
            verdict = Verdict.FAIL
        elif check_result.verdict == Verdict.WARN and verdict == Verdict.PASS:
            verdict = Verdict.WARN
        # Collect remediation notes from checker.
        remediation.extend(check_result.remediation_notes)

    # Scanner verdict contribution.
    if scan_report is not None:
        if scan_report.has_critical_or_high:
            verdict = Verdict.FAIL
        elif not scan_report.is_clean and verdict == Verdict.PASS:
            verdict = Verdict.WARN
        # Collect remediation notes from critical/high findings.
        for finding in scan_report.critical_findings + scan_report.high_findings:
            if finding.remediation and finding.remediation not in remediation:
                note = f"[{finding.severity.value.upper()}] {finding.path}: {finding.remediation}"
                remediation.append(note)

    # License verdict contribution.
    if license_report is not None:
        if license_report.has_critical:
            if verdict != Verdict.FAIL:
                verdict = Verdict.FAIL
        elif license_report.has_warnings and verdict == Verdict.PASS:
            verdict = Verdict.WARN
        # Collect remediation notes from license report.
        for note in license_report.remediation_notes:
            if note not in remediation:
                remediation.append(note)

    # Top-level errors always downgrade to FAIL.
    if effective_errors:
        verdict = Verdict.FAIL

    return ProvenanceReport(
        model_id=model_id,
        revision=revision,
        source=source,
        verdict=verdict,
        manifest=manifest,
        check_result=check_result,
        scan_report=scan_report,
        license_report=license_report,
        remediation_notes=remediation,
        errors=effective_errors,
    )


# ---------------------------------------------------------------------------
# Rich console renderer
# ---------------------------------------------------------------------------


def _render_rich(
    report: ProvenanceReport,
    console: Console,
) -> None:
    """Render a :class:`ProvenanceReport` to the Rich console.

    Produces:
    1. A header panel with the model ID and overall verdict.
    2. A per-file hash / scan status table (if a check result is available).
    3. A scan findings table (if findings exist).
    4. A license compliance section.
    5. Remediation notes.

    Args:
        report: The :class:`ProvenanceReport` to render.
        console: A :class:`~rich.console.Console` instance to write to.
    """
    verdict_emoji, verdict_style = _VERDICT_STYLE.get(
        report.verdict, ("?", "white")
    )

    # ---- Header panel --------------------------------------------------
    header_text = Text()
    header_text.append(
        f"Model Provenance Report\n",
        style="bold white",
    )
    header_text.append(f"  Model:    ", style="dim")
    header_text.append(f"{report.model_id}\n", style="bold cyan")
    header_text.append(f"  Revision: ", style="dim")
    header_text.append(f"{report.revision}\n", style="cyan")
    header_text.append(f"  Source:   ", style="dim")
    header_text.append(f"{report.source}\n", style="cyan")
    header_text.append(f"  Timestamp: ", style="dim")
    header_text.append(f"{report.timestamp}\n", style="cyan")
    if report.aggregate_sha256:
        header_text.append(f"  Aggregate SHA-256: ", style="dim")
        header_text.append(f"{report.aggregate_sha256[:32]}…\n", style="cyan")

    console.print()
    console.print(
        Panel(
            header_text,
            title=f"{verdict_emoji}  Verdict: {report.verdict.value.upper()}",
            title_align="left",
            border_style=verdict_style,
            expand=False,
        )
    )
    console.print()

    # ---- Per-file hash status table ------------------------------------
    if report.check_result is not None and report.check_result.file_results:
        _render_file_table(report.check_result, console)

    # ---- Scan findings table ------------------------------------------
    if report.scan_report is not None and not report.scan_report.is_clean:
        _render_scan_table(report.scan_report, console)
    elif report.scan_report is not None:
        console.print("[green]✅ Scan:[/green] No suspicious files detected.")
        console.print()

    # ---- License section ----------------------------------------------
    if report.license_report is not None:
        _render_license_section(report.license_report, console)

    # ---- Errors section -----------------------------------------------
    if report.errors:
        console.print("[bold red]Errors encountered during audit:[/bold red]")
        for err in report.errors:
            console.print(f"  [red]• {err}[/red]")
        console.print()

    # ---- Remediation notes --------------------------------------------
    if report.remediation_notes:
        console.print("[bold yellow]Remediation Notes:[/bold yellow]")
        for note in report.remediation_notes:
            console.print(f"  [yellow]• {note}[/yellow]")
        console.print()


def _render_file_table(
    check_result: CheckResult,
    console: Console,
) -> None:
    """Render a Rich table of per-file hash check results.

    Args:
        check_result: The :class:`~model_provenance.checker.CheckResult` to
            render.
        console: The Rich console to write to.
    """
    table = Table(
        title=f"File Hash Verification — {check_result.model_id}@{check_result.revision}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
        show_lines=False,
    )
    table.add_column("File", style="cyan", no_wrap=False, max_width=50)
    table.add_column("Type", style="dim", width=8)
    table.add_column("SHA-256 (prefix)", style="dim", width=18)
    table.add_column("DB Status", width=14)
    table.add_column("Size", justify="right", style="dim", width=10)
    table.add_column("Detail", style="dim", no_wrap=False, max_width=40)

    for fcr in check_result.file_results:
        symbol, status_style = _STATUS_STYLE.get(
            fcr.status, (fcr.status.value, "white")
        )
        sha_prefix = fcr.fingerprint.sha256[:16] + "…" if fcr.fingerprint.sha256 else "—"
        size_str = _human_size(fcr.fingerprint.size_bytes)
        detail_short = fcr.detail[:40] + "…" if len(fcr.detail) > 40 else fcr.detail

        table.add_row(
            fcr.path,
            fcr.fingerprint.file_type,
            sha_prefix,
            Text(symbol, style=status_style),
            size_str,
            detail_short,
        )

    console.print(table)

    # Summary line below the table.
    cr = check_result
    coverage_pct = f"{cr.db_coverage * 100:.0f}%" if cr.db_coverage is not None else "N/A"
    console.print(
        f"  [dim]Files: {cr.file_count} total | "
        f"[green]{len(cr.matches)} match[/green] | "
        f"[red]{len(cr.mismatches)} mismatch[/red] | "
        f"[yellow]{len(cr.unknowns)} unknown[/yellow] | "
        f"[red]{len(cr.errors)} error[/red] | "
        f"DB coverage: {coverage_pct}[/dim]"
    )
    console.print()


def _render_scan_table(
    scan_report: ScanReport,
    console: Console,
) -> None:
    """Render a Rich table of suspicious file scan findings.

    Args:
        scan_report: The :class:`~model_provenance.scanner.ScanReport` to
            render.
        console: The Rich console to write to.
    """
    table = Table(
        title=f"Suspicious File Scan — {scan_report.model_id}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
        show_lines=True,
    )
    table.add_column("File", style="cyan", no_wrap=False, max_width=40)
    table.add_column("Severity", width=14)
    table.add_column("Category", style="dim", width=18)
    table.add_column("Title", no_wrap=False, max_width=40)
    table.add_column("Remediation", style="dim", no_wrap=False, max_width=40)

    for finding in scan_report.findings:
        sev_symbol, sev_style = _SEVERITY_STYLE.get(
            finding.severity, (finding.severity.value, "white")
        )
        rem_short = (
            finding.remediation[:38] + "…"
            if len(finding.remediation) > 38
            else finding.remediation
        )
        table.add_row(
            finding.path,
            Text(sev_symbol, style=sev_style),
            finding.category.value,
            finding.title,
            rem_short,
        )

    console.print(table)
    console.print(
        f"  [dim]Findings: {scan_report.finding_count} total | "
        f"[bold red]{len(scan_report.critical_findings)} critical[/bold red] | "
        f"[red]{len(scan_report.high_findings)} high[/red] | "
        f"[yellow]{len(scan_report.medium_findings)} medium[/yellow] | "
        f"[cyan]{len(scan_report.low_findings)} low[/cyan][/dim]"
    )
    console.print()


def _render_license_section(
    license_report: LicenseReport,
    console: Console,
) -> None:
    """Render the license and compliance section of the report.

    Args:
        license_report: The :class:`~model_provenance.license_check.LicenseReport`
            to render.
        console: The Rich console to write to.
    """
    # Determine section colour based on restriction level.
    level = license_report.restriction_level
    if level == LicenseRestrictionLevel.PERMISSIVE:
        level_style = "green"
        level_icon = "✅"
    elif level in (LicenseRestrictionLevel.COPYLEFT, LicenseRestrictionLevel.CONDITIONAL):
        level_style = "yellow"
        level_icon = "⚠️"
    elif level in (LicenseRestrictionLevel.NON_COMMERCIAL, LicenseRestrictionLevel.PROPRIETARY):
        level_style = "red"
        level_icon = "❌"
    else:
        level_style = "yellow"
        level_icon = "❓"

    commercial_str = (
        "[green]✅ allowed[/green]"
        if license_report.allows_commercial_use
        else "[red]❌ NOT allowed[/red]"
    )
    osi_str = (
        "[green]✅ OSI approved[/green]"
        if license_report.is_osi_approved
        else "[dim]not OSI approved[/dim]"
    )

    license_text = Text()
    license_text.append(f"  SPDX ID:          ", style="dim")
    license_text.append(f"{license_report.spdx_id}\n", style="bold")
    license_text.append(f"  Restriction:      ", style="dim")
    license_text.append(
        f"{level_icon} {level.value.replace('_', '-').title()}\n",
        style=level_style,
    )
    license_text.append(f"  Commercial Use:   ", style="dim")

    console.print(
        Panel(
            license_text,
            title="License & Compliance",
            border_style=level_style,
            expand=False,
        )
    )

    # Commercial and OSI flags inline.
    console.print(
        f"  Commercial use: {commercial_str}  |  {osi_str}  |  "
        f"Summary: [italic]{license_report.summary}[/italic]"
    )
    console.print()

    # Compliance notes table (only non-info).
    warning_notes = [
        n for n in license_report.compliance_notes
        if n.severity in ("warning", "critical")
    ]
    if warning_notes:
        comp_table = Table(
            title="Compliance Notes",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold magenta",
            expand=False,
            show_lines=False,
        )
        comp_table.add_column("Framework", style="dim", width=14)
        comp_table.add_column("Severity", width=10)
        comp_table.add_column("Title", no_wrap=False, max_width=50)
        comp_table.add_column("Remediation", style="dim", no_wrap=False, max_width=40)

        for note in warning_notes:
            sev_style = "red" if note.severity == "critical" else "yellow"
            rem_short = (
                note.remediation[:38] + "…"
                if len(note.remediation) > 38
                else note.remediation
            )
            comp_table.add_row(
                note.framework.value,
                Text(note.severity.upper(), style=sev_style),
                note.title,
                rem_short,
            )

        console.print(comp_table)
        console.print()


# ---------------------------------------------------------------------------
# Public rendering API
# ---------------------------------------------------------------------------


def render_report(
    report: ProvenanceReport,
    fmt: str = "rich",
    output: TextIO | None = None,
    *,
    force_terminal: bool = False,
) -> str:
    """Render a :class:`ProvenanceReport` to the requested output format.

    Supports three formats:

    - ``'rich'`` — Rich terminal table output (default).  Writes to *output*
      (defaults to ``sys.stdout``) and also returns the rendered string.
    - ``'json'`` — Pretty-printed JSON string.  Writes to *output* if
      provided and returns the string.
    - ``'yaml'`` — YAML document string.  Writes to *output* if provided
      and returns the string.

    Args:
        report: The :class:`ProvenanceReport` to render.
        fmt: Output format — ``'rich'``, ``'json'``, or ``'yaml'``.
            Case-insensitive.  Defaults to ``'rich'``.
        output: Optional file-like object to write the rendered report to.
            For ``'rich'`` format, if ``None``, writes to ``sys.stdout``.
            For ``'json'``/``'yaml'``, if ``None``, only returns the string.
        force_terminal: For ``'rich'`` format, if ``True``, force Rich to
            emit ANSI colour codes even when *output* is not a TTY (useful
            for testing Rich output capture).

    Returns:
        The rendered report as a string (in the requested format).

    Raises:
        ValueError: If *fmt* is not one of ``'rich'``, ``'json'``, or
            ``'yaml'``.
    """
    fmt_lower = fmt.strip().lower()
    if fmt_lower not in ("rich", "json", "yaml"):
        raise ValueError(
            f"Unknown output format '{fmt}'. Choose from: rich, json, yaml."
        )

    if fmt_lower == "json":
        rendered = report.to_json()
        if output is not None:
            output.write(rendered)
            output.write("\n")
        return rendered

    if fmt_lower == "yaml":
        rendered = report.to_yaml()
        if output is not None:
            output.write(rendered)
        return rendered

    # --- Rich format ---
    # Capture to a string buffer so we can both write to output AND return.
    string_buffer = StringIO()
    capture_console = Console(
        file=string_buffer,
        force_terminal=force_terminal,
        highlight=False,
        markup=True,
    )
    _render_rich(report, capture_console)
    rendered = string_buffer.getvalue()

    # Write to the real console / provided output.
    if output is None:
        real_console = Console(file=sys.stdout, highlight=False, markup=True)
        _render_rich(report, real_console)
    else:
        output.write(rendered)

    return rendered


def render_rich_to_console(
    report: ProvenanceReport,
    console: Console | None = None,
) -> None:
    """Render a :class:`ProvenanceReport` directly to a Rich Console.

    Convenience wrapper for use in the CLI where an existing
    :class:`~rich.console.Console` instance is already available.

    Args:
        report: The :class:`ProvenanceReport` to render.
        console: A :class:`~rich.console.Console` instance.  If ``None``,
            a new console writing to ``sys.stdout`` is created.
    """
    if console is None:
        console = Console(highlight=False, markup=True)
    _render_rich(report, console)


def write_report_to_file(
    report: ProvenanceReport,
    path: str | Path,
    fmt: str = "json",
) -> None:
    """Serialise a :class:`ProvenanceReport` and write it to a file.

    Supports ``'json'``, ``'yaml'``, and ``'rich'`` formats. For ``'rich'``
    format the output is plain text (no ANSI codes) since the file is
    unlikely to be viewed in a terminal.

    Args:
        report: The :class:`ProvenanceReport` to write.
        path: Destination file path.  Parent directories are created if
            needed.
        fmt: Output format — ``'json'``, ``'yaml'``, or ``'rich'``.

    Raises:
        ValueError: If *fmt* is invalid.
        OSError: If the file cannot be written.
    """
    fmt_lower = fmt.strip().lower()
    if fmt_lower not in ("rich", "json", "yaml"):
        raise ValueError(
            f"Unknown output format '{fmt}'. Choose from: rich, json, yaml."
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt_lower == "json":
        content = report.to_json()
    elif fmt_lower == "yaml":
        content = report.to_yaml()
    else:
        # Rich format without ANSI codes.
        string_buffer = StringIO()
        plain_console = Console(
            file=string_buffer,
            force_terminal=False,
            no_color=True,
            highlight=False,
            markup=False,
        )
        _render_rich(report, plain_console)
        content = string_buffer.getvalue()

    out_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with timezone suffix."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _human_size(size_bytes: int) -> str:
    """Format *size_bytes* as a human-readable string.

    Args:
        size_bytes: File size in bytes.

    Returns:
        A compact string like ``'1.2 MiB'``, ``'345 KiB'``, or ``'512 B'``.
    """
    if size_bytes == 0:
        return "—"
    for unit, threshold in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if size_bytes >= threshold:
            return f"{size_bytes / threshold:.1f} {unit}"
    return f"{size_bytes} B"
