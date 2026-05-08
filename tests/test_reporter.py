"""Unit tests for model_provenance.reporter module.

Covers ProvenanceReport construction, verdict computation, serialisation
to JSON and YAML, Rich console rendering (smoke tests), and file output.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
import yaml

from model_provenance.checker import (
    CheckResult,
    FileCheckResult,
    FileCheckStatus,
    Verdict,
)
from model_provenance.fingerprint import FileFingerprint, FingerprintManifest
from model_provenance.license_check import (
    ComplianceFramework,
    ComplianceNote,
    LicenseReport,
    LicenseRestrictionLevel,
)
from model_provenance.reporter import (
    ProvenanceReport,
    _human_size,
    assemble_report,
    render_report,
    render_rich_to_console,
    write_report_to_file,
)
from model_provenance.scanner import (
    FindingCategory,
    FindingSeverity,
    ScanFinding,
    ScanReport,
)
from rich.console import Console


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_manifest(
    model_id: str = "test/model",
    revision: str = "main",
    n_files: int = 2,
) -> FingerprintManifest:
    """Build a minimal FingerprintManifest for testing."""
    files = [
        FileFingerprint(
            path=f"file_{i}.json" if i % 2 == 0 else f"model_{i}.bin",
            sha256="a" * 64,
            size_bytes=100 * (i + 1),
            file_type="config" if i % 2 == 0 else "weight",
        )
        for i in range(n_files)
    ]
    m = FingerprintManifest(model_id=model_id, revision=revision, source="hub")
    m.files = files
    m.aggregate_sha256 = "b" * 64
    return m


def _make_check_result(
    verdict: Verdict = Verdict.PASS,
    n_match: int = 2,
    n_mismatch: int = 0,
    n_unknown: int = 0,
) -> CheckResult:
    """Build a minimal CheckResult for testing."""
    file_results: list[FileCheckResult] = []

    for i in range(n_match):
        fp = FileFingerprint(
            path=f"match_{i}.json",
            sha256="a" * 64,
            size_bytes=100,
            file_type="config",
        )
        file_results.append(
            FileCheckResult(
                fingerprint=fp,
                status=FileCheckStatus.MATCH,
                detail="Hash matches known-good record.",
            )
        )

    for i in range(n_mismatch):
        fp = FileFingerprint(
            path=f"mismatch_{i}.bin",
            sha256="c" * 64,
            size_bytes=200,
            file_type="weight",
        )
        file_results.append(
            FileCheckResult(
                fingerprint=fp,
                status=FileCheckStatus.MISMATCH,
                detail="TAMPER DETECTED: computed=cccc…, expected=aaaa…",
            )
        )

    for i in range(n_unknown):
        fp = FileFingerprint(
            path=f"unknown_{i}.txt",
            sha256="d" * 64,
            size_bytes=50,
            file_type="other",
        )
        file_results.append(
            FileCheckResult(
                fingerprint=fp,
                status=FileCheckStatus.UNKNOWN,
                detail="No known-good hash found.",
            )
        )

    return CheckResult(
        model_id="test/model",
        revision="main",
        verdict=verdict,
        file_results=file_results,
        summary=f"{verdict.value.upper()} — test summary",
        db_coverage=0.9,
    )


def _make_scan_report(
    model_id: str = "test/model",
    n_findings: int = 0,
    severity: FindingSeverity = FindingSeverity.MEDIUM,
) -> ScanReport:
    """Build a minimal ScanReport for testing."""
    report = ScanReport(model_id=model_id)
    report.scanned_files = ["config.json", "model.bin"]
    for i in range(n_findings):
        report.findings.append(
            ScanFinding(
                path=f"suspicious_{i}.sh",
                category=FindingCategory.SHELL_SCRIPT,
                severity=severity,
                title="Shell script detected",
                description="Found a shell script.",
                remediation="Remove the shell script.",
            )
        )
    return report


def _make_license_report(
    spdx_id: str = "apache-2.0",
    restriction_level: LicenseRestrictionLevel = LicenseRestrictionLevel.PERMISSIVE,
    has_warnings: bool = False,
    has_critical: bool = False,
) -> LicenseReport:
    """Build a minimal LicenseReport for testing."""
    notes: list[ComplianceNote] = []
    if has_critical:
        notes.append(
            ComplianceNote(
                framework=ComplianceFramework.EU_AI_ACT,
                severity="critical",
                title="Critical compliance issue",
                detail="Detailed description.",
                remediation="Take immediate action.",
            )
        )
    elif has_warnings:
        notes.append(
            ComplianceNote(
                framework=ComplianceFramework.NIST_RMF,
                severity="warning",
                title="Warning compliance note",
                detail="Review required.",
                remediation="Consult legal counsel.",
            )
        )

    return LicenseReport(
        spdx_id=spdx_id,
        raw_license=spdx_id,
        restriction_level=restriction_level,
        is_restricted=restriction_level != LicenseRestrictionLevel.PERMISSIVE,
        is_osi_approved=restriction_level == LicenseRestrictionLevel.PERMISSIVE,
        allows_commercial_use=restriction_level == LicenseRestrictionLevel.PERMISSIVE,
        allows_redistribution=True,
        requires_attribution=True,
        compliance_notes=notes,
        summary=f"{spdx_id} — {restriction_level.value}",
        remediation_notes=["No action required."] if not has_warnings else ["Review license."],
    )


# ---------------------------------------------------------------------------
# ProvenanceReport — basic construction
# ---------------------------------------------------------------------------


class TestProvenanceReport:
    def test_default_verdict_warn(self) -> None:
        r = ProvenanceReport(model_id="test/model")
        assert r.verdict == Verdict.WARN

    def test_file_count_from_manifest(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            manifest=_make_manifest(n_files=3),
        )
        assert r.file_count == 3

    def test_file_count_from_check_result(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            check_result=_make_check_result(n_match=4),
        )
        assert r.file_count == 4

    def test_file_count_zero_when_no_components(self) -> None:
        r = ProvenanceReport(model_id="test/model")
        assert r.file_count == 0

    def test_has_scan_findings_false_when_clean(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            scan_report=_make_scan_report(n_findings=0),
        )
        assert not r.has_scan_findings

    def test_has_scan_findings_true_when_dirty(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            scan_report=_make_scan_report(n_findings=1),
        )
        assert r.has_scan_findings

    def test_has_scan_findings_false_when_no_scan(self) -> None:
        r = ProvenanceReport(model_id="test/model")
        assert not r.has_scan_findings

    def test_has_license_issues_false_when_permissive(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            license_report=_make_license_report(),
        )
        assert not r.has_license_issues

    def test_has_license_issues_true_when_warnings(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            license_report=_make_license_report(has_warnings=True),
        )
        assert r.has_license_issues

    def test_aggregate_sha256_from_manifest(self) -> None:
        manifest = _make_manifest()
        r = ProvenanceReport(model_id="test/model", manifest=manifest)
        assert r.aggregate_sha256 == "b" * 64

    def test_aggregate_sha256_none_when_no_manifest(self) -> None:
        r = ProvenanceReport(model_id="test/model")
        assert r.aggregate_sha256 is None

    def test_timestamp_format(self) -> None:
        import re
        r = ProvenanceReport(model_id="test/model")
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", r.timestamp)


# ---------------------------------------------------------------------------
# ProvenanceReport.to_dict
# ---------------------------------------------------------------------------


class TestProvenanceReportToDict:
    def _full_report(self) -> ProvenanceReport:
        return ProvenanceReport(
            model_id="test/model",
            revision="main",
            source="hub",
            verdict=Verdict.PASS,
            manifest=_make_manifest(),
            check_result=_make_check_result(),
            scan_report=_make_scan_report(),
            license_report=_make_license_report(),
            remediation_notes=["No action needed."],
            errors=[],
        )

    def test_top_level_keys(self) -> None:
        d = self._full_report().to_dict()
        expected = {
            "model_id", "revision", "source", "timestamp", "verdict",
            "aggregate_sha256", "file_count", "errors", "remediation",
            "manifest", "check_result", "scan_report", "license_report",
        }
        assert set(d.keys()) == expected

    def test_verdict_is_string(self) -> None:
        d = self._full_report().to_dict()
        assert isinstance(d["verdict"], str)
        assert d["verdict"] == "pass"

    def test_manifest_serialised(self) -> None:
        d = self._full_report().to_dict()
        assert d["manifest"] is not None
        assert "files" in d["manifest"]

    def test_check_result_serialised(self) -> None:
        d = self._full_report().to_dict()
        assert d["check_result"] is not None
        assert "file_results" in d["check_result"]

    def test_scan_report_serialised(self) -> None:
        d = self._full_report().to_dict()
        assert d["scan_report"] is not None
        assert "findings" in d["scan_report"]

    def test_license_report_serialised(self) -> None:
        d = self._full_report().to_dict()
        assert d["license_report"] is not None
        assert "spdx_id" in d["license_report"]

    def test_none_components_are_null(self) -> None:
        r = ProvenanceReport(model_id="test/model")
        d = r.to_dict()
        assert d["manifest"] is None
        assert d["check_result"] is None
        assert d["scan_report"] is None
        assert d["license_report"] is None

    def test_errors_included(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            errors=["Something went wrong."],
        )
        d = r.to_dict()
        assert "Something went wrong." in d["errors"]


# ---------------------------------------------------------------------------
# ProvenanceReport.to_json
# ---------------------------------------------------------------------------


class TestProvenanceReportToJson:
    def test_returns_valid_json(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.PASS,
            check_result=_make_check_result(),
        )
        raw = r.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_json_contains_verdict(self) -> None:
        r = ProvenanceReport(model_id="test/model", verdict=Verdict.FAIL)
        parsed = json.loads(r.to_json())
        assert parsed["verdict"] == "fail"

    def test_json_contains_model_id(self) -> None:
        r = ProvenanceReport(model_id="bert-base-uncased")
        parsed = json.loads(r.to_json())
        assert parsed["model_id"] == "bert-base-uncased"

    def test_json_indented(self) -> None:
        r = ProvenanceReport(model_id="test/model")
        raw = r.to_json(indent=4)
        # Should have newlines and 4-space indent.
        assert "\n" in raw

    def test_json_with_scan_findings(self) -> None:
        scan = _make_scan_report(n_findings=2, severity=FindingSeverity.HIGH)
        r = ProvenanceReport(model_id="test/model", scan_report=scan)
        parsed = json.loads(r.to_json())
        assert parsed["scan_report"]["finding_count"] == 2


# ---------------------------------------------------------------------------
# ProvenanceReport.to_yaml
# ---------------------------------------------------------------------------


class TestProvenanceReportToYaml:
    def test_returns_valid_yaml(self) -> None:
        r = ProvenanceReport(model_id="test/model", verdict=Verdict.WARN)
        raw = r.to_yaml()
        parsed = yaml.safe_load(raw)
        assert isinstance(parsed, dict)

    def test_yaml_contains_verdict(self) -> None:
        r = ProvenanceReport(model_id="test/model", verdict=Verdict.PASS)
        parsed = yaml.safe_load(r.to_yaml())
        assert parsed["verdict"] == "pass"

    def test_yaml_contains_model_id(self) -> None:
        r = ProvenanceReport(model_id="gpt2")
        parsed = yaml.safe_load(r.to_yaml())
        assert parsed["model_id"] == "gpt2"

    def test_yaml_with_license_report(self) -> None:
        lr = _make_license_report(spdx_id="mit")
        r = ProvenanceReport(model_id="test/model", license_report=lr)
        parsed = yaml.safe_load(r.to_yaml())
        assert parsed["license_report"]["spdx_id"] == "mit"


# ---------------------------------------------------------------------------
# assemble_report — verdict computation
# ---------------------------------------------------------------------------


class TestAssembleReport:
    def test_pass_when_all_clear(self) -> None:
        report = assemble_report(
            model_id="test/model",
            check_result=_make_check_result(verdict=Verdict.PASS, n_match=3),
            scan_report=_make_scan_report(n_findings=0),
            license_report=_make_license_report(),
        )
        assert report.verdict == Verdict.PASS

    def test_fail_when_check_fails(self) -> None:
        report = assemble_report(
            model_id="test/model",
            check_result=_make_check_result(
                verdict=Verdict.FAIL, n_match=1, n_mismatch=1
            ),
        )
        assert report.verdict == Verdict.FAIL

    def test_warn_when_check_warns(self) -> None:
        report = assemble_report(
            model_id="test/model",
            check_result=_make_check_result(
                verdict=Verdict.WARN, n_match=1, n_unknown=1
            ),
        )
        assert report.verdict == Verdict.WARN

    def test_fail_when_scan_has_critical(self) -> None:
        report = assemble_report(
            model_id="test/model",
            scan_report=_make_scan_report(
                n_findings=1, severity=FindingSeverity.CRITICAL
            ),
        )
        assert report.verdict == Verdict.FAIL

    def test_fail_when_scan_has_high(self) -> None:
        report = assemble_report(
            model_id="test/model",
            scan_report=_make_scan_report(
                n_findings=1, severity=FindingSeverity.HIGH
            ),
        )
        assert report.verdict == Verdict.FAIL

    def test_warn_when_scan_has_medium(self) -> None:
        report = assemble_report(
            model_id="test/model",
            scan_report=_make_scan_report(
                n_findings=1, severity=FindingSeverity.MEDIUM
            ),
        )
        assert report.verdict == Verdict.WARN

    def test_fail_when_license_critical(self) -> None:
        report = assemble_report(
            model_id="test/model",
            license_report=_make_license_report(
                restriction_level=LicenseRestrictionLevel.NON_COMMERCIAL,
                has_critical=True,
            ),
        )
        assert report.verdict == Verdict.FAIL

    def test_warn_when_license_warns(self) -> None:
        report = assemble_report(
            model_id="test/model",
            license_report=_make_license_report(
                restriction_level=LicenseRestrictionLevel.COPYLEFT,
                has_warnings=True,
            ),
        )
        assert report.verdict == Verdict.WARN

    def test_fail_overrides_warn(self) -> None:
        """A FAIL from checker should not be downgraded to WARN by license."""
        report = assemble_report(
            model_id="test/model",
            check_result=_make_check_result(verdict=Verdict.FAIL, n_mismatch=1),
            license_report=_make_license_report(has_warnings=True),
        )
        assert report.verdict == Verdict.FAIL

    def test_errors_cause_fail(self) -> None:
        report = assemble_report(
            model_id="test/model",
            errors=["Something went wrong."],
        )
        assert report.verdict == Verdict.FAIL

    def test_remediation_collected_from_checker(self) -> None:
        check_result = _make_check_result(
            verdict=Verdict.FAIL, n_mismatch=1
        )
        report = assemble_report(
            model_id="test/model",
            check_result=check_result,
        )
        # The checker's remediation_notes should appear in the report.
        assert len(report.remediation_notes) > 0

    def test_remediation_collected_from_license(self) -> None:
        lr = _make_license_report(
            has_warnings=True,
        )
        report = assemble_report(
            model_id="test/model",
            license_report=lr,
        )
        assert any("Review license." in n for n in report.remediation_notes)

    def test_model_id_preserved(self) -> None:
        report = assemble_report(model_id="bert-base-uncased")
        assert report.model_id == "bert-base-uncased"

    def test_revision_preserved(self) -> None:
        report = assemble_report(model_id="test/model", revision="v2.0")
        assert report.revision == "v2.0"

    def test_source_preserved(self) -> None:
        report = assemble_report(model_id="test/model", source="hub")
        assert report.source == "hub"

    def test_pass_with_no_components(self) -> None:
        """No components → no signals → defaults to PASS."""
        report = assemble_report(model_id="test/model")
        # No check_result, no scan, no license → pass (no downgraders)
        assert report.verdict == Verdict.PASS

    def test_clean_scan_does_not_affect_pass_verdict(self) -> None:
        report = assemble_report(
            model_id="test/model",
            check_result=_make_check_result(verdict=Verdict.PASS),
            scan_report=_make_scan_report(n_findings=0),
        )
        assert report.verdict == Verdict.PASS


# ---------------------------------------------------------------------------
# render_report — JSON format
# ---------------------------------------------------------------------------


class TestRenderReportJson:
    def _make_report(self) -> ProvenanceReport:
        return ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.PASS,
            check_result=_make_check_result(),
            scan_report=_make_scan_report(),
            license_report=_make_license_report(),
        )

    def test_returns_string(self) -> None:
        r = self._make_report()
        result = render_report(r, fmt="json")
        assert isinstance(result, str)

    def test_valid_json(self) -> None:
        r = self._make_report()
        result = render_report(r, fmt="json")
        parsed = json.loads(result)
        assert parsed["model_id"] == "test/model"

    def test_writes_to_output(self) -> None:
        r = self._make_report()
        buf = StringIO()
        render_report(r, fmt="json", output=buf)
        buf.seek(0)
        parsed = json.loads(buf.read())
        assert parsed["verdict"] == "pass"

    def test_case_insensitive_format(self) -> None:
        r = self._make_report()
        result = render_report(r, fmt="JSON")
        assert json.loads(result)["model_id"] == "test/model"


# ---------------------------------------------------------------------------
# render_report — YAML format
# ---------------------------------------------------------------------------


class TestRenderReportYaml:
    def _make_report(self) -> ProvenanceReport:
        return ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.WARN,
        )

    def test_returns_string(self) -> None:
        result = render_report(self._make_report(), fmt="yaml")
        assert isinstance(result, str)

    def test_valid_yaml(self) -> None:
        result = render_report(self._make_report(), fmt="yaml")
        parsed = yaml.safe_load(result)
        assert parsed["model_id"] == "test/model"

    def test_writes_to_output(self) -> None:
        r = self._make_report()
        buf = StringIO()
        render_report(r, fmt="yaml", output=buf)
        buf.seek(0)
        parsed = yaml.safe_load(buf.read())
        assert parsed["verdict"] == "warn"

    def test_case_insensitive_format(self) -> None:
        r = self._make_report()
        result = render_report(r, fmt="YAML")
        parsed = yaml.safe_load(result)
        assert parsed["model_id"] == "test/model"


# ---------------------------------------------------------------------------
# render_report — Rich format (smoke tests)
# ---------------------------------------------------------------------------


class TestRenderReportRich:
    def _make_full_report(self) -> ProvenanceReport:
        return ProvenanceReport(
            model_id="test/model",
            revision="main",
            source="hub",
            verdict=Verdict.PASS,
            manifest=_make_manifest(),
            check_result=_make_check_result(),
            scan_report=_make_scan_report(),
            license_report=_make_license_report(),
            remediation_notes=["No action required."],
        )

    def test_returns_non_empty_string(self) -> None:
        r = self._make_full_report()
        result = render_report(r, fmt="rich", force_terminal=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_writes_to_output_buffer(self) -> None:
        r = self._make_full_report()
        buf = StringIO()
        render_report(r, fmt="rich", output=buf, force_terminal=False)
        buf.seek(0)
        content = buf.read()
        assert "test/model" in content or len(content) > 0

    def test_case_insensitive_format(self) -> None:
        r = self._make_full_report()
        result = render_report(r, fmt="RICH", force_terminal=False)
        assert isinstance(result, str)

    def test_invalid_format_raises(self) -> None:
        r = self._make_full_report()
        with pytest.raises(ValueError, match="Unknown output format"):
            render_report(r, fmt="xml")

    def test_rich_with_scan_findings_rendered(self) -> None:
        scan = _make_scan_report(n_findings=2, severity=FindingSeverity.HIGH)
        r = ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.FAIL,
            scan_report=scan,
        )
        result = render_report(r, fmt="rich", force_terminal=False)
        assert isinstance(result, str)

    def test_rich_with_license_warnings(self) -> None:
        lr = _make_license_report(
            restriction_level=LicenseRestrictionLevel.NON_COMMERCIAL,
            has_critical=True,
        )
        r = ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.FAIL,
            license_report=lr,
        )
        result = render_report(r, fmt="rich", force_terminal=False)
        assert isinstance(result, str)

    def test_rich_with_errors(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.FAIL,
            errors=["Read error on config.json"],
        )
        result = render_report(r, fmt="rich", force_terminal=False)
        assert isinstance(result, str)

    def test_rich_with_mismatch_in_check_result(self) -> None:
        check = _make_check_result(
            verdict=Verdict.FAIL, n_match=1, n_mismatch=1
        )
        r = ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.FAIL,
            check_result=check,
        )
        result = render_report(r, fmt="rich", force_terminal=False)
        assert isinstance(result, str)

    def test_rich_unknown_files_shown(self) -> None:
        check = _make_check_result(
            verdict=Verdict.WARN, n_match=1, n_unknown=2
        )
        r = ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.WARN,
            check_result=check,
        )
        result = render_report(r, fmt="rich", force_terminal=False)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# render_rich_to_console
# ---------------------------------------------------------------------------


class TestRenderRichToConsole:
    def test_renders_without_error(self) -> None:
        r = ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.PASS,
        )
        buf = StringIO()
        console = Console(file=buf, no_color=True, highlight=False, markup=False)
        render_rich_to_console(r, console=console)  # Should not raise.

    def test_creates_default_console_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Passing console=None should not raise."""
        r = ProvenanceReport(model_id="test/model")
        # We can't easily suppress stdout here; just verify no exception.
        buf = StringIO()
        import model_provenance.reporter as reporter_mod
        original_console_cls = reporter_mod.Console

        class MockConsole:
            def __init__(self, **kwargs):
                self._buf = StringIO()

            def print(self, *args, **kwargs):
                pass  # silently absorb

        monkeypatch.setattr(reporter_mod, "Console", MockConsole)
        try:
            render_rich_to_console(r, console=None)
        finally:
            monkeypatch.setattr(reporter_mod, "Console", original_console_cls)


# ---------------------------------------------------------------------------
# write_report_to_file
# ---------------------------------------------------------------------------


class TestWriteReportToFile:
    def _make_report(self) -> ProvenanceReport:
        return ProvenanceReport(
            model_id="test/model",
            verdict=Verdict.PASS,
            license_report=_make_license_report(),
        )

    def test_writes_json(self, tmp_path: Path) -> None:
        r = self._make_report()
        out = tmp_path / "report.json"
        write_report_to_file(r, out, fmt="json")
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed["model_id"] == "test/model"

    def test_writes_yaml(self, tmp_path: Path) -> None:
        r = self._make_report()
        out = tmp_path / "report.yaml"
        write_report_to_file(r, out, fmt="yaml")
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert parsed["verdict"] == "pass"

    def test_writes_rich_plain_text(self, tmp_path: Path) -> None:
        r = self._make_report()
        out = tmp_path / "report.txt"
        write_report_to_file(r, out, fmt="rich")
        assert out.exists()
        content = out.read_text()
        assert len(content) > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        r = self._make_report()
        out = tmp_path / "nested" / "dir" / "report.json"
        write_report_to_file(r, out, fmt="json")
        assert out.exists()

    def test_invalid_format_raises(self, tmp_path: Path) -> None:
        r = self._make_report()
        with pytest.raises(ValueError, match="Unknown output format"):
            write_report_to_file(r, tmp_path / "out.xml", fmt="xml")

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        r = self._make_report()
        out = str(tmp_path / "report.json")
        write_report_to_file(r, out, fmt="json")
        assert Path(out).exists()

    def test_json_round_trips(self, tmp_path: Path) -> None:
        r = self._make_report()
        out = tmp_path / "round_trip.json"
        write_report_to_file(r, out, fmt="json")
        parsed = json.loads(out.read_text())
        assert parsed["license_report"]["spdx_id"] == "apache-2.0"

    def test_yaml_round_trips(self, tmp_path: Path) -> None:
        r = self._make_report()
        out = tmp_path / "round_trip.yaml"
        write_report_to_file(r, out, fmt="yaml")
        parsed = yaml.safe_load(out.read_text())
        assert parsed["license_report"]["spdx_id"] == "apache-2.0"


# ---------------------------------------------------------------------------
# _human_size utility
# ---------------------------------------------------------------------------


class TestHumanSize:
    @pytest.mark.parametrize(
        "size_bytes,expected_contains",
        [
            (0, "—"),
            (512, "B"),
            (1024, "KiB"),
            (1024 * 1024, "MiB"),
            (1024 * 1024 * 1024, "GiB"),
            (500, "B"),
            (2048, "KiB"),
            (5 * 1024 * 1024, "MiB"),
        ],
    )
    def test_human_size(self, size_bytes: int, expected_contains: str) -> None:
        result = _human_size(size_bytes)
        assert expected_contains in result
