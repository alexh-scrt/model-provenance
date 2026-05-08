"""Unit tests for model_provenance.checker module.

Covers fingerprint comparison logic, tamper detection, verdict computation,
coverage calculation, and the convenience wrapper functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_provenance.checker import (
    CheckResult,
    FileCheckResult,
    FileCheckStatus,
    FingerprintChecker,
    Verdict,
    _build_summary,
    _compute_coverage,
    _compute_verdict,
    check_file_against_db,
    check_manifest,
)
from model_provenance.db import HashDatabase
from model_provenance.fingerprint import FileFingerprint, FingerprintManifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> HashDatabase:
    """Provide an in-memory HashDatabase with schema initialised."""
    instance = HashDatabase(":memory:")
    instance.init_schema()
    return instance


@pytest.fixture()
def populated_db(db: HashDatabase) -> HashDatabase:
    """Provide a db pre-populated with known-good hashes for 'test/model'."""
    db.add_hash("test/model", "config.json", "a" * 64, revision="main")
    db.add_hash("test/model", "model.bin", "b" * 64, revision="main")
    db.add_hash("test/model", "tokenizer.json", "c" * 64, revision="main")
    return db


def _make_fingerprint(
    path: str,
    sha256: str = "a" * 64,
    size_bytes: int = 100,
    file_type: str = "config",
    error: str | None = None,
) -> FileFingerprint:
    """Helper to construct a FileFingerprint."""
    return FileFingerprint(
        path=path,
        sha256=sha256,
        size_bytes=size_bytes,
        file_type=file_type,
        error=error,
    )


def _make_manifest(
    model_id: str = "test/model",
    revision: str = "main",
    files: list[FileFingerprint] | None = None,
) -> FingerprintManifest:
    """Helper to construct a FingerprintManifest."""
    m = FingerprintManifest(
        model_id=model_id,
        revision=revision,
        source="local",
    )
    m.files = files or []
    return m


# ---------------------------------------------------------------------------
# FileCheckStatus enum
# ---------------------------------------------------------------------------


class TestFileCheckStatus:
    def test_values_are_strings(self) -> None:
        assert FileCheckStatus.MATCH.value == "match"
        assert FileCheckStatus.MISMATCH.value == "mismatch"
        assert FileCheckStatus.UNKNOWN.value == "unknown"
        assert FileCheckStatus.ERROR.value == "error"
        assert FileCheckStatus.NEW.value == "new"

    def test_is_string_subclass(self) -> None:
        assert isinstance(FileCheckStatus.MATCH, str)


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_values(self) -> None:
        assert Verdict.PASS.value == "pass"
        assert Verdict.WARN.value == "warn"
        assert Verdict.FAIL.value == "fail"

    def test_is_string_subclass(self) -> None:
        assert isinstance(Verdict.PASS, str)


# ---------------------------------------------------------------------------
# FileCheckResult
# ---------------------------------------------------------------------------


class TestFileCheckResult:
    def _make(
        self,
        status: FileCheckStatus = FileCheckStatus.MATCH,
        path: str = "config.json",
        sha256: str = "a" * 64,
    ) -> FileCheckResult:
        fp = _make_fingerprint(path=path, sha256=sha256)
        return FileCheckResult(
            fingerprint=fp,
            status=status,
            detail="test detail",
        )

    def test_path_delegates_to_fingerprint(self) -> None:
        fcr = self._make(path="model.bin")
        assert fcr.path == "model.bin"

    def test_is_ok_match(self) -> None:
        assert self._make(status=FileCheckStatus.MATCH).is_ok

    def test_is_ok_new(self) -> None:
        assert self._make(status=FileCheckStatus.NEW).is_ok

    def test_is_ok_false_for_mismatch(self) -> None:
        assert not self._make(status=FileCheckStatus.MISMATCH).is_ok

    def test_is_ok_false_for_unknown(self) -> None:
        assert not self._make(status=FileCheckStatus.UNKNOWN).is_ok

    def test_is_ok_false_for_error(self) -> None:
        assert not self._make(status=FileCheckStatus.ERROR).is_ok

    def test_is_fail_mismatch(self) -> None:
        assert self._make(status=FileCheckStatus.MISMATCH).is_fail

    def test_is_fail_error(self) -> None:
        assert self._make(status=FileCheckStatus.ERROR).is_fail

    def test_is_fail_false_for_match(self) -> None:
        assert not self._make(status=FileCheckStatus.MATCH).is_fail

    def test_is_fail_false_for_unknown(self) -> None:
        assert not self._make(status=FileCheckStatus.UNKNOWN).is_fail

    def test_is_warn_unknown(self) -> None:
        assert self._make(status=FileCheckStatus.UNKNOWN).is_warn

    def test_is_warn_false_for_match(self) -> None:
        assert not self._make(status=FileCheckStatus.MATCH).is_warn

    def test_to_dict_keys(self) -> None:
        fcr = self._make()
        d = fcr.to_dict()
        expected_keys = {
            "path",
            "sha256",
            "file_type",
            "size_bytes",
            "status",
            "known_sha256",
            "detail",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_status_is_string(self) -> None:
        fcr = self._make(status=FileCheckStatus.MATCH)
        assert fcr.to_dict()["status"] == "match"

    def test_to_dict_known_sha256_none_when_no_known_hash(self) -> None:
        fcr = self._make()
        assert fcr.to_dict()["known_sha256"] is None

    def test_to_dict_known_sha256_populated(self) -> None:
        from model_provenance.db import KnownHash

        fp = _make_fingerprint(path="config.json", sha256="a" * 64)
        kh = KnownHash(
            model_id="test/model",
            revision="main",
            file_path="config.json",
            sha256="a" * 64,
        )
        fcr = FileCheckResult(fingerprint=fp, status=FileCheckStatus.MATCH, known_hash=kh)
        assert fcr.to_dict()["known_sha256"] == "a" * 64


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


class TestCheckResult:
    def _make(
        self,
        verdict: Verdict = Verdict.PASS,
        n_match: int = 2,
        n_mismatch: int = 0,
        n_unknown: int = 0,
        n_error: int = 0,
        n_new: int = 0,
    ) -> CheckResult:
        file_results: list[FileCheckResult] = []

        for i in range(n_match):
            fp = _make_fingerprint(path=f"match_{i}.json", sha256="a" * 64)
            file_results.append(
                FileCheckResult(fingerprint=fp, status=FileCheckStatus.MATCH)
            )
        for i in range(n_mismatch):
            fp = _make_fingerprint(path=f"mismatch_{i}.bin", sha256="c" * 64, file_type="weight")
            file_results.append(
                FileCheckResult(fingerprint=fp, status=FileCheckStatus.MISMATCH)
            )
        for i in range(n_unknown):
            fp = _make_fingerprint(path=f"unknown_{i}.txt", sha256="d" * 64, file_type="other")
            file_results.append(
                FileCheckResult(fingerprint=fp, status=FileCheckStatus.UNKNOWN)
            )
        for i in range(n_error):
            fp = _make_fingerprint(
                path=f"error_{i}.bin", sha256="", file_type="weight", error="read error"
            )
            file_results.append(
                FileCheckResult(fingerprint=fp, status=FileCheckStatus.ERROR)
            )
        for i in range(n_new):
            fp = _make_fingerprint(path=f"new_{i}.bin", sha256="e" * 64, file_type="weight")
            file_results.append(
                FileCheckResult(fingerprint=fp, status=FileCheckStatus.NEW)
            )

        return CheckResult(
            model_id="test/model",
            revision="main",
            verdict=verdict,
            file_results=file_results,
            summary="test summary",
            db_coverage=0.8,
        )

    def test_matches_returns_correct_files(self) -> None:
        cr = self._make(n_match=3, n_mismatch=1)
        assert len(cr.matches) == 3

    def test_mismatches_returns_correct_files(self) -> None:
        cr = self._make(n_match=1, n_mismatch=2)
        assert len(cr.mismatches) == 2

    def test_unknowns_returns_correct_files(self) -> None:
        cr = self._make(n_unknown=2)
        assert len(cr.unknowns) == 2

    def test_errors_returns_correct_files(self) -> None:
        cr = self._make(n_error=1)
        assert len(cr.errors) == 1

    def test_new_files_returns_correct(self) -> None:
        cr = self._make(n_new=2)
        assert len(cr.new_files) == 2

    def test_file_count(self) -> None:
        cr = self._make(n_match=2, n_mismatch=1, n_unknown=1)
        assert cr.file_count == 4

    def test_remediation_notes_mismatch(self) -> None:
        cr = self._make(n_mismatch=1)
        notes = cr.remediation_notes
        assert any("CRITICAL" in note or "mismatch" in note.lower() for note in notes)

    def test_remediation_notes_error(self) -> None:
        cr = self._make(n_error=1)
        notes = cr.remediation_notes
        assert any("error" in note.lower() or "WARNING" in note for note in notes)

    def test_remediation_notes_unknown(self) -> None:
        cr = self._make(n_unknown=1)
        notes = cr.remediation_notes
        assert any("unknown" in note.lower() or "INFO" in note for note in notes)

    def test_remediation_notes_empty_when_all_match(self) -> None:
        cr = self._make(n_match=3)
        assert cr.remediation_notes == []

    def test_remediation_mismatch_truncates_at_5(self) -> None:
        cr = self._make(n_mismatch=7)
        notes = cr.remediation_notes
        # Should contain exactly one mismatch note mentioning count
        mismatch_notes = [n for n in notes if "mismatch" in n.lower() or "CRITICAL" in n]
        assert len(mismatch_notes) >= 1
        assert "7" in mismatch_notes[0]
        assert "..." in mismatch_notes[0]

    def test_to_dict_keys(self) -> None:
        cr = self._make(n_match=2)
        d = cr.to_dict()
        expected_keys = {
            "model_id",
            "revision",
            "verdict",
            "summary",
            "db_coverage",
            "file_count",
            "match_count",
            "mismatch_count",
            "unknown_count",
            "error_count",
            "new_count",
            "file_results",
            "remediation",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_verdict_is_string(self) -> None:
        cr = self._make(verdict=Verdict.PASS)
        assert cr.to_dict()["verdict"] == "pass"

    def test_to_dict_file_results_is_list(self) -> None:
        cr = self._make(n_match=2)
        d = cr.to_dict()
        assert isinstance(d["file_results"], list)
        assert len(d["file_results"]) == 2

    def test_to_dict_counts(self) -> None:
        cr = self._make(n_match=3, n_mismatch=1, n_unknown=2)
        d = cr.to_dict()
        assert d["match_count"] == 3
        assert d["mismatch_count"] == 1
        assert d["unknown_count"] == 2
        assert d["error_count"] == 0
        assert d["file_count"] == 6


# ---------------------------------------------------------------------------
# _compute_verdict
# ---------------------------------------------------------------------------


class TestComputeVerdict:
    def _fcr(self, status: FileCheckStatus, path: str = "f.json") -> FileCheckResult:
        fp = _make_fingerprint(path=path, sha256="a" * 64)
        return FileCheckResult(fingerprint=fp, status=status)

    def test_pass_when_all_match(self) -> None:
        results = [self._fcr(FileCheckStatus.MATCH, f"f{i}.json") for i in range(3)]
        assert _compute_verdict(results) == Verdict.PASS

    def test_pass_when_all_new(self) -> None:
        results = [self._fcr(FileCheckStatus.NEW, f"f{i}.bin") for i in range(2)]
        assert _compute_verdict(results) == Verdict.PASS

    def test_warn_when_has_unknown_non_strict(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MATCH, "a.json"),
            self._fcr(FileCheckStatus.UNKNOWN, "b.bin"),
        ]
        assert _compute_verdict(results, strict=False) == Verdict.WARN

    def test_fail_when_has_unknown_strict(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MATCH, "a.json"),
            self._fcr(FileCheckStatus.UNKNOWN, "b.bin"),
        ]
        assert _compute_verdict(results, strict=True) == Verdict.FAIL

    def test_fail_when_has_mismatch(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MATCH, "a.json"),
            self._fcr(FileCheckStatus.MISMATCH, "b.bin"),
        ]
        assert _compute_verdict(results) == Verdict.FAIL

    def test_fail_when_has_error(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MATCH, "a.json"),
            self._fcr(FileCheckStatus.ERROR, "b.bin"),
        ]
        assert _compute_verdict(results) == Verdict.FAIL

    def test_fail_overrides_unknown(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MISMATCH, "a.bin"),
            self._fcr(FileCheckStatus.UNKNOWN, "b.txt"),
        ]
        assert _compute_verdict(results) == Verdict.FAIL

    def test_empty_results_returns_warn(self) -> None:
        assert _compute_verdict([]) == Verdict.WARN

    def test_mixed_match_and_new_is_pass(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MATCH, "a.json"),
            self._fcr(FileCheckStatus.NEW, "b.bin"),
        ]
        assert _compute_verdict(results) == Verdict.PASS


# ---------------------------------------------------------------------------
# _compute_coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def _fcr(self, status: FileCheckStatus, path: str = "f.json") -> FileCheckResult:
        fp = _make_fingerprint(path=path, sha256="a" * 64)
        return FileCheckResult(fingerprint=fp, status=status)

    def test_empty_returns_none(self) -> None:
        assert _compute_coverage([]) is None

    def test_all_match_returns_one(self) -> None:
        results = [self._fcr(FileCheckStatus.MATCH, f"f{i}.json") for i in range(3)]
        assert _compute_coverage(results) == pytest.approx(1.0)

    def test_all_unknown_returns_zero(self) -> None:
        results = [self._fcr(FileCheckStatus.UNKNOWN, f"f{i}.json") for i in range(3)]
        assert _compute_coverage(results) == pytest.approx(0.0)

    def test_half_match_half_unknown(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MATCH, "a.json"),
            self._fcr(FileCheckStatus.UNKNOWN, "b.bin"),
        ]
        assert _compute_coverage(results) == pytest.approx(0.5)

    def test_mismatch_counts_as_known(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MISMATCH, "a.bin"),
            self._fcr(FileCheckStatus.UNKNOWN, "b.txt"),
        ]
        assert _compute_coverage(results) == pytest.approx(0.5)

    def test_new_counts_as_known(self) -> None:
        results = [
            self._fcr(FileCheckStatus.NEW, "a.bin"),
            self._fcr(FileCheckStatus.UNKNOWN, "b.txt"),
        ]
        assert _compute_coverage(results) == pytest.approx(0.5)

    def test_error_does_not_count_as_known(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MATCH, "a.json"),
            self._fcr(FileCheckStatus.ERROR, "b.bin"),
        ]
        assert _compute_coverage(results) == pytest.approx(0.5)

    def test_returns_float_between_0_and_1(self) -> None:
        results = [
            self._fcr(FileCheckStatus.MATCH, "a.json"),
            self._fcr(FileCheckStatus.MATCH, "b.json"),
            self._fcr(FileCheckStatus.UNKNOWN, "c.bin"),
        ]
        cov = _compute_coverage(results)
        assert cov is not None
        assert 0.0 <= cov <= 1.0


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def _make_results(self) -> list[FileCheckResult]:
        return [
            FileCheckResult(
                fingerprint=_make_fingerprint("a.json"),
                status=FileCheckStatus.MATCH,
            ),
            FileCheckResult(
                fingerprint=_make_fingerprint("b.bin", file_type="weight"),
                status=FileCheckStatus.MISMATCH,
            ),
            FileCheckResult(
                fingerprint=_make_fingerprint("c.txt", file_type="other"),
                status=FileCheckStatus.UNKNOWN,
            ),
        ]

    def test_returns_string(self) -> None:
        results = self._make_results()
        summary = _build_summary("test/model", "main", results, Verdict.FAIL)
        assert isinstance(summary, str)

    def test_contains_model_id(self) -> None:
        results = self._make_results()
        summary = _build_summary("bert-base-uncased", "main", results, Verdict.PASS)
        assert "bert-base-uncased" in summary

    def test_contains_revision(self) -> None:
        results = self._make_results()
        summary = _build_summary("test/model", "v1.0", results, Verdict.WARN)
        assert "v1.0" in summary

    def test_contains_verdict(self) -> None:
        results = self._make_results()
        summary = _build_summary("test/model", "main", results, Verdict.FAIL)
        assert "FAIL" in summary

    def test_contains_counts(self) -> None:
        results = self._make_results()
        summary = _build_summary("test/model", "main", results, Verdict.FAIL)
        # Should mention file count numbers
        assert "3" in summary  # total files

    def test_empty_results(self) -> None:
        summary = _build_summary("test/model", "main", [], Verdict.WARN)
        assert isinstance(summary, str)
        assert len(summary) > 0


# ---------------------------------------------------------------------------
# FingerprintChecker — single file checking
# ---------------------------------------------------------------------------


class TestFingerprintCheckerSingleFile:
    def test_match_when_hashes_equal(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(path="config.json", sha256="a" * 64)
        result = checker.check_single_file(fp, model_id="test/model", revision="main")
        assert result.status == FileCheckStatus.MATCH

    def test_mismatch_when_hashes_differ(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        # DB has "a"*64 for config.json; we provide a different hash
        fp = _make_fingerprint(path="config.json", sha256="z" * 64)
        result = checker.check_single_file(fp, model_id="test/model", revision="main")
        assert result.status == FileCheckStatus.MISMATCH

    def test_mismatch_detail_contains_hashes(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(path="config.json", sha256="z" * 64)
        result = checker.check_single_file(fp, model_id="test/model", revision="main")
        assert "zzzz" in result.detail.lower() or "aaaa" in result.detail.lower()

    def test_unknown_when_not_in_db(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(path="nonexistent_file.txt", sha256="d" * 64, file_type="other")
        result = checker.check_single_file(fp, model_id="test/model", revision="main")
        assert result.status == FileCheckStatus.UNKNOWN

    def test_unknown_detail_mentions_no_record(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(path="new_file.bin", sha256="e" * 64, file_type="weight")
        result = checker.check_single_file(fp, model_id="test/model", revision="main")
        assert "no known-good hash" in result.detail.lower() or "not found" in result.detail.lower()

    def test_error_when_fingerprint_has_error(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(
            path="bad.bin", sha256="", file_type="weight", error="Permission denied"
        )
        result = checker.check_single_file(fp, model_id="test/model", revision="main")
        assert result.status == FileCheckStatus.ERROR

    def test_error_detail_contains_fingerprint_error(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(
            path="bad.bin", sha256="", file_type="weight", error="Permission denied"
        )
        result = checker.check_single_file(fp, model_id="test/model", revision="main")
        assert "Permission denied" in result.detail or "Fingerprint error" in result.detail

    def test_unknown_when_wrong_revision(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(path="config.json", sha256="a" * 64)
        # DB has revision="main"; we query with revision="v2.0"
        result = checker.check_single_file(fp, model_id="test/model", revision="v2.0")
        assert result.status == FileCheckStatus.UNKNOWN

    def test_match_has_known_hash_populated(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(path="config.json", sha256="a" * 64)
        result = checker.check_single_file(fp, model_id="test/model", revision="main")
        assert result.known_hash is not None
        assert result.known_hash.sha256 == "a" * 64

    def test_unknown_has_no_known_hash(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(path="ghost.bin", sha256="f" * 64)
        result = checker.check_single_file(fp, model_id="test/model", revision="main")
        assert result.known_hash is None

    def test_different_model_id_gives_unknown(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        fp = _make_fingerprint(path="config.json", sha256="a" * 64)
        result = checker.check_single_file(fp, model_id="other/model", revision="main")
        assert result.status == FileCheckStatus.UNKNOWN


# ---------------------------------------------------------------------------
# FingerprintChecker — manifest checking
# ---------------------------------------------------------------------------


class TestFingerprintCheckerManifest:
    def test_all_match_verdict_pass(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("model.bin", sha256="b" * 64, file_type="weight"),
                _make_fingerprint("tokenizer.json", sha256="c" * 64),
            ]
        )
        result = checker.check_manifest(manifest)
        assert result.verdict == Verdict.PASS

    def test_mismatch_gives_fail(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="z" * 64),  # tampered
                _make_fingerprint("model.bin", sha256="b" * 64, file_type="weight"),
            ]
        )
        result = checker.check_manifest(manifest)
        assert result.verdict == Verdict.FAIL

    def test_unknown_gives_warn_non_strict(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db, strict=False)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("unknown_file.txt", sha256="d" * 64, file_type="other"),
            ]
        )
        result = checker.check_manifest(manifest)
        assert result.verdict == Verdict.WARN

    def test_unknown_gives_fail_strict(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db, strict=True)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("unknown_file.txt", sha256="d" * 64, file_type="other"),
            ]
        )
        result = checker.check_manifest(manifest)
        assert result.verdict == Verdict.FAIL

    def test_error_in_fingerprint_gives_fail(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint(
                    "bad.bin", sha256="", file_type="weight", error="read error"
                ),
            ]
        )
        result = checker.check_manifest(manifest)
        assert result.verdict == Verdict.FAIL

    def test_empty_manifest_gives_warn(self, db: HashDatabase) -> None:
        checker = FingerprintChecker(db=db)
        manifest = _make_manifest(files=[])
        result = checker.check_manifest(manifest)
        assert result.verdict == Verdict.WARN

    def test_result_file_count_matches_manifest(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("model.bin", sha256="b" * 64, file_type="weight"),
            ]
        )
        result = checker.check_manifest(manifest)
        assert result.file_count == 2

    def test_result_model_id_preserved(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(model_id="my-custom-model")
        result = checker.check_manifest(manifest)
        assert result.model_id == "my-custom-model"

    def test_result_revision_from_manifest(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(revision="v1.0")
        result = checker.check_manifest(manifest)
        assert result.revision == "v1.0"

    def test_revision_override(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(revision="main")
        result = checker.check_manifest(manifest, revision="v2.0")
        # v2.0 has no records, so all files should be UNKNOWN
        assert result.revision == "v2.0"

    def test_db_coverage_calculated(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("new_file.txt", sha256="d" * 64, file_type="other"),
            ]
        )
        result = checker.check_manifest(manifest)
        # 1 match out of 2 files = 0.5 coverage
        assert result.db_coverage == pytest.approx(0.5)

    def test_db_coverage_none_for_empty_manifest(self, db: HashDatabase) -> None:
        checker = FingerprintChecker(db=db)
        manifest = _make_manifest(files=[])
        result = checker.check_manifest(manifest)
        assert result.db_coverage is None

    def test_summary_populated(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[_make_fingerprint("config.json", sha256="a" * 64)]
        )
        result = checker.check_manifest(manifest)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_all_file_results_populated(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("model.bin", sha256="b" * 64, file_type="weight"),
                _make_fingerprint("tokenizer.json", sha256="c" * 64),
            ]
        )
        result = checker.check_manifest(manifest)
        assert len(result.file_results) == 3

    def test_mismatches_list_populated(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="z" * 64),  # tampered
                _make_fingerprint("model.bin", sha256="b" * 64, file_type="weight"),
            ]
        )
        result = checker.check_manifest(manifest)
        assert len(result.mismatches) == 1
        assert result.mismatches[0].path == "config.json"

    def test_unknowns_list_populated(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("surprise.bin", sha256="x" * 64, file_type="weight"),
            ]
        )
        result = checker.check_manifest(manifest)
        assert len(result.unknowns) == 1
        assert result.unknowns[0].path == "surprise.bin"

    def test_matches_list_populated(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("model.bin", sha256="b" * 64, file_type="weight"),
            ]
        )
        result = checker.check_manifest(manifest)
        assert len(result.matches) == 2

    def test_errors_list_populated(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[
                _make_fingerprint(
                    "bad.bin", sha256="", file_type="weight", error="read error"
                ),
            ]
        )
        result = checker.check_manifest(manifest)
        assert len(result.errors) == 1

    def test_remediation_notes_for_mismatch(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db)
        manifest = _make_manifest(
            files=[_make_fingerprint("config.json", sha256="tampered" * 8)]
        )
        result = checker.check_manifest(manifest)
        notes = result.remediation_notes
        assert len(notes) > 0
        assert any("mismatch" in n.lower() or "CRITICAL" in n for n in notes)


# ---------------------------------------------------------------------------
# FingerprintChecker — strict mode
# ---------------------------------------------------------------------------


class TestFingerprintCheckerStrictMode:
    def test_strict_false_unknown_is_warn(self, db: HashDatabase) -> None:
        checker = FingerprintChecker(db=db, strict=False)
        fp = _make_fingerprint("config.json", sha256="a" * 64)
        manifest = _make_manifest(files=[fp])
        result = checker.check_manifest(manifest)
        assert result.verdict == Verdict.WARN

    def test_strict_true_unknown_is_fail(self, db: HashDatabase) -> None:
        checker = FingerprintChecker(db=db, strict=True)
        fp = _make_fingerprint("config.json", sha256="a" * 64)
        manifest = _make_manifest(files=[fp])
        result = checker.check_manifest(manifest)
        assert result.verdict == Verdict.FAIL

    def test_strict_true_all_match_is_pass(self, populated_db: HashDatabase) -> None:
        checker = FingerprintChecker(db=populated_db, strict=True)
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("model.bin", sha256="b" * 64, file_type="weight"),
            ]
        )
        result = checker.check_manifest(manifest)
        assert result.verdict == Verdict.PASS


# ---------------------------------------------------------------------------
# check_manifest convenience function
# ---------------------------------------------------------------------------


class TestCheckManifestFunction:
    def test_basic_pass(self, populated_db: HashDatabase) -> None:
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="a" * 64),
                _make_fingerprint("model.bin", sha256="b" * 64, file_type="weight"),
            ]
        )
        result = check_manifest(manifest, db=populated_db)
        assert result.verdict == Verdict.PASS

    def test_fail_on_mismatch(self, populated_db: HashDatabase) -> None:
        manifest = _make_manifest(
            files=[
                _make_fingerprint("config.json", sha256="z" * 64),  # tampered
            ]
        )
        result = check_manifest(manifest, db=populated_db)
        assert result.verdict == Verdict.FAIL

    def test_warn_on_unknown(self, populated_db: HashDatabase) -> None:
        manifest = _make_manifest(
            files=[
                _make_fingerprint("unknown.txt", sha256="d" * 64, file_type="other"),
            ]
        )
        result = check_manifest(manifest, db=populated_db)
        assert result.verdict == Verdict.WARN

    def test_strict_mode(self, populated_db: HashDatabase) -> None:
        manifest = _make_manifest(
            files=[
                _make_fingerprint("unknown.txt", sha256="d" * 64, file_type="other"),
            ]
        )
        result = check_manifest(manifest, db=populated_db, strict=True)
        assert result.verdict == Verdict.FAIL

    def test_revision_override(self, populated_db: HashDatabase) -> None:
        manifest = _make_manifest(revision="main")
        manifest.files = [_make_fingerprint("config.json", sha256="a" * 64)]
        # Querying with a different revision — no records exist for "v99"
        result = check_manifest(manifest, db=populated_db, revision="v99")
        assert result.revision == "v99"
        assert result.verdict == Verdict.WARN  # unknown

    def test_returns_check_result_instance(self, populated_db: HashDatabase) -> None:
        manifest = _make_manifest(files=[])
        result = check_manifest(manifest, db=populated_db)
        assert isinstance(result, CheckResult)


# ---------------------------------------------------------------------------
# check_file_against_db convenience function
# ---------------------------------------------------------------------------


class TestCheckFileAgainstDb:
    def test_match(self, populated_db: HashDatabase) -> None:
        fp = _make_fingerprint("config.json", sha256="a" * 64)
        result = check_file_against_db(fp, model_id="test/model", db=populated_db)
        assert result.status == FileCheckStatus.MATCH

    def test_mismatch(self, populated_db: HashDatabase) -> None:
        fp = _make_fingerprint("config.json", sha256="z" * 64)
        result = check_file_against_db(fp, model_id="test/model", db=populated_db)
        assert result.status == FileCheckStatus.MISMATCH

    def test_unknown(self, populated_db: HashDatabase) -> None:
        fp = _make_fingerprint("new_file.bin", sha256="x" * 64, file_type="weight")
        result = check_file_against_db(fp, model_id="test/model", db=populated_db)
        assert result.status == FileCheckStatus.UNKNOWN

    def test_error_fingerprint(self, populated_db: HashDatabase) -> None:
        fp = _make_fingerprint(
            "bad.bin", sha256="", file_type="weight", error="read failed"
        )
        result = check_file_against_db(fp, model_id="test/model", db=populated_db)
        assert result.status == FileCheckStatus.ERROR

    def test_revision_default_main(self, populated_db: HashDatabase) -> None:
        fp = _make_fingerprint("config.json", sha256="a" * 64)
        result = check_file_against_db(
            fp, model_id="test/model", db=populated_db, revision="main"
        )
        assert result.status == FileCheckStatus.MATCH

    def test_revision_mismatch_gives_unknown(self, populated_db: HashDatabase) -> None:
        fp = _make_fingerprint("config.json", sha256="a" * 64)
        result = check_file_against_db(
            fp, model_id="test/model", db=populated_db, revision="v99"
        )
        assert result.status == FileCheckStatus.UNKNOWN

    def test_returns_file_check_result_instance(self, populated_db: HashDatabase) -> None:
        fp = _make_fingerprint("config.json", sha256="a" * 64)
        result = check_file_against_db(fp, model_id="test/model", db=populated_db)
        assert isinstance(result, FileCheckResult)


# ---------------------------------------------------------------------------
# Integration: DB seeding + manifest check
# ---------------------------------------------------------------------------


class TestIntegrationWithRealFiles:
    def test_manifest_from_directory_vs_db(self, tmp_path: Path) -> None:
        """End-to-end: build a manifest from a real directory and check against DB."""
        from model_provenance.fingerprint import build_manifest_from_directory

        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        config_content = b'{"model_type": "bert"}'
        weight_content = b"fake model weights"
        (model_dir / "config.json").write_bytes(config_content)
        (model_dir / "model.bin").write_bytes(weight_content)

        # Compute expected hashes.
        import hashlib

        config_hash = hashlib.sha256(config_content).hexdigest()
        weight_hash = hashlib.sha256(weight_content).hexdigest()

        # Seed the DB with the correct hashes.
        db = HashDatabase(":memory:")
        db.init_schema()
        db.add_hash("my_model", "config.json", config_hash, revision="local")
        db.add_hash("my_model", "model.bin", weight_hash, revision="local")

        # Build manifest and check.
        manifest = build_manifest_from_directory(model_dir)
        result = check_manifest(manifest, db=db)

        assert result.verdict == Verdict.PASS
        assert len(result.matches) == 2
        assert len(result.mismatches) == 0
        db.close()

    def test_tampered_file_detected(self, tmp_path: Path) -> None:
        """Verify that a modified file triggers a MISMATCH verdict."""
        from model_provenance.fingerprint import build_manifest_from_directory
        import hashlib

        model_dir = tmp_path / "tampered_model"
        model_dir.mkdir()
        original_content = b"original config"
        tampered_content = b"malicious config"

        # The file on disk has been tampered with.
        (model_dir / "config.json").write_bytes(tampered_content)

        # DB has the original (untampered) hash.
        original_hash = hashlib.sha256(original_content).hexdigest()
        db = HashDatabase(":memory:")
        db.init_schema()
        db.add_hash("tampered_model", "config.json", original_hash, revision="local")

        manifest = build_manifest_from_directory(model_dir)
        result = check_manifest(manifest, db=db)

        assert result.verdict == Verdict.FAIL
        assert len(result.mismatches) == 1
        assert result.mismatches[0].path == "config.json"
        db.close()

    def test_new_file_gives_warn(self, tmp_path: Path) -> None:
        """Files not in DB result in WARN (non-strict mode)."""
        from model_provenance.fingerprint import build_manifest_from_directory

        model_dir = tmp_path / "new_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"new config")

        db = HashDatabase(":memory:")
        db.init_schema()
        # No known-good hashes for this model.

        manifest = build_manifest_from_directory(model_dir)
        result = check_manifest(manifest, db=db)

        assert result.verdict == Verdict.WARN
        assert len(result.unknowns) == 1
        db.close()
