"""checker.py: Fingerprint comparison against the known-good hash database.

This module compares a computed :class:`~model_provenance.fingerprint.FingerprintManifest`
against the local SQLite fingerprint database (and optionally the bundled YAML
seed data) to detect tampered, unknown, or mismatched model files.

Each file in the manifest is classified with a :class:`CheckStatus` that
indicates whether it matches a known-good hash, is unknown (not in the DB),
or has been tampered with (hash mismatch). The results are collected into a
:class:`CheckResult` which provides an overall verdict and per-file details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from model_provenance.db import HashDatabase, KnownHash
from model_provenance.fingerprint import FileFingerprint, FingerprintManifest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FileCheckStatus(str, Enum):
    """Status of a single file after comparison against the known-good database.

    Values:
        MATCH: The file's SHA-256 hash matches the known-good record exactly.
        MISMATCH: A known-good record exists but the hashes differ — possible
            tampering.
        UNKNOWN: No known-good record exists for this file; it cannot be
            verified.
        ERROR: The fingerprint for this file could not be computed (e.g. due
            to a read error), so no comparison was possible.
        NEW: The file is not in the database but was added during this run
            (used when auto-learning mode is active).
    """

    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"
    ERROR = "error"
    NEW = "new"


class Verdict(str, Enum):
    """Overall provenance verdict for a model audit.

    Values:
        PASS: All checked files match their known-good hashes and no errors
            occurred.
        WARN: One or more files are unknown (not in the database) but no
            mismatches or errors were found.
        FAIL: One or more files have mismatched hashes or fingerprint errors
            were encountered.
    """

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


# ---------------------------------------------------------------------------
# Per-file result
# ---------------------------------------------------------------------------


@dataclass
class FileCheckResult:
    """The result of comparing a single file fingerprint against the database.

    Attributes:
        fingerprint: The computed :class:`~model_provenance.fingerprint.FileFingerprint`
            for this file.
        status: The comparison outcome as a :class:`FileCheckStatus`.
        known_hash: The :class:`~model_provenance.db.KnownHash` record that
            was found in the database, or ``None`` if no record exists.
        detail: A human-readable explanation of the status (e.g. the
            expected vs. actual hash for a mismatch).
    """

    fingerprint: FileFingerprint
    status: FileCheckStatus
    known_hash: KnownHash | None = None
    detail: str = ""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        """Relative file path (delegated from the fingerprint)."""
        return self.fingerprint.path

    @property
    def is_ok(self) -> bool:
        """Return ``True`` if the file is ``MATCH`` or ``NEW``."""
        return self.status in (FileCheckStatus.MATCH, FileCheckStatus.NEW)

    @property
    def is_fail(self) -> bool:
        """Return ``True`` if the file is ``MISMATCH`` or ``ERROR``."""
        return self.status in (FileCheckStatus.MISMATCH, FileCheckStatus.ERROR)

    @property
    def is_warn(self) -> bool:
        """Return ``True`` if the file status is ``UNKNOWN``."""
        return self.status == FileCheckStatus.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary suitable for JSON / YAML output."""
        return {
            "path": self.path,
            "sha256": self.fingerprint.sha256,
            "file_type": self.fingerprint.file_type,
            "size_bytes": self.fingerprint.size_bytes,
            "status": self.status.value,
            "known_sha256": self.known_hash.sha256 if self.known_hash else None,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Overall check result
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Aggregated result of comparing a full manifest against the database.

    Attributes:
        model_id: Identifier of the model that was checked.
        revision: Git revision that was checked.
        verdict: Overall :class:`Verdict` — ``PASS``, ``WARN``, or ``FAIL``.
        file_results: Ordered list of :class:`FileCheckResult` entries,
            one per file in the manifest.
        summary: Human-readable summary string.
        db_coverage: Fraction (0.0–1.0) of files for which a known-good
            record was found in the database.  ``None`` if there are no
            files in the manifest.
    """

    model_id: str
    revision: str
    verdict: Verdict
    file_results: list[FileCheckResult] = field(default_factory=list)
    summary: str = ""
    db_coverage: float | None = None

    # ------------------------------------------------------------------
    # Filtered views
    # ------------------------------------------------------------------

    @property
    def matches(self) -> list[FileCheckResult]:
        """Return only the files whose status is ``MATCH``."""
        return [r for r in self.file_results if r.status == FileCheckStatus.MATCH]

    @property
    def mismatches(self) -> list[FileCheckResult]:
        """Return only the files whose status is ``MISMATCH``."""
        return [r for r in self.file_results if r.status == FileCheckStatus.MISMATCH]

    @property
    def unknowns(self) -> list[FileCheckResult]:
        """Return only the files whose status is ``UNKNOWN``."""
        return [r for r in self.file_results if r.status == FileCheckStatus.UNKNOWN]

    @property
    def errors(self) -> list[FileCheckResult]:
        """Return only the files whose status is ``ERROR``."""
        return [r for r in self.file_results if r.status == FileCheckStatus.ERROR]

    @property
    def new_files(self) -> list[FileCheckResult]:
        """Return only the files whose status is ``NEW``."""
        return [r for r in self.file_results if r.status == FileCheckStatus.NEW]

    @property
    def file_count(self) -> int:
        """Total number of files checked."""
        return len(self.file_results)

    @property
    def remediation_notes(self) -> list[str]:
        """Generate actionable remediation notes based on the results.

        Returns:
            A list of human-readable strings, one per issue category.
        """
        notes: list[str] = []
        if self.mismatches:
            paths = ", ".join(r.path for r in self.mismatches[:5])
            suffix = "..." if len(self.mismatches) > 5 else ""
            notes.append(
                f"CRITICAL: {len(self.mismatches)} file(s) have hash mismatches "
                f"(possible tampering): {paths}{suffix}. "
                "Do NOT use this model in production without investigation."
            )
        if self.errors:
            paths = ", ".join(r.path for r in self.errors[:5])
            notes.append(
                f"WARNING: {len(self.errors)} file(s) could not be hashed "
                f"(read errors): {paths}. "
                "Check file permissions and integrity."
            )
        if self.unknowns:
            notes.append(
                f"INFO: {len(self.unknowns)} file(s) are not in the known-good "
                "database and cannot be verified. "
                "Consider adding their hashes after manual verification."
            )
        return notes

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary suitable for JSON / YAML output."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "db_coverage": self.db_coverage,
            "file_count": self.file_count,
            "match_count": len(self.matches),
            "mismatch_count": len(self.mismatches),
            "unknown_count": len(self.unknowns),
            "error_count": len(self.errors),
            "new_count": len(self.new_files),
            "file_results": [r.to_dict() for r in self.file_results],
            "remediation": self.remediation_notes,
        }


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------


class FingerprintChecker:
    """Compares a :class:`~model_provenance.fingerprint.FingerprintManifest`
    against a :class:`~model_provenance.db.HashDatabase`.

    Args:
        db: The :class:`~model_provenance.db.HashDatabase` to query for
            known-good hashes.  The caller is responsible for opening and
            closing the database.
        strict: If ``True``, files with ``UNKNOWN`` status (not in DB) will
            contribute to a ``FAIL`` verdict instead of ``WARN``.  Useful for
            environments that require all files to be pre-verified.
    """

    def __init__(
        self,
        db: HashDatabase,
        strict: bool = False,
    ) -> None:
        """Initialise the FingerprintChecker.

        Args:
            db: An initialised :class:`~model_provenance.db.HashDatabase`.
            strict: If ``True``, unknown files cause a ``FAIL`` verdict.
        """
        self._db = db
        self._strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_manifest(
        self,
        manifest: FingerprintManifest,
        revision: str | None = None,
    ) -> CheckResult:
        """Check all files in *manifest* against the known-good database.

        For each :class:`~model_provenance.fingerprint.FileFingerprint` in the
        manifest the method:

        1. Short-circuits with ``ERROR`` if the fingerprint has an error.
        2. Queries the database for a known-good hash keyed on
           ``(model_id, revision, file_path)``.
        3. If found, compares hashes and emits ``MATCH`` or ``MISMATCH``.
        4. If not found, emits ``UNKNOWN``.

        The overall verdict is:
        - ``FAIL`` if any file is ``MISMATCH`` or ``ERROR``, or if *strict*
          mode is on and any file is ``UNKNOWN``.
        - ``WARN`` if any file is ``UNKNOWN`` (and not in strict mode).
        - ``PASS`` otherwise.

        Args:
            manifest: The :class:`~model_provenance.fingerprint.FingerprintManifest`
                to verify.
            revision: Override the revision used for database lookups.  If
                ``None``, uses ``manifest.revision``.

        Returns:
            A fully populated :class:`CheckResult`.
        """
        effective_revision = revision if revision is not None else manifest.revision
        file_results: list[FileCheckResult] = []

        for fp in manifest.files:
            result = self._check_single_file(
                fingerprint=fp,
                model_id=manifest.model_id,
                revision=effective_revision,
            )
            file_results.append(result)
            logger.debug(
                "[%s] %s → %s", manifest.model_id, fp.path, result.status.value
            )

        verdict = _compute_verdict(file_results, strict=self._strict)
        db_coverage = _compute_coverage(file_results)
        summary = _build_summary(manifest.model_id, effective_revision, file_results, verdict)

        return CheckResult(
            model_id=manifest.model_id,
            revision=effective_revision,
            verdict=verdict,
            file_results=file_results,
            summary=summary,
            db_coverage=db_coverage,
        )

    def check_single_file(
        self,
        fingerprint: FileFingerprint,
        model_id: str,
        revision: str = "main",
    ) -> FileCheckResult:
        """Check a single :class:`~model_provenance.fingerprint.FileFingerprint`
        against the database.

        This is the public single-file variant of the internal helper.  Useful
        when you want to check one file without constructing a full manifest.

        Args:
            fingerprint: The file fingerprint to verify.
            model_id: The model identifier to use for the database lookup.
            revision: The revision to use for the database lookup.

        Returns:
            A :class:`FileCheckResult` for the single file.
        """
        return self._check_single_file(
            fingerprint=fingerprint,
            model_id=model_id,
            revision=revision,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_single_file(
        self,
        fingerprint: FileFingerprint,
        model_id: str,
        revision: str,
    ) -> FileCheckResult:
        """Internal implementation of per-file checking.

        Args:
            fingerprint: The file fingerprint to verify.
            model_id: Model identifier for the DB lookup.
            revision: Revision string for the DB lookup.

        Returns:
            A :class:`FileCheckResult`.
        """
        # If the fingerprint itself has an error, we cannot compare.
        if not fingerprint.ok:
            return FileCheckResult(
                fingerprint=fingerprint,
                status=FileCheckStatus.ERROR,
                known_hash=None,
                detail=f"Fingerprint error: {fingerprint.error}",
            )

        known: KnownHash | None = None
        try:
            known = self._db.get_hash(
                model_id=model_id,
                file_path=fingerprint.path,
                revision=revision,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DB lookup failed for %s/%s: %s", model_id, fingerprint.path, exc
            )
            return FileCheckResult(
                fingerprint=fingerprint,
                status=FileCheckStatus.ERROR,
                known_hash=None,
                detail=f"Database lookup error: {exc}",
            )

        if known is None:
            return FileCheckResult(
                fingerprint=fingerprint,
                status=FileCheckStatus.UNKNOWN,
                known_hash=None,
                detail="No known-good hash found in database.",
            )

        if fingerprint.sha256 == known.sha256:
            return FileCheckResult(
                fingerprint=fingerprint,
                status=FileCheckStatus.MATCH,
                known_hash=known,
                detail=f"Hash matches known-good record (source: {known.source}).",
            )

        # Hashes differ — tampering detected.
        return FileCheckResult(
            fingerprint=fingerprint,
            status=FileCheckStatus.MISMATCH,
            known_hash=known,
            detail=(
                f"TAMPER DETECTED: computed={fingerprint.sha256[:16]}…, "
                f"expected={known.sha256[:16]}… (source: {known.source})"
            ),
        )


# ---------------------------------------------------------------------------
# Functional convenience wrappers
# ---------------------------------------------------------------------------


def check_manifest(
    manifest: FingerprintManifest,
    db: HashDatabase,
    revision: str | None = None,
    strict: bool = False,
) -> CheckResult:
    """Convenience function: check a manifest against a database.

    Constructs a :class:`FingerprintChecker` internally and calls
    :meth:`~FingerprintChecker.check_manifest`.

    Args:
        manifest: The :class:`~model_provenance.fingerprint.FingerprintManifest`
            to verify.
        db: An initialised :class:`~model_provenance.db.HashDatabase`.
        revision: Override the revision used for database lookups.
        strict: If ``True``, unknown files cause a ``FAIL`` verdict.

    Returns:
        A fully populated :class:`CheckResult`.
    """
    checker = FingerprintChecker(db=db, strict=strict)
    return checker.check_manifest(manifest, revision=revision)


def check_file_against_db(
    fingerprint: FileFingerprint,
    model_id: str,
    db: HashDatabase,
    revision: str = "main",
) -> FileCheckResult:
    """Convenience function: check a single file fingerprint against the DB.

    Args:
        fingerprint: The :class:`~model_provenance.fingerprint.FileFingerprint`
            to check.
        model_id: Model identifier for the DB lookup.
        db: An initialised :class:`~model_provenance.db.HashDatabase`.
        revision: Revision string for the DB lookup.

    Returns:
        A :class:`FileCheckResult`.
    """
    checker = FingerprintChecker(db=db)
    return checker.check_single_file(
        fingerprint=fingerprint,
        model_id=model_id,
        revision=revision,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_verdict(
    file_results: list[FileCheckResult],
    strict: bool = False,
) -> Verdict:
    """Determine the overall verdict from a list of per-file results.

    Args:
        file_results: Per-file check results.
        strict: If ``True``, ``UNKNOWN`` files also trigger ``FAIL``.

    Returns:
        :class:`Verdict` — ``PASS``, ``WARN``, or ``FAIL``.
    """
    if not file_results:
        # An empty manifest is treated as WARN (nothing to verify).
        return Verdict.WARN

    has_mismatch = any(r.status == FileCheckStatus.MISMATCH for r in file_results)
    has_error = any(r.status == FileCheckStatus.ERROR for r in file_results)
    has_unknown = any(r.status == FileCheckStatus.UNKNOWN for r in file_results)

    if has_mismatch or has_error:
        return Verdict.FAIL

    if has_unknown:
        return Verdict.FAIL if strict else Verdict.WARN

    return Verdict.PASS


def _compute_coverage(file_results: list[FileCheckResult]) -> float | None:
    """Compute the fraction of files that have a known-good DB record.

    Args:
        file_results: Per-file check results.

    Returns:
        A float in ``[0.0, 1.0]``, or ``None`` if *file_results* is empty.
    """
    if not file_results:
        return None
    known_count = sum(
        1
        for r in file_results
        if r.status in (FileCheckStatus.MATCH, FileCheckStatus.MISMATCH, FileCheckStatus.NEW)
    )
    return known_count / len(file_results)


def _build_summary(
    model_id: str,
    revision: str,
    file_results: list[FileCheckResult],
    verdict: Verdict,
) -> str:
    """Build a human-readable one-line summary of the check outcome.

    Args:
        model_id: Model identifier.
        revision: Revision string.
        file_results: Per-file check results.
        verdict: Overall verdict.

    Returns:
        A single-line summary string.
    """
    total = len(file_results)
    matches = sum(1 for r in file_results if r.status == FileCheckStatus.MATCH)
    mismatches = sum(1 for r in file_results if r.status == FileCheckStatus.MISMATCH)
    unknowns = sum(1 for r in file_results if r.status == FileCheckStatus.UNKNOWN)
    errors = sum(1 for r in file_results if r.status == FileCheckStatus.ERROR)

    verdict_label = verdict.value.upper()
    return (
        f"{verdict_label} — {model_id}@{revision}: "
        f"{total} file(s) checked: "
        f"{matches} match, {mismatches} mismatch, "
        f"{unknowns} unknown, {errors} error"
    )
