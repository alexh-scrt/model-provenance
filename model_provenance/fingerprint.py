"""fingerprint.py: SHA-256 fingerprint computation and manifest construction.

This module provides the core cryptographic fingerprinting functionality for
model_provenance. It computes SHA-256 digests for individual files and builds
structured fingerprint manifests that capture all relevant metadata for a
collection of model files.

The manifest data structures defined here are used by all other components
(checker, scanner, reporter) as the common data exchange format.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Block size for streaming file reads (1 MiB)
_READ_BLOCK_SIZE: int = 1024 * 1024

#: File extensions that are considered model weight / parameter files.
WEIGHT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".gguf",
        ".ggml",
        ".pkl",
        ".pickle",
        ".npz",
        ".npy",
        ".h5",
        ".hdf5",
        ".msgpack",
        ".flax",
    }
)

#: File extensions that are considered configuration / metadata files.
CONFIG_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".json",
        ".yaml",
        ".yml",
        ".txt",
        ".md",
        ".toml",
        ".cfg",
        ".ini",
        ".model",
        ".vocab",
        ".merges",
        ".spm",
        ".sentencepiece",
    }
)

#: Directories that are commonly found in model repos but can be skipped.
_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".hf_cache", ".cache", "node_modules"}
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FileFingerprint:
    """Cryptographic fingerprint and metadata for a single file.

    Attributes:
        path: Relative path of the file within the model directory or repo.
        sha256: Lowercase hex-encoded SHA-256 digest of the file contents.
        size_bytes: Size of the file in bytes.
        file_type: Broad category of the file — ``'weight'``, ``'config'``,
            or ``'other'``.
        error: If non-``None``, contains an error message explaining why the
            hash could not be computed for this file.
    """

    path: str
    sha256: str
    size_bytes: int
    file_type: str  # 'weight' | 'config' | 'other'
    error: str | None = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def short_hash(self) -> str:
        """Return the first 16 hex characters of the SHA-256 digest."""
        return self.sha256[:16] if self.sha256 else ""

    @property
    def is_weight(self) -> bool:
        """Return ``True`` if this file is a model weight / parameter file."""
        return self.file_type == "weight"

    @property
    def is_config(self) -> bool:
        """Return ``True`` if this file is a configuration or metadata file."""
        return self.file_type == "config"

    @property
    def ok(self) -> bool:
        """Return ``True`` if the fingerprint was computed without errors."""
        return self.error is None

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary suitable for JSON / YAML output."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "file_type": self.file_type,
            "error": self.error,
        }


@dataclass
class FingerprintManifest:
    """A complete fingerprint manifest for a model.

    Captures the SHA-256 hashes of all files in a model repository or local
    directory together with associated metadata.

    Attributes:
        model_id: Hugging Face model ID (e.g. ``'bert-base-uncased'``) or an
            absolute / relative path for local models.
        revision: Git revision string (branch, tag, or commit SHA). Defaults
            to ``'local'`` for locally computed manifests.
        source: Origin of the manifest — ``'local'`` or ``'hub'``.
        computed_at: ISO-8601 UTC timestamp of when the manifest was computed.
        files: Ordered list of :class:`FileFingerprint` entries.
        aggregate_sha256: SHA-256 hash of the sorted concatenation of all
            individual file hashes — a single digest representing the whole
            model. ``None`` if any file produced an error.
    """

    model_id: str
    revision: str = "local"
    source: str = "local"  # 'local' | 'hub'
    computed_at: str = field(default_factory=lambda: _utc_now_iso())
    files: list[FileFingerprint] = field(default_factory=list)
    aggregate_sha256: str | None = None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get(self, relative_path: str) -> FileFingerprint | None:
        """Look up a :class:`FileFingerprint` by its relative path.

        Args:
            relative_path: The relative file path to look up (using forward
                slashes regardless of OS).

        Returns:
            The matching :class:`FileFingerprint`, or ``None`` if not found.
        """
        normalised = relative_path.replace("\\", "/")
        for fp in self.files:
            if fp.path == normalised:
                return fp
        return None

    @property
    def weight_files(self) -> list[FileFingerprint]:
        """Return only the weight-type file fingerprints."""
        return [f for f in self.files if f.is_weight]

    @property
    def config_files(self) -> list[FileFingerprint]:
        """Return only the config-type file fingerprints."""
        return [f for f in self.files if f.is_config]

    @property
    def errored_files(self) -> list[FileFingerprint]:
        """Return file fingerprints where an error occurred during hashing."""
        return [f for f in self.files if not f.ok]

    @property
    def total_size_bytes(self) -> int:
        """Return the sum of all file sizes in bytes."""
        return sum(f.size_bytes for f in self.files)

    @property
    def file_count(self) -> int:
        """Return the total number of files in the manifest."""
        return len(self.files)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary suitable for JSON / YAML output."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "source": self.source,
            "computed_at": self.computed_at,
            "aggregate_sha256": self.aggregate_sha256,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "files": [f.to_dict() for f in self.files],
        }


# ---------------------------------------------------------------------------
# Low-level hashing helpers
# ---------------------------------------------------------------------------


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Reads the file in 1 MiB streaming chunks so that arbitrarily large weight
    files can be processed without loading them entirely into RAM.

    Args:
        path: Absolute or relative path to the file to hash.

    Returns:
        Lowercase hex-encoded 64-character SHA-256 digest string.

    Raises:
        FileNotFoundError: If *path* does not exist.
        PermissionError: If the process lacks read permission for *path*.
        OSError: For other I/O errors encountered during reading.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_READ_BLOCK_SIZE)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 digest of an in-memory byte string.

    Useful for fingerprinting small configuration blobs fetched from the
    Hugging Face Hub API without writing them to disk first.

    Args:
        data: Raw byte content to hash.

    Returns:
        Lowercase hex-encoded 64-character SHA-256 digest string.
    """
    return hashlib.sha256(data).hexdigest()


def classify_file(path: str | Path) -> str:
    """Classify a file as ``'weight'``, ``'config'``, or ``'other'``.

    The classification is based solely on the file extension.

    Args:
        path: File path (only the suffix is examined).

    Returns:
        ``'weight'`` if the extension matches :data:`WEIGHT_EXTENSIONS`,
        ``'config'`` if it matches :data:`CONFIG_EXTENSIONS`,
        ``'other'`` otherwise.
    """
    suffix = Path(path).suffix.lower()
    if suffix in WEIGHT_EXTENSIONS:
        return "weight"
    if suffix in CONFIG_EXTENSIONS:
        return "config"
    return "other"


def _aggregate_hash(fingerprints: list[FileFingerprint]) -> str:
    """Compute a single aggregate SHA-256 from a sorted list of file hashes.

    The aggregate is computed by hashing the lexicographically sorted
    concatenation of ``<path>:<sha256>\n`` strings. This makes the aggregate
    deterministic regardless of the order files appear on disk.

    Args:
        fingerprints: List of :class:`FileFingerprint` objects. All must have
            a non-empty :attr:`~FileFingerprint.sha256` value.

    Returns:
        Lowercase hex-encoded SHA-256 digest.
    """
    hasher = hashlib.sha256()
    # Sort by path so the aggregate is order-independent.
    for fp in sorted(fingerprints, key=lambda f: f.path):
        hasher.update(f"{fp.path}:{fp.sha256}\n".encode())
    return hasher.hexdigest()


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with timezone suffix."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Fingerprint manifest builders
# ---------------------------------------------------------------------------


def fingerprint_file(
    file_path: Path,
    base_dir: Path | None = None,
) -> FileFingerprint:
    """Compute the SHA-256 fingerprint for a single local file.

    Args:
        file_path: Path to the file to fingerprint.
        base_dir: If provided, the ``path`` field in the returned
            :class:`FileFingerprint` will be relative to *base_dir* using
            forward slashes. If ``None``, the ``path`` field is set to the
            string representation of *file_path* as given.

    Returns:
        A populated :class:`FileFingerprint`. If the file cannot be read, the
        ``sha256`` field is set to an empty string and ``error`` contains the
        error description.
    """
    if base_dir is not None:
        try:
            relative = file_path.resolve().relative_to(base_dir.resolve())
            path_str = str(relative).replace("\\", "/")
        except ValueError:
            # file_path is not under base_dir — fall back to the given path.
            path_str = str(file_path).replace("\\", "/")
    else:
        path_str = str(file_path).replace("\\", "/")

    file_type = classify_file(file_path)

    try:
        size_bytes = file_path.stat().st_size
        sha256 = compute_sha256(file_path)
        return FileFingerprint(
            path=path_str,
            sha256=sha256,
            size_bytes=size_bytes,
            file_type=file_type,
        )
    except FileNotFoundError as exc:
        return FileFingerprint(
            path=path_str,
            sha256="",
            size_bytes=0,
            file_type=file_type,
            error=f"FileNotFoundError: {exc}",
        )
    except PermissionError as exc:
        return FileFingerprint(
            path=path_str,
            sha256="",
            size_bytes=0,
            file_type=file_type,
            error=f"PermissionError: {exc}",
        )
    except OSError as exc:
        return FileFingerprint(
            path=path_str,
            sha256="",
            size_bytes=0,
            file_type=file_type,
            error=f"OSError: {exc}",
        )


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


def build_manifest_from_directory(
    directory: str | Path,
    model_id: str | None = None,
    revision: str = "local",
) -> FingerprintManifest:
    """Build a :class:`FingerprintManifest` by fingerprinting a local directory.

    Recursively walks *directory*, skipping hidden directories and caches,
    and computes a SHA-256 digest for every regular file found.

    Args:
        directory: Path to the local model directory to fingerprint.
        model_id: Human-readable identifier for the model. Defaults to the
            directory's name if not provided.
        revision: Git revision or version label. Defaults to ``'local'``.

    Returns:
        A fully populated :class:`FingerprintManifest` with an aggregate hash
        computed from all successfully hashed files. If any files produced
        errors, :attr:`~FingerprintManifest.aggregate_sha256` is set to
        ``None``.

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

    manifest = FingerprintManifest(
        model_id=effective_model_id,
        revision=revision,
        source="local",
    )

    fingerprints: list[FileFingerprint] = []
    for file_path in sorted(_iter_files(dir_path)):
        fp = fingerprint_file(file_path, base_dir=dir_path)
        fingerprints.append(fp)

    manifest.files = fingerprints

    # Compute aggregate only when all files succeeded.
    successful = [f for f in fingerprints if f.ok and f.sha256]
    if successful and not manifest.errored_files:
        manifest.aggregate_sha256 = _aggregate_hash(successful)
    else:
        manifest.aggregate_sha256 = None

    return manifest


def build_manifest_from_file_map(
    file_map: dict[str, str],
    model_id: str,
    revision: str = "main",
    source: str = "hub",
) -> FingerprintManifest:
    """Build a :class:`FingerprintManifest` from a pre-computed path→hash map.

    This factory is used when file hashes have already been obtained from an
    external source (e.g. the Hugging Face Hub API) rather than computed
    locally.

    Args:
        file_map: Dictionary mapping relative file paths (using forward slashes)
            to their SHA-256 hex digests.
        model_id: Hugging Face model ID or any human-readable label.
        revision: Git revision string.
        source: Data origin label — typically ``'hub'``.

    Returns:
        A :class:`FingerprintManifest` with one :class:`FileFingerprint` per
        entry in *file_map* and an aggregate hash computed from all entries.

    Raises:
        ValueError: If *file_map* is empty.
    """
    if not file_map:
        raise ValueError("file_map must contain at least one entry.")

    fingerprints: list[FileFingerprint] = []
    for path_str, sha256 in sorted(file_map.items()):
        normalised = path_str.replace("\\", "/")
        file_type = classify_file(normalised)
        fingerprints.append(
            FileFingerprint(
                path=normalised,
                sha256=sha256.lower().strip(),
                # Size is not known from a hash map alone.
                size_bytes=0,
                file_type=file_type,
            )
        )

    aggregate = _aggregate_hash([f for f in fingerprints if f.sha256])

    return FingerprintManifest(
        model_id=model_id,
        revision=revision,
        source=source,
        files=fingerprints,
        aggregate_sha256=aggregate,
    )


def fingerprint_bytes_entry(
    path: str,
    data: bytes,
) -> FileFingerprint:
    """Create a :class:`FileFingerprint` for an in-memory byte blob.

    Useful when file content has been downloaded into memory (e.g. a model
    config fetched from the HF Hub API) and you need a fingerprint without
    writing to disk.

    Args:
        path: Relative path label for the entry (used as the manifest key).
        data: Raw file content.

    Returns:
        A :class:`FileFingerprint` with :attr:`~FileFingerprint.size_bytes`
        set to ``len(data)``.
    """
    normalised = path.replace("\\", "/")
    sha256 = compute_sha256_bytes(data)
    file_type = classify_file(normalised)
    return FileFingerprint(
        path=normalised,
        sha256=sha256,
        size_bytes=len(data),
        file_type=file_type,
    )
