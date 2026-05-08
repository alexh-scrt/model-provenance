"""fetcher.py: Fetch model file listings and metadata from Hugging Face Hub or local dirs.

This module abstracts over two model sources:

1. **Hugging Face Hub** — uses the ``huggingface_hub`` library to list files
   in a remote model repository without downloading the full weights, and
   retrieves model card metadata (author, license, tags, pipeline tag, etc.).

2. **Local directory** — scans a local path and builds an equivalent file
   listing so that the rest of the pipeline can treat both sources uniformly.

All results are returned as :class:`ModelFileListing` instances, which contain
a list of :class:`RemoteFileInfo` records and a :class:`ModelCardInfo` record.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RemoteFileInfo:
    """Metadata for a single file in a model repository.

    Attributes:
        path: Relative path within the model repository (forward-slash
            separated).
        size_bytes: File size in bytes.  May be ``0`` if not available from
            the source.
        sha256: Pre-computed SHA-256 hex digest as reported by the remote
            source.  May be an empty string if not available (common for
            files on HF Hub where only the LFS SHA-256 is exposed).
        lfs_sha256: SHA-256 hex digest of the LFS pointer blob itself (not
            the file content).  Populated only for large files stored in
            Git LFS on Hugging Face Hub.
        blob_id: The Git blob object ID (SHA-1) for this file revision.
        is_lfs: ``True`` if the file is stored in Git LFS.
    """

    path: str
    size_bytes: int = 0
    sha256: str = ""
    lfs_sha256: str = ""
    blob_id: str = ""
    is_lfs: bool = False

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "lfs_sha256": self.lfs_sha256,
            "blob_id": self.blob_id,
            "is_lfs": self.is_lfs,
        }


@dataclass
class ModelCardInfo:
    """Metadata extracted from a model card.

    Attributes:
        model_id: Hugging Face model repository ID or local directory path.
        author: Repository author / organisation (if available).
        license: SPDX license identifier string (e.g. ``'apache-2.0'``), or
            ``None`` if not specified in the model card.
        pipeline_tag: Task pipeline tag (e.g. ``'text-classification'``),
            or ``None`` if unset.
        tags: List of string tags attached to the model.
        library_name: ML framework used (e.g. ``'transformers'``), or
            ``None`` if unset.
        language: List of BCP-47 language codes the model supports.
        datasets: List of dataset identifiers used for training.
        raw_metadata: The full raw metadata dictionary from the model card
            YAML front-matter.  Empty dict if not available.
    """

    model_id: str
    author: str | None = None
    license: str | None = None
    pipeline_tag: str | None = None
    tags: list[str] = field(default_factory=list)
    library_name: str | None = None
    language: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "model_id": self.model_id,
            "author": self.author,
            "license": self.license,
            "pipeline_tag": self.pipeline_tag,
            "tags": self.tags,
            "library_name": self.library_name,
            "language": self.language,
            "datasets": self.datasets,
        }


@dataclass
class ModelFileListing:
    """Complete file listing and metadata for a model.

    Attributes:
        model_id: Hugging Face model repository ID or local directory path.
        revision: Git revision (branch, tag, or commit SHA), or ``'local'``
            for locally scanned directories.
        source: ``'hub'`` for Hugging Face Hub models, ``'local'`` for
            locally scanned directories.
        files: List of :class:`RemoteFileInfo` entries for every file in the
            model repository / directory.
        card: Model card metadata.
        fetch_error: If non-``None``, describes an error that occurred while
            fetching the listing (the listing may be partial).
    """

    model_id: str
    revision: str = "main"
    source: str = "hub"
    files: list[RemoteFileInfo] = field(default_factory=list)
    card: ModelCardInfo = field(default_factory=lambda: ModelCardInfo(model_id=""))
    fetch_error: str | None = None

    @property
    def file_count(self) -> int:
        """Return the total number of files in the listing."""
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        """Return the sum of all reported file sizes."""
        return sum(f.size_bytes for f in self.files)

    def get_file(self, path: str) -> RemoteFileInfo | None:
        """Look up a :class:`RemoteFileInfo` by its relative path.

        Args:
            path: Relative file path (forward-slash separated).

        Returns:
            Matching :class:`RemoteFileInfo`, or ``None``.
        """
        normalised = path.replace("\\", "/")
        for f in self.files:
            if f.path == normalised:
                return f
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "source": self.source,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "files": [f.to_dict() for f in self.files],
            "card": self.card.to_dict(),
            "fetch_error": self.fetch_error,
        }


# ---------------------------------------------------------------------------
# Hugging Face Hub fetcher
# ---------------------------------------------------------------------------


def fetch_hub_listing(
    model_id: str,
    revision: str = "main",
    token: str | None = None,
) -> ModelFileListing:
    """Fetch the file listing and model card for a Hugging Face Hub model.

    Uses the ``huggingface_hub`` library to list all files in the repository
    at the given revision without downloading the model weights.  Model card
    metadata is fetched separately via the Hub API.

    Args:
        model_id: Hugging Face repository ID in the format
            ``'owner/model-name'`` or ``'model-name'``.
        revision: Git revision string (branch, tag, or full commit SHA).
            Defaults to ``'main'``.
        token: Optional Hugging Face API token for private repositories.
            If ``None``, the library falls back to the cached login token.

    Returns:
        A :class:`ModelFileListing` populated with file info and model card
        metadata.  If any step fails, the error is captured in
        :attr:`~ModelFileListing.fetch_error` and a partial result is
        returned rather than raising.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed (should not
            happen given the declared dependencies).
    """
    try:
        from huggingface_hub import HfApi, ModelCard  # type: ignore[import]
        from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for remote model fetching. "
            "Install it with: pip install huggingface-hub"
        ) from exc

    listing = ModelFileListing(
        model_id=model_id,
        revision=revision,
        source="hub",
        card=ModelCardInfo(model_id=model_id),
    )

    api = HfApi(token=token)

    # ---- File listing -------------------------------------------------------
    try:
        repo_files = api.list_repo_tree(
            repo_id=model_id,
            revision=revision,
            recursive=True,
            repo_type="model",
        )
        for item in repo_files:
            # RepoFile has: rfilename, size, blob_id, lfs (optional)
            # RepoFolder has: path — skip folders
            if not hasattr(item, "rfilename"):
                continue
            rfi = _hub_repo_file_to_remote_info(item)
            listing.files.append(rfi)
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        listing.fetch_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Could not list files for %s@%s: %s", model_id, revision, exc)
        return listing
    except Exception as exc:  # noqa: BLE001
        # Attempt fallback using list_repo_files (older API)
        try:
            listing.files = _fetch_files_fallback(api, model_id, revision)
        except Exception as inner_exc:  # noqa: BLE001
            listing.fetch_error = f"{type(exc).__name__}: {exc}; fallback also failed: {inner_exc}"
            logger.warning(
                "File listing failed for %s@%s: %s", model_id, revision, exc
            )
            return listing

    # ---- Model card metadata ------------------------------------------------
    try:
        card_info = _fetch_hub_model_card(api, model_id, revision, token)
        listing.card = card_info
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not fetch model card for %s: %s", model_id, exc
        )
        # Non-fatal — listing is still useful without card metadata.

    return listing


def _hub_repo_file_to_remote_info(item: Any) -> RemoteFileInfo:
    """Convert a ``huggingface_hub`` ``RepoFile`` object to a :class:`RemoteFileInfo`.

    Args:
        item: A ``RepoFile`` (or compatible) object from ``list_repo_tree``.

    Returns:
        A populated :class:`RemoteFileInfo`.
    """
    path: str = getattr(item, "rfilename", "").replace("\\", "/")
    size_bytes: int = getattr(item, "size", 0) or 0
    blob_id: str = getattr(item, "blob_id", "") or ""

    lfs_sha256: str = ""
    is_lfs: bool = False
    lfs_info = getattr(item, "lfs", None)
    if lfs_info is not None:
        is_lfs = True
        # lfs_info may be a dict or a BlobLfsInfo object
        if isinstance(lfs_info, dict):
            lfs_sha256 = str(lfs_info.get("sha256", "") or "")
            if not size_bytes:
                size_bytes = int(lfs_info.get("size", 0) or 0)
        else:
            lfs_sha256 = str(getattr(lfs_info, "sha256", "") or "")
            if not size_bytes:
                size_bytes = int(getattr(lfs_info, "size", 0) or 0)

    return RemoteFileInfo(
        path=path,
        size_bytes=size_bytes,
        sha256="",  # Content SHA-256 not available without downloading
        lfs_sha256=lfs_sha256,
        blob_id=blob_id,
        is_lfs=is_lfs,
    )


def _fetch_files_fallback(
    api: Any,
    model_id: str,
    revision: str,
) -> list[RemoteFileInfo]:
    """Fallback file listing using ``HfApi.list_repo_files``.

    This API returns only file paths (no size or LFS info), so the resulting
    :class:`RemoteFileInfo` objects will have ``size_bytes=0``.

    Args:
        api: An ``HfApi`` instance.
        model_id: Hugging Face model repository ID.
        revision: Git revision string.

    Returns:
        List of :class:`RemoteFileInfo` with paths populated.
    """
    files: list[RemoteFileInfo] = []
    for file_path in api.list_repo_files(
        repo_id=model_id, revision=revision, repo_type="model"
    ):
        files.append(RemoteFileInfo(path=str(file_path).replace("\\", "/")))
    return files


def _fetch_hub_model_card(
    api: Any,
    model_id: str,
    revision: str,
    token: str | None,
) -> ModelCardInfo:
    """Fetch and parse model card metadata from Hugging Face Hub.

    Args:
        api: An ``HfApi`` instance.
        model_id: Hugging Face model repository ID.
        revision: Git revision string.
        token: Optional HF API token.

    Returns:
        A :class:`ModelCardInfo` populated from the model card YAML front-matter.
    """
    info = ModelCardInfo(model_id=model_id)

    try:
        model_info = api.model_info(
            repo_id=model_id,
            revision=revision,
            token=token,
        )
        # Extract fields safely
        info.author = getattr(model_info, "author", None) or None
        info.pipeline_tag = getattr(model_info, "pipeline_tag", None) or None
        info.library_name = getattr(model_info, "library_name", None) or None

        # Tags
        raw_tags = getattr(model_info, "tags", None) or []
        info.tags = [str(t) for t in raw_tags]

        # License: try card_data first, then tags
        card_data = getattr(model_info, "card_data", None)
        if card_data is not None:
            lic = getattr(card_data, "license", None)
            if lic:
                info.license = str(lic)

            lang = getattr(card_data, "language", None) or []
            info.language = [str(l) for l in lang] if lang else []

            ds = getattr(card_data, "datasets", None) or []
            info.datasets = [str(d) for d in ds] if ds else []

            # Store raw card data
            try:
                info.raw_metadata = dict(card_data.__dict__) if hasattr(card_data, "__dict__") else {}
            except Exception:  # noqa: BLE001
                info.raw_metadata = {}
        else:
            # Fall back to scanning tags for a license entry
            for tag in info.tags:
                if tag.startswith("license:"):
                    info.license = tag[len("license:"):].strip()
                    break

    except Exception as exc:  # noqa: BLE001
        logger.debug("model_info() failed for %s: %s", model_id, exc)
        # Try fetching the raw ModelCard text as a last resort
        try:
            from huggingface_hub import ModelCard  # type: ignore[import]
            card = ModelCard.load(model_id, token=token)
            if card.data:
                lic = getattr(card.data, "license", None)
                if lic:
                    info.license = str(lic)
        except Exception:  # noqa: BLE001
            pass

    return info


# ---------------------------------------------------------------------------
# Local directory fetcher
# ---------------------------------------------------------------------------


_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "__pycache__", ".hf_cache", ".cache", "node_modules"}
)


def fetch_local_listing(
    directory: str | Path,
    model_id: str | None = None,
    revision: str = "local",
) -> ModelFileListing:
    """Build a :class:`ModelFileListing` by scanning a local directory.

    Recursively lists all regular files under *directory*, skipping hidden /
    cache directories.  Model card metadata is extracted from a
    ``README.md`` file if present.

    Args:
        directory: Path to the local model directory.
        model_id: Human-readable model identifier.  Defaults to the
            directory's base name.
        revision: Version label for the listing.  Defaults to ``'local'``.

    Returns:
        A :class:`ModelFileListing` with all discovered files and any
        available model card metadata.

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

    listing = ModelFileListing(
        model_id=effective_model_id,
        revision=revision,
        source="local",
        card=ModelCardInfo(model_id=effective_model_id),
    )

    for file_path in sorted(_iter_files_local(dir_path)):
        try:
            rel = file_path.resolve().relative_to(dir_path)
            rel_str = str(rel).replace("\\", "/")
        except ValueError:
            rel_str = str(file_path).replace("\\", "/")

        try:
            size = file_path.stat().st_size
        except OSError:
            size = 0

        rfi = RemoteFileInfo(
            path=rel_str,
            size_bytes=size,
        )
        listing.files.append(rfi)

    # Attempt to parse README.md as model card
    readme_path = dir_path / "README.md"
    if readme_path.exists():
        try:
            listing.card = _parse_local_model_card(readme_path, effective_model_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not parse local README.md for %s: %s", effective_model_id, exc)

    return listing


def _iter_files_local(directory: Path):
    """Recursively yield all regular files under *directory*, skipping caches.

    Args:
        directory: Root directory path.

    Yields:
        Absolute :class:`~pathlib.Path` objects for each regular file.
    """
    for entry in os.scandir(directory):
        if entry.name in _SKIP_DIRS:
            continue
        if entry.is_dir(follow_symlinks=False):
            yield from _iter_files_local(Path(entry.path))
        elif entry.is_file(follow_symlinks=False):
            yield Path(entry.path)


def _parse_local_model_card(readme_path: Path, model_id: str) -> ModelCardInfo:
    """Parse a local ``README.md`` YAML front-matter block into :class:`ModelCardInfo`.

    Only the YAML front-matter delimited by ``---`` blocks is parsed; the
    rest of the Markdown is ignored.

    Args:
        readme_path: Absolute path to the ``README.md`` file.
        model_id: Model identifier to embed in the result.

    Returns:
        A :class:`ModelCardInfo` populated from any available front-matter.
    """
    import yaml  # imported locally to keep module-level imports clean

    info = ModelCardInfo(model_id=model_id)
    text = readme_path.read_text(encoding="utf-8", errors="replace")

    front_matter = _extract_yaml_front_matter(text)
    if not front_matter:
        return info

    try:
        data: Any = yaml.safe_load(front_matter)
    except yaml.YAMLError as exc:
        logger.debug("YAML parse error in README.md front-matter: %s", exc)
        return info

    if not isinstance(data, dict):
        return info

    info.raw_metadata = data

    lic = data.get("license")
    if lic:
        info.license = str(lic)

    lang = data.get("language")
    if isinstance(lang, list):
        info.language = [str(l) for l in lang]
    elif isinstance(lang, str):
        info.language = [lang]

    tags = data.get("tags")
    if isinstance(tags, list):
        info.tags = [str(t) for t in tags]

    library_name = data.get("library_name")
    if library_name:
        info.library_name = str(library_name)

    pipeline_tag = data.get("pipeline_tag")
    if pipeline_tag:
        info.pipeline_tag = str(pipeline_tag)

    datasets = data.get("datasets")
    if isinstance(datasets, list):
        info.datasets = [str(d) for d in datasets]

    return info


def _extract_yaml_front_matter(text: str) -> str:
    """Extract the YAML front-matter block from a Markdown document.

    Looks for the opening ``---`` on the very first line and the closing
    ``---`` on a subsequent line.

    Args:
        text: Full text content of the Markdown file.

    Returns:
        The raw YAML string between the ``---`` delimiters, or an empty
        string if no valid front-matter block is found.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""

    end_index: int | None = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = i
            break

    if end_index is None:
        return ""

    return "\n".join(lines[1:end_index])


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def fetch_model_listing(
    model_source: str | Path,
    revision: str = "main",
    token: str | None = None,
    local: bool = False,
) -> ModelFileListing:
    """Unified entry point: fetch a model listing from Hub or local directory.

    If *local* is ``True`` or *model_source* is an existing directory path,
    delegates to :func:`fetch_local_listing`.  Otherwise, treats
    *model_source* as a Hugging Face Hub model ID and delegates to
    :func:`fetch_hub_listing`.

    Args:
        model_source: Either a Hugging Face Hub model ID (e.g.
            ``'bert-base-uncased'``) or a local filesystem path.
        revision: Git revision string for Hub models.  Ignored for local
            listings.
        token: Optional Hugging Face API token.
        local: If ``True``, treat *model_source* as a local path regardless
            of whether it looks like a Hub model ID.

    Returns:
        A :class:`ModelFileListing`.

    Raises:
        NotADirectoryError: If *local* is ``True`` but *model_source* is not
            a valid directory.
    """
    source_path = Path(str(model_source))

    if local or source_path.exists():
        return fetch_local_listing(
            directory=source_path,
            model_id=str(model_source),
            revision=revision,
        )

    return fetch_hub_listing(
        model_id=str(model_source),
        revision=revision,
        token=token,
    )
