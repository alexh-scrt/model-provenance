"""db.py: SQLite-backed fingerprint database for storing and querying known-good hashes.

This module manages a local SQLite database that persists known-good SHA-256
fingerprints for AI model files. It supports seeding from the bundled YAML
database, querying individual file hashes, and adding new entries.

The database is stored at ``~/.model-provenance/hashes.db`` by default, but
the location can be overridden for testing or custom deployments.
"""

from __future__ import annotations

import importlib.resources
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default directory for all model-provenance runtime data.
_DEFAULT_DATA_DIR: Path = Path.home() / ".model-provenance"

#: Default SQLite database file name.
_DEFAULT_DB_NAME: str = "hashes.db"

#: DDL statement for the known_hashes table.
_CREATE_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS known_hashes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT    NOT NULL,
    revision    TEXT    NOT NULL DEFAULT 'main',
    file_path   TEXT    NOT NULL,
    sha256      TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'unknown',
    notes       TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'utc')),
    UNIQUE (model_id, revision, file_path)
)
"""

#: Index on model_id for fast lookups.
_CREATE_INDEX_SQL: str = """
CREATE INDEX IF NOT EXISTS idx_known_hashes_model_id
    ON known_hashes (model_id, revision)
"""


# ---------------------------------------------------------------------------
# KnownHash record
# ---------------------------------------------------------------------------


class KnownHash:
    """A single known-good hash record from the database.

    Attributes:
        model_id: Hugging Face model repository ID (e.g. ``'bert-base-uncased'``).
        revision: Git revision (branch, tag, or full commit SHA).
        file_path: Relative path of the file within the model repository.
        sha256: Lowercase hex-encoded SHA-256 digest.
        source: Where the hash was obtained (``'official'``, ``'community'``,
            ``'computed'``, or ``'unknown'``).
        notes: Optional human-readable notes.
        created_at: ISO-8601 UTC timestamp of when the record was inserted.
    """

    __slots__ = ("model_id", "revision", "file_path", "sha256", "source", "notes", "created_at")

    def __init__(
        self,
        model_id: str,
        revision: str,
        file_path: str,
        sha256: str,
        source: str = "unknown",
        notes: str | None = None,
        created_at: str | None = None,
    ) -> None:
        """Initialise a KnownHash record.

        Args:
            model_id: Hugging Face model repository ID.
            revision: Git revision string.
            file_path: Relative file path within the repository.
            sha256: Lowercase hex SHA-256 digest.
            source: Provenance source label.
            notes: Optional human-readable notes.
            created_at: Optional ISO-8601 creation timestamp.
        """
        self.model_id = model_id
        self.revision = revision
        self.file_path = file_path
        self.sha256 = sha256.lower().strip()
        self.source = source
        self.notes = notes
        self.created_at = created_at or ""

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dictionary."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "file_path": self.file_path,
            "sha256": self.sha256,
            "source": self.source,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KnownHash(model_id={self.model_id!r}, revision={self.revision!r}, "
            f"file_path={self.file_path!r}, sha256={self.sha256[:16]!r}...)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KnownHash):
            return NotImplemented
        return (
            self.model_id == other.model_id
            and self.revision == other.revision
            and self.file_path == other.file_path
            and self.sha256 == other.sha256
        )


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------


class HashDatabase:
    """Manages the local SQLite fingerprint database.

    Provides methods to initialise the schema, seed from the bundled YAML
    file, store new hashes, and query existing ones.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``~/.model-provenance/hashes.db``.  Pass ``':memory:'`` for an
            in-memory database (useful for testing).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialise the HashDatabase.

        Args:
            db_path: Path to the SQLite database file, or ``':memory:'`` for
                an in-memory database.  If ``None``, defaults to the standard
                user data directory.
        """
        if db_path is None:
            self._db_path: str = str(_DEFAULT_DATA_DIR / _DEFAULT_DB_NAME)
        elif str(db_path) == ":memory:":
            self._db_path = ":memory:"
        else:
            self._db_path = str(db_path)

        self._ensure_data_dir()
        self._connection: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_data_dir(self) -> None:
        """Create the parent directory for the database file if needed."""
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Return a cached sqlite3 connection, opening one if necessary."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None,  # autocommit
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
        return self._connection

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that wraps operations in an explicit transaction."""
        conn = self._get_connection()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _row_to_known_hash(row: sqlite3.Row) -> KnownHash:
        """Convert a :class:`sqlite3.Row` to a :class:`KnownHash`."""
        return KnownHash(
            model_id=row["model_id"],
            revision=row["revision"],
            file_path=row["file_path"],
            sha256=row["sha256"],
            source=row["source"],
            notes=row["notes"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Public API — lifecycle
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create the database schema if it does not already exist.

        Safe to call multiple times (uses ``CREATE TABLE IF NOT EXISTS``).
        """
        conn = self._get_connection()
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
        logger.debug("Database schema initialised at %s", self._db_path)

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "HashDatabase":
        self.init_schema()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API — seeding
    # ------------------------------------------------------------------

    def seed_from_yaml(self, yaml_path: str | Path | None = None) -> int:
        """Seed the database from a known-good YAML hash file.

        Loads entries from *yaml_path* (or the bundled
        ``data/known_hashes.yaml`` if not specified) and inserts them using
        ``INSERT OR IGNORE`` so that existing records are never overwritten.

        Args:
            yaml_path: Path to the YAML seed file.  If ``None``, the bundled
                ``data/known_hashes.yaml`` is used.

        Returns:
            The number of new rows inserted.

        Raises:
            FileNotFoundError: If *yaml_path* is specified but does not exist.
            yaml.YAMLError: If the YAML file cannot be parsed.
        """
        self.init_schema()

        if yaml_path is None:
            yaml_content = _load_bundled_yaml()
        else:
            path = Path(yaml_path)
            if not path.exists():
                raise FileNotFoundError(f"YAML seed file not found: {path}")
            yaml_content = path.read_text(encoding="utf-8")

        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            logger.warning("YAML seed file has unexpected top-level structure; skipping.")
            return 0

        models: list[dict] = data.get("known_models", [])
        if not isinstance(models, list):
            logger.warning("'known_models' key missing or not a list; skipping.")
            return 0

        inserted = 0
        with self._transaction() as conn:
            for entry in models:
                if not isinstance(entry, dict):
                    continue
                model_id: str = str(entry.get("model_id", "")).strip()
                revision: str = str(entry.get("revision", "main")).strip()
                source: str = str(entry.get("source", "unknown")).strip()
                notes: str | None = entry.get("notes")
                files: dict = entry.get("files", {})

                if not model_id or not isinstance(files, dict):
                    continue

                for file_path, sha256 in files.items():
                    if not file_path or not sha256:
                        continue
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO known_hashes
                            (model_id, revision, file_path, sha256, source, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            model_id,
                            revision,
                            str(file_path).replace("\\", "/"),
                            str(sha256).lower().strip(),
                            source,
                            notes,
                        ),
                    )
                    inserted += cursor.rowcount

        logger.debug("Seeded %d new records into %s", inserted, self._db_path)
        return inserted

    # ------------------------------------------------------------------
    # Public API — writes
    # ------------------------------------------------------------------

    def add_hash(
        self,
        model_id: str,
        file_path: str,
        sha256: str,
        revision: str = "main",
        source: str = "computed",
        notes: str | None = None,
        overwrite: bool = False,
    ) -> bool:
        """Insert or update a known-good hash record.

        Args:
            model_id: Hugging Face model ID or any model identifier.
            file_path: Relative path of the file within the model repo.
            sha256: SHA-256 hex digest of the file.
            revision: Git revision string.  Defaults to ``'main'``.
            source: Provenance label.  Defaults to ``'computed'``.
            notes: Optional notes.
            overwrite: If ``True``, an existing record with the same
                ``(model_id, revision, file_path)`` key will be updated.
                If ``False`` (default), the existing record is kept.

        Returns:
            ``True`` if a new row was inserted, ``False`` if the record
            already existed (and *overwrite* was ``False``) or if the row
            was updated.
        """
        self.init_schema()
        normalised_path = file_path.replace("\\", "/")
        normalised_sha256 = sha256.lower().strip()

        if overwrite:
            sql = """
                INSERT INTO known_hashes (model_id, revision, file_path, sha256, source, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (model_id, revision, file_path)
                DO UPDATE SET sha256=excluded.sha256, source=excluded.source,
                              notes=excluded.notes,
                              created_at=datetime('now', 'utc')
            """
        else:
            sql = """
                INSERT OR IGNORE INTO known_hashes
                    (model_id, revision, file_path, sha256, source, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """

        with self._transaction() as conn:
            cursor = conn.execute(
                sql,
                (model_id, revision, normalised_path, normalised_sha256, source, notes),
            )
            return cursor.rowcount > 0

    def delete_hash(
        self,
        model_id: str,
        revision: str,
        file_path: str,
    ) -> bool:
        """Delete a specific known-good hash record.

        Args:
            model_id: Model identifier.
            revision: Git revision string.
            file_path: Relative file path.

        Returns:
            ``True`` if the record was deleted, ``False`` if it did not exist.
        """
        self.init_schema()
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM known_hashes WHERE model_id=? AND revision=? AND file_path=?",
                (model_id, revision, file_path.replace("\\", "/")),
            )
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Public API — queries
    # ------------------------------------------------------------------

    def get_hash(
        self,
        model_id: str,
        file_path: str,
        revision: str = "main",
    ) -> KnownHash | None:
        """Look up the known-good hash for a specific model file.

        Args:
            model_id: Hugging Face model ID or model identifier.
            file_path: Relative path of the file within the model repo.
            revision: Git revision string.  Defaults to ``'main'``.

        Returns:
            A :class:`KnownHash` if a record was found, or ``None``.
        """
        self.init_schema()
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT model_id, revision, file_path, sha256, source, notes, created_at
            FROM known_hashes
            WHERE model_id=? AND revision=? AND file_path=?
            """,
            (model_id, revision, file_path.replace("\\", "/")),
        ).fetchone()
        return self._row_to_known_hash(row) if row else None

    def get_all_hashes_for_model(
        self,
        model_id: str,
        revision: str = "main",
    ) -> list[KnownHash]:
        """Return all known-good hash records for a given model and revision.

        Args:
            model_id: Hugging Face model ID or model identifier.
            revision: Git revision string.  Defaults to ``'main'``.

        Returns:
            Possibly-empty list of :class:`KnownHash` records, ordered by
            ``file_path``.
        """
        self.init_schema()
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT model_id, revision, file_path, sha256, source, notes, created_at
            FROM known_hashes
            WHERE model_id=? AND revision=?
            ORDER BY file_path
            """,
            (model_id, revision),
        ).fetchall()
        return [self._row_to_known_hash(r) for r in rows]

    def list_models(self) -> list[tuple[str, str]]:
        """Return the distinct ``(model_id, revision)`` pairs stored in the DB.

        Returns:
            List of ``(model_id, revision)`` tuples, ordered alphabetically.
        """
        self.init_schema()
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT DISTINCT model_id, revision
            FROM known_hashes
            ORDER BY model_id, revision
            """
        ).fetchall()
        return [(row["model_id"], row["revision"]) for row in rows]

    def iter_all_hashes(self) -> Iterator[KnownHash]:
        """Iterate over every record in the database.

        Yields:
            :class:`KnownHash` records in ``(model_id, revision, file_path)``
            order.
        """
        self.init_schema()
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT model_id, revision, file_path, sha256, source, notes, created_at
            FROM known_hashes
            ORDER BY model_id, revision, file_path
            """
        )
        for row in cursor:
            yield self._row_to_known_hash(row)

    def count(self) -> int:
        """Return the total number of records in the database."""
        self.init_schema()
        conn = self._get_connection()
        row = conn.execute("SELECT COUNT(*) AS cnt FROM known_hashes").fetchone()
        return int(row["cnt"])

    def has_model(self, model_id: str, revision: str = "main") -> bool:
        """Return ``True`` if any hashes are stored for the given model.

        Args:
            model_id: Hugging Face model ID or model identifier.
            revision: Git revision string.  Defaults to ``'main'``.
        """
        self.init_schema()
        conn = self._get_connection()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM known_hashes WHERE model_id=? AND revision=?",
            (model_id, revision),
        ).fetchone()
        return int(row["cnt"]) > 0


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_default_db() -> HashDatabase:
    """Return a :class:`HashDatabase` pointing to the default user database.

    The database is located at ``~/.model-provenance/hashes.db``.  The schema
    is initialised automatically on first access.

    Returns:
        An open :class:`HashDatabase` instance.
    """
    db = HashDatabase()
    db.init_schema()
    return db


def _load_bundled_yaml() -> str:
    """Load the bundled ``data/known_hashes.yaml`` seed file.

    Tries ``importlib.resources`` first (works when the package is installed)
    then falls back to a path relative to this source file.

    Returns:
        Raw YAML text content.

    Raises:
        FileNotFoundError: If the bundled YAML cannot be located.
    """
    # Attempt 1: importlib.resources (installed package)
    try:
        ref = importlib.resources.files("model_provenance").joinpath(
            "../data/known_hashes.yaml"
        )
        return ref.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError, ModuleNotFoundError, AttributeError):
        pass

    # Attempt 2: path relative to this source file (development checkout)
    candidate = Path(__file__).parent.parent / "data" / "known_hashes.yaml"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")

    raise FileNotFoundError(
        "Bundled known_hashes.yaml not found. "
        "Ensure the package was installed correctly or provide an explicit path."
    )
