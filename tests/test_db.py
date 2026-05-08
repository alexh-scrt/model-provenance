"""Unit tests for model_provenance.db module.

Covers schema initialisation, YAML seeding, hash insertion / lookup / deletion,
and the module-level convenience helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from model_provenance.db import HashDatabase, KnownHash, _load_bundled_yaml


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
def sample_yaml(tmp_path: Path) -> Path:
    """Write a minimal known_hashes.yaml to a temp file."""
    data = {
        "known_models": [
            {
                "model_id": "test/model-a",
                "revision": "main",
                "source": "official",
                "notes": "Test model A",
                "files": {
                    "config.json": "a" * 64,
                    "model.bin": "b" * 64,
                },
            },
            {
                "model_id": "test/model-b",
                "revision": "v1.0",
                "source": "community",
                "notes": None,
                "files": {
                    "weights.safetensors": "c" * 64,
                },
            },
        ]
    }
    path = tmp_path / "known_hashes.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# KnownHash
# ---------------------------------------------------------------------------


class TestKnownHash:
    def test_sha256_normalised_to_lowercase(self) -> None:
        kh = KnownHash(
            model_id="m", revision="main", file_path="f.bin", sha256="A" * 64
        )
        assert kh.sha256 == "a" * 64

    def test_to_dict_keys(self) -> None:
        kh = KnownHash(
            model_id="m",
            revision="main",
            file_path="config.json",
            sha256="a" * 64,
            source="official",
            notes="note",
        )
        d = kh.to_dict()
        assert set(d.keys()) == {
            "model_id",
            "revision",
            "file_path",
            "sha256",
            "source",
            "notes",
            "created_at",
        }

    def test_equality(self) -> None:
        kh1 = KnownHash(model_id="m", revision="main", file_path="f", sha256="a" * 64)
        kh2 = KnownHash(model_id="m", revision="main", file_path="f", sha256="a" * 64)
        assert kh1 == kh2

    def test_inequality_different_hash(self) -> None:
        kh1 = KnownHash(model_id="m", revision="main", file_path="f", sha256="a" * 64)
        kh2 = KnownHash(model_id="m", revision="main", file_path="f", sha256="b" * 64)
        assert kh1 != kh2

    def test_inequality_different_model(self) -> None:
        kh1 = KnownHash(model_id="m1", revision="main", file_path="f", sha256="a" * 64)
        kh2 = KnownHash(model_id="m2", revision="main", file_path="f", sha256="a" * 64)
        assert kh1 != kh2


# ---------------------------------------------------------------------------
# HashDatabase — schema init
# ---------------------------------------------------------------------------


class TestHashDatabaseInit:
    def test_init_schema_idempotent(self, db: HashDatabase) -> None:
        # Calling again must not raise.
        db.init_schema()
        db.init_schema()

    def test_count_zero_after_init(self, db: HashDatabase) -> None:
        assert db.count() == 0

    def test_context_manager(self) -> None:
        with HashDatabase(":memory:") as instance:
            assert instance.count() == 0

    def test_disk_db_creates_parent_dir(self, tmp_path: Path) -> None:
        db_path = tmp_path / "subdir" / "hashes.db"
        instance = HashDatabase(db_path)
        instance.init_schema()
        assert db_path.exists()
        instance.close()


# ---------------------------------------------------------------------------
# HashDatabase — seeding
# ---------------------------------------------------------------------------


class TestHashDatabaseSeed:
    def test_seed_from_yaml_inserts_records(self, db: HashDatabase, sample_yaml: Path) -> None:
        count = db.seed_from_yaml(sample_yaml)
        assert count == 3  # 2 files for model-a + 1 for model-b

    def test_seed_idempotent(self, db: HashDatabase, sample_yaml: Path) -> None:
        first = db.seed_from_yaml(sample_yaml)
        second = db.seed_from_yaml(sample_yaml)
        assert second == 0  # no new rows
        assert db.count() == first

    def test_seed_file_not_found_raises(self, db: HashDatabase, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            db.seed_from_yaml(tmp_path / "no_such_file.yaml")

    def test_seed_from_bundled_yaml(self, db: HashDatabase) -> None:
        # The bundled YAML should be discoverable and parsable.
        count = db.seed_from_yaml()  # None → bundled file
        assert count >= 0  # seed data may already have records from previous call

    def test_seed_records_retrievable(self, db: HashDatabase, sample_yaml: Path) -> None:
        db.seed_from_yaml(sample_yaml)
        kh = db.get_hash("test/model-a", "config.json", revision="main")
        assert kh is not None
        assert kh.sha256 == "a" * 64
        assert kh.source == "official"
        assert kh.notes == "Test model A"

    def test_seed_empty_yaml_returns_zero(self, db: HashDatabase, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("known_models: []", encoding="utf-8")
        assert db.seed_from_yaml(path) == 0

    def test_seed_yaml_bad_structure_returns_zero(self, db: HashDatabase, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("this_is_not_known_models: true", encoding="utf-8")
        assert db.seed_from_yaml(path) == 0

    def test_seed_sha256_stored_lowercase(self, db: HashDatabase, tmp_path: Path) -> None:
        data = {
            "known_models": [
                {
                    "model_id": "test/upper",
                    "revision": "main",
                    "source": "computed",
                    "files": {"config.json": "A" * 64},
                }
            ]
        }
        path = tmp_path / "upper.yaml"
        path.write_text(yaml.dump(data), encoding="utf-8")
        db.seed_from_yaml(path)
        kh = db.get_hash("test/upper", "config.json", revision="main")
        assert kh is not None
        assert kh.sha256 == "a" * 64


# ---------------------------------------------------------------------------
# HashDatabase — add_hash
# ---------------------------------------------------------------------------


class TestHashDatabaseAddHash:
    def test_add_hash_returns_true_for_new(self, db: HashDatabase) -> None:
        result = db.add_hash("mymodel", "config.json", "a" * 64)
        assert result is True

    def test_add_hash_returns_false_for_duplicate(self, db: HashDatabase) -> None:
        db.add_hash("mymodel", "config.json", "a" * 64)
        result = db.add_hash("mymodel", "config.json", "b" * 64)  # same key
        assert result is False

    def test_add_hash_overwrite(self, db: HashDatabase) -> None:
        db.add_hash("mymodel", "config.json", "a" * 64)
        db.add_hash("mymodel", "config.json", "b" * 64, overwrite=True)
        kh = db.get_hash("mymodel", "config.json")
        assert kh is not None
        assert kh.sha256 == "b" * 64

    def test_add_hash_default_source_computed(self, db: HashDatabase) -> None:
        db.add_hash("mymodel", "config.json", "a" * 64)
        kh = db.get_hash("mymodel", "config.json")
        assert kh is not None
        assert kh.source == "computed"

    def test_add_hash_custom_revision(self, db: HashDatabase) -> None:
        db.add_hash("mymodel", "config.json", "a" * 64, revision="v2.0")
        kh = db.get_hash("mymodel", "config.json", revision="v2.0")
        assert kh is not None
        # Should NOT be found under default revision
        assert db.get_hash("mymodel", "config.json", revision="main") is None

    def test_add_hash_normalises_sha256(self, db: HashDatabase) -> None:
        db.add_hash("mymodel", "config.json", "A" * 64)
        kh = db.get_hash("mymodel", "config.json")
        assert kh is not None
        assert kh.sha256 == "a" * 64

    def test_add_hash_normalises_path_separator(self, db: HashDatabase) -> None:
        db.add_hash("mymodel", "sub\\config.json", "a" * 64)
        kh = db.get_hash("mymodel", "sub/config.json")
        assert kh is not None

    def test_count_increases(self, db: HashDatabase) -> None:
        assert db.count() == 0
        db.add_hash("m", "a.json", "a" * 64)
        db.add_hash("m", "b.bin", "b" * 64)
        assert db.count() == 2


# ---------------------------------------------------------------------------
# HashDatabase — delete_hash
# ---------------------------------------------------------------------------


class TestHashDatabaseDeleteHash:
    def test_delete_existing(self, db: HashDatabase) -> None:
        db.add_hash("mymodel", "config.json", "a" * 64)
        result = db.delete_hash("mymodel", "main", "config.json")
        assert result is True
        assert db.get_hash("mymodel", "config.json") is None

    def test_delete_nonexistent_returns_false(self, db: HashDatabase) -> None:
        result = db.delete_hash("mymodel", "main", "ghost.json")
        assert result is False

    def test_delete_reduces_count(self, db: HashDatabase) -> None:
        db.add_hash("m", "a.json", "a" * 64)
        db.add_hash("m", "b.bin", "b" * 64)
        db.delete_hash("m", "main", "a.json")
        assert db.count() == 1


# ---------------------------------------------------------------------------
# HashDatabase — queries
# ---------------------------------------------------------------------------


class TestHashDatabaseQueries:
    def _populate(self, db: HashDatabase) -> None:
        db.add_hash("bert-base", "config.json", "a" * 64, revision="main")
        db.add_hash("bert-base", "model.bin", "b" * 64, revision="main")
        db.add_hash("bert-base", "config.json", "c" * 64, revision="v2")
        db.add_hash("gpt2", "config.json", "d" * 64, revision="main")

    def test_get_hash_existing(self, db: HashDatabase) -> None:
        self._populate(db)
        kh = db.get_hash("bert-base", "config.json", revision="main")
        assert kh is not None
        assert kh.sha256 == "a" * 64

    def test_get_hash_missing_returns_none(self, db: HashDatabase) -> None:
        self._populate(db)
        assert db.get_hash("bert-base", "missing.txt") is None

    def test_get_all_hashes_for_model(self, db: HashDatabase) -> None:
        self._populate(db)
        hashes = db.get_all_hashes_for_model("bert-base", revision="main")
        assert len(hashes) == 2
        paths = {h.file_path for h in hashes}
        assert paths == {"config.json", "model.bin"}

    def test_get_all_hashes_empty_for_unknown(self, db: HashDatabase) -> None:
        self._populate(db)
        assert db.get_all_hashes_for_model("unknown-model") == []

    def test_list_models(self, db: HashDatabase) -> None:
        self._populate(db)
        models = db.list_models()
        model_ids = [m[0] for m in models]
        assert "bert-base" in model_ids
        assert "gpt2" in model_ids

    def test_list_models_includes_revision(self, db: HashDatabase) -> None:
        self._populate(db)
        pairs = set(db.list_models())
        assert ("bert-base", "main") in pairs
        assert ("bert-base", "v2") in pairs

    def test_iter_all_hashes(self, db: HashDatabase) -> None:
        self._populate(db)
        all_hashes = list(db.iter_all_hashes())
        assert len(all_hashes) == 4

    def test_has_model_true(self, db: HashDatabase) -> None:
        self._populate(db)
        assert db.has_model("bert-base", revision="main") is True

    def test_has_model_false(self, db: HashDatabase) -> None:
        self._populate(db)
        assert db.has_model("nonexistent") is False

    def test_has_model_wrong_revision(self, db: HashDatabase) -> None:
        self._populate(db)
        assert db.has_model("bert-base", revision="nonexistent-rev") is False


# ---------------------------------------------------------------------------
# _load_bundled_yaml
# ---------------------------------------------------------------------------


class TestLoadBundledYaml:
    def test_returns_string(self) -> None:
        content = _load_bundled_yaml()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_valid_yaml(self) -> None:
        content = _load_bundled_yaml()
        data = yaml.safe_load(content)
        assert isinstance(data, dict)
        assert "known_models" in data
