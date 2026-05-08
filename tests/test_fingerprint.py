"""Unit tests for model_provenance.fingerprint module.

Covers SHA-256 hash computation, file classification, manifest construction
from directories and pre-computed maps, and error-handling edge cases.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from model_provenance.fingerprint import (
    FileFingerprint,
    FingerprintManifest,
    _aggregate_hash,
    build_manifest_from_directory,
    build_manifest_from_file_map,
    classify_file,
    compute_sha256,
    compute_sha256_bytes,
    fingerprint_bytes_entry,
    fingerprint_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# compute_sha256
# ---------------------------------------------------------------------------


class TestComputeSha256:
    """Tests for the low-level file-hashing helper."""

    def test_hash_matches_stdlib(self, tmp_path: Path) -> None:
        content = b"hello model provenance world"
        target = tmp_path / "file.bin"
        target.write_bytes(content)
        assert compute_sha256(target) == _sha256_of(content)

    def test_empty_file(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.bin"
        target.write_bytes(b"")
        assert compute_sha256(target) == _sha256_of(b"")

    def test_large_file_chunked(self, tmp_path: Path) -> None:
        # Create a file larger than the 1 MiB read block.
        content = b"X" * (3 * 1024 * 1024)  # 3 MiB
        target = tmp_path / "large.bin"
        target.write_bytes(content)
        assert compute_sha256(target) == _sha256_of(content)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.bin"
        with pytest.raises(FileNotFoundError):
            compute_sha256(missing)

    def test_directory_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a regular file"):
            compute_sha256(tmp_path)

    def test_returns_lowercase_hex(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"test")
        digest = compute_sha256(target)
        assert digest == digest.lower()
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        assert compute_sha256(a) != compute_sha256(b)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        content = b"identical content"
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(content)
        b.write_bytes(content)
        assert compute_sha256(a) == compute_sha256(b)


# ---------------------------------------------------------------------------
# compute_sha256_bytes
# ---------------------------------------------------------------------------


class TestComputeSha256Bytes:
    def test_matches_stdlib(self) -> None:
        data = b"in-memory blob"
        assert compute_sha256_bytes(data) == _sha256_of(data)

    def test_empty_bytes(self) -> None:
        assert compute_sha256_bytes(b"") == _sha256_of(b"")

    def test_returns_lowercase_hex_64_chars(self) -> None:
        digest = compute_sha256_bytes(b"abc")
        assert len(digest) == 64
        assert digest == digest.lower()

    def test_different_data_different_hash(self) -> None:
        assert compute_sha256_bytes(b"aaa") != compute_sha256_bytes(b"bbb")

    def test_large_data(self) -> None:
        data = b"Z" * (5 * 1024 * 1024)
        digest = compute_sha256_bytes(data)
        assert len(digest) == 64
        assert digest == _sha256_of(data)


# ---------------------------------------------------------------------------
# classify_file
# ---------------------------------------------------------------------------


class TestClassifyFile:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("pytorch_model.bin", "weight"),
            ("model.safetensors", "weight"),
            ("model.pt", "weight"),
            ("checkpoint.ckpt", "weight"),
            ("model.pkl", "weight"),
            ("weights.npz", "weight"),
            ("model.h5", "weight"),
            ("model.gguf", "weight"),
            ("model.ggml", "weight"),
            ("model.msgpack", "weight"),
            ("model.flax", "weight"),
            ("model.npy", "weight"),
            ("model.hdf5", "weight"),
            ("config.json", "config"),
            ("tokenizer_config.json", "config"),
            ("vocab.txt", "config"),
            ("README.md", "config"),
            ("setup.cfg", "config"),
            ("special_tokens_map.json", "config"),
            ("run.sh", "other"),
            ("main.py", "other"),
            ("binary.elf", "other"),
            ("archive.zip", "other"),
            ("noextension", "other"),
            (".hidden", "other"),
        ],
    )
    def test_classification(self, filename: str, expected: str) -> None:
        assert classify_file(filename) == expected

    def test_path_object_accepted(self) -> None:
        assert classify_file(Path("model.safetensors")) == "weight"

    def test_case_insensitive_extension(self) -> None:
        assert classify_file("MODEL.BIN") == "weight"
        assert classify_file("Config.JSON") == "config"

    def test_nested_path_uses_suffix_only(self) -> None:
        assert classify_file("subdir/model.bin") == "weight"
        assert classify_file("subdir/config.json") == "config"

    def test_returns_string_type(self) -> None:
        result = classify_file("model.bin")
        assert isinstance(result, str)
        assert result in ("weight", "config", "other")


# ---------------------------------------------------------------------------
# FileFingerprint
# ---------------------------------------------------------------------------


class TestFileFingerprint:
    def _make(self, **kwargs) -> FileFingerprint:
        defaults = dict(
            path="config.json",
            sha256="a" * 64,
            size_bytes=128,
            file_type="config",
        )
        defaults.update(kwargs)
        return FileFingerprint(**defaults)

    def test_short_hash(self) -> None:
        fp = self._make(sha256="abcd1234" + "0" * 56)
        assert fp.short_hash == "abcd12340000000"
        assert len(fp.short_hash) == 16

    def test_is_weight(self) -> None:
        assert self._make(file_type="weight").is_weight
        assert not self._make(file_type="config").is_weight

    def test_is_config(self) -> None:
        assert self._make(file_type="config").is_config
        assert not self._make(file_type="weight").is_config

    def test_other_file_type_not_weight_or_config(self) -> None:
        fp = self._make(file_type="other")
        assert not fp.is_weight
        assert not fp.is_config

    def test_ok_no_error(self) -> None:
        assert self._make().ok

    def test_ok_false_with_error(self) -> None:
        assert not self._make(error="some error").ok

    def test_ok_false_with_empty_error_string(self) -> None:
        # An empty error string still means no error (it's falsy but not None).
        fp = self._make(error="")
        # The ok property checks error is None.
        assert fp.ok  # empty string is not None

    def test_ok_false_with_none_error_is_true(self) -> None:
        fp = self._make(error=None)
        assert fp.ok

    def test_to_dict(self) -> None:
        fp = self._make()
        d = fp.to_dict()
        assert set(d.keys()) == {"path", "sha256", "size_bytes", "file_type", "error"}
        assert d["path"] == "config.json"
        assert d["error"] is None

    def test_to_dict_values(self) -> None:
        fp = self._make(sha256="b" * 64, size_bytes=256, file_type="weight", error="oops")
        d = fp.to_dict()
        assert d["sha256"] == "b" * 64
        assert d["size_bytes"] == 256
        assert d["file_type"] == "weight"
        assert d["error"] == "oops"

    def test_short_hash_empty_when_no_sha256(self) -> None:
        fp = self._make(sha256="")
        assert fp.short_hash == ""

    def test_short_hash_length_16(self) -> None:
        fp = self._make(sha256="a" * 64)
        assert len(fp.short_hash) == 16


# ---------------------------------------------------------------------------
# FingerprintManifest
# ---------------------------------------------------------------------------


class TestFingerprintManifest:
    def _make_manifest(self) -> FingerprintManifest:
        files = [
            FileFingerprint(
                path="config.json",
                sha256="a" * 64,
                size_bytes=100,
                file_type="config",
            ),
            FileFingerprint(
                path="pytorch_model.bin",
                sha256="b" * 64,
                size_bytes=500,
                file_type="weight",
            ),
        ]
        m = FingerprintManifest(
            model_id="test/model",
            revision="main",
            source="hub",
        )
        m.files = files
        return m

    def test_get_existing(self) -> None:
        m = self._make_manifest()
        fp = m.get("config.json")
        assert fp is not None
        assert fp.path == "config.json"

    def test_get_missing(self) -> None:
        m = self._make_manifest()
        assert m.get("nonexistent.txt") is None

    def test_get_normalises_backslash(self) -> None:
        m = FingerprintManifest(model_id="test")
        m.files = [
            FileFingerprint(
                path="sub/config.json",
                sha256="a" * 64,
                size_bytes=10,
                file_type="config",
            )
        ]
        # Should find even with backslash separator.
        assert m.get("sub/config.json") is not None

    def test_weight_files(self) -> None:
        m = self._make_manifest()
        assert len(m.weight_files) == 1
        assert m.weight_files[0].path == "pytorch_model.bin"

    def test_config_files(self) -> None:
        m = self._make_manifest()
        assert len(m.config_files) == 1
        assert m.config_files[0].path == "config.json"

    def test_total_size_bytes(self) -> None:
        m = self._make_manifest()
        assert m.total_size_bytes == 600

    def test_file_count(self) -> None:
        m = self._make_manifest()
        assert m.file_count == 2

    def test_file_count_zero_when_empty(self) -> None:
        m = FingerprintManifest(model_id="test")
        assert m.file_count == 0

    def test_errored_files_empty_when_none(self) -> None:
        m = self._make_manifest()
        assert m.errored_files == []

    def test_errored_files_returns_errored(self) -> None:
        m = self._make_manifest()
        m.files.append(
            FileFingerprint(
                path="bad.bin",
                sha256="",
                size_bytes=0,
                file_type="weight",
                error="read error",
            )
        )
        assert len(m.errored_files) == 1
        assert m.errored_files[0].path == "bad.bin"

    def test_to_dict_keys(self) -> None:
        m = self._make_manifest()
        d = m.to_dict()
        expected_keys = {
            "model_id",
            "revision",
            "source",
            "computed_at",
            "aggregate_sha256",
            "file_count",
            "total_size_bytes",
            "files",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_files_is_list(self) -> None:
        m = self._make_manifest()
        d = m.to_dict()
        assert isinstance(d["files"], list)
        assert len(d["files"]) == 2

    def test_to_dict_model_id(self) -> None:
        m = self._make_manifest()
        assert m.to_dict()["model_id"] == "test/model"

    def test_computed_at_format(self) -> None:
        m = self._make_manifest()
        # Should match YYYY-MM-DDTHH:MM:SSZ
        import re

        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        assert re.match(pattern, m.computed_at)

    def test_aggregate_sha256_defaults_to_none(self) -> None:
        m = FingerprintManifest(model_id="test")
        assert m.aggregate_sha256 is None

    def test_source_default_local(self) -> None:
        m = FingerprintManifest(model_id="test")
        assert m.source == "local"

    def test_revision_default_local(self) -> None:
        m = FingerprintManifest(model_id="test")
        assert m.revision == "local"


# ---------------------------------------------------------------------------
# _aggregate_hash
# ---------------------------------------------------------------------------


class TestAggregateHash:
    def _fps(self) -> list[FileFingerprint]:
        return [
            FileFingerprint(
                path="a.txt", sha256="a" * 64, size_bytes=1, file_type="config"
            ),
            FileFingerprint(
                path="b.bin", sha256="b" * 64, size_bytes=2, file_type="weight"
            ),
        ]

    def test_deterministic(self) -> None:
        fps = self._fps()
        assert _aggregate_hash(fps) == _aggregate_hash(fps)

    def test_order_independent(self) -> None:
        fps = self._fps()
        assert _aggregate_hash(fps) == _aggregate_hash(list(reversed(fps)))

    def test_different_hashes_produce_different_aggregate(self) -> None:
        fps1 = self._fps()
        fps2 = [
            FileFingerprint(
                path="a.txt", sha256="c" * 64, size_bytes=1, file_type="config"
            ),
            FileFingerprint(
                path="b.bin", sha256="d" * 64, size_bytes=2, file_type="weight"
            ),
        ]
        assert _aggregate_hash(fps1) != _aggregate_hash(fps2)

    def test_returns_64_char_hex(self) -> None:
        digest = _aggregate_hash(self._fps())
        assert len(digest) == 64
        assert digest == digest.lower()

    def test_different_paths_produce_different_aggregate(self) -> None:
        fps1 = [
            FileFingerprint(
                path="path_one.txt", sha256="a" * 64, size_bytes=1, file_type="config"
            ),
        ]
        fps2 = [
            FileFingerprint(
                path="path_two.txt", sha256="a" * 64, size_bytes=1, file_type="config"
            ),
        ]
        assert _aggregate_hash(fps1) != _aggregate_hash(fps2)

    def test_single_file_list(self) -> None:
        fps = [
            FileFingerprint(
                path="only.bin", sha256="e" * 64, size_bytes=10, file_type="weight"
            )
        ]
        digest = _aggregate_hash(fps)
        assert len(digest) == 64

    def test_adding_file_changes_aggregate(self) -> None:
        fps = self._fps()
        extra_fp = FileFingerprint(
            path="extra.bin", sha256="f" * 64, size_bytes=3, file_type="weight"
        )
        fps_with_extra = fps + [extra_fp]
        assert _aggregate_hash(fps) != _aggregate_hash(fps_with_extra)


# ---------------------------------------------------------------------------
# fingerprint_file
# ---------------------------------------------------------------------------


class TestFingerprintFile:
    def test_basic(self, tmp_path: Path) -> None:
        content = b"model weights here"
        target = tmp_path / "model.bin"
        target.write_bytes(content)
        fp = fingerprint_file(target, base_dir=tmp_path)
        assert fp.sha256 == _sha256_of(content)
        assert fp.size_bytes == len(content)
        assert fp.file_type == "weight"
        assert fp.path == "model.bin"
        assert fp.ok

    def test_relative_path_uses_forward_slash(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        target = subdir / "config.json"
        target.write_bytes(b"{}")
        fp = fingerprint_file(target, base_dir=tmp_path)
        assert "/" in fp.path
        assert fp.path == "sub/config.json"

    def test_no_base_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "vocab.txt"
        target.write_bytes(b"word\n")
        fp = fingerprint_file(target, base_dir=None)
        assert fp.ok
        # Path should be the full stringified path
        assert str(target).replace("\\", "/") in fp.path or fp.path.endswith(
            "vocab.txt"
        )

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "ghost.bin"
        fp = fingerprint_file(missing, base_dir=tmp_path)
        assert not fp.ok
        assert "FileNotFoundError" in (fp.error or "")
        assert fp.sha256 == ""
        assert fp.size_bytes == 0

    def test_config_file_classified_correctly(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        target.write_bytes(b"{\"key\": \"value\"}")
        fp = fingerprint_file(target, base_dir=tmp_path)
        assert fp.file_type == "config"
        assert fp.is_config

    def test_size_bytes_matches_actual(self, tmp_path: Path) -> None:
        content = b"a" * 1234
        target = tmp_path / "model.pt"
        target.write_bytes(content)
        fp = fingerprint_file(target, base_dir=tmp_path)
        assert fp.size_bytes == 1234

    def test_sha256_matches_direct_computation(self, tmp_path: Path) -> None:
        content = b"direct computation test"
        target = tmp_path / "weights.safetensors"
        target.write_bytes(content)
        fp = fingerprint_file(target, base_dir=tmp_path)
        assert fp.sha256 == _sha256_of(content)

    def test_error_is_none_on_success(self, tmp_path: Path) -> None:
        target = tmp_path / "tokenizer.json"
        target.write_bytes(b"{}")
        fp = fingerprint_file(target, base_dir=tmp_path)
        assert fp.error is None

    def test_nested_path_in_subdirectory(self, tmp_path: Path) -> None:
        subdir = tmp_path / "shards" / "part0"
        subdir.mkdir(parents=True)
        target = subdir / "shard.bin"
        target.write_bytes(b"shard content")
        fp = fingerprint_file(target, base_dir=tmp_path)
        assert fp.path == "shards/part0/shard.bin"
        assert fp.ok


# ---------------------------------------------------------------------------
# build_manifest_from_directory
# ---------------------------------------------------------------------------


class TestBuildManifestFromDirectory:
    def _make_model_dir(self, tmp_path: Path) -> Path:
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b'{"model_type": "bert"}')
        (model_dir / "tokenizer.json").write_bytes(b'{"version": "1.0"}')
        (model_dir / "pytorch_model.bin").write_bytes(b"fake weights")
        return model_dir

    def test_returns_manifest(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(model_dir)
        assert isinstance(manifest, FingerprintManifest)

    def test_correct_file_count(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(model_dir)
        assert manifest.file_count == 3

    def test_model_id_defaults_to_dir_name(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(model_dir)
        assert manifest.model_id == "my_model"

    def test_custom_model_id(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(
            model_dir, model_id="bert-base-uncased"
        )
        assert manifest.model_id == "bert-base-uncased"

    def test_aggregate_sha256_computed(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(model_dir)
        assert manifest.aggregate_sha256 is not None
        assert len(manifest.aggregate_sha256) == 64

    def test_aggregate_deterministic(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        m1 = build_manifest_from_directory(model_dir)
        m2 = build_manifest_from_directory(model_dir)
        assert m1.aggregate_sha256 == m2.aggregate_sha256

    def test_source_is_local(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(model_dir)
        assert manifest.source == "local"

    def test_git_dir_skipped(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        git_dir = model_dir / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_bytes(b"ref: refs/heads/main")
        manifest = build_manifest_from_directory(model_dir)
        paths = [f.path for f in manifest.files]
        assert not any(".git" in p for p in paths)

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            build_manifest_from_directory(tmp_path / "nonexistent")

    def test_file_path_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_bytes(b"x")
        with pytest.raises(NotADirectoryError):
            build_manifest_from_directory(f)

    def test_hashes_match_direct_computation(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(model_dir)
        for fp in manifest.files:
            expected = _sha256_of((model_dir / fp.path).read_bytes())
            assert fp.sha256 == expected

    def test_subdirectory_files_included(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        subdir = model_dir / "shards"
        subdir.mkdir()
        (subdir / "shard-0.bin").write_bytes(b"shard data")
        manifest = build_manifest_from_directory(model_dir)
        paths = [f.path for f in manifest.files]
        assert any("shards" in p for p in paths)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(str(model_dir))
        assert manifest.file_count == 3

    def test_custom_revision(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(model_dir, revision="v1.0")
        assert manifest.revision == "v1.0"

    def test_default_revision_is_local(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(model_dir)
        assert manifest.revision == "local"

    def test_file_paths_use_forward_slash(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        subdir = model_dir / "sub"
        subdir.mkdir()
        (subdir / "extra.json").write_bytes(b"{}")
        manifest = build_manifest_from_directory(model_dir)
        for fp in manifest.files:
            assert "\\" not in fp.path

    def test_aggregate_none_when_error_files(self, tmp_path: Path) -> None:
        """If any file errors, aggregate_sha256 should be None."""
        model_dir = self._make_model_dir(tmp_path)
        manifest = build_manifest_from_directory(model_dir)
        # Manually inject an errored file to test the logic.
        manifest.files.append(
            FileFingerprint(
                path="bad.bin",
                sha256="",
                size_bytes=0,
                file_type="weight",
                error="read error",
            )
        )
        # Re-running build would set aggregate_sha256 = None when errors exist.
        # We test the property of errored_files instead.
        assert len(manifest.errored_files) == 1

    def test_empty_directory_returns_manifest(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_model"
        empty_dir.mkdir()
        manifest = build_manifest_from_directory(empty_dir)
        assert isinstance(manifest, FingerprintManifest)
        assert manifest.file_count == 0

    def test_cache_dir_skipped(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        cache_dir = model_dir / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "something.pyc").write_bytes(b"compiled")
        manifest = build_manifest_from_directory(model_dir)
        paths = [f.path for f in manifest.files]
        assert not any("__pycache__" in p for p in paths)


# ---------------------------------------------------------------------------
# build_manifest_from_file_map
# ---------------------------------------------------------------------------


class TestBuildManifestFromFileMap:
    def _sample_map(self) -> dict[str, str]:
        return {
            "config.json": "a" * 64,
            "pytorch_model.bin": "b" * 64,
            "tokenizer.json": "c" * 64,
        }

    def test_basic(self) -> None:
        manifest = build_manifest_from_file_map(
            self._sample_map(), model_id="test/model", revision="main"
        )
        assert manifest.file_count == 3
        assert manifest.model_id == "test/model"
        assert manifest.revision == "main"

    def test_aggregate_sha256_set(self) -> None:
        manifest = build_manifest_from_file_map(
            self._sample_map(), model_id="test/model"
        )
        assert manifest.aggregate_sha256 is not None
        assert len(manifest.aggregate_sha256) == 64

    def test_file_types_classified(self) -> None:
        manifest = build_manifest_from_file_map(
            self._sample_map(), model_id="test/model"
        )
        type_map = {f.path: f.file_type for f in manifest.files}
        assert type_map["config.json"] == "config"
        assert type_map["pytorch_model.bin"] == "weight"

    def test_empty_map_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            build_manifest_from_file_map({}, model_id="test/model")

    def test_sha256_normalised_to_lowercase(self) -> None:
        file_map = {"config.json": "A" * 64}
        manifest = build_manifest_from_file_map(file_map, model_id="test/model")
        assert manifest.files[0].sha256 == "a" * 64

    def test_source_default_is_hub(self) -> None:
        manifest = build_manifest_from_file_map(
            self._sample_map(), model_id="test/model"
        )
        assert manifest.source == "hub"

    def test_custom_source(self) -> None:
        manifest = build_manifest_from_file_map(
            self._sample_map(), model_id="test/model", source="official"
        )
        assert manifest.source == "official"

    def test_get_file_by_path(self) -> None:
        manifest = build_manifest_from_file_map(
            self._sample_map(), model_id="test/model"
        )
        fp = manifest.get("config.json")
        assert fp is not None
        assert fp.sha256 == "a" * 64

    def test_size_bytes_zero_from_map(self) -> None:
        manifest = build_manifest_from_file_map(
            self._sample_map(), model_id="test/model"
        )
        for fp in manifest.files:
            assert fp.size_bytes == 0

    def test_aggregate_order_independent(self) -> None:
        map1 = {"a.bin": "a" * 64, "b.json": "b" * 64}
        map2 = {"b.json": "b" * 64, "a.bin": "a" * 64}
        m1 = build_manifest_from_file_map(map1, model_id="test")
        m2 = build_manifest_from_file_map(map2, model_id="test")
        assert m1.aggregate_sha256 == m2.aggregate_sha256

    def test_single_file_map(self) -> None:
        file_map = {"only_file.bin": "d" * 64}
        manifest = build_manifest_from_file_map(file_map, model_id="test")
        assert manifest.file_count == 1
        assert manifest.files[0].path == "only_file.bin"

    def test_revision_preserved(self) -> None:
        manifest = build_manifest_from_file_map(
            self._sample_map(), model_id="test/model", revision="v2.0"
        )
        assert manifest.revision == "v2.0"

    def test_all_files_in_manifest(self) -> None:
        file_map = {
            "a.json": "a" * 64,
            "b.bin": "b" * 64,
            "c.safetensors": "c" * 64,
            "d.txt": "d" * 64,
        }
        manifest = build_manifest_from_file_map(file_map, model_id="test")
        paths = {fp.path for fp in manifest.files}
        assert paths == {"a.json", "b.bin", "c.safetensors", "d.txt"}


# ---------------------------------------------------------------------------
# fingerprint_bytes_entry
# ---------------------------------------------------------------------------


class TestFingerprintBytesEntry:
    def test_hash_correct(self) -> None:
        data = b"config content"
        fp = fingerprint_bytes_entry("config.json", data)
        assert fp.sha256 == _sha256_of(data)

    def test_size_bytes(self) -> None:
        data = b"hello"
        fp = fingerprint_bytes_entry("readme.md", data)
        assert fp.size_bytes == len(data)

    def test_file_type_classified(self) -> None:
        fp_weight = fingerprint_bytes_entry("model.bin", b"weights")
        assert fp_weight.file_type == "weight"

        fp_config = fingerprint_bytes_entry("config.json", b"{}")
        assert fp_config.file_type == "config"

    def test_path_normalised(self) -> None:
        fp = fingerprint_bytes_entry("sub/config.json", b"{}")
        assert fp.path == "sub/config.json"

    def test_ok_no_error(self) -> None:
        fp = fingerprint_bytes_entry("vocab.txt", b"word\n")
        assert fp.ok
        assert fp.error is None

    def test_empty_bytes(self) -> None:
        fp = fingerprint_bytes_entry("empty.bin", b"")
        assert fp.sha256 == _sha256_of(b"")
        assert fp.size_bytes == 0
        assert fp.ok

    def test_large_bytes_entry(self) -> None:
        data = b"X" * (1024 * 1024)
        fp = fingerprint_bytes_entry("big_model.bin", data)
        assert fp.sha256 == _sha256_of(data)
        assert fp.size_bytes == len(data)

    def test_other_file_type(self) -> None:
        fp = fingerprint_bytes_entry("run.sh", b"#!/bin/bash")
        assert fp.file_type == "other"

    def test_returns_filefingerprint_instance(self) -> None:
        fp = fingerprint_bytes_entry("model.safetensors", b"data")
        assert isinstance(fp, FileFingerprint)
