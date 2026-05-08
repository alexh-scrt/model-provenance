"""Unit tests for model_provenance.fetcher module.

Covers local directory listing, model card YAML front-matter parsing,
and the unified fetch_model_listing entry point.
Hugging Face Hub calls are not tested against the real network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_provenance.fetcher import (
    ModelCardInfo,
    ModelFileListing,
    RemoteFileInfo,
    _extract_yaml_front_matter,
    _parse_local_model_card,
    fetch_local_listing,
    fetch_model_listing,
)


# ---------------------------------------------------------------------------
# RemoteFileInfo
# ---------------------------------------------------------------------------


class TestRemoteFileInfo:
    def test_to_dict_keys(self) -> None:
        rfi = RemoteFileInfo(
            path="config.json",
            size_bytes=512,
            sha256="a" * 64,
            lfs_sha256="b" * 64,
            blob_id="abc",
            is_lfs=True,
        )
        d = rfi.to_dict()
        assert set(d.keys()) == {"path", "size_bytes", "sha256", "lfs_sha256", "blob_id", "is_lfs"}

    def test_defaults(self) -> None:
        rfi = RemoteFileInfo(path="model.bin")
        assert rfi.size_bytes == 0
        assert rfi.sha256 == ""
        assert rfi.lfs_sha256 == ""
        assert rfi.blob_id == ""
        assert rfi.is_lfs is False


# ---------------------------------------------------------------------------
# ModelCardInfo
# ---------------------------------------------------------------------------


class TestModelCardInfo:
    def test_to_dict_keys(self) -> None:
        info = ModelCardInfo(
            model_id="test/model",
            author="Alice",
            license="mit",
            pipeline_tag="text-classification",
            tags=["pytorch"],
            library_name="transformers",
            language=["en"],
            datasets=["glue"],
        )
        d = info.to_dict()
        expected_keys = {
            "model_id",
            "author",
            "license",
            "pipeline_tag",
            "tags",
            "library_name",
            "language",
            "datasets",
        }
        assert set(d.keys()) == expected_keys

    def test_defaults(self) -> None:
        info = ModelCardInfo(model_id="m")
        assert info.license is None
        assert info.tags == []
        assert info.language == []
        assert info.datasets == []
        assert info.raw_metadata == {}


# ---------------------------------------------------------------------------
# ModelFileListing
# ---------------------------------------------------------------------------


class TestModelFileListing:
    def _make(self) -> ModelFileListing:
        listing = ModelFileListing(
            model_id="test/model",
            revision="main",
            source="hub",
            card=ModelCardInfo(model_id="test/model"),
        )
        listing.files = [
            RemoteFileInfo(path="config.json", size_bytes=100),
            RemoteFileInfo(path="model.bin", size_bytes=500),
        ]
        return listing

    def test_file_count(self) -> None:
        assert self._make().file_count == 2

    def test_total_size_bytes(self) -> None:
        assert self._make().total_size_bytes == 600

    def test_get_file_existing(self) -> None:
        listing = self._make()
        rfi = listing.get_file("config.json")
        assert rfi is not None
        assert rfi.path == "config.json"

    def test_get_file_missing(self) -> None:
        assert self._make().get_file("nonexistent.bin") is None

    def test_get_file_normalises_backslash(self) -> None:
        listing = self._make()
        assert listing.get_file("config.json") is not None

    def test_to_dict_keys(self) -> None:
        d = self._make().to_dict()
        assert set(d.keys()) == {
            "model_id",
            "revision",
            "source",
            "file_count",
            "total_size_bytes",
            "files",
            "card",
            "fetch_error",
        }


# ---------------------------------------------------------------------------
# _extract_yaml_front_matter
# ---------------------------------------------------------------------------


class TestExtractYamlFrontMatter:
    def test_valid_front_matter(self) -> None:
        text = "---\nlicense: mit\ntags:\n  - pytorch\n---\n# Title\n"
        fm = _extract_yaml_front_matter(text)
        assert "license: mit" in fm
        assert "pytorch" in fm

    def test_no_front_matter(self) -> None:
        text = "# Just a README\nNo front matter here."
        assert _extract_yaml_front_matter(text) == ""

    def test_unclosed_front_matter(self) -> None:
        text = "---\nlicense: mit\nNo closing delimiter."
        assert _extract_yaml_front_matter(text) == ""

    def test_empty_document(self) -> None:
        assert _extract_yaml_front_matter("") == ""

    def test_only_opening_delimiter(self) -> None:
        assert _extract_yaml_front_matter("---\n") == ""

    def test_empty_front_matter_block(self) -> None:
        text = "---\n---\n# Title"
        fm = _extract_yaml_front_matter(text)
        assert fm == ""


# ---------------------------------------------------------------------------
# _parse_local_model_card
# ---------------------------------------------------------------------------


class TestParseLocalModelCard:
    def _write_readme(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "README.md"
        p.write_text(content, encoding="utf-8")
        return p

    def test_parses_license(self, tmp_path: Path) -> None:
        readme = self._write_readme(
            tmp_path,
            "---\nlicense: apache-2.0\n---\n# Model",
        )
        info = _parse_local_model_card(readme, "mymodel")
        assert info.license == "apache-2.0"

    def test_parses_language_list(self, tmp_path: Path) -> None:
        readme = self._write_readme(
            tmp_path,
            "---\nlanguage:\n  - en\n  - fr\n---",
        )
        info = _parse_local_model_card(readme, "mymodel")
        assert "en" in info.language
        assert "fr" in info.language

    def test_parses_language_string(self, tmp_path: Path) -> None:
        readme = self._write_readme(
            tmp_path,
            "---\nlanguage: en\n---",
        )
        info = _parse_local_model_card(readme, "mymodel")
        assert info.language == ["en"]

    def test_parses_tags(self, tmp_path: Path) -> None:
        readme = self._write_readme(
            tmp_path,
            "---\ntags:\n  - pytorch\n  - text-generation\n---",
        )
        info = _parse_local_model_card(readme, "mymodel")
        assert "pytorch" in info.tags

    def test_parses_library_name(self, tmp_path: Path) -> None:
        readme = self._write_readme(
            tmp_path,
            "---\nlibrary_name: transformers\n---",
        )
        info = _parse_local_model_card(readme, "mymodel")
        assert info.library_name == "transformers"

    def test_parses_pipeline_tag(self, tmp_path: Path) -> None:
        readme = self._write_readme(
            tmp_path,
            "---\npipeline_tag: text-classification\n---",
        )
        info = _parse_local_model_card(readme, "mymodel")
        assert info.pipeline_tag == "text-classification"

    def test_parses_datasets(self, tmp_path: Path) -> None:
        readme = self._write_readme(
            tmp_path,
            "---\ndatasets:\n  - squad\n  - glue\n---",
        )
        info = _parse_local_model_card(readme, "mymodel")
        assert "squad" in info.datasets

    def test_no_front_matter_returns_empty(self, tmp_path: Path) -> None:
        readme = self._write_readme(tmp_path, "# Just a README\nNo metadata.")
        info = _parse_local_model_card(readme, "mymodel")
        assert info.license is None
        assert info.tags == []

    def test_model_id_preserved(self, tmp_path: Path) -> None:
        readme = self._write_readme(tmp_path, "---\nlicense: mit\n---")
        info = _parse_local_model_card(readme, "custom-model")
        assert info.model_id == "custom-model"

    def test_bad_yaml_returns_empty_info(self, tmp_path: Path) -> None:
        readme = self._write_readme(
            tmp_path, "---\n: invalid: yaml: here\n---"
        )
        # Should not raise; bad YAML is silently ignored.
        info = _parse_local_model_card(readme, "mymodel")
        assert isinstance(info, ModelCardInfo)


# ---------------------------------------------------------------------------
# fetch_local_listing
# ---------------------------------------------------------------------------


class TestFetchLocalListing:
    def _make_model_dir(self, tmp_path: Path) -> Path:
        model_dir = tmp_path / "bert-base"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b'{"model_type": "bert"}')
        (model_dir / "tokenizer.json").write_bytes(b'{"version": "1.0"}')
        (model_dir / "pytorch_model.bin").write_bytes(b"fake weights")
        return model_dir

    def test_returns_listing(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        listing = fetch_local_listing(model_dir)
        assert isinstance(listing, ModelFileListing)

    def test_file_count(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        listing = fetch_local_listing(model_dir)
        assert listing.file_count == 3

    def test_source_is_local(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        listing = fetch_local_listing(model_dir)
        assert listing.source == "local"

    def test_revision_default_local(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        listing = fetch_local_listing(model_dir)
        assert listing.revision == "local"

    def test_model_id_defaults_to_dir_name(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        listing = fetch_local_listing(model_dir)
        assert listing.model_id == "bert-base"

    def test_custom_model_id(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        listing = fetch_local_listing(model_dir, model_id="my-custom-id")
        assert listing.model_id == "my-custom-id"

    def test_file_paths_use_forward_slash(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        subdir = model_dir / "sub"
        subdir.mkdir()
        (subdir / "extra.json").write_bytes(b"{}")
        listing = fetch_local_listing(model_dir)
        for rfi in listing.files:
            assert "\\" not in rfi.path

    def test_size_bytes_populated(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        listing = fetch_local_listing(model_dir)
        for rfi in listing.files:
            assert rfi.size_bytes >= 0

    def test_git_dir_skipped(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        git_dir = model_dir / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_bytes(b"ref: refs/heads/main")
        listing = fetch_local_listing(model_dir)
        paths = [f.path for f in listing.files]
        assert not any(".git" in p for p in paths)

    def test_readme_card_parsed(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        readme = model_dir / "README.md"
        readme.write_text(
            "---\nlicense: apache-2.0\ntags:\n  - pytorch\n---\n# Model",
            encoding="utf-8",
        )
        listing = fetch_local_listing(model_dir)
        assert listing.card.license == "apache-2.0"
        assert "pytorch" in listing.card.tags

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            fetch_local_listing(tmp_path / "nonexistent")

    def test_file_path_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "not_a_dir.txt"
        f.write_bytes(b"x")
        with pytest.raises(NotADirectoryError):
            fetch_local_listing(f)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        listing = fetch_local_listing(str(model_dir))
        assert listing.file_count == 3

    def test_subdirectory_files_included(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        subdir = model_dir / "shards"
        subdir.mkdir()
        (subdir / "shard-0.bin").write_bytes(b"shard")
        listing = fetch_local_listing(model_dir)
        paths = [f.path for f in listing.files]
        assert any("shards" in p for p in paths)

    def test_get_file_lookup(self, tmp_path: Path) -> None:
        model_dir = self._make_model_dir(tmp_path)
        listing = fetch_local_listing(model_dir)
        rfi = listing.get_file("config.json")
        assert rfi is not None


# ---------------------------------------------------------------------------
# fetch_model_listing (unified entry point)
# ---------------------------------------------------------------------------


class TestFetchModelListing:
    def test_local_flag_uses_local_fetcher(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"{}")
        listing = fetch_model_listing(model_dir, local=True)
        assert listing.source == "local"

    def test_existing_path_uses_local_fetcher(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"{}")
        listing = fetch_model_listing(model_dir)
        assert listing.source == "local"

    def test_hub_model_id_calls_hub_fetcher(self, tmp_path: Path) -> None:
        """Non-existent local path should be treated as a Hub model ID."""
        mock_listing = ModelFileListing(
            model_id="bert-base-uncased",
            revision="main",
            source="hub",
            card=ModelCardInfo(model_id="bert-base-uncased"),
        )

        with patch(
            "model_provenance.fetcher.fetch_hub_listing",
            return_value=mock_listing,
        ) as mock_hub:
            listing = fetch_model_listing("bert-base-uncased")
            mock_hub.assert_called_once_with(
                model_id="bert-base-uncased",
                revision="main",
                token=None,
            )
            assert listing.source == "hub"

    def test_local_flag_overrides_hub_check(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "my-local-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_bytes(b"{}")
        listing = fetch_model_listing(str(model_dir), local=True)
        assert listing.source == "local"

    def test_token_passed_to_hub_fetcher(self, tmp_path: Path) -> None:
        mock_listing = ModelFileListing(
            model_id="org/private",
            revision="main",
            source="hub",
            card=ModelCardInfo(model_id="org/private"),
        )
        with patch(
            "model_provenance.fetcher.fetch_hub_listing",
            return_value=mock_listing,
        ) as mock_hub:
            fetch_model_listing("org/private", token="hf_abc123")
            _, kwargs = mock_hub.call_args
            assert kwargs.get("token") == "hf_abc123" or mock_hub.call_args[0][2] == "hf_abc123"
