import pytest

from ml.models.registry import ModelRegistry


def test_next_version_is_v1_when_empty(tmp_path):
    registry = ModelRegistry(tmp_path)

    assert registry.next_version() == "v1"


def test_next_version_is_v1_when_directory_does_not_exist(tmp_path):
    registry = ModelRegistry(tmp_path / "does_not_exist")

    assert registry.next_version() == "v1"


def test_next_version_increments_past_existing_versions(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    registry = ModelRegistry(tmp_path)

    assert registry.next_version() == "v3"


def test_next_version_ignores_non_version_directories(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "preprocessing").mkdir()
    registry = ModelRegistry(tmp_path)

    assert registry.next_version() == "v2"


def test_latest_version_raises_when_no_versions_exist(tmp_path):
    registry = ModelRegistry(tmp_path)

    with pytest.raises(FileNotFoundError):
        registry.latest_version()


def test_latest_version_returns_highest_number(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v3").mkdir()
    (tmp_path / "v2").mkdir()
    registry = ModelRegistry(tmp_path)

    assert registry.latest_version() == "v3"


def test_resolve_with_explicit_version(tmp_path):
    (tmp_path / "v1").mkdir()
    registry = ModelRegistry(tmp_path)

    assert registry.resolve("v1") == tmp_path / "v1"


def test_resolve_without_version_uses_latest(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    registry = ModelRegistry(tmp_path)

    assert registry.resolve() == tmp_path / "v2"


def test_resolve_raises_for_missing_version(tmp_path):
    registry = ModelRegistry(tmp_path)

    with pytest.raises(FileNotFoundError):
        registry.resolve("v99")
