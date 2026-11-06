"""Offline FakeGCS tests for ``storage.create_only_upload`` (no temp, no copy, no delete)."""

from __future__ import annotations

from pathlib import Path

import pytest
from google.api_core.exceptions import Forbidden, PreconditionFailed

from berlin_lst_downscaling.data.io import storage


class _FakeBlob:
    def __init__(self, bucket: _FakeBucket, name: str) -> None:
        self.bucket, self.name = bucket, name

    def exists(self) -> bool:
        return False if self.bucket.race else self.name in self.bucket.objects

    def upload_from_filename(
        self, filename: str, *, if_generation_match: int | None = None, **kw: object
    ) -> None:
        self.bucket.calls.append(("upload_from_filename", self.name, if_generation_match))
        if self.bucket.fail_upload is not None:
            raise self.bucket.fail_upload
        if if_generation_match == 0 and self.name in self.bucket.objects:
            raise PreconditionFailed("exists")
        self.bucket.objects[self.name] = Path(filename).read_bytes()

    def delete(self, *a: object, **k: object) -> None:
        self.bucket.calls.append(("delete", self.name, None))
        raise Forbidden("no delete")


class _FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, str | None, int | None]] = []
        self.fail_upload: Exception | None = None
        self.race = False

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self, name)

    def copy_blob(self, *a: object, **k: object) -> None:
        self.calls.append(("copy_blob", None, None))


class _FakeClient:
    def __init__(self) -> None:
        self.buckets: dict[str, _FakeBucket] = {}

    def bucket(self, name: str) -> _FakeBucket:
        return self.buckets.setdefault(name, _FakeBucket())


@pytest.fixture
def bucket(monkeypatch: pytest.MonkeyPatch) -> _FakeBucket:
    client = _FakeClient()
    monkeypatch.setattr(storage, "_gcs_client", lambda: client)
    return client.bucket("b")


@pytest.fixture
def local(tmp_path: Path) -> Path:
    path = tmp_path / "x.bin"
    path.write_bytes(b"new-bytes")
    return path


def _ops(bucket: _FakeBucket) -> list[str]:
    return [call[0] for call in bucket.calls]


def test_create_only_upload_gcs_success_makes_no_delete_or_copy(
    bucket: _FakeBucket, local: Path
) -> None:
    assert storage.create_only_upload(local, "gs://b/p/x.bin") == "gs://b/p/x.bin"
    assert bucket.objects["p/x.bin"] == b"new-bytes"
    assert bucket.calls == [("upload_from_filename", "p/x.bin", 0)]
    assert not any(".tmp/" in key for key in bucket.objects)


def test_create_only_upload_gcs_existing_object_raises_and_keeps_bytes(
    bucket: _FakeBucket, local: Path
) -> None:
    bucket.objects["p/x.bin"] = b"old"
    with pytest.raises(FileExistsError):
        storage.create_only_upload(local, "gs://b/p/x.bin")
    assert bucket.objects["p/x.bin"] == b"old"
    assert bucket.calls == []


def test_create_only_upload_gcs_precondition_race_propagates(
    bucket: _FakeBucket, local: Path
) -> None:
    bucket.race = True
    bucket.objects["p/x.bin"] = b"old"
    with pytest.raises(PreconditionFailed):
        storage.create_only_upload(local, "gs://b/p/x.bin")
    assert bucket.objects["p/x.bin"] == b"old"
    assert "delete" not in _ops(bucket)


def test_create_only_upload_gcs_upload_error_propagates(bucket: _FakeBucket, local: Path) -> None:
    bucket.fail_upload = Forbidden("denied")
    with pytest.raises(Forbidden):
        storage.create_only_upload(local, "gs://b/p/x.bin")
    assert "delete" not in _ops(bucket)
    assert "p/x.bin" not in bucket.objects


def test_create_only_upload_missing_source_raises(bucket: _FakeBucket, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        storage.create_only_upload(tmp_path / "missing.bin", "gs://b/p/x.bin")
    assert bucket.calls == []


def test_create_only_upload_local_refuses_overwrite(tmp_path: Path, local: Path) -> None:
    dst = tmp_path / "out" / "x.bin"
    storage.create_only_upload(local, str(dst))
    assert dst.read_bytes() == b"new-bytes"
    other = tmp_path / "other.bin"
    other.write_bytes(b"other")
    with pytest.raises(FileExistsError):
        storage.create_only_upload(other, str(dst))
    assert dst.read_bytes() == b"new-bytes"
