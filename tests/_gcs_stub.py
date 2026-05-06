"""In-memory google-cloud-storage stub used by viewer + sync tests.

Install before importing modules that do `from google.cloud import storage`:

    from tests._gcs_stub import install_fake_storage
    client = install_fake_storage()        # injects sys.modules entry
    client.bucket("b").blobs["foo"] = b"x" # seed data
"""
from __future__ import annotations

import io
import sys
import types


class FakeBucket:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blobs: dict[str, bytes] = {}

    def blob(self, name: str) -> "FakeBlob":
        return FakeBlob(self, name)

    def delete_blob(self, name: str, client=None) -> None:
        self.blobs.pop(name, None)


class FakeBlob:
    def __init__(self, bucket: FakeBucket, name: str) -> None:
        self._bucket = bucket
        self.name = name

    @property
    def size(self) -> int | None:
        data = self._bucket.blobs.get(self.name)
        return len(data) if data is not None else None

    def exists(self, client=None) -> bool:
        return self.name in self._bucket.blobs

    def reload(self, client=None) -> None:
        if self.name not in self._bucket.blobs:
            raise FileNotFoundError(self.name)

    def download_as_text(self) -> str:
        return self._bucket.blobs[self.name].decode("utf-8")

    def download_as_bytes(self) -> bytes:
        return self._bucket.blobs[self.name]

    def upload_from_filename(self, path: str, client=None) -> None:
        with open(path, "rb") as f:
            self._bucket.blobs[self.name] = f.read()

    def upload_from_string(self, data, content_type: str | None = None,
                           client=None) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._bucket.blobs[self.name] = data

    def open(self, mode: str):
        if mode != "rb":
            raise ValueError(f"only 'rb' supported, got {mode!r}")
        return io.BytesIO(self._bucket.blobs[self.name])


class _FakeIterator:
    def __init__(self, blobs, prefixes) -> None:
        self._blobs = list(blobs)
        self.prefixes = list(prefixes)

    def __iter__(self):
        return iter(self._blobs)


class FakeClient:
    def __init__(self) -> None:
        self._buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        b = self._buckets.get(name)
        if b is None:
            b = FakeBucket(name)
            self._buckets[name] = b
        return b

    def list_blobs(self, bucket: FakeBucket, prefix: str = "",
                   delimiter: str | None = None,
                   max_results: int | None = None) -> _FakeIterator:
        names = sorted(bucket.blobs)
        prefix = prefix or ""
        if delimiter is None:
            matched = [n for n in names if n.startswith(prefix)]
            if max_results is not None:
                matched = matched[:max_results]
            return _FakeIterator([FakeBlob(bucket, n) for n in matched], [])
        result_names: list[str] = []
        result_prefixes: set[str] = set()
        for n in names:
            if not n.startswith(prefix):
                continue
            tail = n[len(prefix):]
            if delimiter in tail:
                head = tail.split(delimiter, 1)[0]
                result_prefixes.add(prefix + head + delimiter)
            else:
                result_names.append(n)
        return _FakeIterator(
            [FakeBlob(bucket, n) for n in result_names],
            sorted(result_prefixes),
        )


def install_fake_storage(client: FakeClient | None = None) -> FakeClient:
    """Inject a fake `google.cloud.storage` module and return its Client.

    Idempotent: if a fake is already installed, returns the existing client
    so multiple test modules loaded into the same interpreter share state.
    """
    existing = sys.modules.get("google.cloud.storage")
    if existing is not None and hasattr(existing, "_fake_client"):
        return existing._fake_client  # type: ignore[attr-defined]
    chosen = client or FakeClient()
    if "google" not in sys.modules:
        sys.modules["google"] = types.ModuleType("google")
    if "google.cloud" not in sys.modules:
        sys.modules["google.cloud"] = types.ModuleType("google.cloud")
    storage_mod = types.ModuleType("google.cloud.storage")
    storage_mod.Client = lambda *a, **k: chosen
    storage_mod._fake_client = chosen  # type: ignore[attr-defined]
    sys.modules["google.cloud.storage"] = storage_mod
    sys.modules["google.cloud"].storage = storage_mod  # type: ignore[attr-defined]
    return chosen


def uninstall_fake_storage() -> None:
    for name in ("google.cloud.storage", "google.cloud", "google"):
        sys.modules.pop(name, None)
