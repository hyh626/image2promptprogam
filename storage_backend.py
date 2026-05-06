"""Read-only storage backend used by view_eval_results.py and check_eval_storage.py.

A `Backend` is anything that can list directories and read files by string
relative paths. Concrete implementations:

- `LocalBackend(root: Path)` — backed by the filesystem.
- `GCSBackend(bucket, prefix)` — backed by google-cloud-storage with a
  small TTL cache for metadata calls. The SDK is imported lazily so the
  module costs nothing when not used.

`make_backend(root)` dispatches on `gs://` prefix vs. a local path.
"""
from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from threading import RLock
from typing import Any, Iterator


# --------------------------- abstract base ---------------------------


class Backend:
    """Read-only storage backend. Paths are forward-slash strings relative to root."""

    root_label: str = ""

    def list_dir(self, rel: str) -> tuple[list[dict], list[dict]]:
        raise NotImplementedError

    def is_file(self, rel: str) -> bool:
        raise NotImplementedError

    def is_dir(self, rel: str) -> bool:
        raise NotImplementedError

    def exists(self, rel: str) -> bool:
        return self.is_file(rel) or self.is_dir(rel)

    def read_text(self, rel: str) -> str:
        raise NotImplementedError

    def read_bytes(self, rel: str) -> bytes:
        # Default impl: collect from stream(). Concrete classes can override.
        chunks: list[bytes] = []
        for chunk in self.stream(rel):
            chunks.append(chunk)
        return b"".join(chunks)

    def file_size(self, rel: str) -> int:
        raise NotImplementedError

    def stream(self, rel: str) -> Iterator[bytes]:
        raise NotImplementedError

    def content_type(self, rel: str) -> str:
        return mimetypes.guess_type(rel)[0] or "application/octet-stream"

    def format_path(self, rel: str) -> str:
        """Human-readable absolute path, used in error messages."""
        return f"{self.root_label}/{rel}" if rel else self.root_label


# --------------------------- helpers ---------------------------


def _clean_rel(rel: str) -> str:
    """Reject path traversal and normalize to forward slashes."""
    cleaned = (rel or "").replace("\\", "/").strip("/")
    if not cleaned:
        return ""
    parts = cleaned.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise PermissionError(f"path escapes root or contains dotted segments: {rel!r}")
    return "/".join(parts)


# --------------------------- local ---------------------------


class LocalBackend(Backend):
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        if not self._root.is_dir():
            raise FileNotFoundError(f"{self._root} is not a directory")
        self.root_label = self._root.name or str(self._root)
        self._abs_root_str = str(self._root)

    def _abs(self, rel: str) -> Path:
        return self._root / rel if rel else self._root

    def list_dir(self, rel: str) -> tuple[list[dict], list[dict]]:
        target = self._abs(rel)
        subdirs: list[dict] = []
        files: list[dict] = []
        if not target.is_dir():
            return subdirs, files
        try:
            children = sorted(target.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))
        except PermissionError:
            return subdirs, files
        for child in children:
            if child.name.startswith("."):
                continue
            child_rel = f"{rel}/{child.name}" if rel else child.name
            if child.is_dir():
                subdirs.append({"name": child.name, "path": child_rel})
            else:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                files.append({"name": child.name, "path": child_rel, "size": size})
        return subdirs, files

    def is_file(self, rel: str) -> bool:
        return self._abs(rel).is_file()

    def is_dir(self, rel: str) -> bool:
        return self._abs(rel).is_dir()

    def read_text(self, rel: str) -> str:
        return self._abs(rel).read_text(encoding="utf-8")

    def read_bytes(self, rel: str) -> bytes:
        return self._abs(rel).read_bytes()

    def file_size(self, rel: str) -> int:
        return self._abs(rel).stat().st_size

    def stream(self, rel: str) -> Iterator[bytes]:
        with self._abs(rel).open("rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                yield chunk

    def format_path(self, rel: str) -> str:
        return str(self._abs(rel))


# --------------------------- gcs ---------------------------


class GCSBackend(Backend):
    """Direct read-through GCS backend with a small TTL cache for metadata."""

    def __init__(self, bucket: str, prefix: str, cache_ttl: float = 30.0) -> None:
        try:
            from google.cloud import storage  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "google-cloud-storage is required for gs:// roots. "
                "Install with `pip install google-cloud-storage`."
            ) from exc
        self._storage = storage
        self._client = storage.Client()
        self._bucket_name = bucket
        self._bucket = self._client.bucket(bucket)
        self._prefix = prefix.strip("/")
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._lock = RLock()
        self._ttl = cache_ttl
        self.root_label = (
            f"gs://{bucket}" + (f"/{self._prefix}" if self._prefix else "")
        )

    def _full(self, rel: str) -> str:
        rel = rel.strip("/")
        if self._prefix and rel:
            return f"{self._prefix}/{rel}"
        return self._prefix or rel

    def _cache_get(self, key: tuple[str, str]) -> Any:
        with self._lock:
            v = self._cache.get(key)
            if v is None:
                return None
            ts, val = v
            if time.time() - ts > self._ttl:
                self._cache.pop(key, None)
                return None
            return val

    def _cache_set(self, key: tuple[str, str], val: Any) -> None:
        with self._lock:
            self._cache[key] = (time.time(), val)

    def list_dir(self, rel: str) -> tuple[list[dict], list[dict]]:
        key = ("list", rel)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        full = self._full(rel)
        list_prefix = (full + "/") if full else ""
        iterator = self._client.list_blobs(self._bucket, prefix=list_prefix, delimiter="/")
        # Exhausting blobs is required before iterator.prefixes is populated.
        blobs = list(iterator)
        prefixes = list(iterator.prefixes)

        subdirs: list[dict] = []
        for p in sorted(prefixes):
            name = p[len(list_prefix):].rstrip("/")
            if not name or name.startswith("."):
                continue
            sub_rel = f"{rel}/{name}" if rel else name
            subdirs.append({"name": name, "path": sub_rel})
            self._cache_set(("isdir", sub_rel), True)

        files: list[dict] = []
        for b in blobs:
            name = b.name[len(list_prefix):]
            # Skip directory placeholder objects ("foo/") and dotfiles.
            if not name or name.endswith("/") or name.startswith("."):
                continue
            if "/" in name:
                continue
            file_rel = f"{rel}/{name}" if rel else name
            size = b.size or 0
            files.append({"name": name, "path": file_rel, "size": size})
            self._cache_set(("isfile", file_rel), True)
            self._cache_set(("size", file_rel), size)

        result = (subdirs, files)
        self._cache_set(key, result)
        return result

    def is_file(self, rel: str) -> bool:
        key = ("isfile", rel)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        if not rel:
            self._cache_set(key, False)
            return False
        blob = self._bucket.blob(self._full(rel))
        exists = blob.exists(self._client)
        self._cache_set(key, exists)
        return exists

    def is_dir(self, rel: str) -> bool:
        key = ("isdir", rel)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        full = self._full(rel)
        list_prefix = (full + "/") if full else ""
        # Cheap probe: does any blob exist under this prefix?
        probe = self._client.list_blobs(self._bucket, prefix=list_prefix, max_results=1)
        result = any(True for _ in probe)
        if not result and not rel:
            result = True  # root always exists
        self._cache_set(key, result)
        return result

    def read_text(self, rel: str) -> str:
        blob = self._bucket.blob(self._full(rel))
        return blob.download_as_text()

    def read_bytes(self, rel: str) -> bytes:
        blob = self._bucket.blob(self._full(rel))
        return blob.download_as_bytes()

    def file_size(self, rel: str) -> int:
        key = ("size", rel)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        blob = self._bucket.blob(self._full(rel))
        blob.reload(client=self._client)
        size = blob.size or 0
        self._cache_set(key, size)
        return size

    def stream(self, rel: str) -> Iterator[bytes]:
        blob = self._bucket.blob(self._full(rel))
        with blob.open("rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                yield chunk

    def format_path(self, rel: str) -> str:
        full = self._full(rel)
        return f"gs://{self._bucket_name}/{full}" if full else f"gs://{self._bucket_name}"


# --------------------------- factory ---------------------------


def make_backend(root: str | Path, gcs_cache_ttl: float = 30.0) -> Backend:
    """Dispatch on `gs://` prefix vs. local path."""
    root_str = str(root)
    if root_str.startswith("gs://"):
        without_scheme = root_str[len("gs://"):]
        if "/" in without_scheme:
            bucket, prefix = without_scheme.split("/", 1)
        else:
            bucket, prefix = without_scheme, ""
        if not bucket:
            raise ValueError(f"missing bucket in GCS root: {root_str!r}")
        return GCSBackend(bucket, prefix, cache_ttl=gcs_cache_ttl)
    return LocalBackend(Path(root_str))
