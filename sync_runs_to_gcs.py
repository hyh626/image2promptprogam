#!/usr/bin/env python3
"""Sync local experiment artifacts to GCS so the viewer can read them remotely.

Mirrors the canonical layout from EVAL_STORAGE_SCHEMA.md to a `gs://` URI:

    <local-root>/experiments/runs/<run_id>/...   ->   <gcs-root>/experiments/runs/<run_id>/...
    <local-root>/experiments/leader/...          ->   <gcs-root>/experiments/leader/...
    <local-root>/experiments/logbook.md          ->   <gcs-root>/experiments/logbook.md
    <local-root>/eval_data/images/manifest.json  ->   <gcs-root>/eval_data/images/manifest.json

By default `eval_data/images/<split>/*.png` are NOT uploaded — image bytes
are usually already in a separate dataset bucket. Pass --include-images to
upload them too. `holdout/` is always excluded.

Usage:
    python sync_runs_to_gcs.py --src . --dst gs://image2promptdata/experiments
    python sync_runs_to_gcs.py --src . --dst gs://bucket/path --runs <run_id_a> <run_id_b>
    python sync_runs_to_gcs.py --src . --dst gs://... --dry-run

By default only files that are missing or have a different size remotely
are uploaded. Use --force to re-upload everything. Use --delete to remove
remote files that no longer exist locally (scoped to synced subtrees).

Concurrency: --workers controls parallel uploads (default 8). Ctrl-C is
safe; in-flight uploads finish, no partial files are left.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# --------------------------- targets ---------------------------

SYNC_TARGETS_ALWAYS = (
    "experiments/runs",
    "experiments/leader",
    "experiments/logbook.md",
    "eval_data/images/manifest.json",
)

EXCLUDE_PATTERNS = (
    "*/.git/*",
    "*/__pycache__/*",
    "*/.DS_Store",
    "*/cache/*",
    "*/weights/*",
    "*/.venv/*",
    "*.pyc",
)

EXCLUDE_PATHS_PREFIXES = (
    "eval_data/images/holdout/",
)


# --------------------------- model ---------------------------


@dataclass
class Plan:
    uploads: list[tuple[Path, str]]      # (local_path, remote_object_name)
    skips: list[tuple[Path, str, str]]   # (local_path, remote_object_name, reason)
    deletes: list[str]                   # remote_object_name


@dataclass
class Stats:
    uploaded: int = 0
    skipped: int = 0
    deleted: int = 0
    bytes_up: int = 0
    failures: int = 0


# --------------------------- helpers ---------------------------


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"--dst must start with gs://, got {uri!r}")
    rest = uri[len("gs://"):]
    if rest.startswith("/") or not rest:
        raise ValueError(f"missing bucket in {uri!r}")
    rest = rest.rstrip("/")
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
    else:
        bucket, prefix = rest, ""
    if not bucket:
        raise ValueError(f"missing bucket in {uri!r}")
    return bucket, prefix


def is_excluded(rel_path: str) -> bool:
    if any(rel_path.startswith(p) for p in EXCLUDE_PATHS_PREFIXES):
        return True
    return any(fnmatch.fnmatch("/" + rel_path, p) for p in EXCLUDE_PATTERNS)


def iter_files(src: Path, sub: str) -> list[tuple[Path, str]]:
    """Return [(local_path, rel_posix_path)] under src/sub."""
    target = src / sub
    out: list[tuple[Path, str]] = []
    if target.is_file():
        rel = target.relative_to(src).as_posix()
        if not is_excluded(rel):
            out.append((target, rel))
        return out
    if not target.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(target):
        # Skip hidden dirs and obvious caches early
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", ".venv")]
        for fname in filenames:
            p = Path(dirpath) / fname
            rel = p.relative_to(src).as_posix()
            if is_excluded(rel):
                continue
            out.append((p, rel))
    return out


def filter_by_runs(entries: list[tuple[Path, str]], runs: list[str]) -> list[tuple[Path, str]]:
    if not runs:
        return entries
    keep: list[tuple[Path, str]] = []
    run_prefixes = tuple(f"experiments/runs/{r}/" for r in runs)
    run_files = tuple(f"experiments/runs/{r}" for r in runs)
    for p, rel in entries:
        if rel.startswith("experiments/runs/"):
            if any(rel.startswith(rp) or rel == rf for rp, rf in zip(run_prefixes, run_files)):
                keep.append((p, rel))
        else:
            keep.append((p, rel))
    return keep


def remote_object(prefix: str, rel: str) -> str:
    if prefix:
        return f"{prefix.strip('/')}/{rel}"
    return rel


def remote_index(client, bucket, prefix: str) -> dict[str, int]:
    """Return {object_name: size} under prefix, used to skip unchanged files."""
    list_prefix = (prefix.strip("/") + "/") if prefix.strip("/") else ""
    out: dict[str, int] = {}
    for blob in client.list_blobs(bucket, prefix=list_prefix):
        out[blob.name] = blob.size or 0
    return out


def md5_local(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------- planning ---------------------------


def build_plan(
    src: Path,
    bucket,
    prefix: str,
    client,
    runs: list[str],
    include_images: bool,
    delete_orphans: bool,
    force: bool,
) -> Plan:
    targets = list(SYNC_TARGETS_ALWAYS)
    if include_images:
        targets.append("eval_data/images")

    local_entries: list[tuple[Path, str]] = []
    for sub in targets:
        local_entries.extend(iter_files(src, sub))
    local_entries = filter_by_runs(local_entries, runs)

    # Build a deterministic local map.
    local_map: dict[str, tuple[Path, int]] = {}
    for path, rel in local_entries:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        obj = remote_object(prefix, rel)
        local_map[obj] = (path, size)

    # Snapshot remote.
    if runs:
        remote_objs: dict[str, int] = {}
        seen = set()
        for r in runs:
            run_prefix = remote_object(prefix, f"experiments/runs/{r}/")
            for blob in client.list_blobs(bucket, prefix=run_prefix):
                if blob.name in seen:
                    continue
                seen.add(blob.name)
                remote_objs[blob.name] = blob.size or 0
        # Plus singleton files that the run-scoped sync should still cover.
        for sub in ("experiments/leader", "experiments/logbook.md",
                    "eval_data/images/manifest.json"):
            target_prefix = remote_object(prefix, sub)
            for blob in client.list_blobs(bucket, prefix=target_prefix):
                remote_objs[blob.name] = blob.size or 0
    else:
        remote_objs = remote_index(client, bucket, prefix)

    uploads: list[tuple[Path, str]] = []
    skips: list[tuple[Path, str, str]] = []
    for obj, (path, size) in sorted(local_map.items()):
        rsize = remote_objs.get(obj)
        if force or rsize is None or rsize != size:
            uploads.append((path, obj))
        else:
            skips.append((path, obj, "size match"))

    deletes: list[str] = []
    if delete_orphans:
        # Only delete inside the subtrees we manage.
        scope_prefixes = tuple(remote_object(prefix, t) + "/" for t in targets if not t.endswith(".json") and not t.endswith(".md"))
        scope_files = tuple(remote_object(prefix, t) for t in targets if t.endswith(".json") or t.endswith(".md"))
        if runs:
            scope_prefixes = tuple(remote_object(prefix, f"experiments/runs/{r}/") for r in runs) + scope_prefixes
        for obj in remote_objs:
            in_scope = obj in scope_files or any(obj.startswith(p) for p in scope_prefixes)
            if in_scope and obj not in local_map:
                deletes.append(obj)

    return Plan(uploads=uploads, skips=skips, deletes=sorted(deletes))


# --------------------------- execution ---------------------------


def upload_one(client, bucket, path: Path, obj: str, dry_run: bool) -> tuple[str, int, str | None]:
    if dry_run:
        try:
            return obj, path.stat().st_size, None
        except OSError as exc:
            return obj, 0, str(exc)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return obj, 0, str(exc)
    blob = bucket.blob(obj)
    try:
        blob.upload_from_filename(str(path), client=client)
    except Exception as exc:  # noqa: BLE001
        return obj, size, f"{type(exc).__name__}: {exc}"
    return obj, size, None


def delete_one(client, bucket, obj: str, dry_run: bool) -> tuple[str, str | None]:
    if dry_run:
        return obj, None
    try:
        bucket.delete_blob(obj, client=client)
    except Exception as exc:  # noqa: BLE001
        return obj, f"{type(exc).__name__}: {exc}"
    return obj, None


def run_plan(plan: Plan, client, bucket, dry_run: bool, workers: int, quiet: bool) -> Stats:
    stats = Stats(skipped=len(plan.skips))
    if not plan.uploads and not plan.deletes:
        return stats

    started = time.time()
    if plan.uploads:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(upload_one, client, bucket, p, o, dry_run) for p, o in plan.uploads]
            for fut in concurrent.futures.as_completed(futures):
                obj, size, err = fut.result()
                if err:
                    stats.failures += 1
                    print(f"FAIL {obj}: {err}", file=sys.stderr)
                else:
                    stats.uploaded += 1
                    stats.bytes_up += size
                    if not quiet:
                        verb = "WOULD UPLOAD" if dry_run else "uploaded"
                        print(f"{verb} {obj} ({size} B)")

    if plan.deletes:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(delete_one, client, bucket, o, dry_run) for o in plan.deletes]
            for fut in concurrent.futures.as_completed(futures):
                obj, err = fut.result()
                if err:
                    stats.failures += 1
                    print(f"FAIL delete {obj}: {err}", file=sys.stderr)
                else:
                    stats.deleted += 1
                    if not quiet:
                        verb = "WOULD DELETE" if dry_run else "deleted"
                        print(f"{verb} {obj}")

    elapsed = time.time() - started
    if not quiet:
        print(
            f"--- {'plan' if dry_run else 'sync'} done in {elapsed:.1f}s: "
            f"{stats.uploaded} uploaded, {stats.skipped} unchanged, "
            f"{stats.deleted} deleted, {stats.failures} failed, "
            f"{stats.bytes_up / 1e6:.2f} MB",
            file=sys.stderr,
        )
    return stats


# --------------------------- entry point ---------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=Path.cwd(),
                        help="local repo root (default: cwd)")
    parser.add_argument("--dst", required=True,
                        help="GCS destination, e.g. gs://image2promptdata/experiments")
    parser.add_argument("--runs", nargs="*", default=[],
                        help="restrict run-folder uploads to these run_ids (still syncs leader/logbook/manifest)")
    parser.add_argument("--include-images", action="store_true",
                        help="also upload eval_data/images/{train,eval,val}/* (holdout always excluded)")
    parser.add_argument("--delete", dest="delete_orphans", action="store_true",
                        help="delete remote objects within synced subtrees that no longer exist locally")
    parser.add_argument("--force", action="store_true",
                        help="re-upload even when remote size matches local")
    parser.add_argument("--workers", type=int, default=8, help="parallel upload workers (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would happen without uploading or deleting")
    parser.add_argument("--quiet", action="store_true", help="suppress per-file output")
    args = parser.parse_args(argv)

    src = args.src.resolve()
    if not src.is_dir():
        print(f"--src {src} is not a directory", file=sys.stderr)
        return 1

    try:
        bucket_name, prefix = parse_gcs_uri(args.dst)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        from google.cloud import storage  # type: ignore[import-not-found]
    except ImportError:
        print("google-cloud-storage is required. Install with `pip install google-cloud-storage`.", file=sys.stderr)
        return 1

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    plan = build_plan(
        src=src,
        bucket=bucket,
        prefix=prefix,
        client=client,
        runs=list(args.runs),
        include_images=args.include_images,
        delete_orphans=args.delete_orphans,
        force=args.force,
    )

    if not args.quiet:
        print(
            f"Plan: {len(plan.uploads)} upload(s), {len(plan.skips)} unchanged, "
            f"{len(plan.deletes)} delete(s); src={src}, dst={args.dst}",
            file=sys.stderr,
        )

    stats = run_plan(plan, client, bucket, args.dry_run, args.workers, args.quiet)
    return 0 if stats.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
