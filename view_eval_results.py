#!/usr/bin/env python3
"""Web viewer for image-to-prompt autoresearch experiment results.

Browse a directory tree, see a summary table when entering a folder that
contains experiment runs (per EVAL_STORAGE_SCHEMA.md), and click a run to
visualize its per-image targets, generations, prompts, and scores.

Roots can be local paths or GCS URIs:

    python view_eval_results.py --root /local/repo
    python view_eval_results.py --root gs://image2promptdata/experiments

GCS reads stream through the google-cloud-storage SDK; nothing is copied
to local disk. Authentication uses Application Default Credentials
(`gcloud auth application-default login` or
GOOGLE_APPLICATION_CREDENTIALS).
"""
from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

# --------------------------- backend abstraction ---------------------------


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

    def file_size(self, rel: str) -> int:
        raise NotImplementedError

    def stream(self, rel: str) -> Iterator[bytes]:
        raise NotImplementedError

    def content_type(self, rel: str) -> str:
        return mimetypes.guess_type(rel)[0] or "application/octet-stream"


def _clean_rel(rel: str) -> str:
    """Reject path traversal and normalize to forward slashes."""
    cleaned = (rel or "").replace("\\", "/").strip("/")
    if not cleaned:
        return ""
    parts = cleaned.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise PermissionError(f"path escapes root or contains dotted segments: {rel!r}")
    return "/".join(parts)


class LocalBackend(Backend):
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        if not self._root.is_dir():
            raise FileNotFoundError(f"{self._root} is not a directory")
        self.root_label = self._root.name or str(self._root)

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

    def file_size(self, rel: str) -> int:
        return self._abs(rel).stat().st_size

    def stream(self, rel: str) -> Iterator[bytes]:
        with self._abs(rel).open("rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                yield chunk


class GCSBackend(Backend):
    """Direct read-through GCS backend, with a small TTL cache for metadata calls."""

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
        self._bucket = self._client.bucket(bucket)
        self._prefix = prefix.strip("/")
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}
        self._lock = RLock()
        self._ttl = cache_ttl
        self.root_label = f"gs://{bucket}" + (f"/{self._prefix}" if self._prefix else "")

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


BACKEND: Backend  # set in main()


# --------------------------- view builders ---------------------------


def safe_load_json(rel: str) -> Any:
    if not BACKEND.is_file(rel):
        return None
    try:
        return json.loads(BACKEND.read_text(rel))
    except Exception:
        return None


def classify_dir(rel: str) -> str:
    if BACKEND.is_file(f"{rel}/run.json" if rel else "run.json"):
        return "run"
    if BACKEND.is_dir(rel):
        subdirs, _ = BACKEND.list_dir(rel)
        for s in subdirs:
            child_run_json = f"{s['path']}/run.json"
            if BACKEND.is_file(child_run_json):
                return "runs_container"
    return "dir"


def parent_of(rel: str) -> str:
    if not rel or "/" not in rel:
        return ""
    return rel.rsplit("/", 1)[0]


def find_manifest(start_rel: str) -> str | None:
    cur = start_rel
    while True:
        candidate = f"{cur}/eval_data/images/manifest.json" if cur else "eval_data/images/manifest.json"
        if BACKEND.is_file(candidate):
            return candidate
        if not cur:
            return None
        cur = parent_of(cur)


def load_manifest_for(start_rel: str) -> dict[str, dict[str, Any]]:
    mp = find_manifest(start_rel)
    if not mp:
        return {}
    try:
        data = json.loads(BACKEND.read_text(mp))
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    manifest_dir_rel = parent_of(mp)
    for split_name, entries in (data.get("splits") or {}).items():
        for entry in entries or []:
            if isinstance(entry, dict) and "image_id" in entry:
                merged = dict(entry)
                merged["__split__"] = split_name
                merged["__manifest_dir__"] = manifest_dir_rel
                out[entry["image_id"]] = merged
    return out


def breadcrumb(rel: str) -> list[dict[str, str]]:
    parts = [] if not rel else rel.split("/")
    out = [{"name": BACKEND.root_label or "/", "path": ""}]
    accum = ""
    for part in parts:
        accum = f"{accum}/{part}" if accum else part
        out.append({"name": part, "path": accum})
    return out


def build_summary(container_rel: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subdirs, _ = BACKEND.list_dir(container_rel)
    for child in subdirs:
        run_json_path = f"{child['path']}/run.json"
        if not BACKEND.is_file(run_json_path):
            continue
        run = safe_load_json(run_json_path) or {}
        agg = safe_load_json(f"{child['path']}/aggregate.json") or {}
        gate = safe_load_json(f"{child['path']}/gate.json") or {}
        rows.append({
            "run_id": child["name"],
            "path": child["path"],
            "driver": run.get("driver"),
            "name": run.get("name"),
            "status": run.get("status"),
            "started_at": run.get("started_at"),
            "harness_variant": run.get("harness_variant"),
            "split": run.get("split"),
            "n_images": agg.get("n_images") or len(run.get("image_ids") or []),
            "composite": agg.get("composite"),
            "composite_unweighted": agg.get("composite_unweighted"),
            "means": agg.get("means") or {},
            "decision": gate.get("decision"),
            "single_run_gate": gate.get("single_run_gate"),
            "three_seed_gate": gate.get("three_seed_gate"),
        })
    rows.sort(
        key=lambda r: (
            -(r.get("composite") if isinstance(r.get("composite"), (int, float)) else -1),
            r.get("started_at") or "",
        )
    )
    return rows


def build_run_detail(run_rel: str) -> dict[str, Any]:
    run = safe_load_json(f"{run_rel}/run.json") or {}
    agg = safe_load_json(f"{run_rel}/aggregate.json") or {}
    gate = safe_load_json(f"{run_rel}/gate.json") or {}
    cfg = safe_load_json(f"{run_rel}/config.json") or {}
    manifest = load_manifest_for(run_rel)

    images: list[dict[str, Any]] = []
    for image_id in run.get("image_ids") or []:
        img_dir = f"{run_rel}/per_image/{image_id}"
        scores_doc = safe_load_json(f"{img_dir}/scores.json") or {}
        prompt_text = ""
        prompt_path = f"{img_dir}/prompt.txt"
        if BACKEND.is_file(prompt_path):
            try:
                prompt_text = BACKEND.read_text(prompt_path)
            except Exception:
                prompt_text = ""

        target_url = None
        meta = manifest.get(image_id)
        if meta:
            split = meta["__split__"]
            mdir = meta["__manifest_dir__"]
            filename = meta.get("filename")
            if filename:
                target_rel = f"{mdir}/{split}/{filename}" if mdir else f"{split}/{filename}"
                target_url = "/api/file?path=" + urllib.parse.quote(target_rel)

        gen_path = f"{img_dir}/generated.png"
        gen_url = None
        if BACKEND.is_file(gen_path):
            gen_url = "/api/file?path=" + urllib.parse.quote(gen_path)

        seed_entries: list[dict[str, Any]] = []
        seeds_dir = f"{img_dir}/seeds"
        if BACKEND.is_dir(seeds_dir):
            seed_subs, _ = BACKEND.list_dir(seeds_dir)
            for sub in seed_subs:
                seed_scores = safe_load_json(f"{sub['path']}/scores.json") or {}
                seed_gen = f"{sub['path']}/generated.png"
                seed_url = (
                    "/api/file?path=" + urllib.parse.quote(seed_gen)
                    if BACKEND.is_file(seed_gen)
                    else None
                )
                seed_entries.append({
                    "seed": sub["name"],
                    "scores": seed_scores.get("scores") or {},
                    "generated_url": seed_url,
                })

        images.append({
            "image_id": image_id,
            "scores": scores_doc.get("scores") or {},
            "judge": scores_doc.get("judge"),
            "prompt": prompt_text,
            "target_url": target_url,
            "generated_url": gen_url,
            "seeds": seed_entries,
        })

    return {
        "run": run,
        "config": cfg,
        "aggregate": agg,
        "gate": gate,
        "images": images,
    }


def inspect_dir(rel: str) -> dict[str, Any]:
    kind = classify_dir(rel)
    subdirs, files = BACKEND.list_dir(rel)

    enriched_subdirs: list[dict] = []
    for s in subdirs:
        if BACKEND.is_file(f"{s['path']}/run.json"):
            child_kind = "run"
        else:
            grand_subs, _ = BACKEND.list_dir(s["path"])
            child_kind = "dir"
            for gs in grand_subs:
                if BACKEND.is_file(f"{gs['path']}/run.json"):
                    child_kind = "runs_container"
                    break
        enriched_subdirs.append({**s, "kind": child_kind})

    result: dict[str, Any] = {
        "path": rel,
        "kind": kind,
        "breadcrumb": breadcrumb(rel),
        "subdirs": enriched_subdirs,
        "files": files,
        "root_name": BACKEND.root_label or "/",
    }
    if kind == "runs_container":
        result["summary"] = build_summary(rel)
    elif kind == "run":
        result["run_detail"] = build_run_detail(rel)
    return result


# --------------------------- HTTP handler ---------------------------


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Results Viewer</title>
<style>
  :root {
    --fg: #1a1a1a; --muted: #666; --bg: #fafafa; --panel: #fff;
    --border: #e5e5e5; --accent: #2257d6; --good: #1f8f3a; --bad: #c83232;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: var(--fg); background: var(--bg); }
  .layout { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
  .sidebar { background: var(--panel); border-right: 1px solid var(--border); overflow-y: auto; padding: 12px; }
  .main { padding: 18px 22px; overflow-y: auto; }
  .bc { font-size: 13px; margin-bottom: 14px; word-break: break-all; }
  .bc a { color: var(--accent); text-decoration: none; }
  .bc a:hover { text-decoration: underline; }
  .bc .sep { color: var(--muted); margin: 0 4px; }
  ul.dirlist { list-style: none; padding: 0; margin: 0; }
  ul.dirlist li { padding: 5px 6px; border-radius: 4px; display: flex; align-items: center; gap: 6px; }
  ul.dirlist li:hover { background: #f1f3f7; }
  ul.dirlist a { color: var(--fg); text-decoration: none; flex: 1; word-break: break-all; }
  .badge { font-size: 11px; padding: 1px 6px; border-radius: 9px; background: #eef2ff; color: var(--accent); }
  .badge.run { background: #e8f5ec; color: var(--good); }
  .badge.exps { background: #fff4e0; color: #b76b00; }
  .hint { color: var(--muted); font-style: italic; }
  h2 { margin: 0 0 6px 0; font-size: 18px; }
  h3 { margin: 22px 0 8px 0; font-size: 15px; }
  h4 { margin: 0 0 6px 0; font-size: 13px; word-break: break-all; }
  table { border-collapse: collapse; width: 100%; background: var(--panel); border: 1px solid var(--border); }
  th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; font-size: 13px; }
  th { background: #f5f6f9; font-weight: 600; position: sticky; top: 0; }
  tr.row { cursor: pointer; }
  tr.row:hover { background: #f5f8ff; }
  td.composite { font-weight: 600; font-variant-numeric: tabular-nums; }
  td.metric { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #333; }
  .meta-row { color: var(--muted); font-size: 13px; }
  .meta-row b { color: var(--fg); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 6px 0; }
  figure { margin: 0; text-align: center; }
  figure img { max-width: 100%; max-height: 220px; border: 1px solid var(--border); border-radius: 4px; background: #f5f5f5; }
  figure figcaption { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .scores { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; margin: 6px 0; word-break: break-all; }
  .scores .pos { color: var(--good); }
  .scores .neg { color: var(--bad); }
  details { margin-top: 6px; }
  summary { cursor: pointer; font-size: 12px; color: var(--muted); }
  pre { white-space: pre-wrap; background: #f7f7f9; padding: 8px; border-radius: 4px; font-size: 12px; max-height: 200px; overflow: auto; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 9px; font-size: 11px; }
  .pill.promoted { background: #e8f5ec; color: var(--good); }
  .pill.rejected { background: #fdecec; color: var(--bad); }
  .pill.reverted { background: #fff4e0; color: #b76b00; }
  .pill.no_leader { background: #eef2ff; color: var(--accent); }
  .pill.fail { background: #fdecec; color: var(--bad); }
  .pill.pass { background: #e8f5ec; color: var(--good); }
  .heading { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .heading code { background: #f1f3f7; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
  .empty { color: var(--muted); padding: 16px 0; }
  .loading { color: var(--muted); padding: 8px 0; font-style: italic; }
</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar" id="sidebar"></aside>
  <main class="main" id="main"></main>
</div>
<script>
const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function fmt(n, d=4) {
  if (n == null || typeof n !== 'number' || !isFinite(n)) return '—';
  return n.toFixed(d);
}
function pathLink(path, label, extraClass='') {
  return `<a href="#${encodeURIComponent(path)}" data-path="${escapeHtml(path)}" class="link ${extraClass}">${escapeHtml(label)}</a>`;
}

async function loadPath(path) {
  $('main').innerHTML = '<p class="loading">Loading…</p>';
  try {
    const r = await fetch('/api/inspect?path=' + encodeURIComponent(path || ''));
    if (!r.ok) {
      const err = await r.json().catch(() => ({error: r.statusText}));
      $('main').innerHTML = `<p class="empty">Error: ${escapeHtml(err.error || 'unknown')}</p>`;
      return;
    }
    const data = await r.json();
    render(data);
    document.title = (data.path || data.root_name) + ' — eval viewer';
  } catch (e) {
    $('main').innerHTML = `<p class="empty">Error: ${escapeHtml(e.message)}</p>`;
  }
}

function render(d) {
  $('sidebar').innerHTML = renderSidebar(d);
  $('main').innerHTML = renderMain(d);
}

function renderSidebar(d) {
  let html = '<div class="bc">';
  d.breadcrumb.forEach((b, i) => {
    if (i > 0) html += '<span class="sep">/</span>';
    html += pathLink(b.path, b.name);
  });
  html += '</div>';
  if (!d.subdirs.length) {
    html += '<p class="hint">(no subdirectories)</p>';
  } else {
    html += '<ul class="dirlist">';
    for (const s of d.subdirs) {
      const cls = s.kind === 'run' ? 'run' : (s.kind === 'runs_container' ? 'exps' : '');
      const label = s.kind === 'run' ? 'run' : (s.kind === 'runs_container' ? 'experiments' : '');
      const badge = label ? `<span class="badge ${cls}">${label}</span>` : '';
      html += `<li>${pathLink(s.path, s.name)} ${badge}</li>`;
    }
    html += '</ul>';
  }
  if (d.files && d.files.length) {
    html += '<h3 style="font-size:12px;color:var(--muted);margin:14px 0 4px;">files</h3><ul class="dirlist">';
    for (const f of d.files) {
      html += `<li><span style="color:var(--muted);flex:1;">${escapeHtml(f.name)}</span></li>`;
    }
    html += '</ul>';
  }
  return html;
}

function renderMain(d) {
  if (d.kind === 'run') return renderRun(d.run_detail || {});
  if (d.kind === 'runs_container') return renderSummary(d);
  return renderPlainDir(d);
}

function renderPlainDir(d) {
  const here = d.path || d.root_name;
  let html = `<h2>${escapeHtml(here)}</h2>`;
  html += `<p class="hint">Pick a subdirectory in the sidebar.`;
  const children = d.subdirs || [];
  const exps = children.filter(c => c.kind === 'runs_container');
  const runs = children.filter(c => c.kind === 'run');
  if (exps.length) {
    html += ` This folder has <b>${exps.length}</b> experiment folder${exps.length === 1 ? '' : 's'}: `;
    html += exps.map(s => pathLink(s.path, s.name)).join(', ');
    html += '.';
  } else if (runs.length) {
    html += ` This folder has <b>${runs.length}</b> run folder${runs.length === 1 ? '' : 's'}.`;
  }
  html += '</p>';
  return html;
}

function decisionPill(d) {
  if (!d) return '';
  const cls = d === 'promoted' ? 'promoted' :
              d === 'rejected' ? 'rejected' :
              d === 'reverted_after_reeval' ? 'reverted' :
              d === 'no_leader' ? 'no_leader' : '';
  const label = d === 'reverted_after_reeval' ? 'reverted' : d;
  return `<span class="pill ${cls}">${escapeHtml(label)}</span>`;
}

function gatePill(g) {
  if (!g) return '';
  const cls = g === 'pass' ? 'pass' : (g === 'fail' ? 'fail' : '');
  return `<span class="pill ${cls}">${escapeHtml(g)}</span>`;
}

function metricCols(rows) {
  const keys = new Set();
  for (const r of rows) Object.keys(r.means || {}).forEach(k => keys.add(k));
  return Array.from(keys).sort();
}

function renderSummary(d) {
  const rows = d.summary || [];
  let html = `<h2>${escapeHtml(d.path || d.root_name)} <span style="color:var(--muted);font-weight:400;">— ${rows.length} run${rows.length === 1 ? '' : 's'}</span></h2>`;
  if (!rows.length) return html + '<p class="empty">No runs found in this folder.</p>';

  const cols = metricCols(rows);
  html += '<div style="overflow-x:auto;"><table><thead><tr>';
  html += '<th>run_id</th><th>driver</th><th>split</th><th>status</th><th>started</th>';
  html += '<th>composite</th>';
  for (const k of cols) html += `<th>${escapeHtml(k)}</th>`;
  html += '<th>gate</th><th>decision</th></tr></thead><tbody>';
  for (const r of rows) {
    html += `<tr class="row" data-path="${escapeHtml(r.path)}">`;
    html += `<td>${escapeHtml(r.run_id)}</td>`;
    html += `<td>${escapeHtml(r.driver || '')}</td>`;
    html += `<td>${escapeHtml(r.split || '')}</td>`;
    html += `<td>${escapeHtml(r.status || '')}</td>`;
    html += `<td>${escapeHtml(r.started_at || '')}</td>`;
    html += `<td class="composite">${fmt(r.composite, 4)}</td>`;
    for (const k of cols) {
      const v = (r.means || {})[k];
      html += `<td class="metric">${fmt(v, 3)}</td>`;
    }
    html += `<td>${gatePill(r.single_run_gate)}</td>`;
    html += `<td>${decisionPill(r.decision)}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  return html;
}

function renderScores(scores) {
  if (!scores || !Object.keys(scores).length) return '<span class="hint">no scores</span>';
  return Object.entries(scores).map(([k, v]) => `${escapeHtml(k)}=<b>${fmt(v, 3)}</b>`).join(' · ');
}

function renderJudge(judge) {
  if (!judge) return '';
  return Object.entries(judge).map(([k, v]) => `${escapeHtml(k)}=${escapeHtml(String(v))}`).join(' · ');
}

function renderRun(d) {
  const run = d.run || {};
  const agg = d.aggregate || {};
  const gate = d.gate || {};
  const cfg = d.config || {};
  let html = `<div class="heading"><h2>${escapeHtml(run.run_id || '(unnamed run)')}</h2>`;
  html += decisionPill(gate.decision);
  html += gatePill(gate.single_run_gate);
  html += '</div>';
  html += `<p class="meta-row">`;
  html += `<b>${escapeHtml(run.driver || '?')}</b> · `;
  html += `harness=<code>${escapeHtml(run.harness_variant || '?')}</code> · `;
  html += `split=<code>${escapeHtml(run.split || '?')}</code> · `;
  html += `seeds=<code>${escapeHtml(JSON.stringify(run.seeds || []))}</code> · `;
  html += `status=<b>${escapeHtml(run.status || '?')}</b>`;
  html += '</p>';
  html += `<p class="meta-row">`;
  html += `composite=<b>${fmt(agg.composite, 4)}</b>`;
  if (agg.means) html += ` · means: ${renderScores(agg.means)}`;
  html += '</p>';
  if (run.hypothesis) html += `<p class="meta-row"><i>${escapeHtml(run.hypothesis)}</i></p>`;
  if (run.takeaway) html += `<p class="meta-row">→ ${escapeHtml(run.takeaway)}</p>`;

  if (gate && Object.keys(gate).length) {
    html += '<details><summary>gate detail</summary><pre>' + escapeHtml(JSON.stringify(gate, null, 2)) + '</pre></details>';
  }
  if (cfg && Object.keys(cfg).length) {
    html += '<details><summary>config</summary><pre>' + escapeHtml(JSON.stringify(cfg, null, 2)) + '</pre></details>';
  }

  const imgs = d.images || [];
  html += `<h3>Per-image (${imgs.length})</h3>`;
  if (!imgs.length) return html + '<p class="empty">no images recorded</p>';

  html += '<div class="grid">';
  for (const img of imgs) {
    html += `<div class="card"><h4>${escapeHtml(img.image_id)}</h4>`;
    html += '<div class="pair">';
    if (img.target_url) {
      html += `<figure><a href="${img.target_url}" target="_blank"><img src="${img.target_url}" alt="target"></a><figcaption>target</figcaption></figure>`;
    } else {
      html += '<figure><div style="height:140px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;">target unavailable</div><figcaption>target</figcaption></figure>';
    }
    if (img.generated_url) {
      html += `<figure><a href="${img.generated_url}" target="_blank"><img src="${img.generated_url}" alt="generated"></a><figcaption>generated</figcaption></figure>`;
    } else {
      html += '<figure><div style="height:140px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;">no generated.png</div><figcaption>generated</figcaption></figure>';
    }
    html += '</div>';
    html += `<div class="scores">${renderScores(img.scores)}</div>`;
    if (img.judge) html += `<div class="scores" style="color:var(--muted);">judge: ${renderJudge(img.judge)}</div>`;
    if (img.seeds && img.seeds.length) {
      html += '<details><summary>per-seed (' + img.seeds.length + ')</summary>';
      for (const s of img.seeds) {
        html += `<div style="margin-top:6px;">seed ${escapeHtml(s.seed)}: ${renderScores(s.scores)}`;
        if (s.generated_url) html += ` <a href="${s.generated_url}" target="_blank">image</a>`;
        html += '</div>';
      }
      html += '</details>';
    }
    if (img.prompt) {
      html += `<details><summary>prompt (${img.prompt.length} chars)</summary><pre>${escapeHtml(img.prompt)}</pre></details>`;
    }
    html += '</div>';
  }
  html += '</div>';
  return html;
}

document.addEventListener('click', (e) => {
  const a = e.target.closest('[data-path]');
  if (!a) return;
  e.preventDefault();
  const path = a.getAttribute('data-path');
  history.pushState({path}, '', '#' + encodeURIComponent(path));
  loadPath(path);
});

window.addEventListener('popstate', () => {
  const path = decodeURIComponent(location.hash.slice(1)) || '';
  loadPath(path);
});

const initial = decodeURIComponent(location.hash.slice(1)) || '';
loadPath(initial);
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            return self._send_html(INDEX_HTML)
        if parsed.path == "/api/inspect":
            return self._handle_inspect(parsed)
        if parsed.path == "/api/file":
            return self._handle_file(parsed)
        self.send_error(404, "not found")

    def _handle_inspect(self, parsed: urllib.parse.ParseResult) -> None:
        qs = urllib.parse.parse_qs(parsed.query)
        raw = qs.get("path", [""])[0]
        try:
            rel = _clean_rel(raw)
        except PermissionError as exc:
            return self._send_json({"error": str(exc)}, status=403)
        if not BACKEND.is_dir(rel):
            return self._send_json({"error": f"not a directory: {raw!r}"}, status=404)
        try:
            data = inspect_dir(rel)
        except Exception as exc:  # surface backend errors as JSON
            return self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return self._send_json(data)

    def _handle_file(self, parsed: urllib.parse.ParseResult) -> None:
        qs = urllib.parse.parse_qs(parsed.query)
        raw = qs.get("path", [""])[0]
        try:
            rel = _clean_rel(raw)
        except PermissionError as exc:
            return self.send_error(403, str(exc))
        if not BACKEND.is_file(rel):
            return self.send_error(404, "file not found")
        ctype = BACKEND.content_type(rel)
        try:
            size = BACKEND.file_size(rel)
        except Exception as exc:
            return self.send_error(500, str(exc))
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if size:
            self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "private, max-age=60")
        self.end_headers()
        try:
            for chunk in BACKEND.stream(rel):
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_html(self, body: str) -> None:
        body_bytes = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_backend(root: str, gcs_cache_ttl: float) -> Backend:
    if root.startswith("gs://"):
        without_scheme = root[len("gs://"):]
        if "/" in without_scheme:
            bucket, prefix = without_scheme.split("/", 1)
        else:
            bucket, prefix = without_scheme, ""
        if not bucket:
            raise ValueError(f"missing bucket in GCS root: {root!r}")
        return GCSBackend(bucket, prefix, cache_ttl=gcs_cache_ttl)
    return LocalBackend(Path(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(Path.cwd()),
                        help="local path or gs://bucket/prefix (default: cwd)")
    parser.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind host (default: 127.0.0.1; use 0.0.0.0 to expose on LAN)")
    parser.add_argument("--open", action="store_true",
                        help="open the viewer in the default browser on start")
    parser.add_argument("--gcs-cache-ttl", type=float, default=30.0,
                        help="seconds to cache GCS metadata listings (default: 30)")
    args = parser.parse_args(argv)

    global BACKEND
    try:
        BACKEND = make_backend(args.root, args.gcs_cache_ttl)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"failed to open root: {exc}", file=sys.stderr)
        return 1

    server = ThreadingServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving {BACKEND.root_label} at {url}", file=sys.stderr)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", file=sys.stderr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
