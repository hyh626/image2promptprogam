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
import os
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

from storage_backend import (  # noqa: F401  re-exported for tests/back-compat
    Backend,
    GCSBackend,
    LocalBackend,
    _clean_rel,
    make_backend,
)


BACKEND: Backend  # set in main()

# Precomputed summary file written by sync_runs_to_gcs.py at
# `<runs_container>/_index.json`. The viewer prefers it over walking
# every run on the bucket; absent, it falls back to the slow per-run
# walk (current behaviour).
RUNS_INDEX_FILENAME = "_index.json"


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


def load_runs_index(container_rel: str) -> dict | None:
    """Read `<container_rel>/_index.json` if present, else None.

    The file is produced by `sync_runs_to_gcs.py` and lets us render
    Summary + Timeline without fanning out N×M metadata reads for every
    page load. Best-effort: any read/parse failure falls through to the
    slow walk path.
    """
    rel = (
        f"{container_rel}/{RUNS_INDEX_FILENAME}" if container_rel
        else RUNS_INDEX_FILENAME
    )
    if not BACKEND.is_file(rel):
        return None
    try:
        data = json.loads(BACKEND.read_text(rel))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        return None
    return data


def _summary_row_from_index(entry: dict[str, Any], container_rel: str) -> dict[str, Any]:
    rid = entry.get("run_id") or ""
    path = entry.get("path") or (
        f"{container_rel}/{rid}" if container_rel else rid
    )
    return {
        "run_id": rid,
        "path": path,
        "driver": entry.get("driver"),
        "name": entry.get("name"),
        "status": entry.get("status"),
        "started_at": entry.get("started_at"),
        "harness_variant": entry.get("harness_variant"),
        "split": entry.get("split"),
        "n_images": entry.get("n_images") or len(entry.get("image_ids") or []),
        "composite": entry.get("composite"),
        "composite_unweighted": entry.get("composite_unweighted"),
        "means": entry.get("means") or {},
        "decision": entry.get("decision"),
        "single_run_gate": entry.get("single_run_gate"),
        "three_seed_gate": entry.get("three_seed_gate"),
    }


def build_summary(container_rel: str, *, index: dict | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if index is None:
        index = load_runs_index(container_rel)

    if index is not None:
        for entry in index.get("runs") or []:
            rows.append(_summary_row_from_index(entry, container_rel))
    else:
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


def build_timeline(container_rel: str, *, index: dict | None = None) -> dict[str, Any]:
    """Per-image evolution across all runs under `container_rel`.

    Returns a structure suitable for rendering a pivot grid:

        {
          "runs": [{run_id, path, name, driver, started_at, decision,
                    composite, single_run_gate, is_leader_promotion}, ...],
          "image_ids": ["hero_landscape_01", ...],   # union, order = first-seen
          "manifest": {<image_id>: {target_url, split, category}},
          "cells": {
              <image_id>: {
                  <run_id>: {generated_url, scores, decision, prompt_chars}
              }
          }
        }
    """
    runs: list[dict[str, Any]] = []
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    image_ids: list[str] = []
    seen_images: set[str] = set()

    if index is None:
        index = load_runs_index(container_rel)

    if index is not None:
        # Fast path: everything we need is in `_index.json`. No per-run
        # is_file/list_dir calls — useful on GCS where each is a HEAD.
        for entry in index.get("runs") or []:
            rid = entry.get("run_id") or ""
            path = entry.get("path") or (
                f"{container_rel}/{rid}" if container_rel else rid
            )
            decision = entry.get("decision")
            composite = entry.get("composite")
            runs.append({
                "run_id": rid,
                "path": path,
                "name": entry.get("name"),
                "driver": entry.get("driver"),
                "split": entry.get("split"),
                "started_at": entry.get("started_at"),
                "decision": decision,
                "single_run_gate": entry.get("single_run_gate"),
                "composite": composite,
                "is_leader_promotion": decision in ("promoted", "no_leader"),
                "hypothesis": entry.get("hypothesis"),
            })

            entry_cells = entry.get("cells") or {}
            for image_id in entry.get("image_ids") or []:
                if image_id not in seen_images:
                    seen_images.add(image_id)
                    image_ids.append(image_id)
                cinfo = entry_cells.get(image_id) or {}
                cell: dict[str, Any] = {
                    "scores": cinfo.get("scores") or {},
                    "decision": decision,
                    "single_run_gate": entry.get("single_run_gate"),
                    "composite": composite,
                }
                if cinfo.get("has_generated"):
                    cell["generated_url"] = (
                        "/api/file?path="
                        + urllib.parse.quote(f"{path}/per_image/{image_id}/generated.png")
                    )
                cells.setdefault(image_id, {})[rid] = cell
    else:
        subdirs, _ = BACKEND.list_dir(container_rel)
        for child in subdirs:
            run_json_rel = f"{child['path']}/run.json"
            if not BACKEND.is_file(run_json_rel):
                continue
            run = safe_load_json(run_json_rel) or {}
            agg = safe_load_json(f"{child['path']}/aggregate.json") or {}
            gate = safe_load_json(f"{child['path']}/gate.json") or {}
            runs.append({
                "run_id": child["name"],
                "path": child["path"],
                "name": run.get("name"),
                "driver": run.get("driver"),
                "split": run.get("split"),
                "started_at": run.get("started_at"),
                "decision": gate.get("decision"),
                "single_run_gate": gate.get("single_run_gate"),
                "composite": agg.get("composite"),
                "is_leader_promotion": gate.get("decision") in ("promoted", "no_leader"),
                "hypothesis": run.get("hypothesis"),
            })

            for image_id in run.get("image_ids") or []:
                if image_id not in seen_images:
                    seen_images.add(image_id)
                    image_ids.append(image_id)
                img_dir = f"{child['path']}/per_image/{image_id}"
                scores_doc = safe_load_json(f"{img_dir}/scores.json") or {}
                gen_path = f"{img_dir}/generated.png"
                cell: dict[str, Any] = {
                    "scores": scores_doc.get("scores") or {},
                    "decision": gate.get("decision"),
                    "single_run_gate": gate.get("single_run_gate"),
                    "composite": agg.get("composite"),
                }
                if BACKEND.is_file(gen_path):
                    cell["generated_url"] = "/api/file?path=" + urllib.parse.quote(gen_path)
                cells.setdefault(image_id, {})[child["name"]] = cell

    # Sort runs chronologically (oldest first; promotion order reads naturally).
    runs.sort(key=lambda r: r.get("started_at") or "")

    # Resolve target images via manifest (walking up from the container).
    manifest = load_manifest_for(container_rel) if image_ids else {}
    manifest_block: dict[str, dict[str, Any]] = {}
    for image_id in image_ids:
        meta = manifest.get(image_id)
        if not meta:
            manifest_block[image_id] = {"target_url": None, "split": None,
                                        "category": None}
            continue
        split = meta["__split__"]
        mdir = meta["__manifest_dir__"]
        filename = meta.get("filename")
        target_url = None
        if filename:
            target_rel = f"{mdir}/{split}/{filename}" if mdir else f"{split}/{filename}"
            target_url = "/api/file?path=" + urllib.parse.quote(target_rel)
        manifest_block[image_id] = {
            "target_url": target_url,
            "split": split,
            "category": meta.get("category"),
        }

    return {
        "runs": runs,
        "image_ids": image_ids,
        "manifest": manifest_block,
        "cells": cells,
    }


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

    # Hide internal viewer files (currently `_index.json`) from the
    # sidebar's file list. The browser doesn't navigate into files
    # anyway, but the underscore prefix is a clear "system" signal.
    files = [f for f in files if not f["name"].startswith("_")]

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
        index = load_runs_index(rel)
        result["has_index"] = index is not None
        if index is not None and index.get("generated_at"):
            result["index_generated_at"] = index.get("generated_at")
        result["summary"] = build_summary(rel, index=index)
        result["timeline"] = build_timeline(rel, index=index)
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
  /* `minmax(0, 1fr)` lets the main column shrink past its content and
     keeps wide tables (e.g. timeline with many runs) inside the
     viewport instead of stretching the layout grid. */
  .layout { display: grid; grid-template-columns: 320px minmax(0, 1fr); height: 100vh; }
  .sidebar { background: var(--panel); border-right: 1px solid var(--border); overflow-y: auto; padding: 12px; }
  .main { padding: 18px 22px; overflow-y: auto; min-width: 0; }
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
  .tabs { display: flex; gap: 4px; margin: 0 0 12px 0; border-bottom: 1px solid var(--border); }
  .tabs .tab { background: none; border: 1px solid transparent; border-bottom: none; padding: 6px 12px; font-size: 13px; cursor: pointer; color: var(--muted); border-radius: 4px 4px 0 0; }
  .tabs .tab.active { color: var(--fg); border-color: var(--border); background: var(--panel); margin-bottom: -1px; }
  .tabs .tab:hover { color: var(--fg); }
  /* The earlier `table { width: 100% }` rule (plus tbody-th's max-width)
     would squeeze cells until no horizontal scrollbar ever appeared.
     Force fixed per-column widths so the table is predictably wider
     than the viewport on busy datasets, and let the wrap scroll on x. */
  .timeline-wrap { overflow-x: auto; overflow-y: hidden; max-width: 100%; }
  table.timeline { width: auto; border-collapse: separate; border-spacing: 0; background: var(--panel); border: 1px solid var(--border); }
  table.timeline th, table.timeline td { padding: 6px 8px; border-bottom: 1px solid var(--border); border-right: 1px solid var(--border); vertical-align: top; }
  table.timeline thead th { background: #f5f6f9; font-size: 12px; text-align: left; }
  table.timeline tbody th { text-align: left; font-weight: 600; font-size: 12.5px; word-break: break-word; }
  /* Per-column widths win against `tbody th` via the `table.timeline`
     class qualifier (higher specificity). */
  table.timeline .tcell-img-head, table.timeline .tcell-img { width: 180px; min-width: 180px; max-width: 180px; }
  table.timeline .tcell-target-head, table.timeline .tcell-target { width: 180px; min-width: 180px; max-width: 180px; }
  table.timeline .tcol, table.timeline .tcell { width: 200px; min-width: 200px; max-width: 200px; }
  .tcol-head a { color: var(--accent); text-decoration: none; word-break: break-all; }
  .tcol-head a:hover { text-decoration: underline; }
  .tcol-meta { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .tcell { text-align: center; }
  .tcell img { max-width: 160px; max-height: 130px; border: 1px solid var(--border); border-radius: 3px; background: #f5f5f5; }
  .tcell.promoted { background: #f4faf6; }
  .tcell.rejected { background: #fdf6f6; }
  .tcell-img { background: #fafbfd; }
  .tcell-target img { max-height: 130px; }
  .tplaceholder { height: 100px; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 11px; background: #f9f9f9; border: 1px dashed var(--border); border-radius: 3px; }
  .tscores { font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; font-size: 11px; margin-top: 4px; word-break: break-all; }
  .tscores.tdetail { color: var(--muted); font-size: 10.5px; }
  .tscores .pos { color: var(--good); }
  .tscores .neg { color: var(--bad); }
  .tcell-target-head, .tcell-img-head { background: #f0f2f7; }
  /* Pin image_id + target columns to the left so the row identity and
     ground-truth target stay on screen while the run columns scroll. */
  table.timeline .tcell-img-head, table.timeline .tcell-img {
    position: sticky;
    left: 0;
    z-index: 2;
  }
  table.timeline .tcell-target-head, table.timeline .tcell-target {
    position: sticky;
    left: 180px;
    z-index: 2;
    /* Right-edge shadow shows when the run columns are scrolled
       behind the pinned cluster; harmless when nothing is scrolled. */
    box-shadow: 6px 0 8px -6px rgba(0,0,0,0.18);
  }
  table.timeline thead th.tcell-img-head, table.timeline thead th.tcell-target-head { z-index: 3; }
  .toolrow { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin: 0 0 10px 0; padding: 8px 10px; background: var(--panel); border: 1px solid var(--border); border-radius: 4px; }
  .toolrow label { font-size: 12.5px; color: var(--fg); display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
  .toolrow .stat { font-size: 12px; color: var(--muted); }
  .toolrow .stat b { color: var(--fg); }
  .pager { display: inline-flex; align-items: center; gap: 6px; }
  .pager button { font-size: 12px; padding: 3px 9px; border: 1px solid var(--border); background: var(--panel); border-radius: 3px; cursor: pointer; color: var(--fg); }
  .pager button:hover:not(:disabled) { background: #f1f3f7; }
  .pager button:disabled { opacity: 0.5; cursor: default; }
  .pager .pageinfo { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .index-badge { font-size: 11px; color: var(--muted); }
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

function pathFromHash() {
  const raw = decodeURIComponent(location.hash.slice(1) || '');
  return raw.split('?')[0];
}

async function loadPath(path) {
  // Drop any '?tab=…' suffix that may have been included in the hash.
  path = (path || '').split('?')[0];
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
  window.__lastData = d;
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

function currentTab() {
  const t = (location.hash.match(/[?&]tab=([^&]+)/) || [])[1];
  return t === 'timeline' || t === 'leader-only' ? t : 'summary';
}
function setTab(tab) {
  const path = decodeURIComponent((location.hash.split('?')[0] || '').replace(/^#/, ''));
  const newHash = '#' + encodeURIComponent(path) + (tab === 'summary' ? '' : '?tab=' + tab);
  history.replaceState({path}, '', newHash);
}

// Page state for the runs-container views. `showRejected` defaults to
// false so a flurry of failed candidates does not drown out the
// promoted leader chain on first load. Persisted across re-renders for
// the lifetime of the page.
const TIMELINE_PAGE_SIZE = 12;
const RUNS_VIEW = {
  showRejected: false,
  timelinePage: 0,
};
function isRejected(decision) { return decision === 'rejected'; }

function renderMain(d) {
  if (d.kind === 'run') return renderRun(d.run_detail || {});
  if (d.kind === 'runs_container') return renderRunsContainer(d);
  return renderPlainDir(d);
}

function renderRunsContainer(d) {
  const tab = currentTab();
  const tabBtn = (id, label) =>
    `<button class="tab ${tab === id ? 'active' : ''}" data-tab="${id}">${escapeHtml(label)}</button>`;
  const tabs =
    `<div class="tabs">${tabBtn('summary', 'Summary')}` +
    `${tabBtn('timeline', 'Timeline')}` +
    `${tabBtn('leader-only', 'Leader chain')}</div>`;

  const toolRow = renderToolRow(d, tab);
  let body;
  if (tab === 'timeline') body = renderTimeline(d.timeline || {}, false);
  else if (tab === 'leader-only') body = renderTimeline(d.timeline || {}, true);
  else body = renderSummary(d);
  return tabs + toolRow + body;
}

function renderToolRow(d, tab) {
  const summary = d.summary || [];
  const timelineRuns = (d.timeline && d.timeline.runs) || [];
  const totalRuns = tab === 'summary' ? summary.length : timelineRuns.length;
  const rejectedCount = (tab === 'summary' ? summary : timelineRuns)
    .filter(r => isRejected(r.decision)).length;

  const indexBadge = d.has_index
    ? `<span class="index-badge" title="Loaded from precomputed _index.json${d.index_generated_at ? ' generated ' + escapeHtml(d.index_generated_at) : ''}">indexed</span>`
    : '<span class="index-badge" title="No _index.json on this prefix; the viewer walked every run. Re-run sync_runs_to_gcs.py to write one.">unindexed</span>';

  let pager = '';
  if (tab !== 'summary') {
    // Pagination is only needed for the column-heavy timeline grid.
    const visible = filteredTimelineRuns(d.timeline || {}, tab === 'leader-only').length;
    const pageCount = Math.max(1, Math.ceil(visible / TIMELINE_PAGE_SIZE));
    if (RUNS_VIEW.timelinePage >= pageCount) RUNS_VIEW.timelinePage = pageCount - 1;
    if (RUNS_VIEW.timelinePage < 0) RUNS_VIEW.timelinePage = 0;
    const page = RUNS_VIEW.timelinePage;
    const from = visible === 0 ? 0 : page * TIMELINE_PAGE_SIZE + 1;
    const to = Math.min(visible, (page + 1) * TIMELINE_PAGE_SIZE);
    pager = `<span class="pager">
      <button data-page-delta="-1" ${page === 0 ? 'disabled' : ''}>‹ prev</button>
      <span class="pageinfo">runs ${from}–${to} of ${visible}</span>
      <button data-page-delta="1" ${page >= pageCount - 1 ? 'disabled' : ''}>next ›</button>
    </span>`;
  }

  const stat = `<span class="stat">${totalRuns} run${totalRuns === 1 ? '' : 's'}` +
    (rejectedCount ? ` · <b>${rejectedCount}</b> rejected` : '') +
    `</span>`;
  const checkbox = `<label><input type="checkbox" data-toggle-rejected ${RUNS_VIEW.showRejected ? 'checked' : ''}> show rejected</label>`;
  return `<div class="toolrow">${checkbox}${stat}${pager}${indexBadge}</div>`;
}

function filteredTimelineRuns(t, leaderOnly) {
  let runs = t.runs || [];
  if (leaderOnly) runs = runs.filter(r => r.is_leader_promotion);
  if (!RUNS_VIEW.showRejected) runs = runs.filter(r => !isRejected(r.decision));
  return runs;
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
  const all = d.summary || [];
  const rows = RUNS_VIEW.showRejected ? all : all.filter(r => !isRejected(r.decision));
  const hidden = all.length - rows.length;
  let html = `<h2>${escapeHtml(d.path || d.root_name)} <span style="color:var(--muted);font-weight:400;">— ${rows.length} run${rows.length === 1 ? '' : 's'}${hidden ? ` (+${hidden} rejected hidden)` : ''}</span></h2>`;
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

function renderTimeline(t, leaderOnly) {
  const allRuns = t.runs || [];
  const filtered = filteredTimelineRuns(t, leaderOnly);
  const totalPages = Math.max(1, Math.ceil(filtered.length / TIMELINE_PAGE_SIZE));
  if (RUNS_VIEW.timelinePage >= totalPages) RUNS_VIEW.timelinePage = totalPages - 1;
  if (RUNS_VIEW.timelinePage < 0) RUNS_VIEW.timelinePage = 0;
  const start = RUNS_VIEW.timelinePage * TIMELINE_PAGE_SIZE;
  const runs = filtered.slice(start, start + TIMELINE_PAGE_SIZE);
  const imageIds = t.image_ids || [];
  const cells = t.cells || {};
  const manifest = t.manifest || {};

  const hidden = allRuns.length - filtered.length;
  const subtitle = leaderOnly
    ? `Leader chain — ${filtered.length} promoted run${filtered.length === 1 ? '' : 's'}, ${imageIds.length} image${imageIds.length === 1 ? '' : 's'}`
    : `All runs — ${filtered.length} run${filtered.length === 1 ? '' : 's'}, ${imageIds.length} image${imageIds.length === 1 ? '' : 's'}`;

  if (!filtered.length || !imageIds.length) {
    const empty = hidden
      ? `Nothing matches the current filter (${hidden} rejected hidden — toggle "show rejected" above).`
      : 'Nothing to compare yet.';
    return `<p class="hint">${escapeHtml(subtitle)}</p><p class="empty">${escapeHtml(empty)}</p>`;
  }

  const hiddenSuffix = hidden ? ` · <b>${hidden}</b> rejected hidden` : '';
  let html = `<p class="meta-row">${escapeHtml(subtitle)}${hiddenSuffix ? ' (' + hiddenSuffix.replace(/<[^>]+>/g, '') + ')' : ''}. Images on the rows, runs on the columns (left = oldest).</p>`;
  html += '<div class="timeline-wrap"><table class="timeline">';
  // Header row: "image" + per-run columns
  html += '<thead><tr><th class="tcell-img-head">image</th><th class="tcell-target-head">target</th>';
  for (const r of runs) {
    const cls = r.decision === 'promoted' || r.decision === 'no_leader' ? 'promoted'
              : r.decision === 'rejected' ? 'rejected' : '';
    html += `<th class="tcol ${cls}"><div class="tcol-head">`;
    html += `<a href="#${encodeURIComponent(r.path)}" data-path="${escapeHtml(r.path)}">${escapeHtml(r.name || r.run_id)}</a>`;
    html += `<div class="tcol-meta">${escapeHtml((r.started_at || '').replace('T', ' ').replace('Z', ''))}</div>`;
    html += `<div class="tcol-meta">${decisionPill(r.decision)} composite=<b>${fmt(r.composite, 4)}</b></div>`;
    html += '</div></th>';
  }
  html += '</tr></thead><tbody>';

  for (const imageId of imageIds) {
    const meta = manifest[imageId] || {};
    html += `<tr><th class="tcell-img"><div>${escapeHtml(imageId)}</div>`;
    if (meta.split) html += `<div class="tcol-meta">split: <code>${escapeHtml(meta.split)}</code></div>`;
    if (meta.category) html += `<div class="tcol-meta">${escapeHtml(meta.category)}</div>`;
    html += '</th>';
    if (meta.target_url) {
      html += `<td class="tcell tcell-target"><a href="${meta.target_url}" target="_blank"><img src="${meta.target_url}" alt="target"></a></td>`;
    } else {
      html += '<td class="tcell tcell-target"><div class="tplaceholder">no target</div></td>';
    }
    const row = cells[imageId] || {};
    let prevComposite = null;
    for (const r of runs) {
      const c = row[r.run_id];
      const cls = r.decision === 'promoted' || r.decision === 'no_leader' ? 'promoted'
                : r.decision === 'rejected' ? 'rejected' : '';
      html += `<td class="tcell ${cls}">`;
      if (c) {
        if (c.generated_url) {
          html += `<a href="${c.generated_url}" target="_blank"><img src="${c.generated_url}" alt="generated"></a>`;
        } else {
          html += '<div class="tplaceholder">no png</div>';
        }
        // Compose a compact score line; bold composite delta vs previous shown run.
        const composite = c.composite;
        let delta = '';
        if (typeof prevComposite === 'number' && typeof composite === 'number') {
          const d = composite - prevComposite;
          const sign = d > 0 ? '+' : '';
          const dcls = d > 0 ? 'pos' : (d < 0 ? 'neg' : '');
          delta = ` <span class="${dcls}">(${sign}${d.toFixed(3)})</span>`;
        }
        html += `<div class="tscores">composite=<b>${fmt(composite, 4)}</b>${delta}</div>`;
        const s = c.scores || {};
        const compact = Object.entries(s).map(([k, v]) => `${escapeHtml(k.replace('s_', ''))}=${fmt(v, 3)}`).join(' · ');
        if (compact) html += `<div class="tscores tdetail">${compact}</div>`;
      } else {
        html += '<div class="tplaceholder">not in run</div>';
      }
      html += '</td>';
      if (c && typeof c.composite === 'number') prevComposite = c.composite;
    }
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  return html;
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
  const tabBtn = e.target.closest('[data-tab]');
  if (tabBtn) {
    e.preventDefault();
    setTab(tabBtn.getAttribute('data-tab'));
    // Reset pagination when crossing tabs; the run set may be a
    // different size and "page 5" can become out-of-range.
    RUNS_VIEW.timelinePage = 0;
    if (window.__lastData) render(window.__lastData);
    return;
  }
  const pageBtn = e.target.closest('[data-page-delta]');
  if (pageBtn && !pageBtn.disabled) {
    e.preventDefault();
    RUNS_VIEW.timelinePage += Number(pageBtn.getAttribute('data-page-delta')) || 0;
    if (window.__lastData) render(window.__lastData);
    return;
  }
  const a = e.target.closest('[data-path]');
  if (!a) return;
  e.preventDefault();
  const path = a.getAttribute('data-path');
  history.pushState({path}, '', '#' + encodeURIComponent(path));
  loadPath(path);
});

document.addEventListener('change', (e) => {
  const cb = e.target.closest('[data-toggle-rejected]');
  if (!cb) return;
  RUNS_VIEW.showRejected = !!cb.checked;
  RUNS_VIEW.timelinePage = 0;
  if (window.__lastData) render(window.__lastData);
});

window.addEventListener('popstate', () => {
  loadPath(pathFromHash());
});

loadPath(pathFromHash());
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


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # In Cloud Run / containerized deploys, configuration arrives via env
    # vars: VIEWER_ROOT, PORT, VIEWER_HOST, VIEWER_GCS_ONLY,
    # VIEWER_GCS_CACHE_TTL. CLI flags still win when explicitly provided.
    env_root = os.environ.get("VIEWER_ROOT")
    env_port = os.environ.get("PORT")
    env_host = os.environ.get("VIEWER_HOST")
    env_ttl = os.environ.get("VIEWER_GCS_CACHE_TTL")

    parser.add_argument("--root", default=env_root if env_root else str(Path.cwd()),
                        help="local path or gs://bucket/prefix (default: cwd; "
                             "or $VIEWER_ROOT)")
    parser.add_argument("--port", type=int,
                        default=int(env_port) if env_port else 8765,
                        help="port (default: 8765; or $PORT)")
    parser.add_argument("--host", default=env_host or "127.0.0.1",
                        help="bind host (default: 127.0.0.1; use 0.0.0.0 to "
                             "expose on LAN; or $VIEWER_HOST)")
    parser.add_argument("--open", action="store_true",
                        help="open the viewer in the default browser on start")
    parser.add_argument("--gcs-cache-ttl", type=float,
                        default=float(env_ttl) if env_ttl else 30.0,
                        help="seconds to cache GCS metadata listings "
                             "(default: 30; or $VIEWER_GCS_CACHE_TTL)")
    parser.add_argument("--gcs-only", action="store_true",
                        default=_env_bool("VIEWER_GCS_ONLY"),
                        help="reject local --root values; only allow gs:// URIs. "
                             "Also enabled by VIEWER_GCS_ONLY=1; the Cloud Run "
                             "image sets this by default since the container "
                             "filesystem has no useful runs to browse.")
    args = parser.parse_args(argv)

    if args.gcs_only and not str(args.root).startswith("gs://"):
        print(
            f"--gcs-only is set but --root {args.root!r} is not a gs:// URI. "
            "Set VIEWER_ROOT=gs://bucket/prefix when deploying.",
            file=sys.stderr,
        )
        return 2

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
