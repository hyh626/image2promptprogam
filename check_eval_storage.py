#!/usr/bin/env python3
"""Verify a built repo conforms to EVAL_STORAGE_SCHEMA.md.

Usage:
    python check_eval_storage.py --root <repo-root> [--verify-hashes] [--json]
    python check_eval_storage.py --root gs://bucket/prefix [--verify-hashes]

Roots may be local directories or `gs://bucket/prefix` URIs. GCS reads go
through the google-cloud-storage SDK with Application Default Credentials.

Exit code: 0 on success, 1 on any violation.
Each violation prints one line: <CODE> <path>: <message>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from storage_backend import Backend, make_backend

SCHEMA_VERSION = "1.0.0"
SPLITS = ("train", "eval", "val", "holdout")
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z__[a-z0-9][a-z0-9-]*__[a-zA-Z0-9_.-]+$")
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_STATUS = {"completed", "failed", "interrupted"}
GATE_DECISIONS = {"promoted", "rejected", "reverted_after_reeval", "no_leader"}
GATE_OUTCOMES = {"pass", "fail"}
JUDGE_KEYS = {"subject", "composition", "lighting", "palette", "style", "texture"}
FLOAT_TOL = 1e-4


@dataclass
class Violation:
    code: str
    path: str
    message: str


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    files_checked: int = 0

    def add(self, code: str, path: Any, message: str) -> None:
        self.violations.append(Violation(code, str(path), message))

    @property
    def ok(self) -> bool:
        return not self.violations


# --------------------------- helpers ---------------------------


def load_json(backend: Backend, rel: str, report: Report) -> dict | list | None:
    """Load and parse JSON at backend://rel. Records E_FILE_MISSING / E_JSON_INVALID."""
    if not backend.is_file(rel):
        report.add("E_FILE_MISSING", backend.format_path(rel), "required file is missing")
        return None
    try:
        text = backend.read_text(rel)
    except Exception as exc:  # network / permissions / etc.
        report.add("E_JSON_INVALID", backend.format_path(rel), f"could not read: {exc}")
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        report.add("E_JSON_INVALID", backend.format_path(rel), f"invalid JSON: {exc}")
        return None
    report.files_checked += 1
    return data


def is_finite_unit_float(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def require_keys(obj: dict, keys: Iterable[str], path: Any, report: Report,
                 code: str = "E_FIELD_MISSING") -> bool:
    ok = True
    for k in keys:
        if k not in obj:
            report.add(code, path, f"missing required field '{k}'")
            ok = False
    return ok


def require_schema_version(obj: dict, path: Any, report: Report) -> None:
    if obj.get("schema_version") != SCHEMA_VERSION:
        report.add(
            "E_SCHEMA_VERSION",
            path,
            f"schema_version must be {SCHEMA_VERSION!r}, got {obj.get('schema_version')!r}",
        )


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def approx_equal(a: float, b: float, tol: float = FLOAT_TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def join(*parts: str) -> str:
    """Join non-empty path parts with '/'."""
    return "/".join(p.strip("/") for p in parts if p)


# --------------------------- manifest ---------------------------


def check_manifest(backend: Backend, report: Report,
                   verify_hashes: bool) -> dict[str, dict[str, dict]]:
    """Return {split: {image_id: entry}} for downstream cross-checks."""
    by_split: dict[str, dict[str, dict]] = {s: {} for s in SPLITS}

    manifest_rel = "eval_data/images/manifest.json"
    manifest_disp = backend.format_path(manifest_rel)
    data = load_json(backend, manifest_rel, report)
    if not isinstance(data, dict):
        if data is not None:
            report.add("E_MANIFEST_SHAPE", manifest_disp, "manifest must be an object")
        return by_split

    require_schema_version(data, manifest_disp, report)
    splits = data.get("splits")
    if not isinstance(splits, dict):
        report.add("E_MANIFEST_SHAPE", manifest_disp, "missing 'splits' object")
        return by_split

    seen_ids: dict[str, str] = {}
    for split_name in SPLITS:
        entries = splits.get(split_name)
        if entries is None:
            report.add("E_MANIFEST_SPLIT", manifest_disp, f"split {split_name!r} missing")
            continue
        if not isinstance(entries, list):
            report.add("E_MANIFEST_SPLIT", manifest_disp,
                       f"split {split_name!r} must be a list")
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                report.add("E_MANIFEST_ENTRY", manifest_disp,
                           f"non-object entry in {split_name}")
                continue
            if not require_keys(
                entry, ("image_id", "filename", "sha256", "width", "height"),
                manifest_disp, report,
            ):
                continue
            image_id = entry["image_id"]
            if image_id in seen_ids:
                report.add(
                    "E_MANIFEST_DUPLICATE", manifest_disp,
                    f"image_id {image_id!r} appears in both "
                    f"{seen_ids[image_id]!r} and {split_name!r}",
                )
                continue
            seen_ids[image_id] = split_name
            if not SHA256_RE.match(str(entry["sha256"])):
                report.add("E_MANIFEST_HASH_FORMAT", manifest_disp,
                           f"image_id {image_id!r} has malformed sha256")
            if not (isinstance(entry["width"], int) and entry["width"] > 0):
                report.add("E_MANIFEST_DIMS", manifest_disp,
                           f"{image_id} width must be positive int")
            if not (isinstance(entry["height"], int) and entry["height"] > 0):
                report.add("E_MANIFEST_DIMS", manifest_disp,
                           f"{image_id} height must be positive int")
            by_split[split_name][image_id] = entry

            if verify_hashes:
                img_rel = join("eval_data", "images", split_name, entry["filename"])
                img_disp = backend.format_path(img_rel)
                if not backend.is_file(img_rel):
                    report.add("E_IMAGE_MISSING", img_disp,
                               "manifest entry references missing file")
                else:
                    try:
                        actual = sha256_bytes(backend.read_bytes(img_rel))
                    except Exception as exc:
                        report.add("E_IMAGE_MISSING", img_disp,
                                   f"could not read for hashing: {exc}")
                        continue
                    if actual != entry["sha256"]:
                        report.add(
                            "E_IMAGE_HASH_MISMATCH", img_disp,
                            f"sha256 mismatch (manifest={entry['sha256']}, actual={actual})",
                        )
    return by_split


# --------------------------- per-run ---------------------------


def check_run(
    backend: Backend,
    run_rel: str,
    report: Report,
    manifest_by_split: dict[str, dict[str, dict]],
    verify_hashes: bool,
) -> dict[str, Any] | None:
    run_id = run_rel.rsplit("/", 1)[-1]
    run_disp = backend.format_path(run_rel)
    if not RUN_ID_RE.match(run_id):
        report.add("E_RUN_ID_FORMAT", run_disp,
                   f"run_id {run_id!r} does not match canonical format")

    run_json_rel = join(run_rel, "run.json")
    run_json_disp = backend.format_path(run_json_rel)
    run_json = load_json(backend, run_json_rel, report)
    if not isinstance(run_json, dict):
        return None

    require_schema_version(run_json, run_json_disp, report)
    require_keys(
        run_json,
        ("run_id", "name", "driver", "harness_variant",
         "started_at", "finished_at", "split", "image_ids",
         "seeds", "status"),
        run_json_disp, report,
    )

    if run_json.get("run_id") != run_id:
        report.add("E_RUN_ID_MISMATCH", run_json_disp,
                   f"run.json#run_id {run_json.get('run_id')!r} "
                   f"does not match directory name {run_id!r}")

    for ts_field in ("started_at", "finished_at"):
        ts = run_json.get(ts_field)
        if isinstance(ts, str) and not ISO_UTC_RE.match(ts):
            report.add("E_TIMESTAMP_FORMAT", run_json_disp,
                       f"{ts_field} {ts!r} must match YYYY-MM-DDTHH:MM:SSZ")

    status = run_json.get("status")
    if status not in RUN_STATUS:
        report.add("E_RUN_STATUS", run_json_disp,
                   f"status {status!r} not in {sorted(RUN_STATUS)}")

    split = run_json.get("split")
    if split not in SPLITS:
        report.add("E_RUN_SPLIT", run_json_disp,
                   f"split {split!r} not in {SPLITS}")

    image_ids = run_json.get("image_ids") or []
    if not isinstance(image_ids, list) or not all(isinstance(x, str) for x in image_ids):
        report.add("E_RUN_IMAGE_IDS", run_json_disp,
                   "image_ids must be a list of strings")
        image_ids = []
    elif split in SPLITS:
        manifest_ids = manifest_by_split.get(split, {})
        for img in image_ids:
            if img not in manifest_ids:
                report.add("E_RUN_IMAGE_UNKNOWN", run_json_disp,
                           f"image_id {img!r} not in manifest split {split!r}")

    seeds = run_json.get("seeds") or []
    if not isinstance(seeds, list) or not seeds or not all(isinstance(x, int) for x in seeds):
        report.add("E_RUN_SEEDS", run_json_disp,
                   "seeds must be a non-empty list of ints")

    config_rel = join(run_rel, "config.json")
    config = load_json(backend, config_rel, report)
    config_disp = backend.format_path(config_rel)
    metrics: list[str] = []
    if isinstance(config, dict):
        require_schema_version(config, config_disp, report)
        require_keys(config, ("harness_variant", "models", "metrics"), config_disp, report)
        metrics = config.get("metrics") or []
        if not isinstance(metrics, list) or not metrics or not all(isinstance(m, str) for m in metrics):
            report.add("E_CONFIG_METRICS", config_disp,
                       "metrics must be a non-empty list of strings")
            metrics = []

    for sibling in ("prompt_strategy.py", "stdout.log"):
        sibling_rel = join(run_rel, sibling)
        if not backend.is_file(sibling_rel):
            report.add("E_FILE_MISSING", backend.format_path(sibling_rel),
                       f"{sibling} missing")

    per_image_rel = join(run_rel, "per_image")
    per_image_disp = backend.format_path(per_image_rel)
    per_image_means: dict[str, list[float]] = {m: [] for m in metrics}
    if not backend.is_dir(per_image_rel):
        report.add("E_FILE_MISSING", per_image_disp, "per_image/ directory missing")
    else:
        for img in image_ids:
            img_rel = join(per_image_rel, img)
            img_disp = backend.format_path(img_rel)
            if not backend.is_dir(img_rel):
                report.add("E_FILE_MISSING", img_disp, f"per_image/{img}/ missing")
                continue
            for fname in ("prompt.txt", "generated.png", "scores.json"):
                fpath_rel = join(img_rel, fname)
                if not backend.is_file(fpath_rel):
                    if fname == "generated.png" and len(seeds) > 1:
                        # multi-seed: top-level generated.png is optional
                        continue
                    report.add("E_FILE_MISSING", backend.format_path(fpath_rel),
                               f"{fname} missing")

            scores_rel = join(img_rel, "scores.json")
            scores = load_json(backend, scores_rel, report)
            if isinstance(scores, dict):
                check_scores(scores, backend.format_path(scores_rel), img, metrics, report,
                             multi_seed=len(seeds) > 1)
                got = scores.get("scores") or {}
                for m in metrics:
                    if m in got and is_finite_unit_float(got[m]):
                        per_image_means[m].append(float(got[m]))

                gen_rel = join(img_rel, "generated.png")
                if verify_hashes and backend.is_file(gen_rel):
                    expected = scores.get("generated_image_sha256")
                    if isinstance(expected, str) and SHA256_RE.match(expected):
                        try:
                            actual = sha256_bytes(backend.read_bytes(gen_rel))
                        except Exception as exc:
                            report.add("E_GENERATED_HASH_MISMATCH", backend.format_path(gen_rel),
                                       f"could not read for hashing: {exc}")
                        else:
                            if actual != expected:
                                report.add(
                                    "E_GENERATED_HASH_MISMATCH",
                                    backend.format_path(gen_rel),
                                    f"sha256 mismatch (expected={expected}, actual={actual})",
                                )

            seeds_rel = join(img_rel, "seeds")
            if len(seeds) > 1:
                if not backend.is_dir(seeds_rel):
                    report.add("E_FILE_MISSING", backend.format_path(seeds_rel),
                               "multi-seed run missing seeds/ subdir")
                else:
                    for seed in seeds:
                        seed_rel = join(seeds_rel, str(seed))
                        if not backend.is_dir(seed_rel):
                            report.add("E_FILE_MISSING", backend.format_path(seed_rel),
                                       f"seeds/{seed}/ missing")
                            continue
                        for fname in ("generated.png", "scores.json"):
                            if not backend.is_file(join(seed_rel, fname)):
                                report.add("E_FILE_MISSING",
                                           backend.format_path(join(seed_rel, fname)),
                                           f"{fname} missing")
                        seed_scores_rel = join(seed_rel, "scores.json")
                        seed_scores = load_json(backend, seed_scores_rel, report)
                        if isinstance(seed_scores, dict):
                            check_scores(
                                seed_scores, backend.format_path(seed_scores_rel),
                                img, metrics, report,
                                multi_seed=False, expect_seed=seed,
                            )

    aggregate_rel = join(run_rel, "aggregate.json")
    aggregate_disp = backend.format_path(aggregate_rel)
    aggregate = load_json(backend, aggregate_rel, report)
    if isinstance(aggregate, dict):
        check_aggregate(aggregate, aggregate_disp, run_id, split, image_ids,
                        metrics, per_image_means, report)

    gate_rel = join(run_rel, "gate.json")
    gate_disp = backend.format_path(gate_rel)
    gate = load_json(backend, gate_rel, report)
    if isinstance(gate, dict) and isinstance(aggregate, dict):
        check_gate(gate, gate_disp, aggregate, metrics, report)

    return {"run_id": run_id, "aggregate": aggregate}


def check_scores(
    scores: dict,
    path: Any,
    expected_image: str,
    metrics: list[str],
    report: Report,
    multi_seed: bool,
    expect_seed: int | None = None,
) -> None:
    require_schema_version(scores, path, report)
    require_keys(
        scores,
        ("image_id", "seed", "scores", "generated_image_sha256", "prompt_sha256"),
        path, report,
    )
    if scores.get("image_id") != expected_image:
        report.add("E_SCORES_IMAGE_MISMATCH", path,
                   f"image_id {scores.get('image_id')!r} != "
                   f"directory image_id {expected_image!r}")

    seed_val = scores.get("seed")
    if expect_seed is not None:
        if seed_val != expect_seed:
            report.add("E_SCORES_SEED_MISMATCH", path,
                       f"seed {seed_val!r} != expected {expect_seed!r}")
    else:
        if multi_seed and seed_val is not None:
            report.add("E_SCORES_SEED_MISMATCH", path,
                       "top-level scores.json must use seed=null for multi-seed runs")

    score_map = scores.get("scores")
    if not isinstance(score_map, dict):
        report.add("E_SCORES_SHAPE", path, "scores must be an object")
        return

    extra = set(score_map) - set(metrics)
    if extra:
        report.add("E_SCORES_UNKNOWN_METRIC", path,
                   f"unknown metric keys: {sorted(extra)}")
    for m in metrics:
        if m not in score_map:
            report.add("E_SCORES_MISSING_METRIC", path, f"missing metric {m!r}")
        elif not is_finite_unit_float(score_map[m]):
            report.add("E_SCORES_RANGE", path,
                       f"metric {m!r} value {score_map[m]!r} not a finite float in [0, 1]")

    judge = scores.get("judge")
    if judge is not None:
        if not isinstance(judge, dict):
            report.add("E_JUDGE_SHAPE", path, "judge must be null or an object")
        else:
            for k, v in judge.items():
                if not (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5):
                    report.add("E_JUDGE_RANGE", path,
                               f"judge[{k!r}] = {v!r} must be int in [1, 5]")

    for hash_field in ("generated_image_sha256", "prompt_sha256"):
        v = scores.get(hash_field)
        if v is not None and not (isinstance(v, str) and SHA256_RE.match(v)):
            report.add("E_HASH_FORMAT", path, f"{hash_field} must be sha256 hex or null")


def check_aggregate(
    aggregate: dict,
    path: Any,
    run_id: str,
    split: str | None,
    image_ids: list[str],
    metrics: list[str],
    per_image_means: dict[str, list[float]],
    report: Report,
) -> None:
    require_schema_version(aggregate, path, report)
    require_keys(
        aggregate,
        ("run_id", "split", "n_images", "seeds", "means", "composite", "composite_unweighted"),
        path, report,
    )
    if aggregate.get("run_id") != run_id:
        report.add("E_AGG_RUN_MISMATCH", path,
                   f"run_id {aggregate.get('run_id')!r} != {run_id!r}")
    if split is not None and aggregate.get("split") != split:
        report.add("E_AGG_SPLIT_MISMATCH", path,
                   f"split {aggregate.get('split')!r} != run split {split!r}")
    if aggregate.get("n_images") != len(image_ids):
        report.add("E_AGG_N_IMAGES", path,
                   f"n_images {aggregate.get('n_images')!r} "
                   f"!= len(image_ids) {len(image_ids)!r}")

    means = aggregate.get("means")
    if not isinstance(means, dict):
        report.add("E_AGG_MEANS_SHAPE", path, "means must be an object")
        return

    for m in metrics:
        if m not in means:
            report.add("E_AGG_MEANS_MISSING", path, f"means missing metric {m!r}")
            continue
        v = means[m]
        if not is_finite_unit_float(v):
            report.add("E_AGG_RANGE", path, f"means[{m!r}] {v!r} not finite float in [0, 1]")
            continue
        observed = per_image_means.get(m) or []
        if len(observed) != len(image_ids):
            continue
        expected = sum(observed) / len(observed) if observed else 0.0
        if not approx_equal(v, expected):
            report.add("E_AGG_MISMATCH", path,
                       f"means[{m!r}] = {v} disagrees with per-image mean {expected:.6f}")

    composite_unweighted = aggregate.get("composite_unweighted")
    if isinstance(composite_unweighted, (int, float)) and isinstance(means, dict) and metrics:
        recomputed = sum(float(means[m]) for m in metrics if m in means) / len(metrics)
        if not approx_equal(composite_unweighted, recomputed):
            report.add("E_COMPOSITE_FORMULA", path,
                       f"composite_unweighted {composite_unweighted} "
                       f"!= mean of means {recomputed:.6f}")


def check_gate(
    gate: dict,
    path: Any,
    aggregate: dict,
    metrics: list[str],
    report: Report,
) -> None:
    require_schema_version(gate, path, report)
    require_keys(
        gate,
        ("leader_run_id", "candidate_means", "candidate_composite",
         "regression_epsilon", "no_regression", "improves_composite",
         "single_run_gate", "decision"),
        path, report,
    )

    decision = gate.get("decision")
    if decision not in GATE_DECISIONS:
        report.add("E_GATE_DECISION", path,
                   f"decision {decision!r} not in {sorted(GATE_DECISIONS)}")

    single = gate.get("single_run_gate")
    if single not in GATE_OUTCOMES:
        report.add("E_GATE_OUTCOME", path,
                   f"single_run_gate {single!r} not in {sorted(GATE_OUTCOMES)}")
    three = gate.get("three_seed_gate")
    if three is not None and three not in GATE_OUTCOMES:
        report.add("E_GATE_OUTCOME", path,
                   f"three_seed_gate {three!r} not in {sorted(GATE_OUTCOMES)} or null")

    cand_means = gate.get("candidate_means")
    if not isinstance(cand_means, dict):
        report.add("E_GATE_MEANS_SHAPE", path, "candidate_means must be an object")
        return

    agg_means = aggregate.get("means") if isinstance(aggregate, dict) else None
    if isinstance(agg_means, dict):
        for m in metrics:
            if m in cand_means and m in agg_means:
                if not approx_equal(cand_means[m], agg_means[m]):
                    report.add(
                        "E_GATE_AGG_MISMATCH", path,
                        f"candidate_means[{m!r}] {cand_means[m]} "
                        f"!= aggregate means {agg_means[m]}",
                    )

    epsilon = gate.get("regression_epsilon")
    leader_run_id = gate.get("leader_run_id")
    leader_means = gate.get("leader_means")
    if leader_run_id is None:
        if decision != "no_leader":
            report.add("E_GATE_NO_LEADER_DECISION", path,
                       f"first run must use decision='no_leader', got {decision!r}")
        if gate.get("no_regression") is not True or gate.get("improves_composite") is not True:
            report.add("E_GATE_NO_LEADER_FLAGS", path,
                       "first run must set no_regression=true and improves_composite=true")
    else:
        if not isinstance(leader_means, dict):
            report.add("E_GATE_LEADER_MEANS", path,
                       "leader_means must be an object when leader_run_id is set")
        elif isinstance(epsilon, (int, float)):
            expected_no_regression = all(
                m in cand_means and m in leader_means
                and float(cand_means[m]) >= float(leader_means[m]) - float(epsilon)
                for m in metrics
            )
            if bool(gate.get("no_regression")) != expected_no_regression:
                report.add("E_GATE_NO_REGRESSION", path,
                           f"no_regression={gate.get('no_regression')} "
                           f"disagrees with computed {expected_no_regression}")
        leader_comp = gate.get("leader_composite")
        cand_comp = gate.get("candidate_composite")
        if isinstance(leader_comp, (int, float)) and isinstance(cand_comp, (int, float)):
            expected_improves = float(cand_comp) > float(leader_comp)
            if bool(gate.get("improves_composite")) != expected_improves:
                report.add("E_GATE_IMPROVES", path,
                           f"improves_composite={gate.get('improves_composite')} "
                           f"disagrees with computed {expected_improves}")


# --------------------------- leader ---------------------------


def check_leader(backend: Backend, report: Report, run_ids: set[str]) -> None:
    pointer_rel = "experiments/leader/pointer.json"
    history_rel = "experiments/leader/history.jsonl"
    pointer_disp = backend.format_path(pointer_rel)
    history_disp = backend.format_path(history_rel)

    pointer_present = backend.is_file(pointer_rel)
    history_present = backend.is_file(history_rel)
    if not pointer_present and not history_present:
        return  # no leader yet — acceptable for a fresh repo

    pointer: dict | None = None
    if pointer_present:
        loaded = load_json(backend, pointer_rel, report)
        if not isinstance(loaded, dict):
            if loaded is not None:
                report.add("E_LEADER_POINTER_SHAPE", pointer_disp,
                           "pointer.json must be an object")
        else:
            pointer = loaded
            require_schema_version(pointer, pointer_disp, report)
            require_keys(pointer, ("run_id", "composite", "means", "promoted_at"),
                         pointer_disp, report)
            ptr_run = pointer.get("run_id")
            if ptr_run is not None and ptr_run not in run_ids:
                report.add("E_LEADER_RUN_MISSING", pointer_disp,
                           f"pointer.run_id {ptr_run!r} has no matching run directory")

    history_lines: list[dict] = []
    if history_present:
        try:
            history_text = backend.read_text(history_rel)
        except Exception as exc:
            report.add("E_HISTORY_JSON", history_disp, f"could not read: {exc}")
            history_text = ""
        report.files_checked += 1
        for lineno, raw in enumerate(history_text.splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                report.add("E_HISTORY_JSON", history_disp,
                           f"line {lineno}: invalid JSON ({exc})")
                continue
            if not isinstance(obj, dict):
                report.add("E_HISTORY_SHAPE", history_disp, f"line {lineno}: not an object")
                continue
            if not all(k in obj for k in ("run_id", "composite", "promoted_at", "previous_run_id")):
                report.add("E_HISTORY_FIELDS", history_disp,
                           f"line {lineno}: missing required fields")
                continue
            if obj["run_id"] not in run_ids:
                report.add("E_HISTORY_RUN_MISSING", history_disp,
                           f"line {lineno}: run_id {obj['run_id']!r} "
                           f"has no matching run directory")
            history_lines.append(obj)

        for i in range(1, len(history_lines)):
            if history_lines[i]["previous_run_id"] != history_lines[i - 1]["run_id"]:
                report.add("E_HISTORY_CHAIN", history_disp,
                           f"line {i + 1}: previous_run_id does not point to prior leader")

    if pointer and history_lines:
        last = history_lines[-1]
        if last["run_id"] != pointer.get("run_id"):
            report.add("E_LEADER_POINTER_HISTORY", pointer_disp,
                       f"pointer.run_id {pointer.get('run_id')!r} "
                       f"!= last history run_id {last['run_id']!r}")
        if isinstance(last.get("composite"), (int, float)) \
                and isinstance(pointer.get("composite"), (int, float)):
            if not approx_equal(last["composite"], pointer["composite"]):
                report.add("E_LEADER_COMPOSITE", pointer_disp,
                           f"pointer.composite {pointer['composite']} "
                           f"!= last history composite {last['composite']}")


# --------------------------- logbook ---------------------------


LOGBOOK_RE = re.compile(
    r"^### (?P<run_id>\S+)\n"
    r"- driver: .+\n"
    r"- hypothesis: .+\n"
    r"- composite: -?\d+\.\d+\n"
    r"- s_\w+: -?\d+\.\d+(?: \| s_\w+: -?\d+\.\d+)*\n"
    r"- gate: (pass|fail)\n"
    r"- 3-seed re-eval: (?:n/a|-?\d+\.\d+ ± -?\d+\.\d+)\n"
    r"- val composite: (?:n/a|-?\d+\.\d+)\n"
    r"- wall_clock: .+\n"
    r"- est_cost_usd: .+\n"
    r"- takeaway: .+\n"
    r"- promoted: (yes|no|reverted)\n",
    re.MULTILINE,
)


def check_logbook(backend: Backend, report: Report, run_ids: set[str]) -> None:
    rel = "experiments/logbook.md"
    disp = backend.format_path(rel)
    if not backend.is_file(rel):
        report.add("E_FILE_MISSING", disp, "logbook.md missing")
        return
    report.files_checked += 1
    try:
        text = backend.read_text(rel)
    except Exception as exc:
        report.add("E_FILE_MISSING", disp, f"could not read: {exc}")
        return
    matches = list(LOGBOOK_RE.finditer(text))
    seen: set[str] = set()
    for m in matches:
        rid = m.group("run_id")
        if rid in seen:
            report.add("E_LOGBOOK_DUPLICATE", disp,
                       f"run_id {rid!r} appears more than once")
        seen.add(rid)
        if rid not in run_ids:
            report.add("E_LOGBOOK_RUN_MISSING", disp,
                       f"logbook entry {rid!r} has no matching run directory")
    for rid in run_ids - seen:
        report.add("E_LOGBOOK_ENTRY_MISSING", disp,
                   f"run {rid!r} has no logbook entry")


# --------------------------- entry point ---------------------------


def check_root(root: Backend | Path | str, verify_hashes: bool) -> Report:
    """Validate the repo at `root`.

    `root` may be:
      - a `Backend` instance (used as-is, e.g. for tests with a fake)
      - a `pathlib.Path` (treated as a local directory)
      - a `str`: either a local path or a `gs://bucket/prefix` URI
    """
    if isinstance(root, Backend):
        backend = root
    else:
        backend = make_backend(root)

    report = Report()

    for required in ("eval_data/images", "experiments/runs"):
        if not backend.is_dir(required):
            report.add("E_FILE_MISSING", backend.format_path(required),
                       "required directory missing")

    manifest_by_split = check_manifest(backend, report, verify_hashes)

    runs_rel = "experiments/runs"
    run_ids: set[str] = set()
    if backend.is_dir(runs_rel):
        subdirs, _ = backend.list_dir(runs_rel)
        for child in sorted(subdirs, key=lambda s: s["name"]):
            result = check_run(backend, child["path"], report,
                               manifest_by_split, verify_hashes)
            if result is not None:
                run_ids.add(child["name"])

    check_leader(backend, report, run_ids)
    if backend.is_file("experiments/logbook.md") or run_ids:
        check_logbook(backend, report, run_ids)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True,
                        help="repo root: local directory or gs://bucket/prefix")
    parser.add_argument("--verify-hashes", action="store_true",
                        help="recompute sha256 of every image referenced in the manifest")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON output")
    parser.add_argument("--gcs-cache-ttl", type=float, default=30.0,
                        help="seconds to cache GCS metadata listings (default: 30)")
    args = parser.parse_args(argv)

    try:
        backend = make_backend(args.root, gcs_cache_ttl=args.gcs_cache_ttl)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"E_ROOT_MISSING {args.root}: {exc}", file=sys.stderr)
        return 1

    report = check_root(backend, args.verify_hashes)

    if args.json:
        out = {
            "ok": report.ok,
            "files_checked": report.files_checked,
            "violations": [
                {"code": v.code, "path": v.path, "message": v.message}
                for v in report.violations
            ],
        }
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        for v in report.violations:
            print(f"{v.code} {v.path}: {v.message}")
        print(
            f"checked {report.files_checked} JSON file(s); "
            f"{len(report.violations)} violation(s)",
            file=sys.stderr,
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
