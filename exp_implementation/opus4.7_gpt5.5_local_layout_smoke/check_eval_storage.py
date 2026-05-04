#!/usr/bin/env python3
"""Verify a built repo conforms to EVAL_STORAGE_SCHEMA.md.

Usage:
    python check_eval_storage.py --root <repo-root> [--verify-hashes] [--json]

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

    def add(self, code: str, path: Path | str, message: str) -> None:
        self.violations.append(Violation(code, str(path), message))

    @property
    def ok(self) -> bool:
        return not self.violations


# --------------------------- helpers ---------------------------


def load_json(path: Path, report: Report) -> dict | list | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        report.add("E_FILE_MISSING", path, "required file is missing")
        return None
    except json.JSONDecodeError as exc:
        report.add("E_JSON_INVALID", path, f"invalid JSON: {exc}")
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


def require_keys(obj: dict, keys: Iterable[str], path: Path, report: Report, code: str = "E_FIELD_MISSING") -> bool:
    ok = True
    for k in keys:
        if k not in obj:
            report.add(code, path, f"missing required field '{k}'")
            ok = False
    return ok


def require_schema_version(obj: dict, path: Path, report: Report) -> None:
    if obj.get("schema_version") != SCHEMA_VERSION:
        report.add(
            "E_SCHEMA_VERSION",
            path,
            f"schema_version must be {SCHEMA_VERSION!r}, got {obj.get('schema_version')!r}",
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def approx_equal(a: float, b: float, tol: float = FLOAT_TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------- manifest ---------------------------


def check_manifest(root: Path, report: Report, verify_hashes: bool) -> dict[str, dict[str, dict]]:
    """Return {split: {image_id: entry}} for downstream cross-checks."""
    by_split: dict[str, dict[str, dict]] = {s: {} for s in SPLITS}

    manifest_path = root / "eval_data" / "images" / "manifest.json"
    data = load_json(manifest_path, report)
    if not isinstance(data, dict):
        if data is not None:
            report.add("E_MANIFEST_SHAPE", manifest_path, "manifest must be an object")
        return by_split

    require_schema_version(data, manifest_path, report)
    splits = data.get("splits")
    if not isinstance(splits, dict):
        report.add("E_MANIFEST_SHAPE", manifest_path, "missing 'splits' object")
        return by_split

    seen_ids: dict[str, str] = {}
    for split_name in SPLITS:
        entries = splits.get(split_name)
        if entries is None:
            report.add("E_MANIFEST_SPLIT", manifest_path, f"split {split_name!r} missing")
            continue
        if not isinstance(entries, list):
            report.add("E_MANIFEST_SPLIT", manifest_path, f"split {split_name!r} must be a list")
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                report.add("E_MANIFEST_ENTRY", manifest_path, f"non-object entry in {split_name}")
                continue
            if not require_keys(
                entry, ("image_id", "filename", "sha256", "width", "height"), manifest_path, report
            ):
                continue
            image_id = entry["image_id"]
            if image_id in seen_ids:
                report.add(
                    "E_MANIFEST_DUPLICATE",
                    manifest_path,
                    f"image_id {image_id!r} appears in both {seen_ids[image_id]!r} and {split_name!r}",
                )
                continue
            seen_ids[image_id] = split_name
            if not SHA256_RE.match(str(entry["sha256"])):
                report.add(
                    "E_MANIFEST_HASH_FORMAT",
                    manifest_path,
                    f"image_id {image_id!r} has malformed sha256",
                )
            if not (isinstance(entry["width"], int) and entry["width"] > 0):
                report.add("E_MANIFEST_DIMS", manifest_path, f"{image_id} width must be positive int")
            if not (isinstance(entry["height"], int) and entry["height"] > 0):
                report.add("E_MANIFEST_DIMS", manifest_path, f"{image_id} height must be positive int")
            by_split[split_name][image_id] = entry

            if verify_hashes:
                img_path = root / "eval_data" / "images" / split_name / entry["filename"]
                if not img_path.exists():
                    report.add("E_IMAGE_MISSING", img_path, "manifest entry references missing file")
                else:
                    actual = sha256_file(img_path)
                    if actual != entry["sha256"]:
                        report.add(
                            "E_IMAGE_HASH_MISMATCH",
                            img_path,
                            f"sha256 mismatch (manifest={entry['sha256']}, actual={actual})",
                        )
    return by_split


# --------------------------- per-run ---------------------------


def check_run(
    run_dir: Path,
    report: Report,
    manifest_by_split: dict[str, dict[str, dict]],
    verify_hashes: bool,
) -> dict[str, Any] | None:
    run_id = run_dir.name
    if not RUN_ID_RE.match(run_id):
        report.add("E_RUN_ID_FORMAT", run_dir, f"run_id {run_id!r} does not match canonical format")

    run_json = load_json(run_dir / "run.json", report)
    if not isinstance(run_json, dict):
        return None

    require_schema_version(run_json, run_dir / "run.json", report)
    require_keys(
        run_json,
        (
            "run_id", "name", "driver", "harness_variant",
            "started_at", "finished_at", "split", "image_ids",
            "seeds", "status",
        ),
        run_dir / "run.json",
        report,
    )

    if run_json.get("run_id") != run_id:
        report.add(
            "E_RUN_ID_MISMATCH",
            run_dir / "run.json",
            f"run.json#run_id {run_json.get('run_id')!r} does not match directory name {run_id!r}",
        )

    for ts_field in ("started_at", "finished_at"):
        ts = run_json.get(ts_field)
        if isinstance(ts, str) and not ISO_UTC_RE.match(ts):
            report.add(
                "E_TIMESTAMP_FORMAT",
                run_dir / "run.json",
                f"{ts_field} {ts!r} must match YYYY-MM-DDTHH:MM:SSZ",
            )

    status = run_json.get("status")
    if status not in RUN_STATUS:
        report.add("E_RUN_STATUS", run_dir / "run.json", f"status {status!r} not in {sorted(RUN_STATUS)}")

    split = run_json.get("split")
    if split not in SPLITS:
        report.add("E_RUN_SPLIT", run_dir / "run.json", f"split {split!r} not in {SPLITS}")

    image_ids = run_json.get("image_ids") or []
    if not isinstance(image_ids, list) or not all(isinstance(x, str) for x in image_ids):
        report.add("E_RUN_IMAGE_IDS", run_dir / "run.json", "image_ids must be a list of strings")
        image_ids = []
    elif split in SPLITS:
        manifest_ids = manifest_by_split.get(split, {})
        for img in image_ids:
            if img not in manifest_ids:
                report.add(
                    "E_RUN_IMAGE_UNKNOWN",
                    run_dir / "run.json",
                    f"image_id {img!r} not in manifest split {split!r}",
                )

    seeds = run_json.get("seeds") or []
    if not isinstance(seeds, list) or not seeds or not all(isinstance(x, int) for x in seeds):
        report.add("E_RUN_SEEDS", run_dir / "run.json", "seeds must be a non-empty list of ints")

    config = load_json(run_dir / "config.json", report)
    metrics: list[str] = []
    if isinstance(config, dict):
        require_schema_version(config, run_dir / "config.json", report)
        require_keys(config, ("harness_variant", "models", "metrics"), run_dir / "config.json", report)
        metrics = config.get("metrics") or []
        if not isinstance(metrics, list) or not metrics or not all(isinstance(m, str) for m in metrics):
            report.add("E_CONFIG_METRICS", run_dir / "config.json", "metrics must be a non-empty list of strings")
            metrics = []

    if not (run_dir / "prompt_strategy.py").exists():
        report.add("E_FILE_MISSING", run_dir / "prompt_strategy.py", "strategy snapshot missing")
    if not (run_dir / "stdout.log").exists():
        report.add("E_FILE_MISSING", run_dir / "stdout.log", "stdout.log missing")

    per_image_dir = run_dir / "per_image"
    per_image_means: dict[str, list[float]] = {m: [] for m in metrics}
    if not per_image_dir.is_dir():
        report.add("E_FILE_MISSING", per_image_dir, "per_image/ directory missing")
    else:
        for img in image_ids:
            img_dir = per_image_dir / img
            if not img_dir.is_dir():
                report.add("E_FILE_MISSING", img_dir, f"per_image/{img}/ missing")
                continue
            for fname in ("prompt.txt", "generated.png", "scores.json"):
                if not (img_dir / fname).exists() and not (
                    fname == "generated.png" and len(seeds) > 1
                ):
                    # For multi-seed runs the top-level generated.png is optional.
                    if fname == "generated.png":
                        report.add("E_FILE_MISSING", img_dir / fname, "generated.png missing")
                    else:
                        report.add("E_FILE_MISSING", img_dir / fname, f"{fname} missing")

            scores = load_json(img_dir / "scores.json", report)
            if isinstance(scores, dict):
                check_scores(scores, img_dir / "scores.json", img, metrics, report,
                             multi_seed=len(seeds) > 1)
                got = scores.get("scores") or {}
                for m in metrics:
                    if m in got and is_finite_unit_float(got[m]):
                        per_image_means[m].append(float(got[m]))

                if verify_hashes and (img_dir / "generated.png").exists():
                    expected = scores.get("generated_image_sha256")
                    if isinstance(expected, str) and SHA256_RE.match(expected):
                        actual = sha256_file(img_dir / "generated.png")
                        if actual != expected:
                            report.add(
                                "E_GENERATED_HASH_MISMATCH",
                                img_dir / "generated.png",
                                f"sha256 mismatch (expected={expected}, actual={actual})",
                            )

            seeds_dir = img_dir / "seeds"
            if len(seeds) > 1:
                if not seeds_dir.is_dir():
                    report.add("E_FILE_MISSING", seeds_dir, "multi-seed run missing seeds/ subdir")
                else:
                    for seed in seeds:
                        seed_dir = seeds_dir / str(seed)
                        if not seed_dir.is_dir():
                            report.add("E_FILE_MISSING", seed_dir, f"seeds/{seed}/ missing")
                            continue
                        for fname in ("generated.png", "scores.json"):
                            if not (seed_dir / fname).exists():
                                report.add("E_FILE_MISSING", seed_dir / fname, f"{fname} missing")
                        seed_scores = load_json(seed_dir / "scores.json", report)
                        if isinstance(seed_scores, dict):
                            check_scores(seed_scores, seed_dir / "scores.json", img, metrics, report,
                                         multi_seed=False, expect_seed=seed)

    aggregate = load_json(run_dir / "aggregate.json", report)
    if isinstance(aggregate, dict):
        check_aggregate(aggregate, run_dir / "aggregate.json", run_id, split, image_ids,
                        metrics, per_image_means, report)

    gate = load_json(run_dir / "gate.json", report)
    if isinstance(gate, dict) and isinstance(aggregate, dict):
        check_gate(gate, run_dir / "gate.json", aggregate, metrics, report)

    return {"run_id": run_id, "aggregate": aggregate}


def check_scores(
    scores: dict,
    path: Path,
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
        path,
        report,
    )
    if scores.get("image_id") != expected_image:
        report.add("E_SCORES_IMAGE_MISMATCH", path,
                   f"image_id {scores.get('image_id')!r} != directory image_id {expected_image!r}")

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
    path: Path,
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
        path,
        report,
    )
    if aggregate.get("run_id") != run_id:
        report.add("E_AGG_RUN_MISMATCH", path,
                   f"run_id {aggregate.get('run_id')!r} != {run_id!r}")
    if split is not None and aggregate.get("split") != split:
        report.add("E_AGG_SPLIT_MISMATCH", path,
                   f"split {aggregate.get('split')!r} != run split {split!r}")
    if aggregate.get("n_images") != len(image_ids):
        report.add(
            "E_AGG_N_IMAGES",
            path,
            f"n_images {aggregate.get('n_images')!r} != len(image_ids) {len(image_ids)!r}",
        )

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
            # Already reported elsewhere; skip mean comparison.
            continue
        expected = sum(observed) / len(observed) if observed else 0.0
        if not approx_equal(v, expected):
            report.add(
                "E_AGG_MISMATCH",
                path,
                f"means[{m!r}] = {v} disagrees with per-image mean {expected:.6f}",
            )

    composite_unweighted = aggregate.get("composite_unweighted")
    if isinstance(composite_unweighted, (int, float)) and isinstance(means, dict) and metrics:
        recomputed = sum(float(means[m]) for m in metrics if m in means) / len(metrics)
        if not approx_equal(composite_unweighted, recomputed):
            report.add(
                "E_COMPOSITE_FORMULA",
                path,
                f"composite_unweighted {composite_unweighted} != mean of means {recomputed:.6f}",
            )


def check_gate(
    gate: dict,
    path: Path,
    aggregate: dict,
    metrics: list[str],
    report: Report,
) -> None:
    require_schema_version(gate, path, report)
    require_keys(
        gate,
        (
            "leader_run_id", "candidate_means", "candidate_composite",
            "regression_epsilon", "no_regression", "improves_composite",
            "single_run_gate", "decision",
        ),
        path,
        report,
    )

    decision = gate.get("decision")
    if decision not in GATE_DECISIONS:
        report.add("E_GATE_DECISION", path, f"decision {decision!r} not in {sorted(GATE_DECISIONS)}")

    single = gate.get("single_run_gate")
    if single not in GATE_OUTCOMES:
        report.add("E_GATE_OUTCOME", path, f"single_run_gate {single!r} not in {sorted(GATE_OUTCOMES)}")
    three = gate.get("three_seed_gate")
    if three is not None and three not in GATE_OUTCOMES:
        report.add("E_GATE_OUTCOME", path, f"three_seed_gate {three!r} not in {sorted(GATE_OUTCOMES)} or null")

    cand_means = gate.get("candidate_means") or {}
    if not isinstance(cand_means, dict):
        report.add("E_GATE_MEANS_SHAPE", path, "candidate_means must be an object")
        return

    agg_means = aggregate.get("means") if isinstance(aggregate, dict) else None
    if isinstance(agg_means, dict):
        for m in metrics:
            if m in cand_means and m in agg_means:
                if not approx_equal(cand_means[m], agg_means[m]):
                    report.add(
                        "E_GATE_AGG_MISMATCH",
                        path,
                        f"candidate_means[{m!r}] {cand_means[m]} != aggregate means {agg_means[m]}",
                    )

    epsilon = gate.get("regression_epsilon")
    leader_run_id = gate.get("leader_run_id")
    leader_means = gate.get("leader_means")
    if leader_run_id is None:
        if decision != "no_leader":
            report.add(
                "E_GATE_NO_LEADER_DECISION",
                path,
                f"first run must use decision='no_leader', got {decision!r}",
            )
        if gate.get("no_regression") is not True or gate.get("improves_composite") is not True:
            report.add(
                "E_GATE_NO_LEADER_FLAGS",
                path,
                "first run must set no_regression=true and improves_composite=true",
            )
    else:
        if not isinstance(leader_means, dict):
            report.add("E_GATE_LEADER_MEANS", path, "leader_means must be an object when leader_run_id is set")
        elif isinstance(epsilon, (int, float)):
            expected_no_regression = all(
                m in cand_means and m in leader_means
                and float(cand_means[m]) >= float(leader_means[m]) - float(epsilon)
                for m in metrics
            )
            if bool(gate.get("no_regression")) != expected_no_regression:
                report.add(
                    "E_GATE_NO_REGRESSION",
                    path,
                    f"no_regression={gate.get('no_regression')} disagrees with computed {expected_no_regression}",
                )
        leader_comp = gate.get("leader_composite")
        cand_comp = gate.get("candidate_composite")
        if isinstance(leader_comp, (int, float)) and isinstance(cand_comp, (int, float)):
            expected_improves = float(cand_comp) > float(leader_comp)
            if bool(gate.get("improves_composite")) != expected_improves:
                report.add(
                    "E_GATE_IMPROVES",
                    path,
                    f"improves_composite={gate.get('improves_composite')} disagrees with computed {expected_improves}",
                )


# --------------------------- leader ---------------------------


def check_leader(root: Path, report: Report, run_ids: set[str]) -> None:
    pointer_path = root / "experiments" / "leader" / "pointer.json"
    history_path = root / "experiments" / "leader" / "history.jsonl"

    pointer = load_json(pointer_path, report) if pointer_path.exists() else None
    if pointer is None and not history_path.exists():
        # No leader yet — acceptable for a fresh repo.
        return

    if pointer is not None:
        if not isinstance(pointer, dict):
            report.add("E_LEADER_POINTER_SHAPE", pointer_path, "pointer.json must be an object")
            pointer = None
        else:
            require_schema_version(pointer, pointer_path, report)
            require_keys(pointer, ("run_id", "composite", "means", "promoted_at"), pointer_path, report)
            ptr_run = pointer.get("run_id")
            if ptr_run is not None and ptr_run not in run_ids:
                report.add(
                    "E_LEADER_RUN_MISSING",
                    pointer_path,
                    f"pointer.run_id {ptr_run!r} has no matching run directory",
                )

    history_lines: list[dict] = []
    if history_path.exists():
        report.files_checked += 1
        with history_path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    report.add("E_HISTORY_JSON", history_path, f"line {lineno}: invalid JSON ({exc})")
                    continue
                if not isinstance(obj, dict):
                    report.add("E_HISTORY_SHAPE", history_path, f"line {lineno}: not an object")
                    continue
                if not all(k in obj for k in ("run_id", "composite", "promoted_at", "previous_run_id")):
                    report.add("E_HISTORY_FIELDS", history_path, f"line {lineno}: missing required fields")
                    continue
                if obj["run_id"] not in run_ids:
                    report.add(
                        "E_HISTORY_RUN_MISSING",
                        history_path,
                        f"line {lineno}: run_id {obj['run_id']!r} has no matching run directory",
                    )
                history_lines.append(obj)

        for i in range(1, len(history_lines)):
            if history_lines[i]["previous_run_id"] != history_lines[i - 1]["run_id"]:
                report.add(
                    "E_HISTORY_CHAIN",
                    history_path,
                    f"line {i + 1}: previous_run_id does not point to prior leader",
                )

    if pointer and history_lines:
        last = history_lines[-1]
        if last["run_id"] != pointer.get("run_id"):
            report.add(
                "E_LEADER_POINTER_HISTORY",
                pointer_path,
                f"pointer.run_id {pointer.get('run_id')!r} != last history run_id {last['run_id']!r}",
            )
        if isinstance(last.get("composite"), (int, float)) and isinstance(pointer.get("composite"), (int, float)):
            if not approx_equal(last["composite"], pointer["composite"]):
                report.add(
                    "E_LEADER_COMPOSITE",
                    pointer_path,
                    f"pointer.composite {pointer['composite']} != last history composite {last['composite']}",
                )


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


def check_logbook(root: Path, report: Report, run_ids: set[str]) -> None:
    path = root / "experiments" / "logbook.md"
    if not path.exists():
        report.add("E_FILE_MISSING", path, "logbook.md missing")
        return
    report.files_checked += 1
    text = path.read_text(encoding="utf-8")
    matches = list(LOGBOOK_RE.finditer(text))
    seen: set[str] = set()
    for m in matches:
        rid = m.group("run_id")
        if rid in seen:
            report.add("E_LOGBOOK_DUPLICATE", path, f"run_id {rid!r} appears more than once")
        seen.add(rid)
        if rid not in run_ids:
            report.add(
                "E_LOGBOOK_RUN_MISSING",
                path,
                f"logbook entry {rid!r} has no matching run directory",
            )
    for rid in run_ids - seen:
        report.add("E_LOGBOOK_ENTRY_MISSING", path, f"run {rid!r} has no logbook entry")


# --------------------------- entry point ---------------------------


def check_root(root: Path, verify_hashes: bool) -> Report:
    report = Report()

    for required in (
        root / "eval_data" / "images",
        root / "experiments" / "runs",
    ):
        if not required.is_dir():
            report.add("E_FILE_MISSING", required, "required directory missing")

    manifest_by_split = check_manifest(root, report, verify_hashes)

    runs_dir = root / "experiments" / "runs"
    run_ids: set[str] = set()
    if runs_dir.is_dir():
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            result = check_run(run_dir, report, manifest_by_split, verify_hashes)
            if result is not None:
                run_ids.add(run_dir.name)

    check_leader(root, report, run_ids)
    if (root / "experiments" / "logbook.md").exists() or run_ids:
        check_logbook(root, report, run_ids)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="repo root to validate")
    parser.add_argument("--verify-hashes", action="store_true",
                        help="recompute sha256 of every image referenced in the manifest")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON output")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"E_ROOT_MISSING {root}: not a directory", file=sys.stderr)
        return 1

    report = check_root(root, args.verify_hashes)

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
