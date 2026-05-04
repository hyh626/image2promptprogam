#!/usr/bin/env python3
"""Run the fixed image-to-prompt-to-image autoresearch evaluation.

Logbook entries use EVAL_STORAGE_SCHEMA.md's canonical format. The driver
and hypothesis fields come from AUTORESEARCH_DRIVER and
AUTORESEARCH_HYPOTHESIS when set; otherwise they are written as placeholders
that a driver agent can replace after the run.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

ROOT = Path(__file__).resolve().parent
EVAL_IMAGES_ROOT = ROOT / "eval_data" / "images"
EXPERIMENTS_ROOT = ROOT / "experiments"
RUNS_ROOT = EXPERIMENTS_ROOT / "runs"
LEADER_ROOT = EXPERIMENTS_ROOT / "leader"
LOGBOOK_PATH = EXPERIMENTS_ROOT / "logbook.md"
PROMPT_STRATEGY_PATH = ROOT / "prompt_strategy.py"

SCHEMA_VERSION = "1.0.0"
HARNESS_VARIANT = "opus4.7"
METRIC_KEYS = ("s_gemini", "s_dino", "s_lpips", "s_color")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SPLITS = ("train", "eval", "val", "holdout")
EXPECTED_COUNTS = {"eval": 20, "val": 5}
REGRESSION_EPSILON = 0.01

GENERATOR_MODEL = "gemini-3.1-flash-image-preview"
VLM_MODEL = "gemini-3.1-flash-lite-preview"
EMBEDDING_MODEL = "gemini-embedding-2"


@dataclass
class TargetImage:
    image_id: str
    path: Path
    width: int
    height: int
    sha256: str


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()

    def emit(self, message: str = "") -> None:
        print(message)
        self._fh.write(message + "\n")
        self._fh.flush()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def slug(value: str, *, default: str = "run") -> str:
    out = value.strip().lower().replace(" ", "-")
    out = re.sub(r"[^a-z0-9_.-]+", "-", out)
    out = re.sub(r"-+", "-", out).strip("-._")
    return out or default


def image_id_for(path: Path) -> str:
    out = path.stem.lower().replace(" ", "_")
    out = re.sub(r"[^a-z0-9_.-]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "image"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def image_files(split: str) -> list[Path]:
    split_dir = EVAL_IMAGES_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        p for p in split_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def has_transparency(image: Image.Image) -> bool:
    if image.mode in ("RGBA", "LA"):
        alpha = image.getchannel("A")
        return alpha.getextrema()[0] < 255
    return "transparency" in image.info


def sync_manifest() -> dict:
    manifest_path = EVAL_IMAGES_ROOT / "manifest.json"
    existing = read_json(manifest_path) or {
        "schema_version": SCHEMA_VERSION,
        "splits": {split: [] for split in SPLITS},
    }
    old_entries = {
        split: {
            item.get("image_id"): item
            for item in existing.get("splits", {}).get(split, [])
            if isinstance(item, dict)
        }
        for split in SPLITS
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "splits": {split: [] for split in SPLITS},
    }
    seen_ids: dict[str, str] = {}
    for split in SPLITS:
        seen_in_split: set[str] = set()
        entries = []
        for path in image_files(split):
            image_id = image_id_for(path)
            if image_id in seen_in_split:
                raise SystemExit(f"duplicate image_id {image_id!r} in {split}; rename one source file")
            if image_id in seen_ids:
                raise SystemExit(
                    f"image_id {image_id!r} appears in both {seen_ids[image_id]} and {split}"
                )
            seen_ids[image_id] = split
            seen_in_split.add(image_id)
            with Image.open(path) as image:
                width, height = image.size
            previous = old_entries.get(split, {}).get(image_id, {})
            entry = {
                "image_id": image_id,
                "filename": path.name,
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
            }
            for key in ("source", "category", "license", "notes"):
                if key in previous:
                    entry[key] = previous[key]
            entries.append(entry)
        manifest["splits"][split] = sorted(entries, key=lambda e: e["image_id"])

    write_json(manifest_path, manifest)
    return manifest


def validate_inputs(split: str) -> list[TargetImage]:
    manifest = sync_manifest()
    errors: list[str] = []

    eval_entries = manifest["splits"]["eval"]
    val_entries = manifest["splits"]["val"]
    if not eval_entries:
        errors.append(
            "eval_images/ is empty; populate it with 20 reference images before running the harness"
        )
    if eval_entries and len(eval_entries) != EXPECTED_COUNTS["eval"]:
        errors.append(f"eval_images/ must contain exactly 20 images, found {len(eval_entries)}")
    if val_entries and len(val_entries) != EXPECTED_COUNTS["val"]:
        errors.append(f"val_images/ must contain exactly 5 images, found {len(val_entries)}")
    if eval_entries and not val_entries:
        errors.append("val_images/ is empty; populate it with 5 held-out images before running the harness")

    eval_hashes = {e["sha256"]: e["image_id"] for e in eval_entries}
    for entry in val_entries:
        if entry["sha256"] in eval_hashes:
            errors.append(
                f"duplicate image bytes between eval/{eval_hashes[entry['sha256']]} "
                f"and val/{entry['image_id']}"
            )

    for check_split in ("eval", "val"):
        for entry in manifest["splits"][check_split]:
            path = EVAL_IMAGES_ROOT / check_split / entry["filename"]
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.width < 512 or image.height < 512:
                        errors.append(f"{check_split}/{entry['filename']} is smaller than 512x512")
                    if has_transparency(image):
                        errors.append(f"{check_split}/{entry['filename']} has transparency")
            except Exception as exc:  # noqa: BLE001 - report all image load failures together.
                errors.append(f"{check_split}/{entry['filename']} does not load with PIL: {exc}")

    if errors:
        raise SystemExit("\n".join(errors))

    target_entries = manifest["splits"][split]
    if split in EXPECTED_COUNTS and len(target_entries) != EXPECTED_COUNTS[split]:
        raise SystemExit(
            f"{split}_images/ must contain exactly {EXPECTED_COUNTS[split]} images, "
            f"found {len(target_entries)}"
        )

    return [
        TargetImage(
            image_id=entry["image_id"],
            path=EVAL_IMAGES_ROOT / split / entry["filename"],
            width=entry["width"],
            height=entry["height"],
            sha256=entry["sha256"],
        )
        for entry in target_entries
    ]


def aspect_ratio(width: int, height: int) -> str:
    actual = width / height
    candidates = {
        "1:1": 1.0,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
    }
    return min(candidates, key=lambda key: abs(candidates[key] - actual))


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def extract_image_bytes(response: object) -> bytes:
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            data = getattr(inline_data, "data", None)
            if data is None:
                continue
            if isinstance(data, str):
                return base64.b64decode(data)
            return bytes(data)
    raise RuntimeError("Gemini image generation response did not include image bytes.")


def generation_config(aspect: str, seed: int) -> list[Any]:
    configs: list[Any] = []
    try:
        configs.append(
            types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                seed=seed,
                image_config=types.ImageConfig(aspect_ratio=aspect),
            )
        )
    except Exception:
        pass
    configs.append(
        {
            "response_modalities": ["IMAGE"],
            "seed": seed,
            "image_config": {"aspect_ratio": aspect},
        }
    )
    return configs


def generate_image(client: genai.Client, prompt: str, ref: TargetImage, seed: int, out_path: Path) -> None:
    from embed_and_score import retry_with_backoff

    aspect = aspect_ratio(ref.width, ref.height)

    def call() -> object:
        last_exc: Exception | None = None
        for config in generation_config(aspect, seed):
            try:
                return client.models.generate_content(
                    model=GENERATOR_MODEL,
                    contents=prompt,
                    config=config,
                )
            except TypeError as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    response = retry_with_backoff(call)
    raw = extract_image_bytes(response)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.convert("RGB").save(out_path, format="PNG")


def metric_mean(scores: list[dict]) -> dict:
    return {
        metric: float(np.mean([float(score[metric]) for score in scores]))
        for metric in METRIC_KEYS
    }


def metric_std(scores: list[dict]) -> dict:
    return {
        metric: float(np.std([float(score[metric]) for score in scores]))
        for metric in METRIC_KEYS
    }


def composite(means: dict) -> float:
    return float(np.mean([float(means[metric]) for metric in METRIC_KEYS]))


def score_record(
    image_id: str,
    seed: int | None,
    scores: dict,
    prompt_sha: str,
    generated_sha: str | None,
    generation_seconds: float | None,
    scoring_seconds: float | None,
    judge: dict | None = None,
    per_seed: list[int] | None = None,
) -> dict:
    record = {
        "schema_version": SCHEMA_VERSION,
        "generated_image_sha256": generated_sha,
        "generation_seconds": generation_seconds,
        "image_id": image_id,
        "judge": judge,
        "prompt_sha256": prompt_sha,
        "scores": scores,
        "scoring_seconds": scoring_seconds,
        "seed": seed,
    }
    if per_seed is not None:
        record["per_seed"] = per_seed
    return record


def run_scoring_pass(
    *,
    client: genai.Client,
    targets: list[TargetImage],
    seeds: list[int],
    run_dir: Path,
    no_judge: bool,
    logger: RunLogger,
    confirmation: bool = False,
) -> tuple[list[dict], dict]:
    from embed_and_score import featurize, featurize_original, similarity, vlm_judge
    import prompt_strategy

    per_image_scores: list[dict] = []
    seed_composites: dict[int, list[float]] = {seed: [] for seed in seeds}
    root = run_dir / "confirmation" if confirmation else run_dir

    for target in targets:
        image_dir = root / "per_image" / target.image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(target.path) as opened:
            original = opened.convert("RGB")

        prompt_start = time.monotonic()
        prompt = prompt_strategy.image_to_prompt(original)
        prompt_seconds = time.monotonic() - prompt_start
        prompt_path = image_dir / "prompt.txt"
        prompt_path.write_text(prompt.rstrip() + "\n", encoding="utf-8")
        prompt_sha = sha256_file(prompt_path)

        orig_feat = featurize_original(target.path)
        seed_scores: list[dict] = []
        for seed in seeds:
            seed_dir = image_dir / "seeds" / str(seed)
            generated_path = seed_dir / "generated.png"

            generation_start = time.monotonic()
            generate_image(client, prompt, target, seed, generated_path)
            generation_seconds = time.monotonic() - generation_start

            with Image.open(generated_path) as regen_opened:
                regen = regen_opened.convert("RGB")

            scoring_start = time.monotonic()
            regen_feat = featurize(regen)
            scores = similarity(orig_feat, regen_feat)
            scoring_seconds = time.monotonic() - scoring_start

            judge = None if no_judge else vlm_judge(original, regen)
            generated_sha = sha256_file(generated_path)
            seed_record = score_record(
                target.image_id,
                seed,
                scores,
                prompt_sha,
                generated_sha,
                generation_seconds,
                scoring_seconds,
                judge=judge,
            )
            write_json(seed_dir / "scores.json", seed_record)
            if judge is not None:
                write_json(seed_dir / "judge.json", {"schema_version": SCHEMA_VERSION, "judge": judge})
            seed_scores.append(scores)
            seed_composites[seed].append(composite(scores))
            logger.emit(
                f"{target.image_id} seed={seed} prompt_seconds={prompt_seconds:.2f} "
                f"generation_seconds={generation_seconds:.2f} scoring_seconds={scoring_seconds:.2f}"
            )

        image_mean = metric_mean(seed_scores)
        per_image_scores.append({"image_id": target.image_id, **image_mean})
        write_json(
            image_dir / "scores.json",
            score_record(
                target.image_id,
                None,
                image_mean,
                prompt_sha,
                None,
                None,
                None,
                judge=None,
                per_seed=seeds,
            ),
        )

    seed_summary = {
        seed: float(np.mean(values)) if values else 0.0
        for seed, values in seed_composites.items()
    }
    return per_image_scores, seed_summary


def print_table(logger: RunLogger, per_image_scores: list[dict], aggregate: dict) -> None:
    logger.emit("image_id    s_gemini  s_dino   s_lpips  s_color")
    for item in per_image_scores:
        logger.emit(
            f"{item['image_id']:<11} "
            f"{item['s_gemini']:.3f}     {item['s_dino']:.3f}    "
            f"{item['s_lpips']:.3f}    {item['s_color']:.3f}"
        )
    logger.emit("------------------------------------------------")
    means = aggregate["means"]
    logger.emit(
        f"{'mean':<11} {means['s_gemini']:.3f}     {means['s_dino']:.3f}    "
        f"{means['s_lpips']:.3f}    {means['s_color']:.3f}"
    )
    logger.emit(f"composite   {aggregate['composite']:.4f}")


def build_aggregate(run_id: str, split: str, seeds: list[int], per_image_scores: list[dict]) -> dict:
    score_maps = [{metric: item[metric] for metric in METRIC_KEYS} for item in per_image_scores]
    means = metric_mean(score_maps)
    stds = metric_std(score_maps)
    comp = composite(means)
    return {
        "schema_version": SCHEMA_VERSION,
        "composite": comp,
        "composite_unweighted": comp,
        "means": means,
        "n_images": len(per_image_scores),
        "run_id": run_id,
        "seeds": seeds,
        "split": split,
        "stds": stds,
        "three_seed": {
            "mean_composite": None,
            "ran": False,
            "std_composite": None,
        },
    }


def load_leader() -> dict | None:
    return read_json(LEADER_ROOT / "pointer.json")


def update_leader(run_id: str, aggregate: dict, previous: dict | None) -> None:
    LEADER_ROOT.mkdir(parents=True, exist_ok=True)
    promoted_at = utc_now()
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "composite": aggregate["composite"],
        "means": aggregate["means"],
        "promoted_at": promoted_at,
        "run_id": run_id,
    }
    tmp = LEADER_ROOT / "pointer.json.tmp"
    write_json(tmp, pointer)
    os.replace(tmp, LEADER_ROOT / "pointer.json")

    history = {
        "composite": aggregate["composite"],
        "previous_run_id": previous["run_id"] if previous else None,
        "promoted_at": promoted_at,
        "run_id": run_id,
    }
    with (LEADER_ROOT / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(history), sort_keys=True) + "\n")


def config_json(args: argparse.Namespace, cli_args: list[str]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_resolution": [448, 448],
        "cli_args": cli_args,
        "extra": {},
        "harness_variant": HARNESS_VARIANT,
        "metrics": list(METRIC_KEYS),
        "models": {
            "color": "hsv-8x8x8-chi2",
            "embedding": EMBEDDING_MODEL,
            "generator": GENERATOR_MODEL,
            "perceptual": "lpips-alex",
            "structural": "facebook/dinov2-base",
            "vlm": VLM_MODEL,
        },
        "promotion_gate": {
            "improvement_strict": True,
            "reeval_seeds": args.seeds,
            "regression_epsilon": REGRESSION_EPSILON,
        },
    }


def append_logbook(
    *,
    run_id: str,
    driver: str,
    aggregate: dict,
    gate_pass: bool,
    reeval_text: str,
    val_text: str,
    wall_clock_seconds: float,
    promoted: str,
) -> None:
    LOGBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOGBOOK_PATH.exists():
        LOGBOOK_PATH.write_text("# Logbook\n", encoding="utf-8")
    hypothesis = os.environ.get("AUTORESEARCH_HYPOTHESIS", "<TODO>")
    takeaway = os.environ.get("AUTORESEARCH_TAKEAWAY", "<TODO>")
    means = aggregate["means"]
    block = (
        f"\n### {run_id}\n"
        f"- driver: {driver}\n"
        f"- hypothesis: {hypothesis}\n"
        f"- composite: {aggregate['composite']:.4f}\n"
        f"- s_gemini: {means['s_gemini']:.3f} | s_dino: {means['s_dino']:.3f} | "
        f"s_lpips: {means['s_lpips']:.3f} | s_color: {means['s_color']:.3f}\n"
        f"- gate: {'pass' if gate_pass else 'fail'}\n"
        f"- 3-seed re-eval: {reeval_text}\n"
        f"- val composite: {val_text}\n"
        f"- wall_clock: {wall_clock_seconds / 60:.1f} min\n"
        f"- est_cost_usd: 0.00\n"
        f"- takeaway: {takeaway}\n"
        f"- promoted: {promoted}\n"
    )
    with LOGBOOK_PATH.open("a", encoding="utf-8") as fh:
        fh.write(block)


def run_harness(args: argparse.Namespace, cli_args: list[str]) -> int:
    load_dotenv()
    split = "val" if args.val else "eval"
    name = args.name or "val"
    if not args.val and not args.name:
        raise SystemExit("--name is required unless --val is set")
    if args.seeds < 3:
        raise SystemExit("--seeds must be >= 3 for eval and val runs")

    targets = validate_inputs(split)
    driver = slug(os.environ.get("AUTORESEARCH_DRIVER", "gpt-5-codex"), default="driver")
    name_slug = slug(name, default="run")
    run_id = f"{utc_stamp()}__{driver}__{name_slug}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    shutil.copy2(PROMPT_STRATEGY_PATH, run_dir / "prompt_strategy.py")
    write_json(run_dir / "config.json", config_json(args, cli_args))

    logger = RunLogger(run_dir / "stdout.log")
    started_at = utc_now()
    start = time.monotonic()
    seeds = list(range(args.seeds))

    try:
        from embed_and_score import gate

        client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
        per_image_scores, _seed_summary = run_scoring_pass(
            client=client,
            targets=targets,
            seeds=seeds,
            run_dir=run_dir,
            no_judge=args.no_judge,
            logger=logger,
        )
        aggregate = build_aggregate(run_id, split, seeds, per_image_scores)
        write_json(run_dir / "aggregate.json", aggregate)
        print_table(logger, per_image_scores, aggregate)
        logger.emit(f"composite={aggregate['composite']:.4f}")

        leader = load_leader()
        leader_means = None if leader is None else leader["means"]
        leader_composite = None if leader is None else float(leader["composite"])
        no_regression, gate_reason = gate(aggregate["means"], leader_means, REGRESSION_EPSILON)
        improves = True if leader is None else aggregate["composite"] > leader_composite
        single_pass = no_regression and improves
        decision = "no_leader" if leader is None else ("promoted" if single_pass else "rejected")
        three_seed_gate: str | None = None
        promoted = "no"
        reeval_text = "n/a"

        if args.val:
            decision = "no_leader" if leader is None else "rejected"
            promoted = "no"
            gate_reason = "validation run; promotion disabled"
        elif leader is None:
            update_leader(run_id, aggregate, None)
            promoted = "yes"
            gate_reason = "no previous leader; first eval run becomes leader"
        elif single_pass:
            logger.emit("Candidate passes gate. Running multi-seed confirmation re-eval...")
            confirmation_seeds = list(range(1000, 1000 + args.seeds))
            confirmation_scores, confirmation_seed_summary = run_scoring_pass(
                client=client,
                targets=targets,
                seeds=confirmation_seeds,
                run_dir=run_dir,
                no_judge=args.no_judge,
                logger=logger,
                confirmation=True,
            )
            confirmation_aggregate = build_aggregate(
                run_id,
                split,
                confirmation_seeds,
                confirmation_scores,
            )
            confirmation_mean = confirmation_aggregate["composite"]
            confirmation_std = float(np.std(list(confirmation_seed_summary.values())))
            reeval_text = f"{confirmation_mean:.4f} \u00b1 {confirmation_std:.4f}"
            confirmation_no_regression, confirmation_reason = gate(
                confirmation_aggregate["means"],
                leader_means,
                REGRESSION_EPSILON,
            )
            confirmation_improves = confirmation_mean > leader_composite
            confirmation_pass = confirmation_no_regression and confirmation_improves
            three_seed_gate = "pass" if confirmation_pass else "fail"
            aggregate["three_seed"] = {
                "mean_composite": confirmation_mean,
                "ran": True,
                "std_composite": confirmation_std,
            }
            write_json(run_dir / "aggregate.json", aggregate)
            write_json(run_dir / "confirmation" / "aggregate.json", confirmation_aggregate)
            if confirmation_pass:
                pointer_aggregate = {
                    **confirmation_aggregate,
                    "run_id": run_id,
                }
                update_leader(run_id, pointer_aggregate, leader)
                promoted = "yes"
                decision = "promoted"
                gate_reason = (
                    f"{gate_reason}; confirmation held: {confirmation_reason}; "
                    f"3-seed mean composite {confirmation_mean:.4f} beats leader"
                )
                logger.emit("PROMOTED. New leader.")
            else:
                promoted = "reverted"
                decision = "reverted_after_reeval"
                gate_reason = (
                    f"{gate_reason}; confirmation failed: {confirmation_reason}; "
                    f"confirmation_improves={confirmation_improves}"
                )
                logger.emit("REVERTED. multi-seed re-eval did not hold.")

        gate_record = {
            "schema_version": SCHEMA_VERSION,
            "candidate_composite": aggregate["composite"],
            "candidate_means": aggregate["means"],
            "decision": decision,
            "improves_composite": improves,
            "leader_composite": leader_composite,
            "leader_means": leader_means,
            "leader_run_id": None if leader is None else leader["run_id"],
            "no_regression": no_regression,
            "reason": gate_reason,
            "regression_epsilon": REGRESSION_EPSILON,
            "single_run_gate": "pass" if single_pass else "fail",
            "three_seed_gate": three_seed_gate,
        }
        if leader is None:
            gate_record["improves_composite"] = True
            gate_record["no_regression"] = True
            gate_record["single_run_gate"] = "pass"
        write_json(run_dir / "gate.json", gate_record)

        wall_clock = time.monotonic() - start
        run_json = {
            "schema_version": SCHEMA_VERSION,
            "driver": driver,
            "est_cost_usd": 0.0,
            "finished_at": utc_now(),
            "git_commit": git_commit(),
            "harness_variant": HARNESS_VARIANT,
            "hypothesis": os.environ.get("AUTORESEARCH_HYPOTHESIS", "<TODO>"),
            "image_ids": [target.image_id for target in targets],
            "name": name,
            "run_id": run_id,
            "seeds": seeds,
            "split": split,
            "started_at": started_at,
            "status": "completed",
            "takeaway": os.environ.get("AUTORESEARCH_TAKEAWAY", "<TODO>"),
            "wall_clock_seconds": wall_clock,
        }
        write_json(run_dir / "run.json", run_json)

        gate_pass = gate_record["single_run_gate"] == "pass"
        append_logbook(
            run_id=run_id,
            driver=driver,
            aggregate=aggregate,
            gate_pass=gate_pass,
            reeval_text=reeval_text,
            val_text=f"{aggregate['composite']:.4f}" if args.val else "n/a",
            wall_clock_seconds=wall_clock,
            promoted=promoted,
        )

        logger.emit(f"gate={'pass' if gate_pass else 'fail'}")
        logger.emit(gate_reason)
        logger.emit(f"promoted={promoted if promoted != 'reverted' else 'reverted_after_re-eval'}")
        return 0
    finally:
        logger.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="short identifier for the run")
    parser.add_argument("--val", action="store_true", help="run on val_images/ without promotion")
    parser.add_argument("--seeds", type=int, default=3, help="generation seeds per target image")
    parser.add_argument("--no-judge", action="store_true", help="skip diagnostic VLM judge")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    return run_harness(args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
