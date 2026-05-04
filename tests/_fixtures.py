"""Build a minimal but schema-valid repo tree for tests."""
from __future__ import annotations

import base64
import json
from pathlib import Path

SCHEMA = "1.0.0"
RUN_ID = "20260504T123456Z__claude-opus-4-7__baseline"
IMAGE_ID = "hero_photo_01"
METRICS = ["s_gemini", "s_dino", "s_lpips", "s_color"]

# 1x1 red PNG
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_minimal_repo(root: Path) -> dict:
    """Create a one-run schema-valid repo and return ids/paths."""
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": SCHEMA,
        "splits": {
            "train": [],
            "eval": [{
                "image_id": IMAGE_ID,
                "filename": f"{IMAGE_ID}.png",
                "sha256": "0" * 64,
                "width": 1024,
                "height": 768,
                "source": "synthetic",
                "category": "test",
                "license": "CC0",
                "notes": "fixture",
            }],
            "val": [],
            "holdout": [],
        },
    }
    _write_json(root / "eval_data" / "images" / "manifest.json", manifest)
    target_image = root / "eval_data" / "images" / "eval" / f"{IMAGE_ID}.png"
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_image.write_bytes(TINY_PNG)

    # also drop a holdout file so we can confirm sync excludes it
    holdout = root / "eval_data" / "images" / "holdout" / "secret.png"
    holdout.parent.mkdir(parents=True, exist_ok=True)
    holdout.write_bytes(TINY_PNG)

    run_dir = root / "experiments" / "runs" / RUN_ID
    means = {m: 0.5 for m in METRICS}
    _write_json(run_dir / "run.json", {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "name": "baseline",
        "driver": "claude-opus-4-7",
        "harness_variant": "opus4.7",
        "git_commit": "deadbeef",
        "started_at": "2026-05-04T12:34:56Z",
        "finished_at": "2026-05-04T12:39:11Z",
        "split": "eval",
        "image_ids": [IMAGE_ID],
        "seeds": [0],
        "wall_clock_seconds": 250.0,
        "est_cost_usd": 0.18,
        "status": "completed",
        "hypothesis": "fixture",
        "takeaway": "fixture",
    })
    _write_json(run_dir / "config.json", {
        "schema_version": SCHEMA,
        "harness_variant": "opus4.7",
        "models": {"vlm": "x", "generator": "y", "embedding": "z",
                   "structural": "w", "perceptual": "v", "color": "u"},
        "canonical_resolution": [448, 448],
        "metrics": METRICS,
        "promotion_gate": {"regression_epsilon": 0.01,
                           "improvement_strict": True, "reeval_seeds": 3},
        "cli_args": [],
        "extra": {},
    })
    (run_dir / "prompt_strategy.py").write_text("# baseline snapshot\n")
    (run_dir / "stdout.log").write_text("ok\n")

    img_dir = run_dir / "per_image" / IMAGE_ID
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "prompt.txt").write_text("describe this image", encoding="utf-8")
    (img_dir / "generated.png").write_bytes(TINY_PNG)
    _write_json(img_dir / "scores.json", {
        "schema_version": SCHEMA,
        "image_id": IMAGE_ID,
        "seed": 0,
        "scores": means,
        "judge": None,
        "generated_image_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "generation_seconds": 6.4,
        "scoring_seconds": 0.4,
    })
    _write_json(run_dir / "aggregate.json", {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "split": "eval",
        "n_images": 1,
        "seeds": [0],
        "means": means,
        "stds": {m: 0.0 for m in METRICS},
        "composite": 0.5,
        "composite_unweighted": 0.5,
        "three_seed": {"ran": False, "mean_composite": None, "std_composite": None},
    })
    _write_json(run_dir / "gate.json", {
        "schema_version": SCHEMA,
        "leader_run_id": None,
        "leader_means": None,
        "leader_composite": None,
        "candidate_means": means,
        "candidate_composite": 0.5,
        "regression_epsilon": 0.01,
        "no_regression": True,
        "improves_composite": True,
        "single_run_gate": "pass",
        "three_seed_gate": None,
        "decision": "no_leader",
        "reason": "fixture",
    })

    leader_dir = root / "experiments" / "leader"
    _write_json(leader_dir / "pointer.json", {
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "composite": 0.5,
        "means": means,
        "promoted_at": "2026-05-04T12:39:14Z",
    })
    (leader_dir / "history.jsonl").write_text(
        json.dumps({
            "run_id": RUN_ID,
            "composite": 0.5,
            "promoted_at": "2026-05-04T12:39:14Z",
            "previous_run_id": None,
        }) + "\n",
        encoding="utf-8",
    )
    (root / "experiments" / "logbook.md").write_text(
        "# Logbook\n\n"
        f"### {RUN_ID}\n"
        "- driver: claude-opus-4-7\n"
        "- hypothesis: fixture\n"
        "- composite: 0.5000\n"
        "- s_gemini: 0.500 | s_dino: 0.500 | s_lpips: 0.500 | s_color: 0.500\n"
        "- gate: pass\n"
        "- 3-seed re-eval: n/a\n"
        "- val composite: n/a\n"
        "- wall_clock: 4.2 min\n"
        "- est_cost_usd: 0.18\n"
        "- takeaway: fixture\n"
        "- promoted: yes\n",
        encoding="utf-8",
    )

    return {"run_id": RUN_ID, "image_id": IMAGE_ID, "metrics": METRICS,
            "run_dir": run_dir, "target_image": target_image}


def seed_bucket_from_local(bucket, src: Path, prefix: str = "") -> int:
    """Mirror every file under `src` into the fake `bucket` (no excludes)."""
    count = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        obj = f"{prefix.strip('/')}/{rel}" if prefix.strip("/") else rel
        bucket.blobs[obj] = path.read_bytes()
        count += 1
    return count
