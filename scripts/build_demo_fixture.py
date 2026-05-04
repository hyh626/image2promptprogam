"""Build a richer demo fixture for the viewer screenshots.

Creates a mock repo at the requested path with:
- 3 images (different categories) seeded with distinct colors
- 3 runs with progressively better composite scores
- A leader pointer + chained history
- A logbook with three entries

This is for documentation screenshots only; not used by tests.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import struct
import sys
import zlib
from pathlib import Path

SCHEMA = "1.0.0"
METRICS = ["s_gemini", "s_dino", "s_lpips", "s_color"]


def _make_png(rgb_color: tuple[int, int, int], size: int = 96) -> bytes:
    """Build a minimal solid-color PNG without using Pillow."""
    r, g, b = rgb_color
    raw = b""
    for y in range(size):
        raw += b"\x00"  # filter type none
        for x in range(size):
            # subtle diagonal gradient so the image isn't a flat block
            shade = (x + y) % 24
            raw += bytes((max(0, min(255, r - shade)),
                          max(0, min(255, g - shade)),
                          max(0, min(255, b - shade))))
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw, 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash() -> str:
    # Deterministic-ish hash placeholder; real bytes don't matter for the demo.
    return "".join(random.choices("0123456789abcdef", k=64))


def _build(root: Path) -> None:
    random.seed(42)

    images = [
        # (image_id, category, color, target_color, gen_color)
        ("hero_landscape_01", "photography_hero", (60, 110, 175), (90, 140, 195)),
        ("flat_vector_team", "flat_vector",       (220, 100, 80),  (200, 110, 95)),
        ("isometric_office", "isometric",         (110, 200, 130), (100, 180, 140)),
    ]

    image_dir = root / "eval_data" / "images" / "eval"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for image_id, category, target_color, _ in images:
        png = _make_png(target_color, size=128)
        (image_dir / f"{image_id}.png").write_bytes(png)
        manifest_entries.append({
            "image_id": image_id,
            "filename": f"{image_id}.png",
            "sha256": _hash(),
            "width": 128,
            "height": 128,
            "source": "demo-stock-2026-04",
            "category": category,
            "license": "CC-BY-4.0",
            "notes": "",
        })

    _write_json(root / "eval_data" / "images" / "manifest.json", {
        "schema_version": SCHEMA,
        "splits": {"train": [], "eval": manifest_entries, "val": [], "holdout": []},
    })

    runs = [
        {
            "run_id": "20260504T100000Z__claude-opus-4-7__baseline",
            "name": "baseline",
            "driver": "claude-opus-4-7",
            "started_at": "2026-05-04T10:00:00Z",
            "finished_at": "2026-05-04T10:04:12Z",
            "wall_clock": 252.4,
            "cost": 0.12,
            "hypothesis": "Direct caption with no decomposition.",
            "takeaway": "Composite 0.585. Good color, weak structural fidelity.",
            "decision": "no_leader",
            "leader_run_id": None,
            "score_offsets": [0.0, 0.0, 0.0, 0.0],
            "promoted": "yes",
        },
        {
            "run_id": "20260504T112800Z__claude-opus-4-7__add_palette_step",
            "name": "add_palette_step",
            "driver": "claude-opus-4-7",
            "started_at": "2026-05-04T11:28:00Z",
            "finished_at": "2026-05-04T11:32:07Z",
            "wall_clock": 247.8,
            "cost": 0.18,
            "hypothesis": "Extract dominant palette before main caption.",
            "takeaway": "+0.04 s_color, +0.02 s_dino, others flat. Promoted.",
            "decision": "promoted",
            "leader_run_id": "20260504T100000Z__claude-opus-4-7__baseline",
            "score_offsets": [0.018, 0.024, 0.005, 0.041],
            "promoted": "yes",
        },
        {
            "run_id": "20260504T133100Z__gpt-5__verbose_caption",
            "name": "verbose_caption",
            "driver": "gpt-5",
            "started_at": "2026-05-04T13:31:00Z",
            "finished_at": "2026-05-04T13:38:55Z",
            "wall_clock": 475.0,
            "cost": 0.31,
            "hypothesis": "Long prose caption with subject/composition/lighting tags.",
            "takeaway": "Big s_gemini gain but s_dino regressed beyond ε. Rejected.",
            "decision": "rejected",
            "leader_run_id": "20260504T112800Z__claude-opus-4-7__add_palette_step",
            "score_offsets": [0.052, -0.038, 0.012, 0.008],
            "promoted": "no",
        },
    ]

    base_scores = {
        "hero_landscape_01": {"s_gemini": 0.66, "s_dino": 0.55, "s_lpips": 0.61, "s_color": 0.71},
        "flat_vector_team":  {"s_gemini": 0.58, "s_dino": 0.49, "s_lpips": 0.54, "s_color": 0.63},
        "isometric_office":  {"s_gemini": 0.72, "s_dino": 0.62, "s_lpips": 0.65, "s_color": 0.69},
    }

    leader_history: list[dict] = []
    pointer_payload: dict | None = None

    for run in runs:
        run_dir = root / "experiments" / "runs" / run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        offsets = dict(zip(METRICS, run["score_offsets"]))

        per_image_means: list[dict[str, float]] = []
        for image_id, _, _, _ in images:
            base = base_scores[image_id]
            per_img = {m: round(max(0.0, min(1.0, base[m] + offsets[m])), 3)
                       for m in METRICS}
            per_image_means.append(per_img)
            img_dir = run_dir / "per_image" / image_id
            img_dir.mkdir(parents=True, exist_ok=True)

            # Generate a colored png slightly different from target
            tgt_color = next(t for iid, _, t, _ in [(i[0], None, i[2], None) for i in images]
                             if iid == image_id)
            shift = int((per_img["s_color"]) * 30)
            gen_color = (max(0, tgt_color[0] - shift),
                         max(0, tgt_color[1] - shift // 2),
                         max(0, tgt_color[2] - shift // 3))
            (img_dir / "generated.png").write_bytes(_make_png(gen_color, size=128))
            (img_dir / "prompt.txt").write_text(
                f"Reproduce the {image_id.replace('_', ' ')} reference image. "
                f"Match palette, composition, and lighting. Avoid extraneous detail.",
                encoding="utf-8",
            )
            _write_json(img_dir / "scores.json", {
                "schema_version": SCHEMA,
                "image_id": image_id,
                "seed": 0,
                "scores": per_img,
                "judge": {"subject": 4, "composition": 3, "lighting": 4,
                          "palette": 5, "style": 3, "texture": 3},
                "generated_image_sha256": _hash(),
                "prompt_sha256": _hash(),
                "generation_seconds": round(random.uniform(4.5, 7.5), 2),
                "scoring_seconds": round(random.uniform(0.3, 0.6), 2),
            })

        means = {m: sum(p[m] for p in per_image_means) / len(per_image_means)
                 for m in METRICS}
        composite = sum(means.values()) / len(means)

        _write_json(run_dir / "run.json", {
            "schema_version": SCHEMA,
            "run_id": run["run_id"],
            "name": run["name"],
            "driver": run["driver"],
            "harness_variant": "opus4.7",
            "git_commit": _hash()[:12],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "split": "eval",
            "image_ids": [iid for iid, _, _, _ in images],
            "seeds": [0],
            "wall_clock_seconds": run["wall_clock"],
            "est_cost_usd": run["cost"],
            "status": "completed",
            "hypothesis": run["hypothesis"],
            "takeaway": run["takeaway"],
        })
        _write_json(run_dir / "config.json", {
            "schema_version": SCHEMA,
            "harness_variant": "opus4.7",
            "models": {
                "vlm": "gemini-3.1-flash-lite-preview",
                "generator": "gemini-3.1-flash-image-preview",
                "embedding": "gemini-embedding-2",
                "structural": "facebook/dinov2-base",
                "perceptual": "lpips-alex",
                "color": "hsv-8x8x8-chi2",
            },
            "canonical_resolution": [448, 448],
            "metrics": METRICS,
            "promotion_gate": {"regression_epsilon": 0.01,
                               "improvement_strict": True, "reeval_seeds": 3},
            "cli_args": ["--name", run["name"]],
            "extra": {},
        })
        (run_dir / "prompt_strategy.py").write_text(
            f'"""Strategy for {run["name"]}."""\n', encoding="utf-8")
        (run_dir / "stdout.log").write_text(
            f"composite={composite}\ngate=pass\npromoted={run['decision']}\n",
            encoding="utf-8",
        )

        _write_json(run_dir / "aggregate.json", {
            "schema_version": SCHEMA,
            "run_id": run["run_id"],
            "split": "eval",
            "n_images": len(images),
            "seeds": [0],
            "means": means,
            "stds": {m: 0.04 for m in METRICS},
            "composite": composite,
            "composite_unweighted": composite,
            "three_seed": {"ran": False, "mean_composite": None, "std_composite": None},
        })

        leader_means = pointer_payload["means"] if pointer_payload else None
        leader_composite = pointer_payload["composite"] if pointer_payload else None
        no_regression = (leader_means is None or
                         all(means[m] >= leader_means[m] - 0.01 for m in METRICS))
        improves = (leader_composite is None or composite > leader_composite)
        single_gate = "pass" if (no_regression and improves) else "fail"
        _write_json(run_dir / "gate.json", {
            "schema_version": SCHEMA,
            "leader_run_id": run["leader_run_id"],
            "leader_means": leader_means,
            "leader_composite": leader_composite,
            "candidate_means": means,
            "candidate_composite": composite,
            "regression_epsilon": 0.01,
            "no_regression": no_regression,
            "improves_composite": improves,
            "single_run_gate": single_gate,
            "three_seed_gate": "pass" if run["decision"] == "promoted" else None,
            "decision": run["decision"],
            "reason": run["takeaway"],
        })

        if run["decision"] in ("promoted", "no_leader"):
            prev = pointer_payload["run_id"] if pointer_payload else None
            new_pointer = {
                "schema_version": SCHEMA,
                "run_id": run["run_id"],
                "composite": composite,
                "means": means,
                "promoted_at": run["finished_at"],
            }
            leader_history.append({
                "run_id": run["run_id"],
                "composite": composite,
                "promoted_at": run["finished_at"],
                "previous_run_id": prev,
            })
            pointer_payload = new_pointer

        # logbook entry
        gate_str = "pass" if single_gate == "pass" else "fail"
        promoted_str = run["promoted"] if run["decision"] != "rejected" else "no"
        logbook_entry = "\n".join([
            f"### {run['run_id']}",
            f"- driver: {run['driver']}",
            f"- hypothesis: {run['hypothesis']}",
            f"- composite: {composite:.4f}",
            "- " + " | ".join(f"{m}: {means[m]:.3f}" for m in METRICS),
            f"- gate: {gate_str}",
            "- 3-seed re-eval: n/a",
            "- val composite: n/a",
            f"- wall_clock: {run['wall_clock'] / 60:.1f} min",
            f"- est_cost_usd: {run['cost']:.2f}",
            f"- takeaway: {run['takeaway']}",
            f"- promoted: {promoted_str}",
            "",
        ]) + "\n"
        logbook_path = root / "experiments" / "logbook.md"
        if not logbook_path.exists():
            logbook_path.write_text("# Logbook\n\n", encoding="utf-8")
        with logbook_path.open("a", encoding="utf-8") as f:
            f.write(logbook_entry)

    # Persist leader pointer + history.
    if pointer_payload:
        _write_json(root / "experiments" / "leader" / "pointer.json", pointer_payload)
    history_path = root / "experiments" / "leader" / "history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "\n".join(json.dumps(line) for line in leader_history) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory (will be created)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    _build(args.out)
    print(f"demo fixture written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
