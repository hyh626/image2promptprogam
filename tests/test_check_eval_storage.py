import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import check_eval_storage


RUN_ID = "20260504T120000Z__baseline__gpt5.5"
IMAGE_ID = "hero_photo_01"
METRICS = {"s_gemini": 0.8, "s_dino": 0.6}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _build_valid_eval_repo(root: Path) -> None:
    input_image = b"fixture input image"
    generated_image = b"fixture generated image"
    prompt = b"A faithful, detailed reproduction."

    image_dir = root / "eval_data" / "images" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{IMAGE_ID}.png").write_bytes(input_image)

    _write_json(
        root / "eval_data" / "images" / "manifest.json",
        {
            "schema_version": check_eval_storage.SCHEMA_VERSION,
            "splits": {
                "train": [
                    {
                        "image_id": IMAGE_ID,
                        "filename": f"{IMAGE_ID}.png",
                        "sha256": _sha256(input_image),
                        "width": 32,
                        "height": 32,
                    }
                ],
                "eval": [],
                "val": [],
                "holdout": [],
            },
        },
    )

    run_dir = root / "experiments" / "runs" / RUN_ID
    per_image_dir = run_dir / "per_image" / IMAGE_ID
    per_image_dir.mkdir(parents=True, exist_ok=True)
    (per_image_dir / "prompt.txt").write_bytes(prompt)
    (per_image_dir / "generated.png").write_bytes(generated_image)
    (run_dir / "prompt_strategy.py").write_text("# fixture strategy\n", encoding="utf-8")
    (run_dir / "stdout.log").write_text("fixture stdout\n", encoding="utf-8")

    _write_json(
        run_dir / "run.json",
        {
            "schema_version": check_eval_storage.SCHEMA_VERSION,
            "run_id": RUN_ID,
            "name": "baseline",
            "driver": "gpt5.5",
            "harness_variant": "opus4.7",
            "started_at": "2026-05-04T12:00:00Z",
            "finished_at": "2026-05-04T12:01:00Z",
            "split": "train",
            "image_ids": [IMAGE_ID],
            "seeds": [0],
            "status": "completed",
        },
    )
    _write_json(
        run_dir / "config.json",
        {
            "schema_version": check_eval_storage.SCHEMA_VERSION,
            "harness_variant": "opus4.7",
            "models": {"generator": "fixture-generator"},
            "metrics": list(METRICS),
        },
    )
    _write_json(
        per_image_dir / "scores.json",
        {
            "schema_version": check_eval_storage.SCHEMA_VERSION,
            "image_id": IMAGE_ID,
            "seed": 0,
            "scores": METRICS,
            "judge": None,
            "generated_image_sha256": _sha256(generated_image),
            "prompt_sha256": _sha256(prompt),
        },
    )

    composite = sum(METRICS.values()) / len(METRICS)
    _write_json(
        run_dir / "aggregate.json",
        {
            "schema_version": check_eval_storage.SCHEMA_VERSION,
            "run_id": RUN_ID,
            "split": "train",
            "n_images": 1,
            "seeds": [0],
            "means": METRICS,
            "composite": composite,
            "composite_unweighted": composite,
        },
    )
    _write_json(
        run_dir / "gate.json",
        {
            "schema_version": check_eval_storage.SCHEMA_VERSION,
            "leader_run_id": None,
            "candidate_means": METRICS,
            "candidate_composite": composite,
            "regression_epsilon": 0.01,
            "no_regression": True,
            "improves_composite": True,
            "single_run_gate": "pass",
            "decision": "no_leader",
        },
    )

    (root / "experiments" / "logbook.md").write_text(
        "\n".join(
            [
                f"### {RUN_ID}",
                "- driver: gpt5.5",
                "- hypothesis: Minimal compliant run.",
                "- composite: 0.7000",
                "- s_gemini: 0.8000 | s_dino: 0.6000",
                "- gate: pass",
                "- 3-seed re-eval: n/a",
                "- val composite: n/a",
                "- wall_clock: 1s",
                "- est_cost_usd: 0.00",
                "- takeaway: Fixture validates checker behavior.",
                "- promoted: no",
                "",
            ]
        ),
        encoding="utf-8",
    )


class CheckEvalStorageTests(unittest.TestCase):
    def test_valid_eval_repo_passes_with_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_valid_eval_repo(root)

            report = check_eval_storage.check_root(root, verify_hashes=True)

            self.assertTrue(
                report.ok,
                msg=[(v.code, v.path, v.message) for v in report.violations],
            )

    def test_out_of_range_score_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_valid_eval_repo(root)
            scores_path = (
                root
                / "experiments"
                / "runs"
                / RUN_ID
                / "per_image"
                / IMAGE_ID
                / "scores.json"
            )
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
            scores["scores"]["s_gemini"] = 1.25
            _write_json(scores_path, scores)

            report = check_eval_storage.check_root(root, verify_hashes=False)

            self.assertFalse(report.ok)
            self.assertIn("E_SCORES_RANGE", {v.code for v in report.violations})


if __name__ == "__main__":
    unittest.main()
