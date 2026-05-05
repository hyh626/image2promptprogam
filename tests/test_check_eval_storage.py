"""Tests for check_eval_storage.py.

Each test starts from a fully-valid fixture and mutates one thing, then
asserts the expected violation code appears in the report. Codes are
documented in EVAL_STORAGE_SCHEMA.md.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import check_eval_storage  # noqa: E402


RUN_ID = "20260504T120000Z__baseline__gpt5.5"
SECOND_RUN_ID = "20260504T130000Z__followup__gpt5.5"
IMAGE_ID = "hero_photo_01"
SECOND_IMAGE_ID = "hero_photo_02"
METRICS = {"s_gemini": 0.8, "s_dino": 0.6}


# --------------------------- helpers ---------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _mutate_json(path: Path, mutator: Callable[[dict], None]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mutator(data)
    _write_json(path, data)


def _codes(report: check_eval_storage.Report) -> list[str]:
    return [v.code for v in report.violations]


def _build_valid_eval_repo(root: Path) -> None:
    """One run, one image, no leader yet — schema-valid."""
    input_image = b"fixture input image"
    generated_image = b"fixture generated image"
    prompt = b"A faithful, detailed reproduction."

    image_dir = root / "eval_data" / "images" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{IMAGE_ID}.png").write_bytes(input_image)

    _write_json(root / "eval_data" / "images" / "manifest.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "splits": {
            "train": [{
                "image_id": IMAGE_ID,
                "filename": f"{IMAGE_ID}.png",
                "sha256": _sha256(input_image),
                "width": 32,
                "height": 32,
            }],
            "eval": [],
            "val": [],
            "holdout": [],
        },
    })

    run_dir = root / "experiments" / "runs" / RUN_ID
    per_image_dir = run_dir / "per_image" / IMAGE_ID
    per_image_dir.mkdir(parents=True, exist_ok=True)
    (per_image_dir / "prompt.txt").write_bytes(prompt)
    (per_image_dir / "generated.png").write_bytes(generated_image)
    (run_dir / "prompt_strategy.py").write_text("# fixture strategy\n", encoding="utf-8")
    (run_dir / "stdout.log").write_text("fixture stdout\n", encoding="utf-8")

    _write_json(run_dir / "run.json", {
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
    })
    _write_json(run_dir / "config.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "harness_variant": "opus4.7",
        "models": {"generator": "fixture-generator"},
        "metrics": list(METRICS),
    })
    _write_json(per_image_dir / "scores.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "image_id": IMAGE_ID,
        "seed": 0,
        "scores": dict(METRICS),
        "judge": None,
        "generated_image_sha256": _sha256(generated_image),
        "prompt_sha256": _sha256(prompt),
    })

    composite = sum(METRICS.values()) / len(METRICS)
    _write_json(run_dir / "aggregate.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "run_id": RUN_ID,
        "split": "train",
        "n_images": 1,
        "seeds": [0],
        "means": dict(METRICS),
        "composite": composite,
        "composite_unweighted": composite,
    })
    _write_json(run_dir / "gate.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "leader_run_id": None,
        "candidate_means": dict(METRICS),
        "candidate_composite": composite,
        "regression_epsilon": 0.01,
        "no_regression": True,
        "improves_composite": True,
        "single_run_gate": "pass",
        "decision": "no_leader",
    })

    (root / "experiments" / "logbook.md").write_text(
        "\n".join([
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
        ]),
        encoding="utf-8",
    )


def _add_leader(root: Path, run_id: str = RUN_ID) -> None:
    """Promote `run_id` as leader. Caller decides whether the run exists."""
    leader_dir = root / "experiments" / "leader"
    leader_dir.mkdir(parents=True, exist_ok=True)
    composite = sum(METRICS.values()) / len(METRICS)
    _write_json(leader_dir / "pointer.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "run_id": run_id,
        "composite": composite,
        "means": dict(METRICS),
        "promoted_at": "2026-05-04T12:01:01Z",
    })
    (leader_dir / "history.jsonl").write_text(
        json.dumps({
            "run_id": run_id,
            "composite": composite,
            "promoted_at": "2026-05-04T12:01:01Z",
            "previous_run_id": None,
        }) + "\n",
        encoding="utf-8",
    )


def _add_second_run_with_leader(root: Path) -> dict[str, dict[str, float]]:
    """Append a second run (and switch the first one to a 'promoted' decision).

    Returns means dicts so callers can mutate them.
    """
    leader_means = dict(METRICS)
    leader_composite = sum(leader_means.values()) / len(leader_means)

    # Build a strictly-better second run.
    cand_means = {k: min(1.0, v + 0.05) for k, v in leader_means.items()}
    cand_composite = sum(cand_means.values()) / len(cand_means)
    second_dir = root / "experiments" / "runs" / SECOND_RUN_ID
    per_image_dir = second_dir / "per_image" / IMAGE_ID
    per_image_dir.mkdir(parents=True, exist_ok=True)
    (per_image_dir / "prompt.txt").write_text("p", encoding="utf-8")
    (per_image_dir / "generated.png").write_bytes(b"img2")
    (second_dir / "prompt_strategy.py").write_text("# v2\n", encoding="utf-8")
    (second_dir / "stdout.log").write_text("ok\n", encoding="utf-8")

    _write_json(second_dir / "run.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "run_id": SECOND_RUN_ID,
        "name": "followup",
        "driver": "gpt5.5",
        "harness_variant": "opus4.7",
        "started_at": "2026-05-04T13:00:00Z",
        "finished_at": "2026-05-04T13:01:00Z",
        "split": "train",
        "image_ids": [IMAGE_ID],
        "seeds": [0],
        "status": "completed",
    })
    _write_json(second_dir / "config.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "harness_variant": "opus4.7",
        "models": {"generator": "fixture-generator"},
        "metrics": list(METRICS),
    })
    _write_json(per_image_dir / "scores.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "image_id": IMAGE_ID,
        "seed": 0,
        "scores": dict(cand_means),
        "judge": None,
        "generated_image_sha256": _sha256(b"img2"),
        "prompt_sha256": _sha256(b"p"),
    })
    _write_json(second_dir / "aggregate.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "run_id": SECOND_RUN_ID,
        "split": "train",
        "n_images": 1,
        "seeds": [0],
        "means": dict(cand_means),
        "composite": cand_composite,
        "composite_unweighted": cand_composite,
    })
    _write_json(second_dir / "gate.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "leader_run_id": RUN_ID,
        "leader_means": leader_means,
        "leader_composite": leader_composite,
        "candidate_means": dict(cand_means),
        "candidate_composite": cand_composite,
        "regression_epsilon": 0.01,
        "no_regression": True,
        "improves_composite": True,
        "single_run_gate": "pass",
        "decision": "promoted",
    })

    # Pointer + chained history reflecting "promote first, then second".
    leader_dir = root / "experiments" / "leader"
    leader_dir.mkdir(parents=True, exist_ok=True)
    _write_json(leader_dir / "pointer.json", {
        "schema_version": check_eval_storage.SCHEMA_VERSION,
        "run_id": SECOND_RUN_ID,
        "composite": cand_composite,
        "means": dict(cand_means),
        "promoted_at": "2026-05-04T13:01:01Z",
    })
    (leader_dir / "history.jsonl").write_text(
        json.dumps({
            "run_id": RUN_ID,
            "composite": leader_composite,
            "promoted_at": "2026-05-04T12:01:01Z",
            "previous_run_id": None,
        }) + "\n" +
        json.dumps({
            "run_id": SECOND_RUN_ID,
            "composite": cand_composite,
            "promoted_at": "2026-05-04T13:01:01Z",
            "previous_run_id": RUN_ID,
        }) + "\n",
        encoding="utf-8",
    )

    # Append second logbook entry.
    with (root / "experiments" / "logbook.md").open("a", encoding="utf-8") as f:
        f.write("\n".join([
            f"### {SECOND_RUN_ID}",
            "- driver: gpt5.5",
            "- hypothesis: Tweak the prompt.",
            "- composite: 0.7500",
            "- s_gemini: 0.8500 | s_dino: 0.6500",
            "- gate: pass",
            "- 3-seed re-eval: n/a",
            "- val composite: n/a",
            "- wall_clock: 2s",
            "- est_cost_usd: 0.00",
            "- takeaway: Better.",
            "- promoted: yes",
            "",
        ]))

    return {"leader": leader_means, "candidate": cand_means}


# --------------------------- base test class ---------------------------


class _CheckerTestBase(unittest.TestCase):
    """Builds a valid fixture in setUp; subclasses mutate before checking."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _build_valid_eval_repo(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # path conveniences
    @property
    def manifest_path(self) -> Path:
        return self.root / "eval_data" / "images" / "manifest.json"

    @property
    def run_dir(self) -> Path:
        return self.root / "experiments" / "runs" / RUN_ID

    @property
    def run_json_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def config_json_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def scores_json_path(self) -> Path:
        return self.run_dir / "per_image" / IMAGE_ID / "scores.json"

    @property
    def aggregate_json_path(self) -> Path:
        return self.run_dir / "aggregate.json"

    @property
    def gate_json_path(self) -> Path:
        return self.run_dir / "gate.json"

    @property
    def logbook_path(self) -> Path:
        return self.root / "experiments" / "logbook.md"

    # checking
    def check(self, verify_hashes: bool = False) -> check_eval_storage.Report:
        return check_eval_storage.check_root(self.root, verify_hashes=verify_hashes)

    # assertions
    def assertHasCode(self, code: str, report: check_eval_storage.Report) -> None:
        codes = _codes(report)
        self.assertIn(code, codes,
                      msg=f"expected {code} in violations; got {codes}")

    def assertNoCode(self, code: str, report: check_eval_storage.Report) -> None:
        codes = _codes(report)
        self.assertNotIn(code, codes,
                         msg=f"unexpected {code} in violations; got {codes}")


# --------------------------- preserved baseline tests ---------------------------


class CheckEvalStorageTests(_CheckerTestBase):
    """Original tests, preserved."""

    def test_valid_eval_repo_passes_with_hash_verification(self) -> None:
        report = self.check(verify_hashes=True)
        self.assertTrue(
            report.ok,
            msg=[(v.code, v.path, v.message) for v in report.violations],
        )

    def test_out_of_range_score_is_reported(self) -> None:
        _mutate_json(self.scores_json_path,
                     lambda d: d["scores"].__setitem__("s_gemini", 1.25))
        self.assertHasCode("E_SCORES_RANGE", self.check())


# --------------------------- helpers ---------------------------


class HelperTests(unittest.TestCase):
    def test_is_finite_unit_float(self) -> None:
        self.assertTrue(check_eval_storage.is_finite_unit_float(0.5))
        self.assertTrue(check_eval_storage.is_finite_unit_float(0))
        self.assertTrue(check_eval_storage.is_finite_unit_float(1))
        self.assertFalse(check_eval_storage.is_finite_unit_float(1.0001))
        self.assertFalse(check_eval_storage.is_finite_unit_float(-0.0001))
        self.assertFalse(check_eval_storage.is_finite_unit_float(float("inf")))
        self.assertFalse(check_eval_storage.is_finite_unit_float(float("nan")))
        self.assertFalse(check_eval_storage.is_finite_unit_float(True))  # bool excluded
        self.assertFalse(check_eval_storage.is_finite_unit_float("0.5"))
        self.assertFalse(check_eval_storage.is_finite_unit_float(None))

    def test_approx_equal(self) -> None:
        self.assertTrue(check_eval_storage.approx_equal(0.5, 0.50005))
        self.assertFalse(check_eval_storage.approx_equal(0.5, 0.6))


# --------------------------- top-level structure ---------------------------


class TopLevelStructureTests(unittest.TestCase):
    def test_missing_eval_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # eval_data dir doesn't exist; only experiments/ is present.
            (root / "experiments" / "runs").mkdir(parents=True)
            report = check_eval_storage.check_root(root, verify_hashes=False)
            self.assertIn("E_FILE_MISSING", _codes(report))


# --------------------------- manifest ---------------------------


class ManifestTests(_CheckerTestBase):
    def test_invalid_json_in_manifest(self) -> None:
        self.manifest_path.write_text("{not valid", encoding="utf-8")
        self.assertHasCode("E_JSON_INVALID", self.check())

    def test_manifest_must_be_object(self) -> None:
        self.manifest_path.write_text("[]", encoding="utf-8")
        self.assertHasCode("E_MANIFEST_SHAPE", self.check())

    def test_missing_splits_object(self) -> None:
        _mutate_json(self.manifest_path, lambda d: d.pop("splits"))
        self.assertHasCode("E_MANIFEST_SHAPE", self.check())

    def test_split_missing(self) -> None:
        _mutate_json(self.manifest_path, lambda d: d["splits"].pop("holdout"))
        self.assertHasCode("E_MANIFEST_SPLIT", self.check())

    def test_split_must_be_list(self) -> None:
        _mutate_json(self.manifest_path,
                     lambda d: d["splits"].__setitem__("eval", "not-a-list"))
        self.assertHasCode("E_MANIFEST_SPLIT", self.check())

    def test_non_object_entry(self) -> None:
        _mutate_json(self.manifest_path,
                     lambda d: d["splits"]["eval"].append("not-an-object"))
        self.assertHasCode("E_MANIFEST_ENTRY", self.check())

    def test_required_field_missing(self) -> None:
        def mutator(d: dict) -> None:
            d["splits"]["train"][0].pop("width")
        _mutate_json(self.manifest_path, mutator)
        self.assertHasCode("E_FIELD_MISSING", self.check())

    def test_duplicate_image_id_across_splits(self) -> None:
        def mutator(d: dict) -> None:
            entry = dict(d["splits"]["train"][0])
            d["splits"]["eval"].append(entry)
        _mutate_json(self.manifest_path, mutator)
        self.assertHasCode("E_MANIFEST_DUPLICATE", self.check())

    def test_malformed_sha256(self) -> None:
        _mutate_json(self.manifest_path,
                     lambda d: d["splits"]["train"][0].__setitem__("sha256", "deadbeef"))
        self.assertHasCode("E_MANIFEST_HASH_FORMAT", self.check())

    def test_non_positive_dimensions(self) -> None:
        def mutator(d: dict) -> None:
            d["splits"]["train"][0]["width"] = 0
            d["splits"]["train"][0]["height"] = -1
        _mutate_json(self.manifest_path, mutator)
        self.assertHasCode("E_MANIFEST_DIMS", self.check())

    def test_schema_version_wrong(self) -> None:
        _mutate_json(self.manifest_path,
                     lambda d: d.__setitem__("schema_version", "0.0.0"))
        self.assertHasCode("E_SCHEMA_VERSION", self.check())

    def test_image_file_missing_with_hash_verification(self) -> None:
        (self.root / "eval_data" / "images" / "train" / f"{IMAGE_ID}.png").unlink()
        self.assertHasCode("E_IMAGE_MISSING", self.check(verify_hashes=True))

    def test_image_hash_mismatch_with_hash_verification(self) -> None:
        (self.root / "eval_data" / "images" / "train" / f"{IMAGE_ID}.png").write_bytes(
            b"different bytes"
        )
        self.assertHasCode("E_IMAGE_HASH_MISMATCH", self.check(verify_hashes=True))

    def test_hash_check_skipped_without_flag(self) -> None:
        (self.root / "eval_data" / "images" / "train" / f"{IMAGE_ID}.png").write_bytes(
            b"different bytes"
        )
        # Without --verify-hashes, repo should still pass.
        self.assertTrue(self.check(verify_hashes=False).ok)


# --------------------------- run.json ---------------------------


class RunJsonTests(_CheckerTestBase):
    def test_run_id_format(self) -> None:
        # Rename the run dir to something that doesn't match the regex.
        bad_dir = self.run_dir.parent / "not-a-valid-run-id"
        self.run_dir.rename(bad_dir)
        # Also fix up internal run_id to match dir name so we get only the
        # format error, not also E_RUN_ID_MISMATCH.
        rj = bad_dir / "run.json"
        _mutate_json(rj, lambda d: d.__setitem__("run_id", "not-a-valid-run-id"))
        self.assertHasCode("E_RUN_ID_FORMAT", self.check())

    def test_run_id_mismatch(self) -> None:
        _mutate_json(self.run_json_path,
                     lambda d: d.__setitem__("run_id", "different-run-id"))
        self.assertHasCode("E_RUN_ID_MISMATCH", self.check())

    def test_timestamp_format(self) -> None:
        _mutate_json(self.run_json_path,
                     lambda d: d.__setitem__("started_at", "yesterday"))
        self.assertHasCode("E_TIMESTAMP_FORMAT", self.check())

    def test_invalid_status(self) -> None:
        _mutate_json(self.run_json_path,
                     lambda d: d.__setitem__("status", "kinda-done"))
        self.assertHasCode("E_RUN_STATUS", self.check())

    def test_invalid_split(self) -> None:
        _mutate_json(self.run_json_path,
                     lambda d: d.__setitem__("split", "fake"))
        self.assertHasCode("E_RUN_SPLIT", self.check())

    def test_image_ids_must_be_list_of_strings(self) -> None:
        _mutate_json(self.run_json_path,
                     lambda d: d.__setitem__("image_ids", [42]))
        self.assertHasCode("E_RUN_IMAGE_IDS", self.check())

    def test_image_id_unknown_to_manifest(self) -> None:
        _mutate_json(self.run_json_path,
                     lambda d: d.__setitem__("image_ids", ["does_not_exist"]))
        self.assertHasCode("E_RUN_IMAGE_UNKNOWN", self.check())

    def test_seeds_must_be_nonempty_int_list(self) -> None:
        _mutate_json(self.run_json_path, lambda d: d.__setitem__("seeds", []))
        self.assertHasCode("E_RUN_SEEDS", self.check())

    def test_run_json_missing(self) -> None:
        self.run_json_path.unlink()
        # When run.json is unreadable check_run returns early; the run is
        # absent from run_ids but the file-missing violation is reported.
        self.assertHasCode("E_FILE_MISSING", self.check())

    def test_run_json_invalid_json(self) -> None:
        self.run_json_path.write_text("{not json", encoding="utf-8")
        self.assertHasCode("E_JSON_INVALID", self.check())


# --------------------------- config.json ---------------------------


class ConfigTests(_CheckerTestBase):
    def test_metrics_must_be_nonempty_string_list(self) -> None:
        _mutate_json(self.config_json_path, lambda d: d.__setitem__("metrics", []))
        self.assertHasCode("E_CONFIG_METRICS", self.check())

    def test_metrics_wrong_type(self) -> None:
        _mutate_json(self.config_json_path,
                     lambda d: d.__setitem__("metrics", "not-a-list"))
        self.assertHasCode("E_CONFIG_METRICS", self.check())

    def test_config_missing(self) -> None:
        self.config_json_path.unlink()
        self.assertHasCode("E_FILE_MISSING", self.check())


# --------------------------- per-image scores.json ---------------------------


class ScoresTests(_CheckerTestBase):
    def test_image_id_mismatch(self) -> None:
        _mutate_json(self.scores_json_path,
                     lambda d: d.__setitem__("image_id", "wrong"))
        self.assertHasCode("E_SCORES_IMAGE_MISMATCH", self.check())

    def test_unknown_metric(self) -> None:
        _mutate_json(self.scores_json_path,
                     lambda d: d["scores"].__setitem__("s_unexpected", 0.5))
        self.assertHasCode("E_SCORES_UNKNOWN_METRIC", self.check())

    def test_missing_metric(self) -> None:
        _mutate_json(self.scores_json_path,
                     lambda d: d["scores"].pop("s_dino"))
        self.assertHasCode("E_SCORES_MISSING_METRIC", self.check())

    def test_scores_object_required(self) -> None:
        _mutate_json(self.scores_json_path,
                     lambda d: d.__setitem__("scores", []))
        self.assertHasCode("E_SCORES_SHAPE", self.check())

    def test_judge_must_be_object_or_null(self) -> None:
        _mutate_json(self.scores_json_path, lambda d: d.__setitem__("judge", []))
        self.assertHasCode("E_JUDGE_SHAPE", self.check())

    def test_judge_value_out_of_range(self) -> None:
        _mutate_json(self.scores_json_path,
                     lambda d: d.__setitem__("judge", {"subject": 6}))
        self.assertHasCode("E_JUDGE_RANGE", self.check())

    def test_hash_format_invalid(self) -> None:
        _mutate_json(self.scores_json_path,
                     lambda d: d.__setitem__("generated_image_sha256", "x"))
        self.assertHasCode("E_HASH_FORMAT", self.check())

    def test_score_below_zero_reported(self) -> None:
        _mutate_json(self.scores_json_path,
                     lambda d: d["scores"].__setitem__("s_gemini", -0.1))
        self.assertHasCode("E_SCORES_RANGE", self.check())

    def test_score_must_not_be_bool(self) -> None:
        _mutate_json(self.scores_json_path,
                     lambda d: d["scores"].__setitem__("s_gemini", True))
        self.assertHasCode("E_SCORES_RANGE", self.check())

    def test_generated_hash_mismatch_with_verify(self) -> None:
        # Rewrite the generated.png so its hash no longer matches the recorded one.
        (self.run_dir / "per_image" / IMAGE_ID / "generated.png").write_bytes(b"new")
        report = self.check(verify_hashes=True)
        self.assertHasCode("E_GENERATED_HASH_MISMATCH", report)

    def test_per_image_dir_missing(self) -> None:
        # Remove the per_image/<image_id>/ directory.
        import shutil
        shutil.rmtree(self.run_dir / "per_image" / IMAGE_ID)
        self.assertHasCode("E_FILE_MISSING", self.check())


# --------------------------- aggregate.json ---------------------------


class AggregateTests(_CheckerTestBase):
    def test_run_id_mismatch(self) -> None:
        _mutate_json(self.aggregate_json_path,
                     lambda d: d.__setitem__("run_id", "different"))
        self.assertHasCode("E_AGG_RUN_MISMATCH", self.check())

    def test_split_mismatch(self) -> None:
        _mutate_json(self.aggregate_json_path,
                     lambda d: d.__setitem__("split", "eval"))
        self.assertHasCode("E_AGG_SPLIT_MISMATCH", self.check())

    def test_n_images_wrong(self) -> None:
        _mutate_json(self.aggregate_json_path,
                     lambda d: d.__setitem__("n_images", 99))
        self.assertHasCode("E_AGG_N_IMAGES", self.check())

    def test_means_missing_metric(self) -> None:
        _mutate_json(self.aggregate_json_path,
                     lambda d: d["means"].pop("s_dino"))
        self.assertHasCode("E_AGG_MEANS_MISSING", self.check())

    def test_means_shape_wrong(self) -> None:
        _mutate_json(self.aggregate_json_path,
                     lambda d: d.__setitem__("means", [0.5, 0.5]))
        self.assertHasCode("E_AGG_MEANS_SHAPE", self.check())

    def test_means_out_of_range(self) -> None:
        _mutate_json(self.aggregate_json_path,
                     lambda d: d["means"].__setitem__("s_gemini", 1.5))
        self.assertHasCode("E_AGG_RANGE", self.check())

    def test_means_disagree_with_per_image(self) -> None:
        _mutate_json(self.aggregate_json_path,
                     lambda d: d["means"].__setitem__("s_gemini", 0.10))
        self.assertHasCode("E_AGG_MISMATCH", self.check())

    def test_composite_formula(self) -> None:
        _mutate_json(self.aggregate_json_path,
                     lambda d: d.__setitem__("composite_unweighted", 0.99))
        self.assertHasCode("E_COMPOSITE_FORMULA", self.check())


# --------------------------- gate.json ---------------------------


class GateNoLeaderTests(_CheckerTestBase):
    def test_first_run_must_use_no_leader_decision(self) -> None:
        _mutate_json(self.gate_json_path,
                     lambda d: d.__setitem__("decision", "promoted"))
        self.assertHasCode("E_GATE_NO_LEADER_DECISION", self.check())

    def test_first_run_flags_must_be_true(self) -> None:
        _mutate_json(self.gate_json_path,
                     lambda d: d.__setitem__("no_regression", False))
        self.assertHasCode("E_GATE_NO_LEADER_FLAGS", self.check())

    def test_invalid_decision_value(self) -> None:
        _mutate_json(self.gate_json_path,
                     lambda d: d.__setitem__("decision", "weird"))
        self.assertHasCode("E_GATE_DECISION", self.check())

    def test_invalid_outcome_value(self) -> None:
        _mutate_json(self.gate_json_path,
                     lambda d: d.__setitem__("single_run_gate", "ok"))
        self.assertHasCode("E_GATE_OUTCOME", self.check())

    def test_three_seed_outcome_invalid(self) -> None:
        _mutate_json(self.gate_json_path,
                     lambda d: d.__setitem__("three_seed_gate", "maybe"))
        self.assertHasCode("E_GATE_OUTCOME", self.check())

    def test_candidate_means_must_be_object(self) -> None:
        _mutate_json(self.gate_json_path,
                     lambda d: d.__setitem__("candidate_means", []))
        self.assertHasCode("E_GATE_MEANS_SHAPE", self.check())

    def test_candidate_disagrees_with_aggregate(self) -> None:
        _mutate_json(self.gate_json_path,
                     lambda d: d["candidate_means"].__setitem__("s_gemini", 0.1))
        self.assertHasCode("E_GATE_AGG_MISMATCH", self.check())


class GateWithLeaderTests(_CheckerTestBase):
    def setUp(self) -> None:
        super().setUp()
        # Two-run history: RUN_ID is the first leader, SECOND_RUN_ID promoted
        # against it. Pointer references SECOND_RUN_ID.
        _add_second_run_with_leader(self.root)

    @property
    def second_gate_path(self) -> Path:
        return self.root / "experiments" / "runs" / SECOND_RUN_ID / "gate.json"

    def test_no_regression_disagrees_with_computed(self) -> None:
        # candidate strictly improves all metrics; if we claim regressed,
        # checker should disagree.
        _mutate_json(self.second_gate_path,
                     lambda d: d.__setitem__("no_regression", False))
        self.assertHasCode("E_GATE_NO_REGRESSION", self.check())

    def test_improves_composite_disagrees_with_computed(self) -> None:
        _mutate_json(self.second_gate_path,
                     lambda d: d.__setitem__("improves_composite", False))
        self.assertHasCode("E_GATE_IMPROVES", self.check())

    def test_leader_means_required_when_leader_set(self) -> None:
        _mutate_json(self.second_gate_path, lambda d: d.__setitem__("leader_means", None))
        self.assertHasCode("E_GATE_LEADER_MEANS", self.check())


# --------------------------- leader / history ---------------------------


class LeaderTests(_CheckerTestBase):
    def setUp(self) -> None:
        super().setUp()
        _add_leader(self.root, run_id=RUN_ID)

    @property
    def pointer_path(self) -> Path:
        return self.root / "experiments" / "leader" / "pointer.json"

    @property
    def history_path(self) -> Path:
        return self.root / "experiments" / "leader" / "history.jsonl"

    def test_pointer_run_missing(self) -> None:
        # Make pointer reference a non-existent run.
        _mutate_json(self.pointer_path,
                     lambda d: d.__setitem__("run_id", "ghost"))
        # also align history so the pointer-vs-history check doesn't drown
        # the missing-run signal.
        self.history_path.write_text(
            json.dumps({
                "run_id": "ghost",
                "composite": 0.5,
                "promoted_at": "2026-05-04T12:01:01Z",
                "previous_run_id": None,
            }) + "\n",
            encoding="utf-8",
        )
        self.assertHasCode("E_LEADER_RUN_MISSING", self.check())

    def test_pointer_shape(self) -> None:
        self.pointer_path.write_text("[]", encoding="utf-8")
        self.assertHasCode("E_LEADER_POINTER_SHAPE", self.check())

    def test_pointer_history_disagree(self) -> None:
        # Append a history line for a real second run, but leave pointer at first.
        _add_second_run_with_leader(self.root)  # rewrites pointer/history both
        # Now make pointer point to first run while history's last line is second.
        _mutate_json(self.pointer_path,
                     lambda d: d.__setitem__("run_id", RUN_ID))
        self.assertHasCode("E_LEADER_POINTER_HISTORY", self.check())

    def test_pointer_composite_disagrees_with_history(self) -> None:
        _mutate_json(self.pointer_path,
                     lambda d: d.__setitem__("composite", 0.999))
        self.assertHasCode("E_LEADER_COMPOSITE", self.check())


class HistoryTests(_CheckerTestBase):
    def setUp(self) -> None:
        super().setUp()
        _add_leader(self.root, run_id=RUN_ID)

    @property
    def history_path(self) -> Path:
        return self.root / "experiments" / "leader" / "history.jsonl"

    def test_history_invalid_json_line(self) -> None:
        self.history_path.write_text("{not json\n", encoding="utf-8")
        self.assertHasCode("E_HISTORY_JSON", self.check())

    def test_history_line_must_be_object(self) -> None:
        self.history_path.write_text("[1, 2, 3]\n", encoding="utf-8")
        self.assertHasCode("E_HISTORY_SHAPE", self.check())

    def test_history_missing_required_fields(self) -> None:
        self.history_path.write_text(
            json.dumps({"run_id": RUN_ID}) + "\n", encoding="utf-8")
        self.assertHasCode("E_HISTORY_FIELDS", self.check())

    def test_history_run_missing(self) -> None:
        self.history_path.write_text(
            json.dumps({
                "run_id": "ghost",
                "composite": 0.5,
                "promoted_at": "2026-05-04T12:01:01Z",
                "previous_run_id": None,
            }) + "\n",
            encoding="utf-8",
        )
        self.assertHasCode("E_HISTORY_RUN_MISSING", self.check())

    def test_history_chain_broken(self) -> None:
        # Two real runs, but second history line claims previous_run_id != first.
        _add_second_run_with_leader(self.root)
        bad = (
            json.dumps({
                "run_id": RUN_ID,
                "composite": 0.7,
                "promoted_at": "2026-05-04T12:01:01Z",
                "previous_run_id": None,
            }) + "\n" +
            json.dumps({
                "run_id": SECOND_RUN_ID,
                "composite": 0.75,
                "promoted_at": "2026-05-04T13:01:01Z",
                "previous_run_id": "wrong-prev",
            }) + "\n"
        )
        self.history_path.write_text(bad, encoding="utf-8")
        # And keep pointer aligned with last history line so we isolate the chain error.
        _mutate_json(self.root / "experiments" / "leader" / "pointer.json",
                     lambda d: (d.__setitem__("run_id", SECOND_RUN_ID),
                                d.__setitem__("composite", 0.75)))
        self.assertHasCode("E_HISTORY_CHAIN", self.check())


# --------------------------- logbook ---------------------------


class LogbookTests(_CheckerTestBase):
    def test_logbook_missing(self) -> None:
        self.logbook_path.unlink()
        self.assertHasCode("E_FILE_MISSING", self.check())

    def test_logbook_entry_missing_for_run(self) -> None:
        self.logbook_path.write_text("# Logbook\n", encoding="utf-8")
        self.assertHasCode("E_LOGBOOK_ENTRY_MISSING", self.check())

    def test_logbook_duplicate_entry(self) -> None:
        text = self.logbook_path.read_text(encoding="utf-8")
        # Append a second copy of the same entry block.
        first_idx = text.find(f"### {RUN_ID}")
        block = text[first_idx:]
        self.logbook_path.write_text(text + "\n" + block, encoding="utf-8")
        self.assertHasCode("E_LOGBOOK_DUPLICATE", self.check())

    def test_logbook_entry_for_unknown_run(self) -> None:
        text = self.logbook_path.read_text(encoding="utf-8")
        ghost_block = text.replace(f"### {RUN_ID}", "### ghost-run")
        self.logbook_path.write_text(text + "\n" + ghost_block, encoding="utf-8")
        self.assertHasCode("E_LOGBOOK_RUN_MISSING", self.check())


# --------------------------- multi-seed runs ---------------------------


class MultiSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _build_valid_eval_repo(self.root)

        self.run_dir = self.root / "experiments" / "runs" / RUN_ID
        self.img_dir = self.run_dir / "per_image" / IMAGE_ID

        # Convert the single-seed fixture to a multi-seed run with seeds 0,1,2.
        seeds = [0, 1, 2]
        _mutate_json(self.run_dir / "run.json",
                     lambda d: d.__setitem__("seeds", seeds))

        # Top-level scores.json: seed=null and aggregated mean per metric.
        scores_path = self.img_dir / "scores.json"
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        scores["seed"] = None
        scores["per_seed"] = seeds
        # Per-seed dirs all use the same metric values, so the mean equals them.
        _write_json(scores_path, scores)

        seeds_dir = self.img_dir / "seeds"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            (seeds_dir / str(s)).mkdir(exist_ok=True)
            (seeds_dir / str(s) / "generated.png").write_bytes(b"img-" + str(s).encode())
            _write_json(seeds_dir / str(s) / "scores.json", {
                "schema_version": check_eval_storage.SCHEMA_VERSION,
                "image_id": IMAGE_ID,
                "seed": s,
                "scores": dict(METRICS),
                "judge": None,
                "generated_image_sha256": _sha256(b"img-" + str(s).encode()),
                "prompt_sha256": _sha256(b"A faithful, detailed reproduction."),
            })

        # Aggregate seeds list reflects multi-seed.
        _mutate_json(self.run_dir / "aggregate.json",
                     lambda d: d.__setitem__("seeds", seeds))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _check(self) -> check_eval_storage.Report:
        return check_eval_storage.check_root(self.root, verify_hashes=False)

    def _codes(self, report: check_eval_storage.Report) -> list[str]:
        return _codes(report)

    def test_multi_seed_valid_layout_passes(self) -> None:
        report = self._check()
        self.assertTrue(
            report.ok,
            msg=[(v.code, v.path, v.message) for v in report.violations],
        )

    def test_top_level_seed_must_be_null(self) -> None:
        scores_path = self.img_dir / "scores.json"
        _mutate_json(scores_path, lambda d: d.__setitem__("seed", 0))
        self.assertIn("E_SCORES_SEED_MISMATCH", self._codes(self._check()))

    def test_per_seed_seed_must_match_directory(self) -> None:
        seed1_scores = self.img_dir / "seeds" / "1" / "scores.json"
        _mutate_json(seed1_scores, lambda d: d.__setitem__("seed", 7))
        self.assertIn("E_SCORES_SEED_MISMATCH", self._codes(self._check()))

    def test_per_seed_dir_missing(self) -> None:
        import shutil
        shutil.rmtree(self.img_dir / "seeds" / "2")
        self.assertIn("E_FILE_MISSING", self._codes(self._check()))


# --------------------------- main() CLI ---------------------------


class MainCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _build_valid_eval_repo(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_main_returns_zero_on_valid(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = check_eval_storage.main(["--root", str(self.root)])
        self.assertEqual(rc, 0)

    def test_main_returns_one_on_violation(self) -> None:
        # Break the manifest so the run check fails.
        _mutate_json(self.root / "eval_data" / "images" / "manifest.json",
                     lambda d: d.__setitem__("schema_version", "0.0.0"))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = check_eval_storage.main(["--root", str(self.root)])
        self.assertEqual(rc, 1)

    def test_main_json_output(self) -> None:
        # Provoke at least one violation.
        _mutate_json(self.root / "experiments" / "runs" / RUN_ID / "run.json",
                     lambda d: d.__setitem__("status", "weird"))
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = check_eval_storage.main(["--root", str(self.root), "--json"])
        self.assertEqual(rc, 1)
        payload = json.loads(out.getvalue())
        self.assertIn("ok", payload)
        self.assertIn("violations", payload)
        self.assertFalse(payload["ok"])
        codes = {v["code"] for v in payload["violations"]}
        self.assertIn("E_RUN_STATUS", codes)

    def test_main_root_missing(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            rc = check_eval_storage.main(["--root", "/this/path/does/not/exist"])
        self.assertEqual(rc, 1)

    def test_main_verify_hashes_catches_mismatch(self) -> None:
        (self.root / "eval_data" / "images" / "train" / f"{IMAGE_ID}.png").write_bytes(
            b"different bytes")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc_ok = check_eval_storage.main(["--root", str(self.root)])
            rc_bad = check_eval_storage.main(["--root", str(self.root), "--verify-hashes"])
        self.assertEqual(rc_ok, 0)
        self.assertEqual(rc_bad, 1)


# --------------------------- GCS backend ---------------------------


from tests._gcs_stub import install_fake_storage  # noqa: E402

# Inject the fake before importing the backend module that lazily imports
# google.cloud.storage. The viewer tests already trigger this, but we call
# install explicitly to be order-independent.
_FAKE_CLIENT = install_fake_storage()


def _seed_bucket_from_local(bucket, src: Path, prefix: str) -> int:
    """Mirror every file under `src` to `bucket` under `prefix/`."""
    count = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        obj = f"{prefix.strip('/')}/{rel}" if prefix.strip("/") else rel
        bucket.blobs[obj] = path.read_bytes()
        count += 1
    return count


class GcsCheckerTests(unittest.TestCase):
    """Validate the checker over a GCS-backed root using the fake stub."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.local = Path(self._tmp.name)
        _build_valid_eval_repo(self.local)
        _FAKE_CLIENT._buckets.clear()
        self.bucket = _FAKE_CLIENT.bucket("checker-bucket")
        n = _seed_bucket_from_local(self.bucket, self.local, prefix="mirror")
        self.assertGreater(n, 0)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        _FAKE_CLIENT._buckets.clear()

    def _check(self, verify_hashes: bool = False) -> check_eval_storage.Report:
        return check_eval_storage.check_root(
            "gs://checker-bucket/mirror", verify_hashes=verify_hashes)

    def test_valid_gcs_root_passes(self) -> None:
        report = self._check()
        self.assertTrue(
            report.ok,
            msg=[(v.code, v.path, v.message) for v in report.violations],
        )

    def test_violation_paths_use_gs_uri(self) -> None:
        # Provoke any violation, then check the path field is a gs:// URL.
        manifest_obj = "mirror/eval_data/images/manifest.json"
        manifest_text = self.bucket.blobs[manifest_obj].decode("utf-8")
        broken = manifest_text.replace('"schema_version": "1.0.0"',
                                        '"schema_version": "0.0.0"', 1)
        self.bucket.blobs[manifest_obj] = broken.encode("utf-8")
        report = self._check()
        codes = [v.code for v in report.violations]
        self.assertIn("E_SCHEMA_VERSION", codes)
        # All violation paths should be gs:// URIs, not local paths.
        for v in report.violations:
            self.assertTrue(
                v.path.startswith("gs://checker-bucket/"),
                msg=f"violation path {v.path!r} is not a gs:// URI",
            )

    def test_score_violation_through_gcs(self) -> None:
        scores_obj = (
            f"mirror/experiments/runs/{RUN_ID}/per_image/{IMAGE_ID}/scores.json")
        text = self.bucket.blobs[scores_obj].decode("utf-8")
        data = json.loads(text)
        data["scores"]["s_gemini"] = 1.5
        self.bucket.blobs[scores_obj] = json.dumps(data).encode("utf-8")
        report = self._check()
        codes = [v.code for v in report.violations]
        self.assertIn("E_SCORES_RANGE", codes)

    def test_hash_verification_through_gcs(self) -> None:
        # Replace the image bytes; recorded sha256 in manifest should mismatch.
        img_obj = f"mirror/eval_data/images/train/{IMAGE_ID}.png"
        self.bucket.blobs[img_obj] = b"different image bytes"
        report = self._check(verify_hashes=True)
        codes = [v.code for v in report.violations]
        self.assertIn("E_IMAGE_HASH_MISMATCH", codes)

    def test_check_root_accepts_backend_instance(self) -> None:
        from storage_backend import GCSBackend
        backend = GCSBackend("checker-bucket", "mirror")
        report = check_eval_storage.check_root(backend, verify_hashes=False)
        self.assertTrue(report.ok)

    def test_main_with_gcs_root(self) -> None:
        # CLI: --root gs://... should validate and exit 0 on a clean repo.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = check_eval_storage.main([
                "--root", "gs://checker-bucket/mirror",
            ])
        self.assertEqual(rc, 0)

    def test_main_with_gcs_root_violation(self) -> None:
        # Break the bucket; CLI should exit 1.
        scores_obj = (
            f"mirror/experiments/runs/{RUN_ID}/per_image/{IMAGE_ID}/scores.json")
        text = self.bucket.blobs[scores_obj].decode("utf-8")
        data = json.loads(text)
        data["scores"]["s_gemini"] = -0.1
        self.bucket.blobs[scores_obj] = json.dumps(data).encode("utf-8")
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = check_eval_storage.main([
                "--root", "gs://checker-bucket/mirror", "--json",
            ])
        self.assertEqual(rc, 1)
        payload = json.loads(out.getvalue())
        self.assertFalse(payload["ok"])
        self.assertIn("E_SCORES_RANGE",
                      {v["code"] for v in payload["violations"]})


class GcsCheckerNoPrefixTests(unittest.TestCase):
    """Same checker against a bucket with no prefix."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.local = Path(self._tmp.name)
        _build_valid_eval_repo(self.local)
        _FAKE_CLIENT._buckets.clear()
        self.bucket = _FAKE_CLIENT.bucket("checker-rootbucket")
        _seed_bucket_from_local(self.bucket, self.local, prefix="")

    def tearDown(self) -> None:
        self._tmp.cleanup()
        _FAKE_CLIENT._buckets.clear()

    def test_no_prefix_root_passes(self) -> None:
        report = check_eval_storage.check_root(
            "gs://checker-rootbucket", verify_hashes=False)
        self.assertTrue(
            report.ok,
            msg=[(v.code, v.path, v.message) for v in report.violations],
        )


class CheckRootTypeDispatchTests(unittest.TestCase):
    """check_root() accepts Path, str, or Backend."""

    def test_accepts_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_valid_eval_repo(root)
            self.assertTrue(check_eval_storage.check_root(root, verify_hashes=False).ok)

    def test_accepts_str_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _build_valid_eval_repo(Path(tmp))
            self.assertTrue(
                check_eval_storage.check_root(tmp, verify_hashes=False).ok)

    def test_rejects_bad_gs_uri(self) -> None:
        with self.assertRaises(ValueError):
            check_eval_storage.check_root("gs:///nope", verify_hashes=False)


if __name__ == "__main__":
    unittest.main()
