"""Tests for sync_runs_to_gcs.py."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from tests._fixtures import RUN_ID, build_minimal_repo  # noqa: E402
from tests._gcs_stub import install_fake_storage  # noqa: E402

# Inject stub before importing the sync module so its lazy storage import
# resolves to our fake.
_FAKE_CLIENT = install_fake_storage()

import sync_runs_to_gcs as s  # noqa: E402


class ParseGcsUriTests(unittest.TestCase):
    def test_full(self) -> None:
        self.assertEqual(s.parse_gcs_uri("gs://b/p/q"), ("b", "p/q"))
        self.assertEqual(s.parse_gcs_uri("gs://b/p/q/"), ("b", "p/q"))

    def test_bucket_only(self) -> None:
        self.assertEqual(s.parse_gcs_uri("gs://just-bucket"), ("just-bucket", ""))
        self.assertEqual(s.parse_gcs_uri("gs://just-bucket/"), ("just-bucket", ""))

    def test_rejects_bad(self) -> None:
        with self.assertRaises(ValueError):
            s.parse_gcs_uri("not-a-gs-uri")
        with self.assertRaises(ValueError):
            s.parse_gcs_uri("gs:///nope")


class ExclusionTests(unittest.TestCase):
    def test_holdout_always_excluded(self) -> None:
        self.assertTrue(s.is_excluded("eval_data/images/holdout/secret.png"))

    def test_cache_and_weights(self) -> None:
        self.assertTrue(s.is_excluded("foo/cache/x.npz"))
        self.assertTrue(s.is_excluded("weights/dino.bin"))
        self.assertTrue(s.is_excluded("a/__pycache__/b.pyc"))

    def test_normal_files_kept(self) -> None:
        self.assertFalse(s.is_excluded(
            "experiments/runs/RID/per_image/img/scores.json"))
        self.assertFalse(s.is_excluded("experiments/logbook.md"))


class IterFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        build_minimal_repo(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_walks_runs(self) -> None:
        entries = s.iter_files(self.root, "experiments/runs")
        rels = {rel for _, rel in entries}
        self.assertIn(f"experiments/runs/{RUN_ID}/run.json", rels)
        self.assertIn(f"experiments/runs/{RUN_ID}/aggregate.json", rels)
        self.assertIn(
            f"experiments/runs/{RUN_ID}/per_image/hero_photo_01/generated.png", rels)

    def test_single_file(self) -> None:
        entries = s.iter_files(self.root, "experiments/logbook.md")
        rels = [rel for _, rel in entries]
        self.assertEqual(rels, ["experiments/logbook.md"])

    def test_holdout_excluded(self) -> None:
        # eval_data/images includes a holdout file; iter_files for that subtree
        # must omit it because of EXCLUDE_PATHS_PREFIXES.
        entries = s.iter_files(self.root, "eval_data/images")
        rels = {rel for _, rel in entries}
        self.assertIn("eval_data/images/manifest.json", rels)
        self.assertIn("eval_data/images/eval/hero_photo_01.png", rels)
        self.assertFalse(any("holdout" in rel for rel in rels))


class FilterByRunsTests(unittest.TestCase):
    def test_filters_run_dirs_only(self) -> None:
        sample = [
            (Path("/x/experiments/runs/A/run.json"),
             "experiments/runs/A/run.json"),
            (Path("/x/experiments/runs/B/run.json"),
             "experiments/runs/B/run.json"),
            (Path("/x/experiments/leader/pointer.json"),
             "experiments/leader/pointer.json"),
            (Path("/x/eval_data/images/manifest.json"),
             "eval_data/images/manifest.json"),
        ]
        out = s.filter_by_runs(sample, ["A"])
        rels = {rel for _, rel in out}
        # Only run "A" kept, but non-run files always pass through.
        self.assertEqual(rels, {
            "experiments/runs/A/run.json",
            "experiments/leader/pointer.json",
            "eval_data/images/manifest.json",
        })

    def test_no_filter_returns_all(self) -> None:
        sample = [(Path("/x/y"), "experiments/runs/A/run.json")]
        self.assertEqual(s.filter_by_runs(sample, []), sample)


class BuildPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        build_minimal_repo(self.root)
        _FAKE_CLIENT._buckets.clear()
        self.bucket = _FAKE_CLIENT.bucket("test-bucket")

    def tearDown(self) -> None:
        self._tmp.cleanup()
        _FAKE_CLIENT._buckets.clear()

    def test_empty_remote_uploads_everything(self) -> None:
        plan = s.build_plan(
            src=self.root, bucket=self.bucket, prefix="mirror",
            client=_FAKE_CLIENT, runs=[], include_images=False,
            delete_orphans=False, force=False)
        self.assertEqual(len(plan.skips), 0)
        objs = {o for _, o in plan.uploads}
        self.assertIn("mirror/eval_data/images/manifest.json", objs)
        self.assertIn(f"mirror/experiments/runs/{RUN_ID}/run.json", objs)
        self.assertIn("mirror/experiments/leader/pointer.json", objs)
        self.assertIn("mirror/experiments/logbook.md", objs)
        # holdout never included unless --include-images, and even then never.
        self.assertFalse(any("holdout" in o for o in objs))

    def test_size_match_skips(self) -> None:
        local_path = self.root / "experiments/logbook.md"
        # Pre-seed remote with identical size.
        self.bucket.blobs["mirror/experiments/logbook.md"] = local_path.read_bytes()
        plan = s.build_plan(
            src=self.root, bucket=self.bucket, prefix="mirror",
            client=_FAKE_CLIENT, runs=[], include_images=False,
            delete_orphans=False, force=False)
        skip_objs = {o for _, o, _ in plan.skips}
        self.assertIn("mirror/experiments/logbook.md", skip_objs)

    def test_force_reuploads(self) -> None:
        local_path = self.root / "experiments/logbook.md"
        self.bucket.blobs["mirror/experiments/logbook.md"] = local_path.read_bytes()
        plan = s.build_plan(
            src=self.root, bucket=self.bucket, prefix="mirror",
            client=_FAKE_CLIENT, runs=[], include_images=False,
            delete_orphans=False, force=True)
        upload_objs = {o for _, o in plan.uploads}
        self.assertIn("mirror/experiments/logbook.md", upload_objs)

    def test_orphan_deletion_in_scope(self) -> None:
        # Add an orphan inside experiments/runs/<ghost>/...
        self.bucket.blobs["mirror/experiments/runs/ghost__nope/run.json"] = b"{}"
        # And one outside any scope — must not be deleted.
        self.bucket.blobs["mirror/unrelated/foo.txt"] = b"keep me"

        plan = s.build_plan(
            src=self.root, bucket=self.bucket, prefix="mirror",
            client=_FAKE_CLIENT, runs=[], include_images=False,
            delete_orphans=True, force=False)
        self.assertIn("mirror/experiments/runs/ghost__nope/run.json", plan.deletes)
        self.assertNotIn("mirror/unrelated/foo.txt", plan.deletes)

    def test_include_images_uploads_train_eval_val_only(self) -> None:
        plan = s.build_plan(
            src=self.root, bucket=self.bucket, prefix="mirror",
            client=_FAKE_CLIENT, runs=[], include_images=True,
            delete_orphans=False, force=False)
        objs = {o for _, o in plan.uploads}
        self.assertIn("mirror/eval_data/images/eval/hero_photo_01.png", objs)
        self.assertFalse(any("holdout" in o for o in objs))

    def test_run_scoped_filters_other_runs(self) -> None:
        # Add a second run on disk.
        other_run = self.root / "experiments" / "runs" / "OTHER"
        (other_run / "per_image" / "x").mkdir(parents=True, exist_ok=True)
        (other_run / "run.json").write_text("{}", encoding="utf-8")
        plan = s.build_plan(
            src=self.root, bucket=self.bucket, prefix="mirror",
            client=_FAKE_CLIENT, runs=[RUN_ID], include_images=False,
            delete_orphans=False, force=False)
        upload_objs = {o for _, o in plan.uploads}
        self.assertTrue(any(RUN_ID in o for o in upload_objs))
        self.assertFalse(any("/runs/OTHER/" in o for o in upload_objs))


class RunPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        build_minimal_repo(self.root)
        _FAKE_CLIENT._buckets.clear()
        self.bucket = _FAKE_CLIENT.bucket("test-bucket")

    def tearDown(self) -> None:
        self._tmp.cleanup()
        _FAKE_CLIENT._buckets.clear()

    def test_dry_run_does_not_upload(self) -> None:
        plan = s.build_plan(
            src=self.root, bucket=self.bucket, prefix="mirror",
            client=_FAKE_CLIENT, runs=[], include_images=False,
            delete_orphans=False, force=False)
        self.assertGreater(len(plan.uploads), 0)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            stats = s.run_plan(plan, _FAKE_CLIENT, self.bucket,
                               dry_run=True, workers=2, quiet=True)
        self.assertEqual(stats.uploaded, len(plan.uploads))
        self.assertEqual(stats.failures, 0)
        self.assertEqual(self.bucket.blobs, {})  # nothing actually uploaded

    def test_actual_upload_populates_remote(self) -> None:
        plan = s.build_plan(
            src=self.root, bucket=self.bucket, prefix="mirror",
            client=_FAKE_CLIENT, runs=[], include_images=False,
            delete_orphans=False, force=False)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            stats = s.run_plan(plan, _FAKE_CLIENT, self.bucket,
                               dry_run=False, workers=4, quiet=True)
        self.assertEqual(stats.failures, 0)
        # Real bytes for an artifact must round-trip.
        local_run = (self.root / "experiments/runs" / RUN_ID / "run.json").read_bytes()
        self.assertEqual(
            self.bucket.blobs[f"mirror/experiments/runs/{RUN_ID}/run.json"],
            local_run,
        )
        # Holdout never uploaded.
        self.assertFalse(any("holdout" in k for k in self.bucket.blobs))

    def test_main_dry_run(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = s.main([
                "--src", str(self.root),
                "--dst", "gs://test-bucket/mirror",
                "--dry-run", "--quiet",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(self.bucket.blobs, {})

    def test_main_actual(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = s.main([
                "--src", str(self.root),
                "--dst", "gs://test-bucket/mirror",
                "--quiet",
            ])
        self.assertEqual(rc, 0)
        self.assertIn(f"mirror/experiments/runs/{RUN_ID}/run.json",
                      self.bucket.blobs)

    def test_main_subsequent_call_skips_unchanged(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(s.main([
                "--src", str(self.root),
                "--dst", "gs://test-bucket/mirror",
                "--quiet",
            ]), 0)
        first_blobs = dict(self.bucket.blobs)

        # Re-run; remote already matches → second pass should produce no
        # new uploads (size-match skip path).
        plan = s.build_plan(
            src=self.root, bucket=self.bucket, prefix="mirror",
            client=_FAKE_CLIENT, runs=[], include_images=False,
            delete_orphans=False, force=False)
        self.assertEqual(plan.uploads, [])
        self.assertGreater(len(plan.skips), 0)
        self.assertEqual(self.bucket.blobs, first_blobs)


if __name__ == "__main__":
    unittest.main()
