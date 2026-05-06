"""Tests for view_eval_results.py."""
from __future__ import annotations

import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from tests._fixtures import (  # noqa: E402
    IMAGE_ID,
    METRICS,
    RUN_ID,
    TINY_PNG,
    build_minimal_repo,
    seed_bucket_from_local,
)
from tests._gcs_stub import install_fake_storage  # noqa: E402

# Install fake GCS BEFORE importing the viewer so GCSBackend's lazy
# `from google.cloud import storage` resolves to our fake.
_FAKE_CLIENT = install_fake_storage()

import view_eval_results as v  # noqa: E402


class CleanRelTests(unittest.TestCase):
    def test_empty_and_normalize(self) -> None:
        self.assertEqual(v._clean_rel(""), "")
        self.assertEqual(v._clean_rel("/"), "")
        self.assertEqual(v._clean_rel("a/b/c"), "a/b/c")
        self.assertEqual(v._clean_rel("a\\b\\c"), "a/b/c")
        self.assertEqual(v._clean_rel("/a/b/"), "a/b")

    def test_rejects_traversal(self) -> None:
        with self.assertRaises(PermissionError):
            v._clean_rel("../etc/passwd")
        with self.assertRaises(PermissionError):
            v._clean_rel("a/../b")
        with self.assertRaises(PermissionError):
            v._clean_rel("a/./b")


class MakeBackendTests(unittest.TestCase):
    def test_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            backend = v.make_backend(td, gcs_cache_ttl=10.0)
            self.assertIsInstance(backend, v.LocalBackend)

    def test_gcs(self) -> None:
        backend = v.make_backend("gs://my-bucket/some/prefix/", gcs_cache_ttl=5.0)
        self.assertIsInstance(backend, v.GCSBackend)
        self.assertEqual(backend.root_label, "gs://my-bucket/some/prefix")
        self.assertEqual(backend._full(""), "some/prefix")
        self.assertEqual(backend._full("a/b"), "some/prefix/a/b")

    def test_gcs_no_prefix(self) -> None:
        backend = v.make_backend("gs://just-a-bucket", gcs_cache_ttl=5.0)
        self.assertEqual(backend.root_label, "gs://just-a-bucket")
        self.assertEqual(backend._full(""), "")
        self.assertEqual(backend._full("foo"), "foo")

    def test_gcs_missing_bucket(self) -> None:
        with self.assertRaises(ValueError):
            v.make_backend("gs:///nope", gcs_cache_ttl=5.0)


class LocalBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.info = build_minimal_repo(self.root)
        self.backend = v.LocalBackend(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_root_label(self) -> None:
        self.assertEqual(self.backend.root_label, self.root.name)

    def test_is_dir_and_file(self) -> None:
        self.assertTrue(self.backend.is_dir(""))
        self.assertTrue(self.backend.is_dir("experiments/runs"))
        self.assertTrue(self.backend.is_file("eval_data/images/manifest.json"))
        self.assertFalse(self.backend.is_file("does/not/exist"))

    def test_list_dir(self) -> None:
        subdirs, files = self.backend.list_dir("experiments")
        names = {s["name"] for s in subdirs}
        self.assertEqual(names, {"runs", "leader"})
        file_names = {f["name"] for f in files}
        self.assertIn("logbook.md", file_names)

    def test_read_text(self) -> None:
        text = self.backend.read_text("experiments/logbook.md")
        self.assertIn("# Logbook", text)

    def test_stream(self) -> None:
        chunks = list(self.backend.stream("eval_data/images/eval/" + IMAGE_ID + ".png"))
        self.assertEqual(b"".join(chunks), TINY_PNG)


class GCSBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.local = Path(self._tmp.name)
        build_minimal_repo(self.local)
        # Reset bucket and seed it from the local fixture.
        _FAKE_CLIENT._buckets.clear()
        bucket = _FAKE_CLIENT.bucket("test-bucket")
        n = seed_bucket_from_local(bucket, self.local, prefix="experiments-mirror")
        self.assertGreater(n, 0)
        self.backend = v.GCSBackend("test-bucket", "experiments-mirror", cache_ttl=60.0)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        _FAKE_CLIENT._buckets.clear()

    def test_full_path_join(self) -> None:
        self.assertEqual(self.backend._full(""), "experiments-mirror")
        self.assertEqual(self.backend._full("a/b"), "experiments-mirror/a/b")

    def test_is_file_and_dir(self) -> None:
        self.assertTrue(self.backend.is_file("eval_data/images/manifest.json"))
        self.assertFalse(self.backend.is_file("eval_data/images"))
        self.assertTrue(self.backend.is_dir("experiments/runs"))
        self.assertTrue(self.backend.is_dir(""))
        self.assertFalse(self.backend.is_file("nope"))

    def test_list_dir_one_level(self) -> None:
        subdirs, files = self.backend.list_dir("experiments")
        self.assertEqual({s["name"] for s in subdirs}, {"runs", "leader"})
        self.assertEqual({f["name"] for f in files}, {"logbook.md"})

    def test_list_dir_skips_holdout_when_listed_directly(self) -> None:
        # holdout/ is present in the bucket but only appears if we list its parent.
        subdirs, files = self.backend.list_dir("eval_data/images")
        names = {s["name"] for s in subdirs}
        self.assertIn("eval", names)
        self.assertIn("holdout", names)  # backend exposes it; sync excludes it.

    def test_read_text_and_stream(self) -> None:
        text = self.backend.read_text("experiments/logbook.md")
        self.assertIn("# Logbook", text)
        chunks = list(self.backend.stream(
            "eval_data/images/eval/" + IMAGE_ID + ".png"))
        self.assertEqual(b"".join(chunks), TINY_PNG)

    def test_file_size(self) -> None:
        self.assertEqual(
            self.backend.file_size("eval_data/images/eval/" + IMAGE_ID + ".png"),
            len(TINY_PNG),
        )


class ViewBuildersTests(unittest.TestCase):
    """High-level view builders should produce expected fields."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        build_minimal_repo(self.root)
        v.BACKEND = v.LocalBackend(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_classify_dir(self) -> None:
        self.assertEqual(v.classify_dir(""), "dir")
        self.assertEqual(v.classify_dir("experiments"), "dir")
        self.assertEqual(v.classify_dir("experiments/runs"), "runs_container")
        self.assertEqual(v.classify_dir("experiments/runs/" + RUN_ID), "run")

    def test_find_manifest_walks_up(self) -> None:
        path = v.find_manifest("experiments/runs/" + RUN_ID)
        self.assertEqual(path, "eval_data/images/manifest.json")

    def test_load_manifest_for_indexes_image_ids(self) -> None:
        index = v.load_manifest_for("experiments/runs/" + RUN_ID)
        self.assertIn(IMAGE_ID, index)
        self.assertEqual(index[IMAGE_ID]["__split__"], "eval")
        self.assertEqual(index[IMAGE_ID]["__manifest_dir__"], "eval_data/images")

    def test_build_summary(self) -> None:
        rows = v.build_summary("experiments/runs")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["run_id"], RUN_ID)
        self.assertEqual(row["composite"], 0.5)
        self.assertEqual(row["decision"], "no_leader")
        self.assertEqual(set(row["means"].keys()), set(METRICS))

    def test_build_run_detail(self) -> None:
        detail = v.build_run_detail("experiments/runs/" + RUN_ID)
        self.assertEqual(detail["run"]["run_id"], RUN_ID)
        self.assertEqual(detail["aggregate"]["composite"], 0.5)
        self.assertEqual(len(detail["images"]), 1)
        img = detail["images"][0]
        self.assertEqual(img["image_id"], IMAGE_ID)
        # Target URL must NOT have doubled "images/images" segments.
        self.assertEqual(
            img["target_url"],
            "/api/file?path=eval_data/images/eval/" + IMAGE_ID + ".png",
        )
        self.assertTrue(img["generated_url"].endswith(
            f"per_image/{IMAGE_ID}/generated.png"))
        self.assertEqual(img["prompt"], "describe this image")
        self.assertEqual(set(img["scores"].keys()), set(METRICS))

    def test_inspect_dir_runs_container(self) -> None:
        data = v.inspect_dir("experiments/runs")
        self.assertEqual(data["kind"], "runs_container")
        self.assertEqual(len(data["summary"]), 1)
        self.assertEqual(data["root_name"], v.BACKEND.root_label)
        # Timeline must come along with the runs_container response.
        self.assertIn("timeline", data)
        t = data["timeline"]
        self.assertEqual(len(t["runs"]), 1)
        self.assertEqual(t["image_ids"], [IMAGE_ID])
        cell = t["cells"][IMAGE_ID][RUN_ID]
        self.assertEqual(set(cell["scores"].keys()), set(METRICS))
        self.assertEqual(cell["decision"], "no_leader")
        self.assertTrue(cell["generated_url"].endswith(
            f"per_image/{IMAGE_ID}/generated.png"))
        self.assertEqual(t["manifest"][IMAGE_ID]["split"], "eval")
        self.assertEqual(
            t["manifest"][IMAGE_ID]["target_url"],
            "/api/file?path=eval_data/images/eval/" + IMAGE_ID + ".png",
        )

    def test_build_timeline_orders_by_started_at(self) -> None:
        # Add a second run dated AFTER the existing fixture run and verify
        # the timeline returns them oldest -> newest.
        from tests._fixtures import RUN_ID as FIX_RUN_ID  # noqa: F401
        second_run_id = "20260504T999999Z__claude-opus-4-7__followup"
        run_dir = self.root / "experiments" / "runs" / second_run_id
        per_image = run_dir / "per_image" / IMAGE_ID
        per_image.mkdir(parents=True, exist_ok=True)
        (per_image / "prompt.txt").write_text("p", encoding="utf-8")
        (per_image / "generated.png").write_bytes(TINY_PNG)
        (run_dir / "prompt_strategy.py").write_text("# v2\n", encoding="utf-8")
        (run_dir / "stdout.log").write_text("ok\n", encoding="utf-8")
        means = {m: 0.6 for m in METRICS}
        composite = sum(means.values()) / len(means)
        json_files = {
            "run.json": {
                "schema_version": "1.0.0",
                "run_id": second_run_id,
                "name": "followup",
                "driver": "claude-opus-4-7",
                "harness_variant": "opus4.7",
                "started_at": "2026-06-01T08:00:00Z",
                "finished_at": "2026-06-01T08:05:00Z",
                "split": "eval",
                "image_ids": [IMAGE_ID],
                "seeds": [0],
                "status": "completed",
            },
            "config.json": {
                "schema_version": "1.0.0",
                "harness_variant": "opus4.7",
                "models": {"x": "y"},
                "metrics": list(METRICS),
            },
            "aggregate.json": {
                "schema_version": "1.0.0",
                "run_id": second_run_id,
                "split": "eval",
                "n_images": 1,
                "seeds": [0],
                "means": means,
                "composite": composite,
                "composite_unweighted": composite,
            },
            "gate.json": {
                "schema_version": "1.0.0",
                "leader_run_id": RUN_ID,
                "leader_means": {m: 0.5 for m in METRICS},
                "leader_composite": 0.5,
                "candidate_means": means,
                "candidate_composite": composite,
                "regression_epsilon": 0.01,
                "no_regression": True,
                "improves_composite": True,
                "single_run_gate": "pass",
                "decision": "promoted",
            },
        }
        for fname, payload in json_files.items():
            (run_dir / fname).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        scores_doc = {
            "schema_version": "1.0.0",
            "image_id": IMAGE_ID,
            "seed": 0,
            "scores": means,
            "judge": None,
            "generated_image_sha256": "a" * 64,
            "prompt_sha256": "b" * 64,
        }
        (per_image / "scores.json").write_text(
            json.dumps(scores_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        t = v.build_timeline("experiments/runs")
        ordered_ids = [r["run_id"] for r in t["runs"]]
        self.assertEqual(ordered_ids, [RUN_ID, second_run_id])
        # Promoted column flagged as leader-chain entry
        self.assertTrue(t["runs"][0]["is_leader_promotion"])  # no_leader counts
        self.assertTrue(t["runs"][1]["is_leader_promotion"])  # promoted counts
        # The cell for the second run should have its own scores/url
        self.assertIn(second_run_id, t["cells"][IMAGE_ID])
        self.assertNotEqual(
            t["cells"][IMAGE_ID][RUN_ID]["composite"],
            t["cells"][IMAGE_ID][second_run_id]["composite"],
        )

    def test_inspect_dir_run(self) -> None:
        data = v.inspect_dir("experiments/runs/" + RUN_ID)
        self.assertEqual(data["kind"], "run")
        self.assertIn("run_detail", data)

    def test_inspect_dir_breadcrumb(self) -> None:
        data = v.inspect_dir("experiments/runs/" + RUN_ID)
        bc_paths = [b["path"] for b in data["breadcrumb"]]
        self.assertEqual(
            bc_paths,
            ["", "experiments", "experiments/runs", "experiments/runs/" + RUN_ID],
        )


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class HttpServerTests(unittest.TestCase):
    """Spin up the actual HTTP server against a local backend."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        build_minimal_repo(cls.root)
        v.BACKEND = v.LocalBackend(cls.root)
        cls.port = _free_port()
        cls.server = v.ThreadingServer(("127.0.0.1", cls.port), v.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        # tiny wait for serve loop
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls._tmp.cleanup()

    def _get(self, path: str) -> tuple[int, bytes, str]:
        url = f"http://127.0.0.1:{self.port}{path}"
        try:
            with urllib.request.urlopen(url) as resp:
                return resp.status, resp.read(), resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read() or b"", exc.headers.get("Content-Type", "")

    def test_index_html(self) -> None:
        status, body, ctype = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Eval Results Viewer", body)
        self.assertTrue(ctype.startswith("text/html"))

    def test_inspect_runs_container(self) -> None:
        status, body, ctype = self._get("/api/inspect?path=experiments%2Fruns")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "application/json")
        data = json.loads(body)
        self.assertEqual(data["kind"], "runs_container")
        self.assertEqual(len(data["summary"]), 1)

    def test_inspect_run(self) -> None:
        status, body, _ = self._get(
            f"/api/inspect?path=experiments%2Fruns%2F{RUN_ID}")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["kind"], "run")
        self.assertEqual(data["run_detail"]["run"]["run_id"], RUN_ID)

    def test_file_serving(self) -> None:
        target_param = (
            "eval_data%2Fimages%2Feval%2F" + IMAGE_ID + ".png")
        status, body, ctype = self._get(f"/api/file?path={target_param}")
        self.assertEqual(status, 200)
        self.assertEqual(body, TINY_PNG)
        self.assertEqual(ctype, "image/png")

    def test_traversal_blocked(self) -> None:
        status, _, _ = self._get("/api/inspect?path=..%2F..%2Fetc")
        self.assertEqual(status, 403)

    def test_missing_returns_404(self) -> None:
        status, _, _ = self._get("/api/inspect?path=does%2Fnot%2Fexist")
        self.assertEqual(status, 404)
        status, _, _ = self._get("/api/file?path=missing.png")
        self.assertEqual(status, 404)


class CloudRunEntrypointTests(unittest.TestCase):
    """`--gcs-only` and the env-var-driven defaults that Cloud Run depends on."""

    def setUp(self) -> None:
        # Snapshot env so individual tests can poke specific vars.
        self._env_keys = (
            "VIEWER_ROOT", "VIEWER_HOST", "VIEWER_GCS_ONLY",
            "VIEWER_GCS_CACHE_TTL", "PORT",
        )
        self._saved = {k: os.environ.pop(k, None) for k in self._env_keys}

    def tearDown(self) -> None:
        for k, val in self._saved.items():
            os.environ.pop(k, None)
            if val is not None:
                os.environ[k] = val

    def test_gcs_only_flag_rejects_local_root(self) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            rc = v.main(["--root", "/tmp", "--gcs-only"])
        self.assertEqual(rc, 2)
        self.assertIn("gs://", err.getvalue())

    def test_gcs_only_env_rejects_local_root(self) -> None:
        os.environ["VIEWER_GCS_ONLY"] = "1"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = v.main(["--root", "/tmp"])
        self.assertEqual(rc, 2)

    def _stub_make_backend(self, captured: dict):
        """Replace make_backend with a recorder that raises before binding."""
        original = v.make_backend

        def fake(root, ttl):
            captured["root"] = root
            captured["ttl"] = ttl
            raise ImportError("stubbed: short-circuit before server bind")

        self.addCleanup(lambda: setattr(v, "make_backend", original))
        v.make_backend = fake

    def test_gcs_only_flag_accepts_gs_root(self) -> None:
        captured: dict = {}
        self._stub_make_backend(captured)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = v.main(["--root", "gs://test-bucket/x", "--gcs-only"])
        # rc=1 because the stubbed make_backend raises ImportError;
        # rc=2 would mean the gcs-only validation rejected the gs:// root.
        self.assertEqual(rc, 1)
        self.assertEqual(captured["root"], "gs://test-bucket/x")

    def test_env_var_defaults_apply(self) -> None:
        os.environ["VIEWER_ROOT"] = "gs://envb/envp"
        os.environ["PORT"] = "5500"
        os.environ["VIEWER_HOST"] = "0.0.0.0"
        os.environ["VIEWER_GCS_CACHE_TTL"] = "12.5"
        os.environ["VIEWER_GCS_ONLY"] = "1"
        captured: dict = {}
        self._stub_make_backend(captured)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = v.main([])
        # Env's VIEWER_ROOT must have flowed into make_backend (not cwd).
        self.assertEqual(captured["root"], "gs://envb/envp")
        self.assertAlmostEqual(captured["ttl"], 12.5)
        self.assertEqual(rc, 1)  # ImportError from stub, not a 2 from validation


if __name__ == "__main__":
    unittest.main()
