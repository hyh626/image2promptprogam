"""Tests for view_eval_results.py."""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
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


if __name__ == "__main__":
    unittest.main()
