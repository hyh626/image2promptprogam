"""Fixed metric stack for the image-prompt autoresearch harness."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

os.environ.setdefault("TORCH_HOME", str(Path("weights") / "torch"))

import lpips  # noqa: E402

load_dotenv()

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT:
    raise RuntimeError(
        "GOOGLE_CLOUD_PROJECT is required. Copy .env.example to .env, set "
        "GOOGLE_CLOUD_PROJECT, and authenticate to Vertex AI."
    )

LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
WEIGHTS_DIR = Path("weights")
CACHE_DIR = Path("cache")
FEATURE_CACHE_DIR = Path("experiments") / "cache" / "features"
ORIGINALS_CACHE = CACHE_DIR / "originals.npz"

CANONICAL_SIZE = (448, 448)
METRIC_KEYS = ("s_gemini", "s_dino", "s_lpips", "s_color")
EMBEDDING_MODEL = "gemini-embedding-2"
VLM_JUDGE_MODEL = "gemini-3.1-flash-lite-preview"

_client = genai.Client(
    vertexai=True,
    project=PROJECT,
    location=LOCATION,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dino_processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base",
    cache_dir=str(WEIGHTS_DIR),
)
dino_model = AutoModel.from_pretrained(
    "facebook/dinov2-base",
    cache_dir=str(WEIGHTS_DIR),
).to(device)
dino_model.eval()

lpips_model = lpips.LPIPS(net="alex").to(device)
lpips_model.eval()

T = TypeVar("T")


def retry_with_backoff(fn: Callable[[], T], *, max_retries: int = 5) -> T:
    """Run a Gemini API operation with exponential backoff and jitter."""
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - surface original on final failure.
            last_exc = exc
            if attempt == max_retries - 1:
                raise
            time.sleep(delay + random.uniform(0.0, delay * 0.25))
            delay *= 2.0
    assert last_exc is not None
    raise last_exc


def _canonical(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(CANONICAL_SIZE, Image.Resampling.LANCZOS)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr
    return arr / norm


def _embedding_values(response: object) -> np.ndarray:
    embeddings = getattr(response, "embeddings", None)
    if embeddings:
        emb = embeddings[0]
        values = getattr(emb, "values", None)
        if values is None and hasattr(emb, "embedding"):
            values = getattr(emb.embedding, "values", None)
        if values is not None:
            return _l2_normalize(np.asarray(values, dtype=np.float32))

    embedding = getattr(response, "embedding", None)
    if embedding is not None:
        values = getattr(embedding, "values", None)
        if values is not None:
            return _l2_normalize(np.asarray(values, dtype=np.float32))

    raise RuntimeError("Gemini embedding response did not include embedding values.")


def _gemini_image_embedding(image: Image.Image) -> np.ndarray:
    data = _png_bytes(image)
    part = types.Part.from_bytes(data=data, mime_type="image/png")

    def call() -> object:
        try:
            return _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[part],
                config=types.EmbedContentConfig(output_dimensionality=3072),
            )
        except (AttributeError, TypeError):
            return _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[part],
            )

    return _embedding_values(retry_with_backoff(call))


def _dino_feature(image: Image.Image) -> np.ndarray:
    inputs = dino_processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = dino_model(**inputs)
    cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
    return _l2_normalize(cls)


def _lpips_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _color_histogram(image: Image.Image) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=(8, 8, 8),
        range=((0, 256), (0, 256), (0, 256)),
    )
    flat = hist.astype(np.float32).reshape(-1)
    total = float(flat.sum())
    if total == 0.0:
        return flat
    return flat / total


def featurize(image: Image.Image) -> dict:
    """Return Gemini, DINOv2, LPIPS, and HSV-histogram features."""
    canonical = _canonical(image)
    return {
        "gemini": _gemini_image_embedding(canonical),
        "dino": _dino_feature(canonical),
        "lpips_tensor": _lpips_tensor(canonical),
        "color_hist": _color_histogram(canonical),
    }


def _load_original_cache() -> dict[str, dict]:
    if not ORIGINALS_CACHE.exists():
        return {}
    data = np.load(ORIGINALS_CACHE, allow_pickle=False)
    entries: dict[str, dict] = {}
    paths = [str(p) for p in data["paths"]]
    hashes = [str(h) for h in data["hashes"]]
    for idx, path in enumerate(paths):
        entries[path] = {
            "hash": hashes[idx],
            "gemini": data["gemini"][idx],
            "dino": data["dino"][idx],
            "color_hist": data["color_hist"][idx],
        }
    return entries


def _save_original_cache(entries: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(entries)
    gemini = np.stack([entries[p]["gemini"] for p in paths]) if paths else np.empty((0, 3072))
    dino = np.stack([entries[p]["dino"] for p in paths]) if paths else np.empty((0, 768))
    color = np.stack([entries[p]["color_hist"] for p in paths]) if paths else np.empty((0, 512))
    np.savez(
        ORIGINALS_CACHE,
        paths=np.array(paths),
        hashes=np.array([entries[p]["hash"] for p in paths]),
        gemini=gemini,
        dino=dino,
        color_hist=color,
    )


def _json_safe(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(float(v)) for v in obj.reshape(-1)]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _slug_image_id(path: Path) -> str:
    stem = path.stem.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_.-]+", "_", stem).strip("_") or "image"


def _infer_split_and_id(path: Path) -> tuple[str | None, str]:
    resolved = path.resolve()
    parts = resolved.parts
    for split in ("train", "eval", "val", "holdout"):
        marker = ("eval_data", "images", split)
        for idx in range(len(parts) - len(marker) + 1):
            if tuple(parts[idx : idx + len(marker)]) == marker:
                return split, _slug_image_id(path)
    return None, _slug_image_id(path)


def _update_feature_index(path: Path, sha: str, feat: dict) -> None:
    split, image_id = _infer_split_and_id(path)
    if split is None:
        return
    feature_path = FEATURE_CACHE_DIR / split / f"{image_id}.json"
    feature_payload = {
        "schema_version": "1.0.0",
        "image_id": image_id,
        "split": split,
        "sha256": sha,
        "features": {
            "gemini": base64.b64encode(np.asarray(feat["gemini"], dtype=np.float32).tobytes()).decode("ascii"),
            "dino": base64.b64encode(np.asarray(feat["dino"], dtype=np.float32).tobytes()).decode("ascii"),
            "color_hist": base64.b64encode(np.asarray(feat["color_hist"], dtype=np.float32).tobytes()).decode("ascii"),
        },
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(feature_path, feature_payload)

    index_path = FEATURE_CACHE_DIR / "index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {"schema_version": "1.0.0", "entries": []}
    else:
        index = {"schema_version": "1.0.0", "entries": []}
    entries = [
        e for e in index.get("entries", [])
        if not (e.get("image_id") == image_id and e.get("split") == split)
    ]
    entries.append(
        {
            "image_id": image_id,
            "split": split,
            "sha256": sha,
            "feature_path": str(feature_path),
            "computed_at": feature_payload["computed_at"],
        }
    )
    index["schema_version"] = "1.0.0"
    index["entries"] = sorted(entries, key=lambda e: (e["split"], e["image_id"]))
    _write_json(index_path, index)


def featurize_original(path: str | Path) -> dict:
    """Featurize an original image, caching non-LPIPS features by path + sha."""
    image_path = Path(path)
    data = image_path.read_bytes()
    sha = _sha256_bytes(data)
    key = str(image_path)
    entries = _load_original_cache()
    cached = entries.get(key)

    image = Image.open(io.BytesIO(data)).convert("RGB")
    canonical = _canonical(image)
    if cached and cached["hash"] == sha:
        return {
            "gemini": cached["gemini"],
            "dino": cached["dino"],
            "lpips_tensor": _lpips_tensor(canonical),
            "color_hist": cached["color_hist"],
        }

    feat = featurize(canonical)
    entries[key] = {
        "hash": sha,
        "gemini": feat["gemini"],
        "dino": feat["dino"],
        "color_hist": feat["color_hist"],
    }
    _save_original_cache(entries)
    _update_feature_index(image_path, sha, feat)
    return feat


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def similarity(feat_a: dict, feat_b: dict) -> dict:
    """Compute all four per-pair similarities in [0, 1], higher is better."""
    eps = 1e-10
    chi2 = 0.5 * np.sum(
        np.square(feat_a["color_hist"] - feat_b["color_hist"])
        / (feat_a["color_hist"] + feat_b["color_hist"] + eps)
    )
    with torch.no_grad():
        lpips_distance = float(lpips_model(feat_a["lpips_tensor"], feat_b["lpips_tensor"]).item())

    return {
        "s_gemini": _clip01(_cosine(feat_a["gemini"], feat_b["gemini"])),
        "s_dino": _clip01(_cosine(feat_a["dino"], feat_b["dino"])),
        "s_lpips": 1.0 - _clip01(lpips_distance),
        "s_color": 1.0 - _clip01(float(chi2) / 2.0),
    }


def compose(per_image_sims: list[dict]) -> dict:
    """Aggregate per-image similarities into per-metric means and composite."""
    if not per_image_sims:
        raise ValueError("compose() requires at least one per-image score.")
    means = {
        metric: float(np.mean([float(s[metric]) for s in per_image_sims]))
        for metric in METRIC_KEYS
    }
    return {
        "means": means,
        "composite": float(np.mean(list(means.values()))),
    }


def gate(
    candidate_means: dict,
    leader_means: dict | None,
    epsilon: float = 0.01,
) -> tuple[bool, str]:
    """Return whether the candidate avoids per-metric regressions."""
    if leader_means is None:
        return True, "no previous leader"
    regressions = []
    for metric in METRIC_KEYS:
        delta = float(candidate_means[metric]) - float(leader_means[metric])
        if delta < -epsilon:
            regressions.append(f"{metric} regressed by {abs(delta):.4f}")
    if regressions:
        return False, "; ".join(regressions)
    return True, f"no metric regressed by more than {epsilon:.2f}"


def _judge_prompt() -> str:
    return (
        "Compare image A (reference) and image B (regeneration). "
        "Return only JSON with integer scores from 1 to 5 for these keys: "
        "subject, composition, lighting, palette, style, texture. "
        "Use 5 for an excellent match and 1 for a poor match."
    )


def _parse_judge_json(text: str) -> dict[str, int]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"VLM judge did not return JSON: {text!r}")
    raw = json.loads(match.group(0))
    keys = ("subject", "composition", "lighting", "palette", "style", "texture")
    parsed: dict[str, int] = {}
    for key in keys:
        value = int(raw[key])
        parsed[key] = max(1, min(5, value))
    return parsed


def vlm_judge(image_a: Image.Image, image_b: Image.Image) -> dict[str, int]:
    """6-axis VLM judge. Diagnostic only; never used in promotion logic."""
    a = types.Part.from_bytes(data=_png_bytes(_canonical(image_a)), mime_type="image/png")
    b = types.Part.from_bytes(data=_png_bytes(_canonical(image_b)), mime_type="image/png")

    def call() -> object:
        return _client.models.generate_content(
            model=VLM_JUDGE_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=_judge_prompt()),
                        types.Part.from_text(text="Image A:"),
                        a,
                        types.Part.from_text(text="Image B:"),
                        b,
                    ],
                )
            ],
        )

    response = retry_with_backoff(call)
    return _parse_judge_json(str(getattr(response, "text", "")))
