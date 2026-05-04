# IMPLEMENTATION.md — building the autoresearch harness

You are the implementation agent. This is a **one-time bootstrap job**:
build the repo described here so that the autoresearch loop documented
in `program.md` can run.

You are NOT running experiments. The autoresearch driver agent will do
that, in a separate session, after you finish. Don't edit
`prompt_strategy.py` beyond shipping the baseline. Don't run the
optimization loop.

`program.md` is the authoritative spec. Read it first. This doc
describes how to bring the spec into being.

---

## 1. What you're building

A Python repo that lets a coding agent run an autonomous research loop
on prompt-from-image generation. The loop:

1. Calls a VLM to caption a reference image into a generation prompt.
2. Generates an image from the prompt.
3. Scores reproduction fidelity with four complementary similarity
   metrics.
4. Promotes a new prompting strategy only if it strictly improves
   without regressing on any individual metric, and survives a
   configurable multi-seed confirmation re-evaluation with at least
   three generations per target image.

The methodology follows Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) pattern: one
file is the agent's degree of freedom, everything else is fixed
infrastructure. Comparability across experiments depends on the harness
being unchanging.

## 2. Stack

| Role | Model / Method | Notes |
|---|---|---|
| **VLM** | `gemini-3.1-flash-lite-preview` (Vertex AI Gemini, `location=global`) | Called from `prompt_strategy.py`. Driver agent will iterate on this. |
| **Image generator** | `gemini-3.1-flash-image-preview` / Nano Banana 2 (Vertex AI Gemini, `location=global`) | Called from `harness.py`. Pass an aspect ratio that matches the reference image. |
| **Semantic similarity** | `gemini-embedding-2` (Vertex AI Gemini, `location=global`) | Multimodal: embeds images and text into shared vector space. |
| **Structural similarity** | DINOv2 ViT-B/14 (`facebook/dinov2-base`, local via `transformers`, Apache 2.0) | Self-supervised vision features. ~330 MB. |
| **Perceptual similarity** | LPIPS, AlexNet backbone (`lpips` package, local) | ~25 MB. |
| **Color similarity** | HSV histogram, chi-square distance (computed locally) | No model needed. |

Python 3.11+, dependencies managed by `uv`. CPU-only execution must
work; GPU is used automatically if present.

### Gemini access

Use Vertex AI for all Gemini calls, not API-key-only Gemini Developer
API mode. Instantiate the Google GenAI SDK with Vertex AI enabled,
project from `GOOGLE_CLOUD_PROJECT`, and location `global`:

```python
_client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)
```

`.env.example` should include:

```text
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_LOCATION=global
```

### Why these specific local metrics

The Gemini embedding alone is insufficient: it captures semantic
content but not structural fidelity. Two photos that both depict
"golden retriever on grass" embed close together while looking
nothing alike. The driver agent will Goodhart any single metric. The
local metrics fill specific gaps:

- **DINOv2** — self-supervised vision transformer features. Strong on
  layout, pose, and object appearance. Nearly orthogonal to semantic
  embedding. We use DINOv2 (Apache 2.0) rather than DINOv3 (Meta's
  custom commercial license) for licensing simplicity; for our
  pair-similarity use case the practical gap is small, since DINOv3's
  improvements are mostly on dense-prediction tasks.
- **LPIPS** — perceptual distance trained against human similarity
  judgments. Catches texture and fine-detail mismatches.
- **HSV histogram + chi-square** — explicit color-palette match. Cheap,
  catches "right object, wrong colors" failures that perceptual metrics
  often miss.

Together with Gemini Embedding 2, these four signals span roughly
orthogonal axes of "do these images match", which is what makes the
no-regression gate (defined in `program.md`) a meaningful guard
against optimization metric leakage.

### Local-metric compute requirements

Per regenerated image, on CPU:

- DINOv2 ViT-B/14 forward pass: ~400 ms
- LPIPS pair comparison: ~30 ms
- HSV histogram + chi-square: ~5 ms

For a 20-image experiment, all local metrics combined add ~10 seconds
on CPU. The bottleneck is the 20 Nano Banana 2 generations (~3–5 min
wall clock); local scoring is in the noise. Original-image features
are cached on first run; only regenerated-image features are computed
per experiment.

## 3. Repo layout to create

```
prompt_strategy.py          ← baseline you ship; driver iterates on it
harness.py                  ← you implement
embed_and_score.py          ← you implement
eval_data/images/eval/      ← canonical 30-image eval split
eval_data/images/val/       ← canonical 5-image validation split
eval_data/images/train/     ← schema split, may stay empty for this harness
eval_data/images/holdout/   ← schema split, keep private/empty here
eval_data/images/manifest.json
experiments/          ← canonical schema runtime outputs
cache/                ← auto-created at runtime, git-ignored
runs/                 ← compatibility symlink/copy to experiments/runs
weights/              ← auto-created at runtime, git-ignored
logbook.md            ← compatibility symlink/copy to experiments/logbook.md
program.md            ← already exists; do not modify
README.md             ← already exists; do not modify
IMPLEMENTATION.md     ← this file; you can leave it in place
AGENTS.md             ← implementation context now; replace with program.md for driver handoff
pyproject.toml        ← you create
.env.example          ← you create
.env                  ← human creates from .env.example; never commit
.gitignore            ← you create
```

## 4. Build sequence

Do this top to bottom. Don't skip the smoke test in step 9.

1. **Initialize the project.** `uv init`, replace the generated
   `pyproject.toml` with the one in section 5, run `uv sync`.
2. **Create `.gitignore`.** Must exclude: `.env`, `weights/`,
   `cache/`, `runs/`, `experiments/runs/`, `experiments/cache/`,
   `eval_data/images/*/*.png`, `__pycache__/`, `*.pyc`, `.venv/`.
3. **Create `.env.example`** with `GOOGLE_GENAI_USE_VERTEXAI=true`,
   `GOOGLE_CLOUD_PROJECT=`, and `GOOGLE_CLOUD_LOCATION=global`.
4. **Implement `embed_and_score.py`** per section 6.
5. **Implement `harness.py`** per section 7.
6. **Ship the baseline `prompt_strategy.py`** per section 8.
7. **Set up image directories.** Create
   `eval_data/images/{train,eval,val,holdout}` and
   `eval_data/images/manifest.json`.
   If empty, see section 9 — likely requires pausing to ask the human.
8. **Prepare driver agent context files.** The prepared task may start
   with implementation-phase `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md`
   wrappers so auto-loaded context does not fight this bootstrap phase.
   At handoff, replace those files with symlinks or copies of
   `program.md` so driver agents receive the research-loop instructions.
9. **Run the smoke test** in section 10.
10. **Initial git commit:** `init: bootstrap autoresearch repo`.
11. **Hand off** with the message in section 11.

### Eval storage schema overlay

`EVAL_STORAGE_SCHEMA.md` is canonical for persisted eval inputs, run
artifacts, leader pointers, and logbook storage. This spec's older `runs/` and
root `logbook.md` references are compatibility paths for driver ergonomics.
Implement the canonical layout under `experiments/` and either:

- symlink `runs -> experiments/runs` and `logbook.md -> experiments/logbook.md`,
  or
- write both paths from one shared persistence layer without allowing them to
  diverge.

Likewise, `eval_data/images/eval` and `eval_data/images/val` are the only real
places for benchmark images.
`python check_eval_storage.py --root .` must pass after eval artifacts exist.

## 5. `pyproject.toml`

```toml
[project]
name = "image-prompt-autoresearch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "google-genai",        # Gemini API (VLM, generator, embeddings)
  "torch",               # for DINOv2 + LPIPS
  "transformers",        # DINOv2 weights from HuggingFace
  "lpips",               # perceptual similarity
  "pillow",              # image I/O
  "numpy",
  "scipy",               # chi-square distance, robust stats
  "python-dotenv",       # loads .env
]
```

Note: torch wheels are large. Don't pin a specific torch version
unless required — let `uv` resolve a compatible one.

## 6. `embed_and_score.py` — implementation contract

### Module-level setup

- Load `dotenv`, read `GOOGLE_CLOUD_PROJECT`, and default
  `GOOGLE_CLOUD_LOCATION` to `global`. Fail fast with a clear error if
  the project is missing.
- Instantiate the Gemini client once in Vertex AI mode:
  `genai.Client(vertexai=True, project=..., location="global")`.
- Load DINOv2 ViT-B/14:
  ```python
  from transformers import AutoModel, AutoImageProcessor
  dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
  dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
  ```
  Set `cache_dir="weights/"`. Move to GPU if available.
- Load LPIPS: `lpips_model = lpips.LPIPS(net='alex')`. Store its
  weights under `weights/`. Move to GPU if available.
- Detect device: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
  All local computation must work CPU-only.

### Canonical preprocessing

All images are resized to **448×448 with PIL `LANCZOS`** before any
feature extraction. This is the canonical resolution; do not change
it. Reason: 448 is a multiple of DINOv2's patch size 14 (448 / 14 = 32)
and provides
enough detail for LPIPS without inflating cost.

### Public functions

```python
def featurize(image: PIL.Image.Image) -> dict:
    """Return all four feature representations for one image.

    Returns:
        {
            "gemini": np.ndarray of shape (3072,),       # L2-normalized
            "dino":   np.ndarray of shape (768,),         # L2-normalized
            "lpips_tensor": torch.Tensor of shape (1, 3, 448, 448),  # for LPIPS pair op
            "color_hist":   np.ndarray of shape (512,),   # 8x8x8 flattened, normalized to sum=1
        }
    """
```

```python
def similarity(feat_a: dict, feat_b: dict) -> dict:
    """Compute all four per-pair similarities. All in [0, 1], higher = better.

    Formulas:
        s_gemini = cosine(feat_a["gemini"], feat_b["gemini"])
                   clamped to [0, 1] (the underlying space is positive
                   in practice; clamp defensively).
        s_dino   = cosine(feat_a["dino"], feat_b["dino"])
                   clamped to [0, 1].
        s_lpips  = 1 - clip(lpips_model(feat_a["lpips_tensor"],
                                        feat_b["lpips_tensor"]).item(), 0, 1)
        s_color  = 1 - clip(chi2_distance(feat_a["color_hist"],
                                          feat_b["color_hist"]) / 2.0, 0, 1)
                   where chi2_distance(p, q) = 0.5 * sum((p-q)^2 / (p+q+eps))
    """
```

```python
def compose(per_image_sims: list[dict]) -> dict:
    """Aggregate per-image similarities into a composite score.

    Returns:
        {
            "means": {
                "gemini": float, "dino": float,
                "lpips": float,  "color": float,
            },
            "composite": float,  # mean of the four means
        }
    """
```

```python
def gate(candidate_means: dict, leader_means: dict | None,
         epsilon: float = 0.01) -> tuple[bool, str]:
    """Anti-Goodhart promotion gate.

    A candidate passes only if no individual metric regresses by more
    than epsilon. If leader_means is None (no leader yet), passes
    automatically.

    Returns (passed, reason). reason is human-readable.
    """
```

```python
def vlm_judge(image_a: PIL.Image.Image,
              image_b: PIL.Image.Image) -> dict[str, int]:
    """6-axis VLM judge using gemini-3.1-flash-lite-preview.

    Returns scores 1-5 on each axis:
        {"subject": int, "composition": int, "lighting": int,
         "palette": int, "style": int, "texture": int}

    DIAGNOSTIC ONLY. Logged but never used in promotion logic.
    """
```

### Caching

Original-image features are cached in `cache/originals.npz`. Cache key
is the file path; cache value includes the sha256 of the file bytes
for invalidation. On every call to `featurize_original(path)` (a thin
wrapper if you find it convenient), check the hash and recompute if
mismatched.

Cache layout:
```python
np.savez(
    "cache/originals.npz",
    paths=np.array([...]),
    hashes=np.array([...]),
    gemini=np.stack([...]),   # (N, 3072)
    dino=np.stack([...]),     # (N, 768)
    color_hist=np.stack([...]),# (N, 512)
)
# LPIPS tensors are recomputed from disk; they're cheap.
```

### Error handling

All Gemini API calls go through one retry-with-backoff helper
(exponential backoff, max 5 retries, jitter, surface the original
exception on final failure). Place this helper in
`embed_and_score.py` and reuse it from `harness.py`.

## 7. `harness.py` — implementation contract

### CLI

```
uv run harness.py --name <name> [--val] [--seeds N] [--no-judge]
```

- `--name <name>`: short identifier for the run. Required unless
  `--val`.
- `--val`: run on `eval_data/images/val` instead of
  `eval_data/images/eval`. No promotion logic. Prints scores only.
- `--seeds N`: run generation and scoring N times per target image
  with different generation seeds, aggregate per-image scores across
  those repeats, and report mean ± std. Default 3. Validate `N >= 3`
  for eval and val runs; the eval logic must never judge a target from
  a single generated image.
- `--no-judge`: skip the VLM-judge call (saves ~$0.05 per
  experiment). Default: judge runs.

### Behavior

For each image in the target set:

1. Read original (PIL.Image).
2. Call `prompt_strategy.image_to_prompt(image)`. Time it.
3. For each configured seed, call Nano Banana 2 with the prompt:
   - Model: `gemini-3.1-flash-image-preview`
   - Use Vertex AI with `location=global`
   - Derive an aspect ratio from the reference image dimensions and
     pass it through the model's supported `aspect_ratio` parameter
     (for example `1:1`, `4:3`, `3:4`, `16:9`, or `9:16` as
     appropriate). Do not force every generation to square unless the
     reference image is square.
4. Save each regenerated image to a deterministic per-seed path, for
   example `runs/<name>/<image_id>/seed_<seed>.png`, and save the
   prompt once for that image.
5. For each generated image, call `featurize`. Read cached features
   for the original.
6. For each generated image, call `similarity(orig_feat, regen_feat)`.
7. Call `vlm_judge(orig, regen)` for each generated image unless
   `--no-judge`.
8. Aggregate the N generated/scored repeats into one per-image score
   using the mean per metric, and retain the per-seed scores for audit.

After all images:

- Call `compose`.
- Print a per-metric breakdown table:
  ```
  image_id    s_gemini  s_dino   s_lpips  s_color
  001         0.812     0.701    0.688    0.767
  002         ...
  ...
  ────────────────────────────────────────────────
  mean        0.789     0.681    0.660    0.741
  composite   0.7178
  ```

### Promotion logic (default mode only)

- Read previous leader from `runs/leader/composite.json` (if
  exists).
- Call `gate(candidate_means, leader_means)`.
- If gate passes AND `composite_candidate > composite_leader`:
  - Print "Candidate passes gate. Running multi-seed confirmation
    re-eval..."
  - Run a confirmation eval with the configured seed count
    (`--seeds`, default 3, minimum 3) and different generation seeds.
  - Compute the multi-seed mean per metric.
  - Re-run `gate` against the previous leader using the multi-seed
    means.
  - If gate still passes AND the multi-seed mean composite still beats
    the leader composite:
    - Copy `runs/<name>/` to `runs/leader/`.
    - Write `runs/leader/composite.json`:
      ```json
      {
        "name": "<name>",
        "means": {...},
        "composite": 0.7421,
        "confirmation_seed_count": 3,
        "confirmation_seed_mean": 0.7398,
        "confirmation_seed_std": 0.0042,
        "timestamp": "..."
      }
      ```
    - Print "PROMOTED. New leader."
  - Else: print "REVERTED. multi-seed re-eval did not hold."

### Logbook entry

After every run, append to `logbook.md` in the format defined in
`program.md` (search for "Logbook entry format"). Always append,
never overwrite.

The driver agent fills in `hypothesis`, `takeaway`, and `driver`
fields in its own session. The harness should leave them as `<TODO>`
placeholders the driver can fill in via str_replace, OR write them
based on environment variables `AUTORESEARCH_HYPOTHESIS` and
`AUTORESEARCH_DRIVER` if set. Either approach works; pick one and
document it in the file's docstring.

### Output discipline

The driver agent reads stdout to decide what to do next. Make output
clear and parseable:

- Print the per-metric table above.
- Print one line: `composite=<value>`
- Print one line: `gate=<pass|fail>` with reason on next line.
- Print one line: `promoted=<yes|no|reverted_after_re-eval>`

These are not machine-required formats, but they should be greppable.

## 8. Baseline `prompt_strategy.py` — ship verbatim

```python
"""
Baseline prompt strategy. The autoresearch driver agent will edit this file
to discover better strategies. The function signature must remain stable.
"""
from google import genai
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)

def image_to_prompt(image: Image.Image) -> str:
    """Given a reference image, return a prompt for Nano Banana 2."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    response = _client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=[
            {"role": "user", "parts": [
                {"text": "Describe this image so it can be regenerated."},
                {"inline_data": {"mime_type": "image/png", "data": buf.getvalue()}},
            ]},
        ],
    )
    return response.text.strip()
```

This is deliberately mediocre. Expect `composite` ≈ 0.55–0.70 on a
reasonable eval set. Beating it within the first few experiments
should be straightforward for the driver.

## 9. Eval and validation images

The eval set is **30 images**, the val set is **5 images**. Once
locked, neither is modified — every experiment is comparable only
because the benchmark is fixed.

### What the human will provide

The eval set should span a range of visual archetypes the driver
agent might be asked to reproduce. For images-used-in-slides as the
target domain, a representative split is roughly:

- 4× hero photography (landscape, urban, abstract, atmospheric)
- 3× conceptual flat-vector illustrations
- 3× isometric or hand-drawn illustrations
- 2× product photography on clean background
- 2× people / portraits (one solo, one group)
- 2× photographic metaphors (cliché stock-photo style)
- 2× background textures or patterns
- 1× UI mockup or data viz (text-rendering stress test)
- 1× icon or pictogram

Difficulty gradient: ~6 easy, ~10 medium, ~4 hard. The hard ones
should be deliberately challenging (distinctive faces, dense
composition, distinctive style) so different strategies separate
visibly.

The val set (5 images) should cover the same archetypes with
different specific content. If eval has a flat-vector "growth chart"
illustration, val should have a flat-vector "team meeting"
illustration — same archetype, different content.

### If `eval_data/images/eval/` is empty when you reach this step

**Stop and ask the human.** Do not scrape, generate, or auto-fill
substitutes. The eval set is the most consequential decision in the
project; locking in random images would silently invalidate every
experiment afterward.

A reasonable hand-off message at that point:

> The harness is built and the smoke test for `embed_and_score.py`
> passes. I need you to populate `eval_data/images/eval/` with 20
> reference images and `eval_data/images/val/` with 5 held-out images
> before I can run the end-to-end smoke test. See IMPLEMENTATION.md
> section 9 for the recommended distribution. Once images are in place,
> I'll re-run the smoke test and finish handoff.

### Validation checks (run after the human populates the directories)

- All files in `eval_data/images/eval/` and `eval_data/images/val/` load with PIL.
- All images are at least 512×512.
- No transparent backgrounds (will confuse generation).
- No duplicate sha256s between `eval_data/images/eval/` and `eval_data/images/val/`.
- File count is exactly 30 and 5.
- `eval_data/images/manifest.json` is present and includes every eval and val image.

If any check fails, stop and report.

## 10. Smoke test — required before declaring done

### Test 1: feature extractors load and run

```bash
uv run python -c "from embed_and_score import featurize; from PIL import Image; print(list(featurize(Image.new('RGB', (448, 448), 'red')).keys()))"
```

Expected output: `['gemini', 'dino', 'lpips_tensor', 'color_hist']`
(in some order). This verifies that all four feature extractors
import, weights download, and a forward pass completes
end-to-end.

If this fails, fix before proceeding. Common failure modes:

- `GOOGLE_CLOUD_PROJECT` missing or Vertex AI credentials unavailable →
  check `.env`, `dotenv` load, and local Google Cloud auth.
- HuggingFace download fails → check network, retry.
- Torch CPU fallback path broken → make sure `device` detection
  doesn't crash without CUDA.

### Test 2: full harness on at least one image

If the human has populated `eval_data/images/eval/` with at least one image:

```bash
uv run harness.py --name smoke_test --no-judge
```

(Use `--no-judge` to keep the smoke test cheap.)

Expected: completes without error, prints the per-metric table,
writes `runs/smoke_test/`, and promotes itself as the first leader
(since there's no previous leader). If this fails, fix before
handing off.

If `eval_data/images/eval/` is empty, skip Test 2 and note in the handoff
that it could not be run.

## 11. Definition of done

Hand off only when ALL of these are true:

- [ ] `uv sync` succeeds from a clean clone.
- [ ] Smoke test 1 passes (`featurize` returns the four expected
      keys).
- [ ] Smoke test 2 passes if `eval_data/images/eval/` is populated
      (full harness run completes end-to-end).
- [ ] Eval and val runs validate `--seeds >= 3` and generate, score,
      aggregate, and persist at least three per-seed results for each
      target image.
- [ ] `program.md` is symlinked or copied to `AGENTS.md`, `CLAUDE.md`,
      and `GEMINI.md` for the driver-agent handoff.
- [ ] `README.md` is in place (it should already exist; do not
      overwrite).
- [ ] `.env.example` exists; `.env` is in `.gitignore`.
- [ ] `weights/`, `cache/`, `runs/` are in `.gitignore`.
- [ ] `logbook.md` exists with a `# Logbook` header and nothing
      else.
- [ ] Initial git commit is made:
      `init: bootstrap autoresearch repo`.

## 12. Handoff message

Write a short message to the human covering:

1. What was built (one sentence).
2. Whether smoke test 2 was run; if not, what's blocking it.
3. What the human should do next:
   - If `eval_data/images/eval/` is empty: populate it per section 9, then
     run `uv run harness.py --name baseline_check` to verify the
     baseline.
   - If `eval_data/images/eval/` was populated and smoke test 2 passed: the
     harness is ready; launch their preferred coding agent with
     the kickoff prompt from `program.md` "Session entrypoint".
4. Any deviations you made from this spec, with brief justification.

Do not include long postambles or summaries. Two or three sentences
plus the next-action list is plenty.

---

## Notes for you (the implementation agent)

- **Don't run experiments.** That's the next agent's job. Your job
  ends at handoff. If you find yourself thinking about prompting
  strategy quality, you're out of scope.
- **Don't change `program.md` or `README.md`.** They are inputs to
  this build, not outputs. If you spot an inconsistency, surface it
  in the handoff message rather than editing.
- **Don't over-engineer.** The harness is small (target: under ~500
  lines of Python total across both files). Resist temptations to
  add config systems, plugin architectures, or abstractions for
  alternative metric backends. The whole point is that the harness
  is fixed.
- **Do print clearly.** The driver agent reads stdout. A scrambled
  output format is a real problem, not a cosmetic one.
- **Do test the CPU fallback.** Many implementation agents
  unconsciously assume CUDA. The driver may run on a laptop without
  one.
