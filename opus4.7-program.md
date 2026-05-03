# program.md — Image → Prompt → Image autoresearch

You are an autonomous research agent. Your job is to discover prompting
strategies that, given an input image, produce a text prompt which causes an
image generator to reproduce that image as faithfully as possible.

You will iterate by editing a single file, running a fixed evaluation, reading
the score, and deciding whether to keep or discard your change. Keep going
until told to stop.

---

## The setup

| Role | Model / Method | What it does |
|---|---|---|
| **VLM** (called from `prompt_strategy.py`) | `gemini-3.1-flash-lite-preview` (Gemini API) | Looks at the input image and produces a prompt |
| **Image generator** (called by harness) | `gemini-3.1-flash-image-preview` / Nano Banana 2 (Gemini API) | Generates an image from the prompt |
| **Semantic similarity** (harness) | `gemini-embedding-2-preview` (Gemini API) | Multimodal embedding — semantic content match |
| **Structural similarity** (harness) | DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, local) | Self-supervised vision features — pose, layout, appearance |
| **Perceptual similarity** (harness) | LPIPS with AlexNet backbone (local) | Perceptual texture / fine-detail match |
| **Color similarity** (harness) | HSV histogram with chi-square distance (local) | Color palette match |

The harness combines all four similarity signals into a composite score with
anti-Goodhart guards. See "The metric" below.

## Repo layout

```
prompt_strategy.py    ← the ONLY file you edit
harness.py            ← runs eval loop. DO NOT MODIFY.
embed_and_score.py    ← all four similarity metrics + compositing. DO NOT MODIFY.
eval_images/          ← 20 fixed reference images. DO NOT MODIFY.
val_images/           ← 5 held-out images for sanity check. DO NOT MODIFY.
cache/                ← cached features for original images (auto-created)
runs/                 ← per-experiment artifacts (auto-created)
weights/              ← downloaded local model weights (auto-created)
logbook.md            ← append every experiment here
.env                  ← GEMINI_API_KEY=...
```

## The single file you edit

`prompt_strategy.py` must expose exactly one function:

```python
def image_to_prompt(image: PIL.Image.Image) -> str:
    """Given a reference image, return a prompt for Nano Banana 2."""
```

Everything inside this file is fair game: system prompt, user prompt, number
of VLM calls, iterative refinement against draft generations, decomposition
into subject/style/composition fields, few-shot examples, thinking levels,
etc. The function may make multiple API calls internally.

## Running an experiment

```bash
uv run harness.py --name <short_descriptive_name>
```

For each of the 20 images in `eval_images/`, the harness:

1. Calls `image_to_prompt(image)` to get a prompt.
2. Calls Nano Banana 2 once with that prompt (1024×1024, default config).
3. Resizes both original and regenerated to canonical 448×448.
4. Computes four similarity signals: Gemini-emb, DINOv3, LPIPS, HSV histogram.
5. Combines them into the composite score (defined below).

Original-image features for all four metrics are computed once and cached in
`cache/originals.npz`. Per-experiment cost is dominated by Nano Banana 2
generation (~3–5 min for 20 images); all four similarity metrics combined add
under 30 seconds.

---

## The metric

### Step 1: per-pair similarities (each in [0, 1], higher = better)

| Signal | Computation | Notes |
|---|---|---|
| `s_gemini` | cosine of `gemini-embedding-2-preview` vectors | 3072-d output. Pre-normalize the original. |
| `s_dino` | cosine of DINOv3 ViT-B/16 CLS-token features | 448×448 input (multiple of patch size 16). Use the model's native preprocessing. |
| `s_lpips` | `1 - clip(lpips_distance, 0, 1)` | LPIPS from the `lpips` package, AlexNet backbone, default normalization. |
| `s_color` | `1 - clip(chi_square_distance / NORM, 0, 1)` | 3D HSV histogram, 8×8×8 bins, normalized. NORM = 2.0 (calibrate on first run if needed). |

All four are computed at canonical 448×448 to make framing/resolution
differences not pollute the signal.

### Step 2: composite score per experiment

For each of the 20 eval images, compute all four similarities. Then:

```
mean_signal[m]  =  mean over 20 eval images of  s_m   for m in {gemini, dino, lpips, color}
composite       =  mean(mean_signal["gemini"], mean_signal["dino"],
                        mean_signal["lpips"],  mean_signal["color"])
```

`composite` is the number you primarily try to improve. But it is not the
only thing the gate checks.

### Step 3: anti-Goodhart promotion gate

A new candidate is **promoted** (becomes the new leader) only if BOTH:

1. **No-regression rule:** for each individual metric m,
   `mean_signal_candidate[m] >= mean_signal_leader[m] - epsilon`,
   with `epsilon = 0.01`.
2. **Improvement rule:** `composite_candidate > composite_leader`.

The no-regression rule defeats the most common failure mode: a strategy that
boosts semantic similarity (`s_gemini`) by abandoning structural fidelity
(`s_dino` collapses), or vice versa. If any single dimension drops by more
than ε, the candidate is rejected even if `composite` improved.

### Step 4: re-eval against lucky seeds

Promoted candidates re-run the full eval **3 more times** with different
generation seeds. The 3-seed mean composite must still beat the previous
leader's composite under the same gate. Otherwise, revert.

### Held-out validation

`uv run harness.py --val` runs the full pipeline (all 4 metrics, no
promotion logic) on `val_images/`. Run this every ~10 promoted leaders. If
`composite_eval` is climbing but `composite_val` is flat or dropping, the
strategy is overfitting to the eval set — back off and try something
qualitatively different.

### Diagnostic, not optimized: VLM judge

A VLM-judge call (using `gemini-3.1-flash-lite-preview`) rates the pair on
6 axes — subject identity, composition, lighting, color palette, style,
texture — each 1–5. This is **logged for inspection only**. Do not optimize
against it; treating it as a target would create a feedback loop with the
prompting model.

---

## Implementation reference (for whoever builds the harness)

### Dependencies (`pyproject.toml`)
```toml
[project]
dependencies = [
  "google-genai",        # Gemini API
  "torch",               # for DINOv3 + LPIPS
  "transformers",        # DINOv3 weights from HuggingFace
  "lpips",               # perceptual similarity
  "pillow",
  "numpy",
  "scipy",               # chi-square distance
  "python-dotenv",
]
```

### `embed_and_score.py` responsibilities

- Load DINOv3 ViT-B/16 once at module import via `transformers`
  (`AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")`,
  paired `AutoImageProcessor`). Cache to `weights/`.
- Load LPIPS once (`lpips.LPIPS(net='alex')`). Cache to `weights/`.
- Detect GPU automatically; fall back to CPU silently. Local metrics MUST
  work CPU-only.
- Expose four functions:
  - `featurize(image: PIL.Image) -> dict[str, np.ndarray]` — returns
    `{"gemini": ..., "dino": ..., "lpips_tensor": ..., "color_hist": ...}`
    for one image. Note: LPIPS needs a tensor not a vector; `s_lpips` is
    computed on the pair, not as cosine of features.
  - `similarity(feat_a, feat_b) -> dict[str, float]` — returns
    `{"gemini": s, "dino": s, "lpips": s, "color": s}` per the formulas
    in the table above.
  - `compose(per_image_sims: list[dict]) -> dict` — returns
    `{"means": {gemini, dino, lpips, color}, "composite": float}`.
  - `gate(candidate_means, leader_means, epsilon=0.01) -> tuple[bool, str]`
    — returns whether the no-regression rule passes and a reason string.
- Cache original-image features in `cache/originals.npz` keyed by file path
  + sha256 of the bytes. Invalidate on hash mismatch.
- Resize all images to 448×448 with PIL `LANCZOS` before any feature
  extraction. Canonical resolution.

### `harness.py` responsibilities

- Imports `prompt_strategy.image_to_prompt`.
- For each eval image: calls `image_to_prompt`, calls Nano Banana 2 with a
  fixed `image_config` (1024×1024, no aspect ratio override), saves the
  regenerated image to `runs/<name>/<image_id>.png` and the prompt to
  `runs/<name>/<image_id>.txt`.
- Computes all per-image similarities, calls `compose`, prints a
  per-metric breakdown table.
- Reads previous leader from `runs/leader/composite.json`. If candidate
  passes `gate` AND improves `composite`, runs 3-seed re-eval; if still
  passes, copies all artifacts to `runs/leader/` and updates
  `composite.json`.
- Appends a logbook entry to `logbook.md` in the format below.
- `--val` flag: same pipeline on `val_images/`, no promotion logic, prints
  scores only.
- `--seeds N` flag: run N seeds and report mean ± std.
- All Gemini API calls go through one retry-with-backoff wrapper.

### Cost & runtime expectations

- **Per experiment:** ~3–5 min wall clock dominated by 20 Nano Banana 2
  generations. Local metrics together add <30s on CPU, <2s on GPU. API cost
  roughly $0.10–$0.30 per experiment depending on prompting complexity.
- **First run:** add ~30s for downloading DINOv3 weights (~340 MB) and
  LPIPS weights (~25 MB) once.
- **No GPU required.** Laptop CPU is fine. GPU is used automatically if
  present.

---

## Workflow per experiment

1. **Hypothesis.** Write one sentence in `logbook.md` describing what you
   are trying and why. "Add explicit color palette extraction step before
   main description" is good. "Try better prompt" is not.
2. **Edit** `prompt_strategy.py`.
3. **Run** `uv run harness.py --name <name>`.
4. **Read** the per-metric breakdown, not just `composite`. If `s_dino`
   jumped but `s_gemini` dropped, that's a signal about what your change
   actually did.
5. **Decide.**
   - Gate passes AND composite improves → 3-seed re-eval. If still wins,
     promote (commit `prompt_strategy.py` with
     `leader: <name> = <composite>`).
   - Otherwise → `git checkout prompt_strategy.py` and try something else.
6. **Log** the entry. Append, never overwrite.
7. **Repeat.** Aim for one experiment per ~10 minutes wall clock.

## Budget guardrails

- **Per experiment:** ≤ 20 input images × your VLM calls per image, plus
  20 generations and 80 local feature extractions. If your strategy needs
  more than ~5 VLM calls per image, the gain needs to be substantial to
  justify it.
- **Per session:** stop after 50 experiments and write a session summary.

## Starting baseline

`prompt_strategy.py` ships with a deliberately mediocre baseline: one VLM
call, system prompt = "Describe this image so it can be regenerated.",
user content = the image. Expect `composite` ≈ 0.55–0.70. Beating this
within the first few experiments should be straightforward.

## What's worth exploring (not exhaustive)

- **Output format:** free-form vs. structured (subject / style /
  composition / lighting / palette / camera).
- **Decomposition:** multiple VLM calls each focused on one aspect, merged.
- **Iterative refinement:** generate a draft, embed it, compare to
  original, ask the VLM what's missing, revise the prompt, regenerate.
  More expensive — measure whether it pays.
- **Prompt length:** very short evocative prompts vs. dense literal ones.
  Nano Banana 2 has strong world knowledge; sometimes short wins.
- **Negative prompts** / what to avoid.
- **Few-shot exemplars** in the system prompt.
- **Thinking level** on the VLM (`minimal`, `low`, `medium`, `high`).
- **Style-vs-content tradeoff:** strategies that capture style well
  sometimes lose subject identity. Watch per-metric scores. If `s_dino` is
  high but `s_gemini` is low, you're matching layout/texture but missing
  subject identity. If the reverse, you're describing what's there but not
  how it looks.

## What NOT to do

- Do not modify `harness.py`, `embed_and_score.py`, `eval_images/`, or
  `val_images/`. These define the benchmark.
- Do not change the image generator model, the embedding model, the local
  metric models, or the canonical 448×448 resolution. Comparability across
  experiments depends on these being fixed.
- Do not memorize the eval set. Hardcoding references to specific eval
  images is cheating; held-out val will catch it.
- Do not optimize against the VLM-judge scores. They are diagnostic only.
- Do not chase a single lucky run. The 3-seed re-eval gate exists for a
  reason.

## Logbook entry format

```
### <name> — <YYYY-MM-DD HH:MM>
- hypothesis: <one sentence>
- composite: 0.7421
- s_gemini: 0.812 | s_dino: 0.701 | s_lpips: 0.688 | s_color: 0.767
- gate vs leader: pass | fail (<which metric regressed by how much>)
- 3-seed re-eval: 0.7398 ± 0.0042   (only if promoted)
- val composite: 0.7301              (only if --val was run)
- wall_clock: 4.2 min
- est_cost_usd: 0.18
- takeaway: <one or two sentences>
- promoted: yes | no
```

## Stopping

When the user says stop, or after 50 experiments, append a "Session
summary" section to `logbook.md` with:
1. Top 3 strategies by `composite`, with their per-metric breakdown and
   `val composite` for honesty.
2. What worked consistently across all four metrics.
3. What looked promising on `composite` but failed the gate or regressed
   on val.
4. Three concrete next experiments worth running with more time.

---

## Kicking off

To start a session, the human should say something like:

> Hi, have a look at program.md and let's kick off a new session. First do
> the setup check (download local model weights, run the baseline once,
> confirm all four metrics produce sane scores), then start iterating.
