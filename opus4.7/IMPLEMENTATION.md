# Implementation Bootstrap

This is a **one-time bootstrap job**. You are setting up the autoresearch infrastructure. You are NOT running experiments — that is a separate driver agent's job, governed by `program.md`. When you finish, hand off and stop.

`program.md` is the authoritative spec for shared definitions: model choices, the four metrics, the promotion gate (`epsilon = 0.01`), the 3-seed re-eval, the held-out validation rule, and the VLM-judge "diagnostic only" stance. This document only adds what is specific to *building* the harness. Where `program.md` already defines something, this doc references it instead of restating.

## What you're building

A fixed, frozen benchmark harness for an autoresearch loop. The driver agent will edit one file (`prompt_strategy.py`), run `uv run harness.py --name <…>`, and read scores. Everything else — image loading, generation, featurization, the four similarity metrics, the gate, 3-seed re-eval, logbook append, VLM-judge — lives in code you write now and nobody touches afterward.

### Driver agent vs. inner VLM (carried over from `program.md`)

The **driver** is a coding agent (Claude Code / Codex / Gemini CLI / etc.) that edits `prompt_strategy.py` and runs experiments. The **inner VLM** is `gemini-3.1-flash-lite-preview`, called from inside `prompt_strategy.image_to_prompt(...)`. They are different — your harness must be agnostic to which driver agent runs it.

## Stack

| Role | Model / Library |
|---|---|
| Inner VLM | `gemini-3.1-flash-lite-preview` (called from baseline `prompt_strategy.py`) |
| Image generator | `gemini-3.1-flash-image-preview` (Nano Banana 2), 1024×1024, defaults |
| Semantic embedding | `gemini-embedding-2-preview`, 3072-d, multimodal |
| Structural | `facebook/dinov3-vitb16-pretrain-lvd1689m` via HuggingFace `transformers` |
| Perceptual | `lpips` package (AlexNet backbone) |
| Color | HSV histogram 8×8×8 + chi-square (numpy/scipy) |

CPU-only execution must work. No CUDA assumption.

## Why these specific local metrics

Gemini multimodal embeddings capture semantic content, not visual fidelity. Two photos of "golden retriever on grass" can embed close together while looking nothing alike. A driver will Goodhart any single metric. Three additional locally-computed metrics span axes roughly orthogonal to the semantic one:

- **DINOv3 ViT-B/16 (structural).** Self-supervised features capturing pose, layout, appearance. Catches "right concept, wrong arrangement" failures the embedding misses.
- **LPIPS / AlexNet (perceptual).** Texture and fine-detail similarity, trained against human similarity judgments. Catches "looks the same to a captioner, looks wrong to a person."
- **HSV 8×8×8 + chi² (color).** Explicit palette match. Catches "right object, wrong colors" — a common Nano Banana 2 failure mode that DINOv3, LPIPS, and Gemini all soft-pedal.

Together with `s_gemini`, the four cover semantics, structure, perception, and color. The promotion gate enforces no-regression on each axis with `epsilon = 0.01`, so the driver cannot trade one for another.

## Local-metric compute requirements

Per 20-image experiment, on CPU: combined local metrics add ~10s. Bottleneck is API generation (~3–5 min for the 20 Nano Banana 2 calls). Original-image features are computed once and cached on disk; each subsequent run only featurizes the 20 regenerated images. DINOv3 weights download once on first run (~330 MB) into `weights/`.

## Repo layout to create

```
.
├── program.md              # already provided; you do not write this
├── IMPLEMENTATION.md       # this file
├── prompt_strategy.py      # baseline you ship; driver edits later
├── harness.py              # the immutable benchmark
├── embed_and_score.py      # featurize / similarity / compose / gate / vlm_judge
├── eval_images/            # human populates; you create the dir empty
├── val_images/             # human populates; you create the dir empty
├── runs/                   # auto-created at runtime
├── cache/                  # auto-created; original-image features
├── weights/                # auto-created; DINOv3 weights
├── logbook.md              # empty starting file with header `# Logbook`
├── pyproject.toml
├── .env.example
├── .gitignore
└── AGENTS.md -> program.md # symlink
```

`README.md` is a separate artifact and is **not** your responsibility. Do not create one.

## Build sequence

1. `uv init --python 3.11`
2. Author `pyproject.toml` (see below).
3. `uv sync`
4. Write `.gitignore` covering `.env`, `runs/`, `cache/`, `weights/`, `__pycache__/`, `*.pyc`, `.venv/`.
5. Write `.env.example` listing required vars: `GEMINI_API_KEY` (required), `HF_HOME` (optional).
6. Implement `embed_and_score.py` per contract below.
7. Implement `harness.py` per contract below.
8. Ship baseline `prompt_strategy.py` (verbatim code below).
9. Create empty `eval_images/` and `val_images/` directories.
10. `ln -s program.md AGENTS.md`.
11. Smoke test 1: `featurize` on a synthetic 448×448 red image.
12. Smoke test 2 (only if eval images already populated): full harness with `--no-judge`.
13. Initial git commit: `git init && git add -A && git commit -m "bootstrap"`.
14. Hand off (template at bottom).

## `pyproject.toml`

Python 3.11+. Dependencies (pin loosely with `>=`, not `==`):

- `google-genai`
- `torch`
- `transformers`
- `lpips`
- `pillow`
- `numpy`
- `scipy`
- `python-dotenv`

No optional script entry needed; `uv run harness.py` is sufficient.

## `embed_and_score.py` implementation contract

### Module-level setup (runs once on import)

- `dotenv.load_dotenv()`
- Gemini client from `google.genai` using `os.environ["GEMINI_API_KEY"]`.
- DINOv3 model + image processor from `transformers`, `cache_dir="weights/"`, `.eval()`.
- LPIPS model `lpips.LPIPS(net='alex')`, `.eval()`.
- Device detection: prefer `cuda` if available, else `mps`, else `cpu`. Move DINOv3 and LPIPS to device. **The CPU path must work end-to-end with no further branching.**

### Canonical preprocessing

A single helper applied everywhere:

```python
def _canon(image: PIL.Image.Image) -> PIL.Image.Image:
    return image.convert("RGB").resize((448, 448), PIL.Image.LANCZOS)
```

Used by `featurize` for DINOv3, LPIPS, and the HSV histogram. The Gemini embedding call is also fed the canonical 448×448 RGB image so all four metrics share the same input crop.

### Public functions

```python
def featurize(image: PIL.Image.Image) -> dict:
    """Returns:
        'gemini':       np.ndarray, shape (3072,), L2-normalized
        'dino':         np.ndarray, shape (768,),  L2-normalized (CLS token)
        'lpips_tensor': torch.Tensor, shape (1, 3, 448, 448), normalized to [-1, 1]
        'color_hist':   np.ndarray, shape (512,),  L1-normalized HSV 8x8x8
    """

def similarity(feat_a: dict, feat_b: dict) -> dict:
    """Returns {'s_gemini': float, 's_dino': float, 's_lpips': float, 's_color': float}.
    Per program.md formulas."""

def compose(per_image_sims: list[dict]) -> dict:
    """Returns {
        'means': {'s_gemini': float, 's_dino': float, 's_lpips': float, 's_color': float},
        'composite': float,
    }"""

def gate(candidate_means: dict, leader_means: dict, epsilon: float = 0.01) -> tuple[bool, str]:
    """Returns (passed, reason). Reason is human-readable: which metric regressed,
    or 'composite did not improve', or 'pass'."""

def vlm_judge(image_a: PIL.Image.Image, image_b: PIL.Image.Image) -> dict:
    """Returns {'subject':1-5, 'composition':1-5, 'lighting':1-5,
              'palette':1-5, 'style':1-5, 'texture':1-5}.
    Diagnostic only — never call from inside gate/compose."""
```

### Cache layout

`cache/originals.npz` stores the four featurization outputs per original image, keyed by `f"{relative_path}::{sha256_first_8}"`. On `featurize_original(path)`: hash the file bytes; if the key exists, load; else compute and write back. Hash-based invalidation means swapping an image at the same path correctly busts the cache.

LPIPS tensors are stored as numpy float16 arrays inside the npz and re-wrapped into torch tensors on load.

### Retry helper

All Gemini API calls (embedding, generation, judge) go through a single `_with_retry(fn, *args, **kwargs)` helper: exponential backoff (base 2s), max 5 retries, ±25% jitter. Logs each retry to stderr. Raises after the final attempt.

## `harness.py` implementation contract

### CLI

```
uv run harness.py --name <str>             # eval, 1 seed, promotion path
uv run harness.py --name <str> --seeds 3   # eval, 3 seeds (re-eval mode)
uv run harness.py --val                    # val set, no promotion, no judge
uv run harness.py --name <str> --no-judge  # skip VLM-judge
```

`argparse`, no extra CLI lib.

### Per-image flow

For each image in `eval_images/` (or `val_images/` if `--val`):

1. Load original via PIL.
2. Call `prompt_strategy.image_to_prompt(original)` → prompt string.
3. For each seed in `[1..N]`:
   - Generate 1024×1024 with Nano Banana 2, deterministic seed (1, 2, 3, ...).
   - Save regen to `runs/<name>/<image_stem>__seed<k>.png`.
   - Save the prompt to `runs/<name>/<image_stem>__seed<k>.prompt.txt`.
   - Featurize regen, call `similarity(orig_feat, regen_feat)`.
   - If not `--no-judge` and not `--val`: `vlm_judge(orig, regen)`, append to `runs/<name>/judge.jsonl`.
4. Aggregate across seeds → per-image similarity dict (mean of seeds).

### Output

Print a per-metric breakdown table: per-image rows + summary row, all four `s_*` columns plus a per-image composite. Then exactly these greppable lines:

```
composite=<value>
gate=<pass|fail: reason>
promoted=<yes|no>
```

Drivers grep on these prefixes — keep them exact.

### Promotion logic

If `--val`: skip everything below; never promote.

Otherwise:

1. Read `runs/leader/composite.json` if it exists. If not, this is the first run → auto-promote regardless.
2. Compute candidate means and composite from the 1-seed run.
3. Call `gate(candidate_means, leader_means)`. If fail, print `gate=fail: <reason>`, `promoted=no`, exit 0.
4. If pass and `--seeds == 1`, automatically re-run the eval with seeds 1–3 in-process (do not re-invoke as a subprocess). Recompute means as mean across all 3 seeds × 20 images. Re-call `gate`. If still pass, proceed; if fail, print rejection reason, exit 0.
5. Copy `runs/<name>/` artifacts to `runs/leader/`, write `runs/leader/composite.json` containing `{means, composite, name, timestamp}`.
6. Print `promoted=yes`.

### Logbook append

Append one block to `logbook.md` per harness invocation (every run, promoted or not). If env vars `AUTORESEARCH_HYPOTHESIS` and `AUTORESEARCH_DRIVER` are set, fill them in; otherwise emit `<TODO: hypothesis>` and `<TODO: driver>` placeholders. Use the format defined in `program.md`. Never overwrite or rewrite older entries.

## Baseline `prompt_strategy.py` — ship verbatim

```python
"""Baseline prompt strategy. Deliberately mediocre.
The driver agent rewrites this file; the harness does not."""

import os
from google import genai
import PIL.Image
from dotenv import load_dotenv

load_dotenv()
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-3.1-flash-lite-preview"


def image_to_prompt(image: PIL.Image.Image) -> str:
    """Take a reference image, return a prompt string for Nano Banana 2."""
    response = _client.models.generate_content(
        model=_MODEL,
        contents=[image, "Describe this image so it can be regenerated."],
    )
    return response.text.strip()
```

One VLM call, one trivial instruction. The driver is meant to beat this.

## Eval / val image guidance

Target: 20 eval images + 5 val images. The images are the kind used **in slides** (hero photos, illustrations, icons, etc.), not slide screenshots themselves. Recommended distribution for the 20-image eval set:

| Count | Archetype |
|---|---|
| 4 | hero photography (landscape, urban, abstract, atmospheric) |
| 3 | conceptual flat-vector illustrations |
| 3 | isometric or hand-drawn illustrations |
| 2 | product photography on clean background |
| 2 | people / portraits (one solo, one group — identity stress test) |
| 2 | photographic metaphors (cliché stock-photo style) |
| 2 | background textures or patterns |
| 1 | UI mockup or data viz (text-rendering stress test) |
| 1 | icon or pictogram |

Difficulty distribution: ~6 easy, ~10 medium, ~4 hard.

Val set covers the same archetypes with **different content** (eval has flat-vector "growth chart" → val has flat-vector "team meeting"). Once the human locks both sets, neither is modified for the duration of the project.

### If `eval_images/` is empty after bootstrap

**Stop. Ask the human. Do not scrape, generate, or auto-fill substitutes.** A wrong eval set silently invalidates every future result.

Sample handoff message for that case:

> Bootstrap is complete and smoke test 1 passed. `eval_images/` and `val_images/` are empty. Per the spec, I will not auto-fill them — silently substituting images would invalidate the benchmark. Please drop 20 reference images into `eval_images/` and 5 into `val_images/` following the archetype distribution in IMPLEMENTATION.md. Once populated, run `uv run harness.py --name baseline_smoke --no-judge` to confirm end-to-end. I will not run experiments.

### Validation checks (run once images are populated)

- `PIL.Image.open(path).load()` succeeds for every file.
- Shortest side ≥ 512px.
- No transparency (`mode in {"RGB", "L"}` after open; reject `"RGBA"` and `"P"` with alpha).
- No sha256 collisions between `eval_images/` and `val_images/`.
- Exact counts: 20 in eval, 5 in val.

## Two-stage smoke test

### Test 1 — always run

Create a synthetic 448×448 solid red `PIL.Image`. Call `featurize(img)`. Assert the four expected keys are present and shapes match the contract. Catches: missing API key, missing weights, broken imports, device misconfiguration.

### Test 2 — only if `eval_images/` is populated

Run `uv run harness.py --name baseline_smoke --no-judge`. Expected outcome:

- Exits 0.
- Prints all three greppable lines.
- Self-promotes (no prior leader → auto-promote).
- Writes `runs/leader/composite.json`.

If Test 2 cannot be run (empty image dir), document this in the handoff message and stop.

## Definition of done

1. `uv sync` succeeds on a clean checkout.
2. `embed_and_score.py` exposes the five public functions with the documented signatures.
3. `harness.py` runs end-to-end with a populated eval set, or stops cleanly with a clear message if the dir is empty.
4. Smoke test 1 passes.
5. Smoke test 2 passes, or is correctly skipped with a documented reason.
6. `program.md` is symlinked to `AGENTS.md`.
7. `logbook.md` exists with a single header line `# Logbook` and nothing else.
8. `.env.example` is present; real `.env` is gitignored.
9. Initial git commit made.

## Handoff message format

When done, hand off with a message containing:

- **What was built:** one short paragraph.
- **Smoke test status:** Test 1 result, Test 2 result (or reason skipped).
- **Next action for the human:** populate image dirs (if needed), set `GEMINI_API_KEY` in `.env`, then start a driver session pointing the chosen agent at `program.md`.
- **Deviations from this spec:** any place you deliberately diverged, with one-line justification each. If none, write "none".

## Notes for the implementation agent

- **Do not run experiments.** Beating the baseline is the driver's job, not yours.
- **Do not modify `program.md`.** It is the driver's contract.
- **Do not over-engineer.** Target ~500 lines total across `embed_and_score.py` and `harness.py`. If you're past 700, you're adding things that are not in the contract.
- **Do not assume CUDA.** Test the CPU path before declaring done.
- **Do not invent additional metrics, gates, or modes.** The set is fixed: four similarities, the gate with `epsilon = 0.01`, 3-seed re-eval, held-out val, VLM-judge diagnostic only.
- **Cross-reference, don't duplicate.** Where `program.md` defines something (gate semantics, metric formulas, logbook format), reference it from comments and docstrings rather than restating it here or in code.
