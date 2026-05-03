# Autoresearch: Image Reproduction via Prompt Strategy

You are a driver agent running an autoresearch loop. Your job is to discover a prompting strategy that causes a text-to-image model to faithfully reproduce a reference image. **The harness already exists. You do not modify it.** You edit one file (`prompt_strategy.py`), run the benchmark, read the score, and iterate.

Comparability across experiments depends on the harness being unchanging. Touching it invalidates every prior score.

## Setup

### Models

| Role | Model |
|---|---|
| Inner VLM (writes prompt from image, called inside `prompt_strategy.py`) | `gemini-3.1-flash-lite-preview` |
| Image generator (called by harness) | `gemini-3.1-flash-image-preview` (Nano Banana 2), 1024×1024, defaults |
| Semantic embedding (orig + regen, shared 3072-d space) | `gemini-embedding-2-preview` |

### Driver agent vs. inner VLM

The **driver** is the coding agent reading this document — Claude Code, Codex CLI, Gemini CLI, OpenCode, Aider, or Cursor. It edits `prompt_strategy.py` and runs experiments.

The **inner VLM** is always `gemini-3.1-flash-lite-preview`. It runs inside `prompt_strategy.image_to_prompt(...)` and turns a reference image into a generation prompt.

These are different. The driver does not generate prompts directly; `prompt_strategy.py` does. The driver designs strategies *for* `prompt_strategy.py`.

## Repo layout

```
.
├── program.md              # this file (you read this)
├── prompt_strategy.py      # YOU EDIT THIS
├── harness.py              # do not modify
├── embed_and_score.py      # do not modify
├── eval_images/            # 20 reference images, do not modify
├── val_images/             # 5 held-out images, do not modify
├── runs/                   # per-experiment artifacts (auto-created)
│   ├── <name>/             # one dir per --name
│   └── leader/             # current best, gate-passed
├── cache/                  # original-image features, do not delete
├── weights/                # DINOv3 weights cache
├── logbook.md              # YOUR memory across runs
├── pyproject.toml
└── .env                    # GEMINI_API_KEY
```

You edit only `prompt_strategy.py` and `logbook.md`. You may read everything else. You modify nothing else.

## The single file you edit

`prompt_strategy.py` exposes one public function:

```python
def image_to_prompt(image: PIL.Image.Image) -> str:
    """Take a reference image, return a prompt string for Nano Banana 2."""
```

The harness imports this, passes each eval image, takes the returned string, and feeds it to the image generator. Everything you experiment with — single-call vs decomposition, output format, thinking levels, few-shot, negative prompts, iterative refinement — happens *inside this function*. Its signature is fixed; its body is yours.

## Running experiments

```bash
uv run harness.py --name <experiment_name>             # eval, 1 seed
uv run harness.py --name <experiment_name> --seeds 3   # eval, 3 seeds (re-eval)
uv run harness.py --val                                # held-out val, no promotion
uv run harness.py --name <experiment_name> --no-judge  # skip VLM-judge to save calls
```

Flags:
- `--name` — directory under `runs/` for artifacts; required for the promotion path.
- `--val` — runs against `val_images/`; never promotes; cheaper sanity check.
- `--seeds N` — number of generation seeds per image (fixed: 1, 2, 3, ...).
- `--no-judge` — skips the VLM-judge call to save API budget.

## The metric

### Per-pair similarities (each in [0, 1], higher = better)

| Name | Formula |
|---|---|
| `s_gemini` | cosine of Gemini multimodal embeddings |
| `s_dino` | cosine of DINOv3 ViT-B/16 CLS-token features |
| `s_lpips` | `1 − clip(lpips_distance, 0, 1)` (AlexNet backbone) |
| `s_color` | `1 − clip(chi2_distance / 2.0, 0, 1)`; HSV 8×8×8 histogram; chi² = `0.5 * Σ((p−q)² / (p+q+ε))` |

All features are computed at canonical **448×448 LANCZOS** (multiple of DINOv3 patch size 16). Same input crop for all four metrics.

### Composite

Composite = mean of the four per-metric means across the 20 eval images.

### Promotion gate

A candidate is promoted only if **both** rules hold against the current leader:

1. **No-regression:** for each metric `m`, `mean_m(candidate) ≥ mean_m(leader) − epsilon` with `epsilon = 0.01`. If any single metric drops by more than 0.01, the candidate is rejected even if composite improved. This defeats the most common failure mode: trading one signal for another.
2. **Improvement:** `composite(candidate) > composite(leader)`.

### 3-seed re-eval

When a 1-seed candidate passes the gate, the harness automatically re-runs the full eval at fixed seeds 1, 2, 3. The 3-seed mean must still pass both rules; otherwise revert to leader. This filters single-seed luck.

### Held-out validation

`val_images/` contains 5 images covering the same archetypes as eval, with different content. Run `uv run harness.py --val` every ~10 promoted leaders. If `composite_eval` climbs while `composite_val` stays flat or drops, you are overfitting the eval set. Stop and rethink.

### VLM-judge — diagnostic only, NOT optimized against

A 6-axis Gemini-Flash-Lite judge scores each (orig, regen) pair on **subject identity, composition, lighting, palette, style, texture**, each 1–5. It is logged per pair for diagnostic insight only. **Do not optimize against it.** The inner VLM is also Gemini; optimizing against a Gemini-judged score creates a feedback loop that rewards prompts the model family happens to find pretty. The four computed metrics above are the actual objective. The judge's value is post-hoc: when a metric moves, the judge axes help you tell *which kind* of failure it was.

## Workflow per experiment

**1. Re-read `logbook.md` before forming a new hypothesis.**

This is the single most important instruction in this document. The logbook is your only persistent memory across runs. Most failure modes — repeating an already-tried experiment, drifting along one axis without checking the gate, losing track of which dimension the leader has stopped improving on — come from skipping this step. Read the whole logbook. Then form your hypothesis.

2. Write a one-line hypothesis: what change, what metric you expect to move, why. Be specific (e.g., "explicit color-palette enumeration should raise `s_color` ≥ 0.02 without dropping `s_dino`").

3. Edit `prompt_strategy.py`. Keep diffs small. One change per experiment.

4. Set `AUTORESEARCH_HYPOTHESIS` and `AUTORESEARCH_DRIVER` env vars so the harness fills the logbook entry automatically.

5. Run `uv run harness.py --name <descriptive_name>`. Names are cheap — use long ones.

6. Read the per-metric breakdown. Look at all four metrics and per-image rows, not just composite. Greppable lines: `composite=`, `gate=`, `promoted=`.

7. Append a logbook entry: hypothesis, diff summary, per-metric numbers, gate result, takeaway. Be honest in the takeaway — especially when the result was negative.

8. After every ~10 promoted leaders, run `uv run harness.py --val` to confirm gains generalize.

## Budget guardrails

- **Per experiment:** one full eval ≈ 20 images × ~10s generation ≈ 3–5 min wall time. With `--seeds 3`, triple it.
- **Per session:** cap at 50 experiments. Past that you are almost certainly Goodharting.
- **VLM-judge** is the most expensive non-generation cost. Use `--no-judge` while iterating fast on a known direction; re-enable for the runs you're going to write up.
- API spend is dominated by Nano Banana 2 generation, not by the VLM or embedding calls.

## What's worth exploring

| Axis | Examples |
|---|---|
| Output format | comma-separated tags; structured JSON-like schema; long prose; bullet list |
| Decomposition | one VLM call vs. separate calls for subject / setting / style / palette |
| Iterative refinement | first-pass description → critique → revised prompt |
| Prompt length | 20 words vs. 80 vs. 300 |
| Negative prompts | append "avoid: blur, watermark, text artifacts" |
| Few-shot | inline example pairs of (image-description → ideal prompt) |
| Thinking levels | enable/disable model thinking; vary thinking budget |
| Style vs. content tradeoff | per-metric inspection: did `s_dino` (structure) win at the expense of `s_color` (palette)? |

Prefer **qualitatively different** experiments over local tweaks of the same idea. The leader can drift far along one direction (more verbose, more structured, etc.) and you'll get diminishing-then-negative returns without noticing. If your last 5 experiments all touched the same axis, switch axes.

## What NOT to do

- Do not modify `harness.py`, `embed_and_score.py`, the image dirs, or anything in `cache/` — invalidates all prior scores.
- Do not change models, model parameters, or generation config. The harness fixes them deliberately.
- Do not memorize the eval set in `prompt_strategy.py` (per-filename branching, hardcoded captions). It defeats validation.
- Do not optimize against the VLM-judge axes. Diagnostic only.
- Do not promote on a single lucky seed; trust the 3-seed re-eval.
- Do not delete or rewrite logbook entries. Append-only.
- Do not chase a metric that's already at 0.95+; the headroom is in the laggards.

## Logbook entry format

Append one block per run, newest at bottom:

```
## <experiment_name> — <YYYY-MM-DD HH:MM>
driver: <claude-code | codex | gemini-cli | opencode | aider | cursor>
hypothesis: <one line>
diff: <one-line summary of what changed in prompt_strategy.py>
seeds: <1 | 3>
metrics:
  s_gemini: <mean> | s_dino: <mean> | s_lpips: <mean> | s_color: <mean>
  composite: <mean>
gate: <pass | fail: reason>
promoted: <yes | no>
takeaway: <one or two sentences, honest>
```

The `driver:` field exists so cross-agent comparisons stay clean.

## Stopping

When you stop (session cap, exhausted ideas, or human says so), append a session summary to `logbook.md`:

```
## SESSION SUMMARY — <YYYY-MM-DD>
top 3 by composite (with val): <names + numbers>
what worked: <2–4 bullets>
promising but failed gate or val: <2–4 bullets>
three concrete next experiments: <numbered>
```

The "three concrete next experiments" matters. The next session reads it as part of step 1.

## Multi-CLI compatibility

The driver loop is agent-agnostic: read files, edit files, run shell, run git. No tool-specific instructions, slash commands, MCP servers, or persistent agent memory. `logbook.md` is the only memory.

### Per-agent setup

| Agent | Auto-loaded file | Setup |
|---|---|---|
| Claude Code | `CLAUDE.md` | `ln -s program.md CLAUDE.md` |
| Codex CLI | `AGENTS.md` | `ln -s program.md AGENTS.md` |
| Gemini CLI | `GEMINI.md` | `ln -s program.md GEMINI.md` |
| OpenCode | `AGENTS.md` | `ln -s program.md AGENTS.md` |
| Aider | none | `--read program.md` on launch |
| Cursor | `.cursorrules` | `ln -s program.md .cursorrules` |

### Sandbox / permissions

The driver needs:
- Network egress to `generativelanguage.googleapis.com` (Gemini API) and `huggingface.co` (DINOv3 weights, first run only).
- Read/write in working directory.
- Permission to run `git` and `uv`.

### Session entrypoint kickoff prompt

Paste this as the first user message of a fresh session:

> You are running an autoresearch loop on prompt strategies for image reproduction. Read `program.md` first. Then read `logbook.md` in full. Then propose your first experiment as a one-line hypothesis, edit `prompt_strategy.py`, and run the harness. Iterate.

### Cross-agent meta-experiment (optional)

To compare driver agents directly: create one git branch per driver (`driver/claude-code`, `driver/codex`, etc.) sharing the same `eval_images/`, `val_images/`, and starting `prompt_strategy.py`. Each driver runs an independent session. Compare top composites, val performance, and number of promotions per 50-experiment session. The `driver:` field in logbook entries makes the comparison clean.
