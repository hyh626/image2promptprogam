# program.md — Image → Prompt → Image autoresearch

You are an autonomous research agent driving an experimental loop. Your
job is to discover prompting strategies that, given an input image,
produce a text prompt which causes an image generator to reproduce that
image as faithfully as possible.

You will iterate by editing a single file, running a fixed evaluation,
reading the score, and deciding whether to keep or discard your change.
Keep going until told to stop.

The harness has already been built. You operate it; you do not modify
it. If you find yourself wanting to change `harness.py` or
`embed_and_score.py`, stop — you are out of scope.

This document is **agent-agnostic**: it should drive equally well
whether you are running inside Claude Code, OpenAI Codex CLI, Gemini
CLI, OpenCode, Aider, Cursor, or any other coding agent that can read
files, edit files, run shell commands, and use git.

The methodology follows Andrej Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) pattern.

---

## The setup

| Role | Model / Method | What it does |
|---|---|---|
| **VLM** (called from `prompt_strategy.py`) | `gemini-3.1-flash-lite-preview` (Gemini API) | Looks at the input image and produces a prompt |
| **Image generator** (called by harness) | `gemini-3.1-flash-image-preview` / Nano Banana 2 (Gemini API) | Generates an image from the prompt |
| **Semantic similarity** (harness) | `gemini-embedding-2-preview` (Gemini API) | Multimodal embedding — semantic content match |
| **Structural similarity** (harness) | DINOv2 ViT-B/14 (local, Apache 2.0) | Self-supervised vision features — pose, layout, appearance |
| **Perceptual similarity** (harness) | LPIPS with AlexNet backbone (local) | Perceptual texture / fine-detail match |
| **Color similarity** (harness) | HSV histogram, chi-square distance (local) | Color palette match |

**You (the driver) are independent of the inner VLM.** You might be
Claude, GPT-5, or Gemini Pro driving this loop; the inner VLM that
writes prompts inside `prompt_strategy.py` is always Gemini Flash-Lite.
Don't confuse the two roles.

## Repo layout

```
prompt_strategy.py    ← the ONLY file you edit
harness.py            ← runs the eval loop. DO NOT MODIFY.
embed_and_score.py    ← four similarity metrics + compositing. DO NOT MODIFY.
eval_data/images/eval/ ← 20 fixed reference images. DO NOT MODIFY.
eval_data/images/val/  ← 5 held-out images. DO NOT MODIFY.
cache/                ← cached features for original images
runs/                 ← per-experiment artifacts
weights/              ← downloaded local model weights
logbook.md            ← every experiment appended here
program.md            ← this file
```

## The single file you edit

`prompt_strategy.py` exposes one function whose signature is fixed:

```python
def image_to_prompt(image: PIL.Image.Image) -> str:
    """Given a reference image, return a prompt for Nano Banana 2."""
```

Everything inside this function is fair game: system prompt, user
prompt, number of VLM calls, iterative refinement against draft
generations, decomposition into subject/style/composition fields,
few-shot examples, thinking levels, etc. The function may make
multiple API calls internally.

## Running an experiment

```bash
uv run harness.py --name <short_descriptive_name>
```

For each of the 20 images in `eval_data/images/eval/`, the harness:

1. Calls `image_to_prompt(image)` to get a prompt.
2. Calls Nano Banana 2 with that prompt for each configured generation
   seed, defaulting to at least three seeds per eval image.
3. Computes four similarity signals between the original and each
   regenerated image, then aggregates the per-seed scores.
4. Combines them into the composite score (defined below).
5. Decides whether to promote against the current leader.
6. Appends a logbook entry.

Per experiment cost scales with `--seeds`; the default 3-seed run is roughly
3x the one-seed generation cost.

Other CLI flags:

```bash
uv run harness.py --val           # run on eval_data/images/val, no promotion
uv run harness.py --name <n> --seeds N   # run N seeds per image, N >= 3
```

---

## The metric

### Per-pair similarities (each in [0, 1], higher = better)

| Signal | Computation |
|---|---|
| `s_gemini` | cosine of `gemini-embedding-2-preview` vectors |
| `s_dino` | cosine of DINOv2 ViT-B/14 CLS-token features |
| `s_lpips` | `1 - clip(lpips_distance, 0, 1)` (LPIPS, AlexNet backbone) |
| `s_color` | `1 - clip(chi_square / 2.0, 0, 1)` (HSV histogram, 8×8×8 bins) |

All four are computed at canonical 448×448 so framing/resolution
differences don't pollute the signal.

### Composite

For each of the 20 eval images, compute all four similarities for each
configured seed, average per image, then aggregate:

```
mean_signal[m]  =  mean over 20 eval images of  s_m
composite       =  mean across the four metrics
```

`composite` is the number you primarily try to improve. But it is not
the only thing the gate checks.

### Anti-Goodhart promotion gate

A new candidate is **promoted** (becomes the new leader) only if BOTH:

1. **No-regression rule:** for each individual metric m,
   `mean_signal_candidate[m] >= mean_signal_leader[m] - 0.01`.
2. **Improvement rule:** `composite_candidate > composite_leader`.

If any single dimension drops by more than 0.01, the candidate is
rejected even if `composite` improved. This defeats the most common
failure mode: boost one signal by abandoning another.

### Multi-seed re-eval

Every eval run generates and scores each target image with a configurable
number of generation seeds. The default is 3, and real eval/val runs must
reject values below 3. When a candidate passes the gate, the harness
automatically runs a confirmation eval with the configured seed count. The
multi-seed mean must still pass the gate. Otherwise, revert.

### Held-out validation

`uv run harness.py --val` runs the full pipeline on `eval_data/images/val/`
without promotion logic. Run this every ~10 promoted leaders. If
`composite_eval` is climbing but `composite_val` is flat or dropping,
you are overfitting to the eval set — back off and try something
qualitatively different.

### Diagnostic, not optimized: VLM judge

The harness also logs a 6-axis VLM-judge score (subject identity,
composition, lighting, color palette, style, texture; each 1–5) per
pair. **Logged for inspection only.** Do not optimize against it;
treating it as a target creates a feedback loop with the prompting
model.

---

## Workflow per experiment

1. **Re-read `logbook.md`** before forming a new hypothesis. This is
   the single most important instruction in the loop. Without it,
   drivers drift into local search and forget what's already been
   tried.
2. **Hypothesis.** Write one sentence describing what you are trying
   and why. "Add explicit color palette extraction step before main
   description" is good. "Try better prompt" is not.
3. **Edit** `prompt_strategy.py`.
4. **Run** `uv run harness.py --name <name>`.
5. **Read** the per-metric breakdown, not just `composite`. If `s_dino`
   jumped but `s_gemini` dropped, that's a signal about what your
   change actually did.
6. **Decide** based on the harness output:
   - Gate passes AND composite improves → harness runs the multi-seed
     confirmation re-eval. If still passes, commit the file with message
     `leader: <name> = <composite>`.
   - Otherwise → `git checkout prompt_strategy.py` and try something
     else.
7. **Log** the entry in `logbook.md`. Append, never overwrite.
8. **Repeat.** Aim for one experiment per ~10 minutes wall clock.

Prefer **qualitatively different** experiments over local optimization
of any single approach. The biggest failure mode is the agent walking
down one direction (more decomposition, more decomposition) instead of
jumping to a different family (iterative refinement, few-shot, terse
prompts). Diversity is a goal, not a side effect.

## Budget guardrails

- **Per experiment:** 20 input images × your VLM calls per image, plus
  20 generations and 80 local feature extractions. If your strategy
  needs more than ~5 VLM calls per image, the gain has to be
  substantial to justify it.
- **Per session:** stop after 50 experiments and write a session
  summary.

## What's worth exploring (not exhaustive)

- **Output format:** free-form vs. structured (subject / style /
  composition / lighting / palette / camera).
- **Decomposition:** multiple VLM calls each focused on one aspect,
  merged.
- **Iterative refinement:** generate a draft, embed it, compare to the
  original, ask the VLM what's missing, revise the prompt, regenerate.
  Costs more per experiment — measure whether it pays.
- **Prompt length:** very short evocative prompts vs. dense literal
  ones. Nano Banana 2 has strong world knowledge; sometimes short wins.
- **Negative prompts** / what to avoid.
- **Few-shot exemplars** in the system prompt.
- **Thinking level** on the inner VLM (`minimal`, `low`, `medium`,
  `high`) — Gemini-specific feature, controls reasoning effort.
- **Style-vs-content tradeoff:** if `s_dino` is high but `s_gemini` is
  low, you're matching layout/texture but missing subject identity. If
  reverse, you're describing what's there but not how it looks.

## What NOT to do

- Do not modify `harness.py`, `embed_and_score.py`,
  `eval_data/images/eval/`, or `eval_data/images/val/`. These define the
  benchmark.
- Do not change the image generator model, the embedding model, the
  local metric models, or the canonical 448×448 resolution.
  Comparability across experiments depends on these being fixed.
- Do not memorize the eval set. Hardcoding references to specific
  eval images is cheating; held-out val will catch it.
- Do not optimize against the VLM-judge scores. They are diagnostic
  only.
- Do not chase a single lucky run. The configurable multi-seed re-eval
  gate exists for a reason, and it must use at least three seeds.

## Logbook entry format

```
### <run_id>
- driver: <e.g. claude-opus-4.7 via claude-code, or gpt-5 via codex>
- hypothesis: <one sentence>
- composite: 0.7421
- s_gemini: 0.812 | s_dino: 0.701 | s_lpips: 0.688 | s_color: 0.767
- gate: pass
- 3-seed re-eval: 0.7398 ± 0.0042   (configured confirmation run; n/a if not run)
- val composite: 0.7301              (only if --val was run)
- wall_clock: 4.2 min
- est_cost_usd: 0.18
- takeaway: <one or two sentences, including any gate-vs-leader detail>
- promoted: yes | no | reverted
```

The `driver` field is important: it lets future readers (or
meta-experiments) compare how different driver agents explored the
space on the same harness.

## Stopping

When the user says stop, or after 50 experiments, append a "Session
summary" section to `logbook.md` with:

1. Top 3 strategies by `composite`, with their per-metric breakdown
   and `val composite` for honesty.
2. What worked consistently across all four metrics.
3. What looked promising on `composite` but failed the gate or
   regressed on val.
4. Three concrete next experiments worth running with more time.

---

## Multi-CLI compatibility

The loop assumes the agent has only these capabilities, which every
modern coding agent provides:

- Read files in the working directory.
- Edit files (at minimum `prompt_strategy.py` and `logbook.md`).
- Run shell commands (`uv run`, `git`, basic POSIX).
- Use git (`checkout`, `commit`, `diff`).

It does NOT assume any specific tool name, slash command, IDE
integration, MCP server, or persistent agent memory. `logbook.md` is
the memory.

### Per-agent context-file setup (one-time)

Symlink `program.md` to whatever filename your agent auto-loads:

| Agent | Auto-loaded file | Setup |
|---|---|---|
| Claude Code | `CLAUDE.md` | `ln -s program.md CLAUDE.md` |
| OpenAI Codex CLI | `AGENTS.md` | `ln -s program.md AGENTS.md` |
| Gemini CLI | `GEMINI.md` | `ln -s program.md GEMINI.md` |
| OpenCode | `AGENTS.md` | `ln -s program.md AGENTS.md` |
| Aider | none auto-loaded | pass `--read program.md` on launch |
| Cursor | `.cursorrules` | `ln -s program.md .cursorrules` |

Prepared implementation folders may initially ship implementation-phase
context wrappers in these files. After the harness is built, replace those
wrappers with symlinks or copies of `program.md` before starting the
research/driver agent.

### Sandboxing & permissions

The loop needs:

- **Network egress** to `generativelanguage.googleapis.com` (Gemini
  API).
- **Read/write** within the working directory.
- **Permission to run** `git` and `uv`.

Whitelist these explicitly when launching the agent if it asks.

### Session entrypoint

Same regardless of which CLI: paste the contents of
[`autoresearch-kickoff-prompt.txt`](autoresearch-kickoff-prompt.txt)
as your first message to the driver agent.

### Cross-agent meta-experiment (optional)

Because `program.md` is fixed and `logbook.md` records the driver,
running the same spec under different agents and comparing outcomes
is a legitimate meta-experiment. Use a separate git branch per
driver (`session/claude-opus`, `session/gpt-5`, `session/gemini-pro`)
so the per-driver logbooks and leader trajectories don't interleave.
