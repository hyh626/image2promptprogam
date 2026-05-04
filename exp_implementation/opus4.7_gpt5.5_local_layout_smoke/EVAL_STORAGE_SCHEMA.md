# EVAL_STORAGE_SCHEMA.md — canonical eval data storage schema

This document defines a **canonical on-disk layout and JSON schema** for
storing experiment artifacts produced by any implementation of the
image-to-prompt autoresearch harness in this repo (`opus4.7/`,
`gpt5.5/`, `gemini3.1pro/`, or future variants).

It is meant to be **appended verbatim** to each spec's `IMPLEMENTATION.md`
so that whichever agent builds the harness lands on the same on-disk
shape. A checker program (`check_eval_storage.py`, in this repo root)
verifies that a built repo conforms.

The goal is comparability:

- Two runs from two different driver agents should be diffable without
  rewriting paths.
- Re-running the same run on a different machine should produce a
  byte-identical structure (modulo timestamps and image bytes).
- A meta-experiment that compares Claude vs. GPT vs. Gemini drivers
  should be a `find` and a `jq` away, not a parse-three-formats job.

If a spec variant's existing instructions disagree with this schema,
this schema wins for the **storage layout and metadata files**. The
spec variant still owns the harness logic, CLI flags, and metric
definitions.

---

## 1. Top-level layout

```text
<repo-root>/
  eval_data/                   # immutable inputs
    images/
      train/<image_id>.<ext>
      eval/<image_id>.<ext>
      val/<image_id>.<ext>
      holdout/<image_id>.<ext>
      manifest.json            # registry of all image_ids by split
  experiments/                 # all run-time outputs (gitignored)
    runs/
      <run_id>/                # one harness invocation
        run.json
        config.json
        prompt_strategy.py     # snapshot of the strategy file used
        aggregate.json
        gate.json
        stdout.log
        stderr.log             # optional
        per_image/
          <image_id>/
            prompt.txt
            generated.png
            scores.json
            judge.json         # optional, VLM-judge diagnostic only
            seeds/             # optional, only when seeds > 1
              <seed>/
                generated.png
                scores.json
    leader/
      pointer.json             # current leader run_id + metrics
      history.jsonl            # append-only promotion history
    cache/
      features/
        <split>/<image_id>.json
        index.json             # cache index keyed by image_id + sha256
    logbook.md                 # human-readable narrative log
  weights/                     # local model weights (gitignored)
```

### Path rules

- `<image_id>` is the basename of the source image, **without
  extension**, lowercased, with spaces replaced by `_`. Example:
  `eval_data/images/eval/Hero Photo 01.PNG` → `image_id = "hero_photo_01"`.
- `<run_id>` is `<UTC-timestamp>__<driver>__<name>` where
  `<UTC-timestamp>` is `YYYYMMDDTHHMMSSZ`, `<driver>` is a slugged
  agent identifier (e.g. `claude-opus-4-7`, `gpt-5`, `gemini-3-1-pro`),
  and `<name>` is the user-supplied experiment name. Example:
  `20260504T123456Z__claude-opus-4-7__add_palette_step`.
- All image filenames inside `per_image/` use `.png` regardless of
  the source format. Source image extension is preserved only inside
  `eval_data/images/`.
- All JSON files use UTF-8, 2-space indentation, sorted keys, and a
  trailing newline. Floats use up to 6 decimal places.

### What is committed vs. ignored

- **Committed:** `eval_data/images/manifest.json`, `logbook.md`,
  `experiments/leader/pointer.json`, `experiments/leader/history.jsonl`.
- **Gitignored:** `eval_data/images/*/*.png`, `experiments/runs/`,
  `experiments/cache/`, `weights/`. (Image bytes are large, run
  artifacts are reproducible.)

A spec variant that already mandates `runs/` at the repo root (e.g.
`opus4.7/`) should treat `experiments/runs/` as the canonical location
and either move existing references or symlink `runs -> experiments/runs`.

---

## 2. JSON schemas

Every schema below is normative. A field marked **required** must be
present with a non-null value (empty list/string/object are allowed
where the type permits). Unknown extra fields are allowed and
preserved.

### 2.1 `eval_data/images/manifest.json`

Lists every image used as an input to any experiment.

```json
{
  "schema_version": "1.0.0",
  "splits": {
    "train": [
      {
        "image_id": "hero_photo_01",
        "filename": "hero_photo_01.png",
        "sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "width": 1024,
        "height": 768,
        "source": "stock-archive-2025-04",
        "category": "photography_hero",
        "license": "CC-BY-4.0",
        "notes": ""
      }
    ],
    "eval": [],
    "val": [],
    "holdout": []
  }
}
```

Required per-image fields: `image_id`, `filename`, `sha256`, `width`,
`height`. Optional but recommended: `source`, `category`, `license`,
`notes`. The same `image_id` MUST NOT appear in two splits.

### 2.2 `experiments/runs/<run_id>/run.json`

Describes the run as a unit.

```json
{
  "schema_version": "1.0.0",
  "run_id": "20260504T123456Z__claude-opus-4-7__add_palette_step",
  "name": "add_palette_step",
  "driver": "claude-opus-4-7",
  "harness_variant": "opus4.7",
  "git_commit": "a1b2c3d4e5f6",
  "started_at": "2026-05-04T12:34:56Z",
  "finished_at": "2026-05-04T12:39:11Z",
  "split": "eval",
  "image_ids": ["hero_photo_01", "hero_photo_02"],
  "seeds": [0],
  "wall_clock_seconds": 254.7,
  "est_cost_usd": 0.18,
  "status": "completed",
  "hypothesis": "Add an explicit color palette extraction step before main description.",
  "takeaway": "s_color +0.04, s_dino flat, s_gemini -0.01 — net positive."
}
```

Required: `schema_version`, `run_id`, `name`, `driver`,
`harness_variant`, `started_at`, `finished_at`, `split`,
`image_ids`, `seeds`, `status`. `status` ∈
`{"completed", "failed", "interrupted"}`. `image_ids` MUST be a
subset of the manifest entries for the named split.

### 2.3 `experiments/runs/<run_id>/config.json`

Frozen snapshot of the configuration that produced this run. The
exact contents depend on the spec variant; the wrapper is fixed:

```json
{
  "schema_version": "1.0.0",
  "harness_variant": "opus4.7",
  "models": {
    "vlm": "gemini-3.1-flash-lite-preview",
    "generator": "gemini-3.1-flash-image-preview",
    "embedding": "gemini-embedding-2",
    "structural": "facebook/dinov2-base",
    "perceptual": "lpips-alex",
    "color": "hsv-8x8x8-chi2"
  },
  "canonical_resolution": [448, 448],
  "metrics": ["s_gemini", "s_dino", "s_lpips", "s_color"],
  "promotion_gate": {
    "regression_epsilon": 0.01,
    "improvement_strict": true,
    "reeval_seeds": 3
  },
  "cli_args": ["--name", "add_palette_step"],
  "extra": {}
}
```

Required: `schema_version`, `harness_variant`, `models`, `metrics`.
`metrics` is the list of metric keys present in every per-image
`scores.json` of this run.

### 2.4 `experiments/runs/<run_id>/per_image/<image_id>/scores.json`

Per-image metrics for the regenerated image.

```json
{
  "schema_version": "1.0.0",
  "image_id": "hero_photo_01",
  "seed": 0,
  "scores": {
    "s_gemini": 0.812,
    "s_dino": 0.701,
    "s_lpips": 0.688,
    "s_color": 0.767
  },
  "judge": null,
  "generated_image_sha256": "f1e2d3c4b5a6...",
  "prompt_sha256": "0badc0ffee...",
  "generation_seconds": 6.4,
  "scoring_seconds": 0.43
}
```

Required: `schema_version`, `image_id`, `seed`, `scores`,
`generated_image_sha256`, `prompt_sha256`. Every metric key listed in
the run's `config.json#metrics` MUST appear in `scores` with a float
in `[0, 1]`. Any key in `scores` that is not in the configured metric
list is rejected.

`judge` is either `null` or an object whose values are integers in
`[1, 5]`. Suggested keys: `subject`, `composition`, `lighting`,
`palette`, `style`, `texture`. Diagnostic only.

### 2.5 `experiments/runs/<run_id>/per_image/<image_id>/seeds/<seed>/scores.json`

Same shape as 2.4, written when `seeds > 1`. The top-level
`scores.json` in `per_image/<image_id>/` for multi-seed runs holds the
per-image **mean across seeds**, with an extra `per_seed` key:

```json
{
  "schema_version": "1.0.0",
  "image_id": "hero_photo_01",
  "seed": null,
  "scores": { "s_gemini": 0.811, "s_dino": 0.700, "s_lpips": 0.685, "s_color": 0.770 },
  "per_seed": [0, 1, 2],
  "generated_image_sha256": null,
  "prompt_sha256": "0badc0ffee...",
  "generation_seconds": null,
  "scoring_seconds": null
}
```

### 2.6 `experiments/runs/<run_id>/aggregate.json`

Aggregated results across the eval set. The composite is the
unweighted mean of per-metric means (the canonical formula in
`opus4.7/program.md`); spec variants that use weights MUST also
record `composite_unweighted` so cross-variant comparisons remain
possible.

```json
{
  "schema_version": "1.0.0",
  "run_id": "20260504T123456Z__claude-opus-4-7__add_palette_step",
  "split": "eval",
  "n_images": 20,
  "seeds": [0],
  "means": {
    "s_gemini": 0.789,
    "s_dino": 0.681,
    "s_lpips": 0.660,
    "s_color": 0.741
  },
  "stds": {
    "s_gemini": 0.041,
    "s_dino": 0.052,
    "s_lpips": 0.038,
    "s_color": 0.061
  },
  "composite": 0.7178,
  "composite_unweighted": 0.7178,
  "three_seed": {
    "ran": false,
    "mean_composite": null,
    "std_composite": null
  }
}
```

Required: `schema_version`, `run_id`, `split`, `n_images`, `seeds`,
`means`, `composite`, `composite_unweighted`. `n_images` MUST equal
`len(run.json#image_ids)`. Each metric mean MUST equal the mean of
the per-image scores within `1e-4`.

### 2.7 `experiments/runs/<run_id>/gate.json`

Records the promotion-gate decision computed at run time.

```json
{
  "schema_version": "1.0.0",
  "leader_run_id": "20260504T100000Z__claude-opus-4-7__baseline",
  "leader_means": {
    "s_gemini": 0.770, "s_dino": 0.670, "s_lpips": 0.650, "s_color": 0.700
  },
  "leader_composite": 0.6975,
  "candidate_means": {
    "s_gemini": 0.789, "s_dino": 0.681, "s_lpips": 0.660, "s_color": 0.741
  },
  "candidate_composite": 0.7178,
  "regression_epsilon": 0.01,
  "no_regression": true,
  "improves_composite": true,
  "single_run_gate": "pass",
  "three_seed_gate": "pass",
  "decision": "promoted",
  "reason": "no metric regressed by more than 0.01; composite +0.0203; 3-seed mean composite 0.7164 still beats leader."
}
```

Required: `schema_version`, `leader_run_id` (may be `null` for the
first run), `candidate_means`, `candidate_composite`,
`regression_epsilon`, `no_regression`, `improves_composite`,
`single_run_gate`, `decision`. `decision` ∈
`{"promoted", "rejected", "reverted_after_reeval", "no_leader"}`.

### 2.8 `experiments/leader/pointer.json`

Single-line pointer to the current leader run. Updated atomically
(write to a temp path, then `os.replace`).

```json
{
  "schema_version": "1.0.0",
  "run_id": "20260504T123456Z__claude-opus-4-7__add_palette_step",
  "composite": 0.7178,
  "means": {
    "s_gemini": 0.789, "s_dino": 0.681, "s_lpips": 0.660, "s_color": 0.741
  },
  "promoted_at": "2026-05-04T12:39:14Z"
}
```

The referenced `run_id` MUST exist under `experiments/runs/`.

### 2.9 `experiments/leader/history.jsonl`

Append-only. One JSON object per line, in promotion order.

```json
{"run_id": "20260504T100000Z__claude-opus-4-7__baseline", "composite": 0.6975, "promoted_at": "2026-05-04T10:05:11Z", "previous_run_id": null}
{"run_id": "20260504T123456Z__claude-opus-4-7__add_palette_step", "composite": 0.7178, "promoted_at": "2026-05-04T12:39:14Z", "previous_run_id": "20260504T100000Z__claude-opus-4-7__baseline"}
```

Required per line: `run_id`, `composite`, `promoted_at`,
`previous_run_id`. The last line's `run_id` MUST equal
`pointer.json#run_id`.

### 2.10 `experiments/cache/features/index.json`

```json
{
  "schema_version": "1.0.0",
  "entries": [
    {
      "image_id": "hero_photo_01",
      "split": "eval",
      "sha256": "9f86d081...",
      "feature_path": "experiments/cache/features/eval/hero_photo_01.json",
      "computed_at": "2026-05-04T09:00:00Z"
    }
  ]
}
```

Per-feature files store the feature payload (e.g. base64-encoded
numpy bytes or arrays). The shape is spec-variant-specific; the index
is fixed.

---

## 3. Cross-file invariants

These are the consistency rules the checker enforces.

1. **Manifest is closed under reference.** Every `image_id` referenced
   by any `run.json`, `scores.json`, or `pointer.json` exists in
   `eval_data/images/manifest.json` under the declared split.
2. **Per-image scores match aggregates.** For each metric in
   `aggregate.json#means`, the value equals the mean of
   `scores.json#scores[m]` across `per_image/*/scores.json`, within
   `1e-4`.
3. **Composite formula.** `composite_unweighted` equals
   `mean(aggregate.means.values())` within `1e-4`.
4. **Gate self-consistency.** `gate.json#no_regression` equals
   `all(candidate[m] >= leader[m] - epsilon)` for all metrics, and
   `improves_composite` equals
   `candidate_composite > leader_composite`. For the first run
   (`leader_run_id == null`), `decision == "no_leader"` and both
   booleans are `true`.
5. **Leader pointer matches history.** `pointer.json#run_id` equals
   the `run_id` on the last line of `history.jsonl`, and that line's
   `composite` equals `pointer.json#composite`.
6. **All leader runs exist.** Every `run_id` in `history.jsonl` and
   `pointer.json` corresponds to a real
   `experiments/runs/<run_id>/run.json`.
7. **Per-image directory completeness.** For each `image_id` in
   `run.json#image_ids`, `per_image/<image_id>/{prompt.txt,
   generated.png, scores.json}` all exist.
8. **Schema version.** All `schema_version` values are `"1.0.0"`. A
   harness that bumps the version MUST also update this document.
9. **Metric range.** Every metric value is a finite float in
   `[0, 1]`.
10. **Image hash integrity.** When a `sha256` is recorded for an
    image (manifest entry, per-image `generated_image_sha256`), it
    matches the bytes on disk. The checker can verify on demand
    (`--verify-hashes`); off by default to keep checks fast.

---

## 4. Logbook entry format

Each run appends a Markdown block to `experiments/logbook.md`.
Whitespace and field order must match exactly so the checker can
parse it.

```markdown
### <run_id>
- driver: <driver>
- hypothesis: <one sentence>
- composite: 0.7178
- s_gemini: 0.789 | s_dino: 0.681 | s_lpips: 0.660 | s_color: 0.741
- gate: pass
- 3-seed re-eval: 0.7164 ± 0.0042
- val composite: n/a
- wall_clock: 4.2 min
- est_cost_usd: 0.18
- takeaway: <one or two sentences>
- promoted: yes
```

Fields with no value MUST be written as `n/a`, not omitted.
The `3-seed re-eval` field name is kept stable for checker compatibility. If
the harness uses a configurable confirmation seed count greater than 3, write
that confirmation run's mean/std in this field and record the exact seed count
in the run artifacts.

---

## 5. How to use this addendum in a spec

1. Append this whole file (or a link to it) to each
   `*/IMPLEMENTATION.md` under a heading like
   `## Appendix: Eval Storage Schema (canonical)`.
2. In the implementation agent's "Definition of done" checklist, add
   one item: `python check_eval_storage.py --root . passes`.
3. If the spec defines its own paths (e.g. `outputs/runs/` or
   `workspace/`), bridge them with symlinks pointing into
   `experiments/`.

---

## 6. Verifying conformance

The checker lives at the repo root: `check_eval_storage.py`.

```bash
python check_eval_storage.py --root /path/to/built/repo
python check_eval_storage.py --root . --verify-hashes
python check_eval_storage.py --root . --json     # machine-readable output
```

Exit code is `0` on success, `1` on any violation. Each violation
prints one line with a stable code (e.g. `E_AGG_MISMATCH`,
`E_LEADER_MISSING`) so CI can grep for specific failures.
