# Program: Prompt-Only Image Reproduction Autoresearch

This repo is a harness for running prompt-only autoresearch experiments.

The goal is to discover prompts that make a fixed image-generation model reproduce a target image as closely as possible. The harness is assumed to already be implemented. This file explains how to use it as a research agent or human researcher.

## Research Goal

Given:

- a target image,
- a fixed image generator,
- a VLM that can analyze images and propose prompt edits,
- an embedding model that can score target/generated similarity,

run iterative experiments to answer:

> Can a modern VLM construct prompts that improve image reproduction under a fixed image-generation model?

A stronger second-stage question is:

> Can the system discover reusable prompt-construction rules that generalize across target images?

## Model Roles

The default intended setup uses three model roles:

| Role | Purpose |
|---|---|
| Multimodal embedding model | Scores similarity between target and generated images. |
| VLM analyst / optimizer | Describes target images, critiques generated images, and proposes prompt revisions. |
| Image generator | Produces images from prompts. This model should remain fixed during an experiment. |

In the Gemini-oriented setup:

| Component | Suggested role |
|---|---|
| Gemini embeddings | Main similarity metric / reward model. |
| Gemini Flash Lite VLM | Image analyst, difference critic, prompt optimizer, and rule extractor. |
| Gemini Nano Banana image model | Fixed image generator / environment. |

Model names should be configured in the repo config rather than hardcoded into experiment logic.

## What This Repo Optimizes

The repo optimizes prompt text, not model weights.

For each target image, the harness repeatedly:

1. builds one or more prompt candidates,
2. generates images from those prompts,
3. scores generated images against the target,
4. asks the VLM to analyze differences,
5. proposes improved prompts,
6. keeps the candidates that score best,
7. logs the full trace.

This is analogous to code autoresearch, except the editable object is a prompt rather than a training script.

## Important Constraints

Keep the image generator fixed within a run:

- same model,
- same resolution / aspect ratio,
- same generation parameters,
- same negative-prompt policy,
- same seed policy,
- same safety and content filters.

Changing generation settings mid-run makes the experiment hard to interpret. If you want to test different generation settings, create a separate experiment run.

## Typical Workflow

### 1. Add target images

Place target images under:

```text
eval_data/images/eval/
```

Recommended starting set:

```text
eval_data/images/eval/
  easy/
  medium/
  hard/
```

Suggested categories:

- easy: single object, clean portrait, simple landscape;
- medium: interiors, multi-object scenes, stylized illustrations;
- hard: crowded scenes, dense layouts, screenshots, infographics, text-heavy images.

### 2. Configure the run

Create or edit a config file under:

```text
configs/
```

A typical run config should define:

```yaml
experiment_name: single_image_debug
target_glob: eval_data/images/eval/easy/*.png

models:
  embedding: gemini-embedding-model
  vlm: gemini-flash-lite
  generator: gemini-nano-banana

generation:
  aspect_ratio: "1:1"
  num_images_per_prompt: 3
  min_images_per_prompt: 3
  seed_policy: fixed_or_random
  negative_prompt_enabled: true

search:
  max_iterations: 8
  beam_width: 2
  mutations_per_prompt: 3
  final_reeval_seeds: 3

scoring:
  primary_metric: composite
  use_whole_image_embedding: true
  use_region_embedding: true
  use_vlm_judge: true
```

Exact config keys may differ by implementation. Use the repo’s example configs as the source of truth.

### 3. Run a single-target experiment

Use the repo CLI to run one target image first.

Example:

```bash
python -m image_prompt_search.runner \
  --config configs/single_image.yaml \
  --target eval_data/images/eval/easy/example.png
```

The first run should be treated as a debugging run. Verify that:

- the target image is read correctly,
- the VLM produces a detailed target decomposition,
- the generator produces images,
- embeddings are computed,
- scores are logged,
- generated images are saved,
- the best prompt improves over the baseline.

### 4. Run a benchmark experiment

After the single-image loop works, run a small benchmark:

```bash
python -m image_prompt_search.runner \
  --config configs/benchmark_small.yaml \
  --target-glob "eval_data/images/eval/**/*.png"
```

Start with 5 target images, then scale to 20–50 images after cost and logging are stable.

## Optimization Loop

The default recommended loop is beam search.

At each iteration:

1. keep the top `B` prompts from the previous round,
2. for each prompt, ask the VLM for `K` revised prompt candidates,
3. generate images for all candidates,
4. score each generated image against the target,
5. keep the top `B` candidates overall,
6. repeat until the iteration budget is reached.

Good initial values:

```text
beam_width = 2 or 3
mutations_per_prompt = 3 or 4
max_iterations = 8
num_images_per_prompt = 3 during search
final_reeval_seeds = 3 to 5
```

Early search should stay cheap by keeping beam width and mutation count modest,
but every real candidate evaluation still generates and scores at least 3
images per target/example. Use more than 3 seeds only for finalists or
late-stage candidates.

## Prompt Representation

The harness should support both freeform and structured prompts.

A structured prompt is recommended because it makes targeted edits and later analysis easier.

Example internal schema:

```json
{
  "subject": "",
  "scene": "",
  "composition": "",
  "style": "",
  "lighting": "",
  "color_palette": "",
  "camera": "",
  "materials_textures": "",
  "important_objects": [],
  "spatial_relations": [],
  "negative_constraints": []
}
```

The structured object is compiled into the final prompt sent to the generator.

During research, prefer edits that explain which field changed and why.

## Scoring Philosophy

Do not rely only on a single whole-image multimodal embedding score.

Whole-image embeddings are useful for semantic similarity, but they can miss:

- exact composition,
- object counts,
- spatial relations,
- color palette,
- lighting,
- local details,
- text and typography.

Use a composite evaluator when available.

Recommended first composite score:

```text
final_score =
  0.45 * whole_image_embedding_similarity
+ 0.25 * region_embedding_similarity
+ 0.30 * vlm_structured_judge_score
```

Additional metrics may be logged as diagnostics:

- color histogram / palette distance,
- brightness and contrast difference,
- OCR similarity for text-heavy images,
- object count and layout consistency,
- pairwise VLM preference among finalists,
- robustness across seeds.

## Region-Aware Similarity

Region-aware scoring mitigates the semantic-only weakness of whole-image embeddings.

Recommended crops:

```text
full image
center crop
3x3 grid cells
optional subject crop
optional background crop
```

Compare corresponding target/generated regions with embeddings and average the results.

This helps catch cases where two images have the same semantic subject but different layout.

## VLM Judge

Use the VLM judge for interpretable scoring and prompt feedback.

The judge should compare the target and generated image on:

- subject match,
- composition match,
- style match,
- color and lighting match,
- important details,
- object count,
- spatial relations,
- text / typography if present.

The VLM judge should also output concrete prompt edits, not only a scalar score.

Example output fields:

```json
{
  "subject_match": 4,
  "composition_match": 2,
  "style_match": 5,
  "color_lighting_match": 3,
  "detail_match": 2,
  "main_differences": [
    "The generated subject is centered, but the target subject is lower-left.",
    "The target background is darker and more blurred.",
    "The generated image is missing the red collar."
  ],
  "recommended_prompt_edits": [
    "Place the main subject in the lower-left foreground.",
    "Use a darker blurred background.",
    "Add a visible red collar."
  ]
}
```

## VLM Pairwise Reranking

For top candidates, pairwise judging is often more reliable than absolute scoring.

A practical pipeline:

```text
generate many candidates
rank by embedding score
take top 5
rerank with VLM pairwise judge
keep best 1 or 2
```

Ask the VLM:

> Which generated image is closer to the target? Judge by composition, subject identity, style, color, lighting, and important local details. Do not reward only broad semantic similarity.

## Seed Policy

Avoid overfitting to a lucky generation.

Recommended schedule:

```text
early search: 1 seed per candidate
late search: 3 seeds for top candidates
final report: 3 to 5 seeds for finalists
```

Report both:

- best-of-k score: best achievable reproduction,
- average-of-k score: reliable reproduction,
- score standard deviation: prompt robustness.

## Logs and Outputs

Each experiment should save a complete trace.

Expected outputs:

```text
outputs/runs/<run_id>/
  config.yaml
  target_summary.json
  candidates.jsonl
  generations/
  best/
  report.md
```

Each candidate record should include:

```json
{
  "run_id": "",
  "target_id": "",
  "iteration": 0,
  "candidate_id": "",
  "parent_candidate_id": null,
  "prompt_structured": {},
  "prompt_text": "",
  "negative_prompt": "",
  "mutation_rationale": "",
  "generation_params": {},
  "seed": null,
  "generated_image_path": "",
  "scores": {
    "whole_image_embedding": null,
    "region_embedding": null,
    "vlm_judge": null,
    "final_score": null
  },
  "vlm_feedback": {
    "main_differences": [],
    "recommended_prompt_edits": []
  }
}
```

## Baselines

For meaningful results, compare against baselines.

Recommended baselines:

1. short VLM caption prompt,
2. dense VLM caption prompt,
3. structured VLM prompt without iterative optimization,
4. iterative critique-guided prompt search,
5. optional human-written prompt if available.

## Ablations

Useful ablations:

| Ablation | Question |
|---|---|
| VLM initialization vs simple caption | Does the VLM starting point help? |
| Freeform vs structured prompt | Does schema-based prompting improve search? |
| Critique-guided mutation vs blind mutation | Does target/generated difference analysis help? |
| Single-seed vs multi-seed scoring | Are improvements robust or just lucky? |
| Embeddings-only vs composite score | Does richer evaluation improve reproduction? |
| Whole-image vs region-aware embeddings | Does regional scoring improve layout matching? |

## Final Report

For each target image, report:

- baseline prompt and score,
- best prompt and score,
- score improvement,
- best generated image,
- score trajectory,
- major failure modes,
- whether improvement was semantic, compositional, stylistic, or color-related.

Across all targets, report:

- mean improvement,
- category-level improvement,
- best/worst target categories,
- prompt patterns that helped,
- recurring failure modes,
- cases where prompt-only search hit a ceiling.

## Expected Failure Modes

Prompt-only reproduction may fail on:

- exact geometry,
- exact object placement,
- exact human identity,
- dense layouts,
- screenshots / UI,
- readable text,
- logos,
- small local details,
- unusual images outside the generator’s prior.

These failures are useful research data. Do not hide them.

## Research Notes

The goal is not pixel-perfect reconstruction.

The goal is to measure and improve how much prompt-only search can reproduce a target image under a fixed generator.

A good result is not merely one impressive example. A good result shows:

- consistent improvement over baseline captions,
- interpretable prompt edits,
- robust final prompts across seeds,
- category-level insights,
- reusable prompt-construction rules.

## Kickoff Prompt

Paste the contents of
[`autoresearch-kickoff-prompt.txt`](autoresearch-kickoff-prompt.txt)
as the first message to the driver agent (works across Claude Code,
Codex CLI, Gemini CLI, Aider, Cursor, etc.).
