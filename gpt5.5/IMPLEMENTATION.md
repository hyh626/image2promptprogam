# Implementation Spec: Prompt-Only Image Reproduction Autoresearch Harness

This file is for the implementation agent building the harness.

The companion `program.md` is the researcher-facing guide. This implementation spec may refer to `program.md`, but it contains the full design context and engineering requirements needed to implement the repo.

## 1. Project Summary

Build a harness for prompt-only autoresearch over image generation.

Given a target image, the harness should use:

- a multimodal embedding model to score target/generated similarity,
- a VLM to analyze images, critique differences, and propose prompt edits,
- a fixed image generator to generate images from prompts,

then run an iterative search loop to find prompts that improve reproduction of the target image.

The intended Gemini-oriented setup is:

| Component | Role |
|---|---|
| Vertex AI Gemini `gemini-embedding-2` | Similarity metric / reward model. |
| Vertex AI Gemini `gemini-3.1-flash-lite-preview` | Target analyzer, generated-image critic, prompt optimizer, rule extractor. |
| Vertex AI Gemini `gemini-3.1-flash-image-preview` | Fixed image generator / environment. |
| DINOv2 ViT-B/14 (`facebook/dinov2-base`) | Local structural similarity signal for pose, layout, and appearance. |

Keep all model names configurable. Do not hardcode Gemini-specific model IDs in core logic.
The default Gemini provider should use Vertex AI with `location: global` for
all Gemini model calls.

## 2. Core Research Question

The harness should support experiments answering:

> Can a modern VLM iteratively construct prompts that improve image reproduction under a fixed image-generation model?

And later:

> Can optimization traces reveal reusable prompt-construction rules that generalize across target images?

## 3. Non-Goals

Do not implement model fine-tuning.

Do not optimize image generator weights.

Do not assume pixel-perfect reconstruction is achievable.

Do not require a specific vendor in the core abstractions.

Do not let the optimization loop mutate generator settings inside one run unless the config explicitly declares that experiment type.

## 4. Required Repo Shape

Recommended structure:

```text
image_prompt_search/
  README.md
  program.md
  IMPLEMENTATION.md
  pyproject.toml
  configs/
    single_image.yaml
    benchmark_small.yaml
  data/
    targets/
  src/
    image_prompt_search/
      __init__.py
      analysis.py
      config.py
      embeddings.py
      generator.py
      image_utils.py
      judge.py
      logger.py
      prompt_builder.py
      prompt_schema.py
      runner.py
      scorer.py
      search.py
      vlm.py
  tests/
    test_prompt_schema.py
    test_scorer.py
    test_search.py
    test_logger.py
  outputs/
    runs/
```

The exact package name may vary, but keep the separation of concerns.

## 5. Main CLI

Implement a CLI entry point that can run one image or a benchmark set.

Example desired commands:

```bash
python -m image_prompt_search.runner \
  --config configs/single_image.yaml \
  --target data/targets/easy/example.png
```

```bash
python -m image_prompt_search.runner \
  --config configs/benchmark_small.yaml \
  --target-glob "data/targets/**/*.png"
```

Useful flags:

```text
--config
--target
--target-glob
--run-id
--output-dir
--max-iterations
--beam-width
--dry-run
--resume
```

The CLI should write a complete run directory under:

```text
outputs/runs/<run_id>/
```

## 6. Configuration

Use YAML or JSON config.

Suggested config schema:

```yaml
experiment_name: single_image_debug
target_glob: data/targets/easy/*.png
output_dir: outputs/runs

models:
  embedding:
    provider: gemini
    model: gemini-embedding-2
    vertexai: true
    location: global
  vlm:
    provider: gemini
    model: gemini-3.1-flash-lite-preview
    vertexai: true
    location: global
  generator:
    provider: gemini
    model: gemini-3.1-flash-image-preview
    vertexai: true
    location: global
  structural:
    provider: local
    model: facebook/dinov2-base

generation:
  aspect_ratio: auto
  resolution: null
  num_images_per_prompt: 1
  seed_policy: random
  fixed_seed: null
  negative_prompt_enabled: true
  extra_params: {}

search:
  algorithm: beam
  max_iterations: 8
  beam_width: 2
  mutations_per_prompt: 3
  final_reeval_seeds: 3
  keep_failed_candidates: true

scoring:
  mode: composite
  weights:
    whole_image_embedding: 0.35
    region_embedding: 0.25
    dino_structure: 0.20
    vlm_judge: 0.20
  region_embedding:
    enabled: true
    crops:
      - full
      - center
      - grid_3x3
  vlm_judge:
    enabled: true
    use_pairwise_rerank: true
    pairwise_top_k: 5
  diagnostics:
    color: true
    ocr: false
    layout: false

cost:
  max_candidates_per_target: null
  max_generation_calls_per_target: null
```

Implementation should validate config early and fail with clear errors.
When `generation.aspect_ratio` is `auto`, infer it from the target image and
pass the nearest supported aspect ratio to
`gemini-3.1-flash-image-preview` through its `aspect_ratio` parameter. Do not
silently force all targets to `1:1`.

For Vertex AI Gemini clients, read `GOOGLE_CLOUD_PROJECT` from the environment
and default location to `global`:

```python
client = genai.Client(
  vertexai=True,
  project=os.environ["GOOGLE_CLOUD_PROJECT"],
  location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)
```

## 7. Provider Interfaces

Define provider interfaces so the system is not locked to one model vendor.

### Embedding Client

Responsibilities:

- embed image,
- embed text if needed,
- compute or return embedding vectors,
- cache embeddings by content hash.

Suggested interface:

```python
class EmbeddingClient:
  def embed_image(self, image_path: str) -> list[float]:
    ...

  def embed_text(self, text: str) -> list[float]:
    ...
```

### VLM Client

Responsibilities:

- analyze target image,
- compare target and generated image,
- propose prompt candidates,
- optionally pairwise-rerank candidates,
- extract final rules from traces.

Suggested interface:

```python
class VLMClient:
  def describe_target(self, image_path: str) -> dict:
    ...

  def propose_initial_prompts(self, target_analysis: dict, n: int) -> list[dict]:
    ...

  def critique_generation(
    self,
    target_image_path: str,
    generated_image_path: str,
    prompt: dict,
    scores: dict,
  ) -> dict:
    ...

  def propose_mutations(
    self,
    target_analysis: dict,
    current_candidate: dict,
    critique: dict,
    n: int,
  ) -> list[dict]:
    ...

  def judge_pairwise(
    self,
    target_image_path: str,
    candidate_a_image_path: str,
    candidate_b_image_path: str,
    context: dict,
  ) -> dict:
    ...
```

### Image Generator Client

Responsibilities:

- generate one or more images from prompt text,
- preserve target framing by passing the configured or inferred
  `aspect_ratio` parameter when the Gemini image model is used,
- save returned images,
- return generation metadata.

Suggested interface:

```python
class ImageGeneratorClient:
  def generate(
    self,
    prompt: str,
    negative_prompt: str | None,
    seed: int | None,
    params: dict,
    output_dir: str,
  ) -> list[dict]:
    ...
```

Return records should include:

```json
{
  "image_path": "",
  "seed": null,
  "params": {},
  "raw_response_metadata": {}
}
```

## 8. Prompt Schema

Implement structured prompts as first-class objects.

Suggested schema:

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

Fields can be optional but should be normalized.

Implement:

- `PromptSpec` dataclass or Pydantic model,
- JSON serialization,
- validation,
- flattening into final prompt text,
- negative prompt extraction.

Flattening should produce clear, generator-friendly text.

Example compiled prompt shape:

```text
Subject: ...
Scene: ...
Composition: ...
Style: ...
Lighting: ...
Color palette: ...
Camera: ...
Important details: ...
Spatial relations: ...
```

Negative constraints should be compiled separately when supported by the generator.

## 9. Target Analysis

For each target image, ask the VLM for:

1. dense caption,
2. structured visual decomposition,
3. top defining visual traits,
4. reproduction-focused instructions,
5. likely hard-to-reproduce details.

Expected schema:

```json
{
  "dense_caption": "",
  "image_type": "portrait|landscape|product|interior|illustration|infographic|ui|other",
  "structured": {
    "main_subjects": [],
    "object_counts": {},
    "composition": "",
    "camera_angle": "",
    "lighting": "",
    "color_palette": [],
    "style": "",
    "background": "",
    "important_details": [],
    "spatial_relations": [],
    "text_or_symbols": []
  },
  "defining_traits": [],
  "reproduction_priorities": [],
  "known_challenges": []
}
```

Save this as:

```text
outputs/runs/<run_id>/<target_id>/target_summary.json
```

## 10. Initial Prompt Generation

From target analysis, generate multiple initial prompt candidates.

At minimum:

1. compact prompt,
2. dense prompt,
3. structured prompt.

Optional:

4. composition-prioritized prompt,
5. style-prioritized prompt.

Each initial candidate should include:

```json
{
  "candidate_id": "",
  "parent_candidate_id": null,
  "prompt_structured": {},
  "prompt_text": "",
  "negative_prompt": "",
  "mutation_rationale": "initial baseline",
  "strategy": "compact|dense|structured|composition|style"
}
```

## 11. Search Algorithm

Implement beam search first.

Algorithm:

```text
target_analysis = describe_target(target)
initial_candidates = propose_initial_prompts(target_analysis)

evaluate initial_candidates
beam = top B candidates

for iteration in 1..max_iterations:
  proposals = []

  for candidate in beam:
    critique = critique_generation(target, candidate.best_image, candidate.prompt, candidate.scores)
    mutations = propose_mutations(target_analysis, candidate, critique, K)
    proposals.extend(mutations)

  evaluate proposals
  pool = beam + proposals
  optionally pairwise-rerank top candidates
  beam = top B candidates from pool

finalists = re-evaluate beam with multiple seeds
write final report
```

Keep parent/child relationships so the trace can be reconstructed.

## 12. Candidate Evaluation

For each candidate:

1. compile prompt text and negative prompt,
2. generate one or more images,
3. score each generation,
4. aggregate score across images/seeds,
5. store best generation and aggregate stats,
6. log all records.

During early search, generate one image per prompt. During final reevaluation, generate 3–5 images per prompt with different seeds.

Aggregation modes:

```text
best_of_k
mean_of_k
mean_top_2
robust_score = mean - penalty * std
```

Default:

```text
search ranking: best_of_1
final report: both best_of_k and mean_of_k
```

## 13. Scoring System

Do not use only a single whole-image embedding score.

Implement a composite scoring system with pluggable metrics.

### 13.1 Whole-Image Embedding Similarity

Compute:

```text
cosine(embed_image(target), embed_image(generated))
```

This is the semantic backbone.

### 13.2 Region-Aware Embedding Similarity

Create corresponding crops from target and generated image.

Required crops:

```text
full image
center crop
3x3 grid cells
```

Optional crops:

```text
subject crop
background crop
salient object crops
```

Compute embedding similarity for each corresponding crop and average.

This mitigates the weakness that whole-image embeddings often ignore layout.

### 13.3 DINOv2 Structural Similarity

Use DINOv2 ViT-B/14 (`facebook/dinov2-base`) locally as a structural
similarity signal. Extract normalized image features for the target and
generated image, compute cosine similarity, and log it as
`dino_structure`. This signal should complement Gemini embeddings by catching
pose, layout, and appearance mismatches.

### 13.4 VLM Structured Judge

Ask the VLM to compare target and generated image and output structured scores.

Expected schema:

```json
{
  "subject_match": 1,
  "composition_match": 1,
  "style_match": 1,
  "color_lighting_match": 1,
  "detail_match": 1,
  "object_count_match": 1,
  "spatial_relation_match": 1,
  "text_typography_match": null,
  "main_differences": [],
  "recommended_prompt_edits": [],
  "overall_judgment": ""
}
```

Normalize numeric fields to 0–1 and average them for `vlm_judge_score`.

### 13.5 Composite Score

Default:

```text
final_score =
  0.35 * whole_image_embedding
+ 0.25 * region_embedding
+ 0.20 * dino_structure
+ 0.20 * vlm_judge
```

Weights should be configurable.

### 13.6 Optional Diagnostics

Implement as logging-only first, then allow inclusion in score later.

Possible diagnostics:

- dominant color palette distance,
- histogram distance,
- brightness / contrast difference,
- saturation difference,
- OCR text similarity,
- object count consistency,
- bounding-box or layout consistency,
- edge-map similarity,
- LPIPS / SSIM if local dependencies are acceptable.

## 14. Pairwise VLM Reranking

Implement optional pairwise reranking for top candidates.

Pipeline:

```text
rank all candidates by composite score
take top pairwise_top_k
ask VLM pairwise comparisons
convert preferences to final order
```

Pairwise prompt should emphasize:

- composition,
- subject identity,
- style,
- color,
- lighting,
- local details,
- text if present.

It should explicitly avoid rewarding only broad semantic similarity.

Suggested instruction:

```text
Given the target image and two generated candidates, choose the candidate that better reproduces the target. Judge by subject, composition, object count, spatial relations, style, color palette, lighting, important local details, and visible text. Do not choose an image merely because it shares the same broad semantic subject.
```

Return schema:

```json
{
  "winner": "A|B|tie",
  "reason": "",
  "dimension_scores": {
    "subject": "A|B|tie",
    "composition": "A|B|tie",
    "style": "A|B|tie",
    "color_lighting": "A|B|tie",
    "details": "A|B|tie"
  }
}
```

A simple tournament or round-robin among top candidates is enough.

## 15. VLM Prompting Requirements

The VLM should produce structured JSON whenever possible.

### Target Description Prompt

Ask for:

- dense caption,
- structured decomposition,
- reproduction priorities,
- important details,
- spatial relations,
- color palette,
- style,
- likely failure cases.

### Critique Prompt

Inputs:

- target image,
- generated image,
- current prompt,
- score breakdown.

Ask for:

- specific differences,
- which prompt fields likely caused the mismatch,
- recommended concrete edits,
- do not be vague,
- output JSON.

### Mutation Prompt

Inputs:

- target analysis,
- current prompt,
- generated-image critique,
- score breakdown,
- parent candidate history if available.

Ask for several candidates:

1. conservative revision,
2. composition-focused revision,
3. style-focused revision,
4. detail-rich revision,
5. simplified revision if requested.

Mutation output schema:

```json
{
  "candidates": [
    {
      "strategy": "",
      "prompt_structured": {},
      "prompt_text": "",
      "negative_prompt": "",
      "mutation_rationale": "",
      "expected_improvement": ""
    }
  ]
}
```

The implementation should validate and repair minor malformed JSON when possible.

## 16. Logging

Logging is critical.

Use JSONL for candidate/generation records.

Run directory layout:

```text
outputs/runs/<run_id>/
  config.yaml
  manifest.json
  report.md
  targets/
    <target_id>/
      target.png
      target_summary.json
      candidates.jsonl
      generations/
        iter_000/
        iter_001/
      best/
        best.png
        best_prompt.json
        best_prompt.txt
      trace.md
```

Candidate record:

```json
{
  "run_id": "",
  "target_id": "",
  "iteration": 0,
  "candidate_id": "",
  "parent_candidate_id": null,
  "strategy": "",
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
    "color": null,
    "ocr": null,
    "layout": null,
    "final_score": null
  },
  "aggregate_scores": {
    "best_of_k": null,
    "mean_of_k": null,
    "std": null
  },
  "vlm_feedback": {
    "main_differences": [],
    "recommended_prompt_edits": []
  },
  "raw_model_metadata": {}
}
```

Manifest:

```json
{
  "run_id": "",
  "created_at": "",
  "config_path": "",
  "target_count": 0,
  "model_roles": {},
  "status": "running|completed|failed",
  "cost_estimate": null
}
```

## 17. Reporting

Generate a Markdown report at the end of each run.

Per target:

- target image path,
- baseline prompt,
- baseline score,
- best prompt,
- best score,
- absolute and relative improvement,
- best generated image path,
- score trajectory,
- top failure modes,
- seed robustness stats.

Across targets:

- mean baseline score,
- mean best score,
- mean improvement,
- category-level improvement,
- best/worst categories,
- common prompt edits that helped,
- common failure modes.

Also generate trace files for debugging.

## 18. Baselines

Implement baseline strategies:

1. short VLM caption prompt,
2. dense VLM caption prompt,
3. structured VLM prompt without iterative optimization,
4. iterative critique-guided prompt search,
5. optional human prompt loaded from file.

The benchmark report should compare the iterative best score to these baselines.

## 19. Ablations

The harness should support ablations by config.

Required ablations:

| Ablation | Config idea |
|---|---|
| VLM initialization vs simple caption | `initialization: simple_caption|vlm_structured` |
| Freeform vs structured prompt | `prompt_mode: freeform|structured` |
| Critique-guided vs blind mutation | `mutation_mode: critique_guided|blind` |
| Single-seed vs multi-seed | `num_images_per_prompt`, `final_reeval_seeds` |
| Embeddings-only vs composite score | scoring weights |
| Whole-image vs region-aware embeddings | region scoring enabled/disabled |
| Pairwise reranking on/off | `use_pairwise_rerank` |

## 20. Metric Calibration

The design should allow later human preference calibration.

Support exporting pairs:

```json
{
  "target_image": "",
  "candidate_a": "",
  "candidate_b": "",
  "score_a": {},
  "score_b": {},
  "human_preference": null
}
```

This enables measuring which metric combination agrees best with humans.

## 21. Mitigating Semantic-Only Embedding Limitations

This is a central design requirement.

Whole-image multimodal embeddings can miss composition, spatial layout, color, text, and local detail.

Mitigation layers to implement:

1. composite score rather than one embedding score,
2. region-aware embedding score,
3. structured VLM visual judge,
4. optional pairwise VLM reranking,
5. color and lighting diagnostics,
6. OCR diagnostics for text-heavy images,
7. object/count/layout diagnostics when available,
8. category-specific scoring profiles,
9. multi-seed robustness evaluation.

Category-specific scoring profiles should be possible in config.

Example:

```yaml
scoring_profiles:
  product_photo:
    whole_image_embedding: 0.30
    region_embedding: 0.25
    vlm_judge: 0.25
    color: 0.10
    layout: 0.10
  infographic:
    whole_image_embedding: 0.20
    region_embedding: 0.25
    vlm_judge: 0.25
    color: 0.10
    ocr: 0.20
```

Initial implementation can log color/OCR/layout diagnostics without including them in the default score.

## 22. Image Utilities

Implement utilities for:

- image loading,
- format conversion,
- resizing,
- center crop,
- 3x3 grid crop,
- content hashing,
- saving generated images,
- optional thumbnail/contact-sheet generation.

Region crop naming should be stable:

```text
full
center
grid_0_0
grid_0_1
...
grid_2_2
```

Cache embeddings by:

```text
model_name + image_content_hash + crop_name
```

## 23. Caching

Add caching early because embedding calls and VLM calls are expensive.

Cache:

- target image embeddings,
- target crop embeddings,
- generated image embeddings,
- VLM target analyses,
- optionally VLM critiques.

Use content hashes, model names, and prompt text where appropriate.

Recommended cache directory:

```text
outputs/cache/
```

or under each run if global cache is not desired.

## 24. Error Handling

The harness should be robust to:

- generation API failure,
- invalid image response,
- embedding API failure,
- VLM malformed JSON,
- VLM refusal or empty answer,
- score NaNs,
- missing generated files,
- interrupted run.

For each failed candidate:

- log the error,
- mark status as failed,
- continue if possible,
- do not crash the full benchmark unless config says fail-fast.

## 25. Resume Support

Implement lightweight resume support.

A run should be resumable from existing JSONL logs when possible.

At minimum:

- avoid overwriting existing run directories unless `--force` is passed,
- write manifest status,
- flush JSONL records after each candidate,
- keep generated images in deterministic paths.

## 26. Dry Run / Mock Mode

Implement a mock mode for tests and local development.

Mock providers should:

- produce deterministic fake embeddings,
- generate placeholder images or copy sample images,
- return deterministic VLM JSON,
- make search tests fast and offline.

This allows unit tests without external API calls.

## 27. Tests

Required tests:

### Prompt schema

- validates minimal prompt,
- validates structured prompt,
- compiles to prompt text,
- extracts negative constraints.

### Scorer

- cosine similarity works,
- composite score respects weights,
- missing optional metric is handled,
- region aggregation works with mock embeddings.

### Search

- beam keeps top candidates,
- parent/child lineage is preserved,
- failed candidates do not kill the run,
- final reevaluation aggregates seeds correctly.

### Logger

- writes JSONL,
- writes manifest,
- creates run directories,
- resumes without corrupting previous records.

### VLM JSON handling

- parses valid JSON,
- repairs simple malformed JSON if implemented,
- fails gracefully otherwise.

## 28. Implementation Order

Build in this order:

1. config loading and validation,
2. prompt schema and prompt compiler,
3. run directory and JSONL logger,
4. mock providers,
5. image utilities and crop generation,
6. embedding scorer with mock embeddings,
7. beam search loop with mock generator,
8. real embedding provider,
9. real generator provider,
10. real VLM target analysis,
11. real VLM critique and mutation,
12. composite scorer,
13. pairwise reranking,
14. final reporting,
15. benchmark runner,
16. ablation configs.

Do not start with all real APIs. First make the harness work in mock mode.

## 29. Acceptance Criteria

The implementation is acceptable when:

1. `program.md` accurately explains how to run research with the harness.
2. A single-image experiment can run end-to-end.
3. The run saves target analysis, candidate logs, generated images, and a report.
4. Beam search preserves candidate lineage.
5. Scoring supports whole-image embedding, region embedding, and VLM judge.
6. The harness does not rely on a single semantic embedding score.
7. Final reevaluation supports multiple seeds.
8. Configs can enable/disable ablations.
9. Mock mode tests pass without external APIs.
10. Real provider hooks are isolated behind interfaces.

## 30. Minimal Viable Experiment

The first working demo should use:

```text
5 target images
max_iterations = 8
beam_width = 2
mutations_per_prompt = 3
1 generated image per candidate during search
3 generated images per finalist
whole-image embedding + region embedding + VLM judge
```

Expected deliverables:

```text
outputs/runs/<run_id>/report.md
outputs/runs/<run_id>/targets/*/target_summary.json
outputs/runs/<run_id>/targets/*/candidates.jsonl
outputs/runs/<run_id>/targets/*/best/best_prompt.txt
outputs/runs/<run_id>/targets/*/best/best.png
```

## 31. Final Design Principle

The harness should not merely chase a scalar score.

It should produce a research trace:

- what the target image contains,
- what each prompt tried to change,
- what the generator produced,
- why the candidate failed or improved,
- which metrics agreed or disagreed,
- what prompt-construction rules emerged.

The value of the repo is not only finding a better prompt for one image. The value is learning which prompt-only strategies work, where they fail, and how those findings generalize.
