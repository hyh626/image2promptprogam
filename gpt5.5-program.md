# Autoprompt Research Program

You are an autonomous research coding agent. Your goal is to improve an image-to-prompt pipeline that reconstructs target images using text-only prompts.

The experiment loop is:

```text
target image
→ VLM prompt constructor
→ text prompt
→ text-to-image generator
→ generated image candidates
→ multimodal and visual similarity evaluator
→ score
→ log result
→ improve the prompt-construction algorithm
```

Lower-level goal: discover prompt-construction strategies that make generated images more similar to the original target images.

Higher-level goal: produce generalizable methods for translating an input image into an effective text prompt for a modern image generation model.

## Fixed model stack

Use these Gemini models unless explicitly changed in config:

- VLM / prompt constructor: `gemini-3.1-flash-lite-preview`
- Similarity embedding model: `gemini-embedding-2`
- Image generator: `gemini-3.1-flash-image-preview`

Important: the image generator must receive only the text prompt produced by the prompt constructor. Do not pass the target image into the image generator, because that turns the task into image editing or image-conditioned generation instead of prompt reconstruction.

The VLM may see the target image. The evaluator may see both target and generated images. The generator may see only text.

## Research question

Given a target image, can we automatically construct a text prompt that causes a text-to-image model to generate an image with high semantic and visual similarity to the target?

The main optimization target is an ensemble similarity score between the target image and generated image. Do not optimize only for global multimodal embedding similarity, because global embeddings can over-reward semantic similarity while missing layout, color, object count, and fine visual details.

## Repository structure

The intended repository structure is:

```text
.
├── program.md
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── targets/
│   │   ├── image_0001.png
│   │   ├── image_0002.png
│   │   └── ...
│   └── metadata.jsonl
├── runs/
│   └── ...
├── src/
│   ├── config.py
│   ├── gemini_client.py
│   ├── prompt_from_image.py
│   ├── generate_image.py
│   ├── embed_image.py
│   ├── crop_regions.py
│   ├── visual_metrics.py
│   ├── score.py
│   ├── run_experiment.py
│   ├── report.py
│   └── utils.py
├── results.tsv
└── notes.md
```

## Editable files

You may edit:

- `src/config.py`
- `src/gemini_client.py`
- `src/prompt_from_image.py`
- `src/generate_image.py`
- `src/embed_image.py`
- `src/crop_regions.py`
- `src/visual_metrics.py`
- `src/score.py`
- `src/run_experiment.py`
- `src/report.py`
- `src/utils.py`
- `README.md`
- `notes.md`
- `program.md`, only to clarify instructions or add results-backed protocol changes

You may create small helper files if needed.

Do not edit:

- target images in `data/targets/`
- previous generated images in `runs/`
- historical rows in `results.tsv`, except to fix clearly broken formatting
- `.env`
- API keys or secrets

## Environment

The user must provide:

```bash
GEMINI_API_KEY=...
```

Use `.env.example` as a template, but never commit real keys.

The code should use the official Google GenAI SDK.

Install with:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` should include:

```text
google-genai
python-dotenv
pillow
numpy
pandas
tqdm
```

Optional dependencies for stronger local visual metrics:

```text
opencv-python
scikit-image
```

Do not require optional dependencies for the baseline to run. If optional dependencies are unavailable, skip those metrics and record that they were skipped.

## Core experiment

Run:

```bash
python -m src.run_experiment
```

Default behavior:

1. Load target images from `data/targets/`.
2. For each target image:
   - ask the VLM to describe the image and construct a generation prompt
   - send only the text prompt to the image generator
   - generate `N` candidate images
   - embed target image with `gemini-embedding-2`
   - embed each generated image with `gemini-embedding-2`
   - compute global image embedding similarity
   - compute regional image embedding similarity
   - compute VLM pairwise visual judge score
   - compute optional object inventory, color, layout, and OCR/text scores
   - compute an ensemble score
   - select the best candidate by ensemble score
   - log all component scores and final scores
3. Append one row per target image and strategy to `results.tsv`.
4. Save generated images, prompts, traces, and score breakdowns in `runs/<timestamp>/`.

## Command-line interface

Implement these arguments:

```bash
python -m src.run_experiment \
  --strategy baseline_dense_caption \
  --max-targets 3 \
  --candidates 2 \
  --scorer ensemble_v1
```

Required arguments and defaults:

```text
--strategy baseline_dense_caption
--max-targets 10
--candidates 4
--scorer ensemble_v1
--run-name optional-human-readable-name
--target-dir data/targets
--output-dir runs
--results-path results.tsv
--resume false
```

Supported scorer names:

```text
global_embedding_v1
regional_embedding_v1
vlm_pairwise_judge_v1
object_inventory_v1
color_layout_v1
ensemble_v1
```

## Required metrics

Primary benchmark metric:

```text
ensemble_v1_score
```

Keep this diagnostic metric:

```text
global_embedding_score
```

Global embedding score is useful, but it must not be the only optimization target.

Required component metrics:

```text
global_embedding_score
regional_embedding_score
vlm_pairwise_judge_score
object_inventory_score
color_palette_score
edge_layout_score
text_similarity_score
caption_fact_score
ensemble_v1_score
```

If a component cannot be computed, set it to null, log the reason, and compute the ensemble from available components after re-normalizing weights.

Secondary metrics:

```text
mean_candidate_ensemble_score
median_candidate_ensemble_score
std_candidate_ensemble_score
best_candidate_index
prompt_length_chars
prompt_length_tokens_if_available
generation_count
cost_estimate_if_available
latency_seconds
failure_count
api_retry_count
```

Optional diagnostic metric:

```text
target_image_to_prompt_embedding_cosine
```

This compares the target image embedding to the final prompt embedding. It is diagnostic only and must not replace generated-image similarity.

## Similarity functions

Use cosine similarity for embedding comparisons:

```text
cosine(a, b) = dot(a, b) / (norm(a) * norm(b))
```

Normalize cosine similarity to `[0, 1]` before mixing it into ensemble scores:

```text
normalized_cosine = (cosine + 1.0) / 2.0
```

Clamp all component scores to `[0, 1]` before computing the ensemble.

The score for one target image is the maximum candidate ensemble score:

```text
score = max ensemble_v1_score(target_image, generated_candidate_i)
```

The score for one run is:

```text
run_score = mean(score over target images)
```

Higher is better.

## Scoring limitation mitigation

Do not rely only on global multimodal embedding similarity. Global embeddings tend to capture semantic similarity better than exact visual reconstruction.

The evaluator must support multiple scoring modes:

```text
global_embedding_v1
regional_embedding_v1
vlm_pairwise_judge_v1
object_inventory_v1
color_layout_v1
ensemble_v1
```

Primary benchmark metric:

```text
ensemble_v1_score
```

Keep global embedding score as a diagnostic metric.

### Ensemble scorer

Compute:

```text
ensemble_v1_score =
  0.30 * global_embedding_score
+ 0.20 * regional_embedding_score
+ 0.15 * vlm_pairwise_judge_score
+ 0.15 * object_inventory_score
+ 0.10 * color_palette_score
+ 0.05 * edge_layout_score
+ 0.05 * text_similarity_score
```

All component scores must be normalized to `[0, 1]`.

If one or more metrics are unavailable, re-normalize weights across the available metrics and log which metrics were missing.

Example:

```text
available_weight_sum = sum(weights for available metrics)
ensemble = sum(metric_score * metric_weight / available_weight_sum)
```

### Anti-gaming rule

A prompt strategy should not be considered better unless it improves both:

```text
ensemble_v1_score
```

and at least one of:

```text
regional_embedding_score
vlm_pairwise_judge_score
object_inventory_score
```

This prevents optimization from exploiting only the global embedding metric.

## Global embedding score

Embed the full target image and the full generated image with `gemini-embedding-2`.

Compute:

```text
global_embedding_score = normalized_cosine(target_full_embedding, generated_full_embedding)
```

Use PNG or JPEG inputs. Convert generated outputs to PNG before embedding if necessary.

## Regional embedding score

Create crops from both target and generated images:

```text
full image
center crop
3x3 grid crops
```

Embed each corresponding crop with `gemini-embedding-2`.

Compute:

```text
regional_embedding_score = weighted_mean(
  full_image_similarity,
  center_crop_similarity,
  matching_grid_crop_similarities
)
```

Default weights:

```text
0.40 full image
0.20 center crop
0.40 3x3 grid mean
```

For 3x3 grid crops, compare matching regions only:

```text
target_top_left ↔ generated_top_left
target_top_center ↔ generated_top_center
...
target_bottom_right ↔ generated_bottom_right
```

Implementation requirements:

- Resize generated image to target image aspect ratio before crop comparison, or resize both to a common size.
- Preserve aspect ratio when possible.
- Save crop debug images only when `--debug-crops true` is set.
- Cache crop embeddings by image file hash plus crop name.

## VLM pairwise visual judge score

Use `gemini-3.1-flash-lite-preview` to compare target and generated images directly.

The VLM must return JSON:

```json
{
  "subject_identity": 0.0,
  "object_count": 0.0,
  "pose_action": 0.0,
  "spatial_layout": 0.0,
  "background_scene": 0.0,
  "camera_framing": 0.0,
  "lighting": 0.0,
  "color_palette": 0.0,
  "style_medium": 0.0,
  "small_details": 0.0,
  "overall_visual_match": 0.0,
  "major_mismatches": []
}
```

Each numeric value must be between `0.0` and `1.0`.

The judge prompt must emphasize visual similarity, not artistic quality.

Use this rubric:

```text
You are evaluating whether a generated image visually reconstructs a target image.
Compare target and generated image.
Score each field from 0.0 to 1.0.
Do not reward artistic quality unless it improves match to the target.
Do not reward semantic similarity alone if the layout, count, color, style, or details are wrong.
Return JSON only.
```

Compute:

```text
vlm_pairwise_judge_score = weighted_mean(rubric_fields)
```

Default rubric weights:

```text
subject_identity: 0.15
object_count: 0.10
pose_action: 0.10
spatial_layout: 0.15
background_scene: 0.10
camera_framing: 0.10
lighting: 0.08
color_palette: 0.08
style_medium: 0.06
small_details: 0.08
overall_visual_match: 0.10
```

Normalize weights if they do not sum exactly to `1.0`.

Store the raw VLM judge response in the trace JSON.

## Object inventory score

Use the VLM to extract structured visual facts from both images independently.

The VLM must return JSON:

```json
{
  "main_subjects": [
    {
      "name": "",
      "count": 0,
      "position": "",
      "pose_or_action": "",
      "attributes": [],
      "clothing_or_surface_details": []
    }
  ],
  "secondary_objects": [
    {
      "name": "",
      "count": 0,
      "position": "",
      "attributes": []
    }
  ],
  "background": "",
  "scene_type": "",
  "camera_framing": "",
  "lighting": "",
  "dominant_colors": [],
  "visible_text": [],
  "style_or_medium": "",
  "important_small_details": []
}
```

Compare target inventory and generated inventory.

Compute a normalized overlap score from:

```text
main subject match
main subject count match
secondary object match
relative position match
pose/action match
background/scene match
camera/framing match
lighting match
dominant color match
visible text match
style/medium match
small detail match
```

Implementation can start simple:

1. Serialize both JSON objects into compact text.
2. Ask the VLM to compare them and return a score from `0.0` to `1.0` plus mismatch labels.
3. Later replace or augment this with deterministic field-level matching.

The VLM inventory comparison must return JSON:

```json
{
  "object_inventory_score": 0.0,
  "matched_facts": [],
  "missing_facts": [],
  "extra_facts": [],
  "mismatch_labels": []
}
```

Store both inventories and comparison JSON in the trace.

## Color and layout metrics

Use simple local image processing to compute non-semantic visual metrics.

These metrics intentionally catch differences that embeddings may ignore.

Required local metrics:

```text
color_palette_score
brightness_similarity_score
contrast_similarity_score
edge_layout_score
```

### Color palette score

Compute dominant color similarity between target and generated image.

Minimum implementation:

1. Resize both images to a small common size.
2. Convert to RGB.
3. Compute a normalized RGB histogram for each image.
4. Compare histograms using histogram intersection or cosine similarity.

Return a score in `[0, 1]`.

A better implementation may use k-means dominant colors, but do not require it for the first milestone.

### Brightness and contrast similarity

Compute image brightness and contrast from grayscale images.

```text
brightness = mean(grayscale_pixels)
contrast = std(grayscale_pixels)
```

Convert differences into similarity scores:

```text
brightness_similarity = 1.0 - abs(brightness_a - brightness_b) / 255.0
contrast_similarity = 1.0 - abs(contrast_a - contrast_b) / 128.0
```

Clamp to `[0, 1]`.

### Edge layout score

Compute an edge map for each image.

Minimum implementation without OpenCV:

1. Convert to grayscale.
2. Resize both images to `256x256`.
3. Use simple finite differences or PIL filters to approximate edges.
4. Compare edge maps with cosine similarity or mean absolute difference.

Preferred implementation with OpenCV, if available:

1. Convert to grayscale.
2. Resize both images to `256x256`.
3. Run Canny edge detection.
4. Compare edge maps with normalized intersection or cosine similarity.

Return a score in `[0, 1]`.

### Combined color-layout score

Compute:

```text
color_layout_v1_score =
  0.45 * color_palette_score
+ 0.15 * brightness_similarity_score
+ 0.15 * contrast_similarity_score
+ 0.25 * edge_layout_score
```

## Text/OCR similarity score

If the target image contains visible text, score whether the generated image reproduces it.

Use VLM extraction first. Optional OCR libraries may be added later.

Extract visible text from each image using the VLM:

```json
{
  "visible_text": [
    {
      "text": "",
      "location": "",
      "confidence": "low|medium|high"
    }
  ],
  "has_visible_text": true
}
```

If target has no visible text:

```text
text_similarity_score = 1.0
```

If target has visible text and generated image has none:

```text
text_similarity_score = 0.0
```

Otherwise compare extracted strings with normalized string similarity.

Minimum implementation:

```text
text_similarity_score = token overlap F1 between target visible text and generated visible text
```

Also log whether text rendering failure is likely a generator limitation.

## Caption fact score

Use the VLM to caption both images independently.

For each image, request structured facts:

```json
{
  "caption": "",
  "atomic_visual_facts": [
    "",
    ""
  ]
}
```

Then compare target facts and generated facts.

Minimum implementation:

1. Extract facts for target image.
2. Extract facts for generated image.
3. Ask the VLM to compute fact overlap and missing/extra facts.
4. Return `caption_fact_score` in `[0, 1]`.

The VLM fact comparison must return JSON:

```json
{
  "caption_fact_score": 0.0,
  "matched_facts": [],
  "missing_facts": [],
  "extra_facts": []
}
```

This metric is diagnostic in `ensemble_v1`. Do not overweight it because it is still semantic.

## Failure-aware penalties

Create explicit penalties for common reconstruction failures.

Failure labels:

```text
subject_mismatch
count_mismatch
layout_mismatch
style_mismatch
color_mismatch
lighting_mismatch
text_rendering_failure
identity_failure
missing_small_objects
extra_objects
overly_generic_prompt
prompt_too_long
safety_filter_or_generation_failure
api_failure
```

Penalty defaults:

```text
subject_mismatch: -0.15
count_mismatch: -0.10
layout_mismatch: -0.10
style_mismatch: -0.07
color_mismatch: -0.07
lighting_mismatch: -0.05
text_rendering_failure: -0.05
identity_failure: -0.10
missing_small_objects: -0.05
extra_objects: -0.05
overly_generic_prompt: -0.05
prompt_too_long: -0.03
safety_filter_or_generation_failure: -0.20
api_failure: no score; mark candidate failed
```

Penalty application:

```text
penalized_ensemble_score = clamp(ensemble_v1_score + sum(applicable_penalties), 0, 1)
```

Use `penalized_ensemble_score` to pick the best candidate, but log both raw and penalized scores.

Do not apply penalties unless there is clear evidence from VLM judge, inventory comparison, OCR/text extraction, or generation failure logs.

## Results format

Append to `results.tsv` with these columns:

```text
timestamp
run_id
git_commit
strategy_name
target_image
candidate_count
best_candidate_index
best_raw_ensemble_score
best_penalized_ensemble_score
global_embedding_score
regional_embedding_score
vlm_pairwise_judge_score
object_inventory_score
color_palette_score
edge_layout_score
text_similarity_score
caption_fact_score
mean_candidate_ensemble_score
std_candidate_ensemble_score
best_candidate_path
prompt_path
prompt_length_chars
vlm_model
embedding_model
generator_model
scorer_name
latency_seconds
failure_labels
notes
```

Also save a JSON trace per target:

```text
runs/<run_id>/traces/<target_stem>.json
```

Trace schema:

```json
{
  "target_image": "...",
  "strategy_name": "...",
  "scorer_name": "ensemble_v1",
  "vlm_model": "gemini-3.1-flash-lite-preview",
  "embedding_model": "gemini-embedding-2",
  "generator_model": "gemini-3.1-flash-image-preview",
  "prompt": "...",
  "negative_prompt": "...",
  "vlm_raw_response": "...",
  "target_inventory": {},
  "generated_candidates": [
    {
      "index": 0,
      "path": "...",
      "scores": {
        "global_embedding_score": 0.0,
        "regional_embedding_score": 0.0,
        "vlm_pairwise_judge_score": 0.0,
        "object_inventory_score": 0.0,
        "color_palette_score": 0.0,
        "brightness_similarity_score": 0.0,
        "contrast_similarity_score": 0.0,
        "edge_layout_score": 0.0,
        "text_similarity_score": 0.0,
        "caption_fact_score": 0.0,
        "ensemble_v1_score": 0.0,
        "penalized_ensemble_score": 0.0
      },
      "failure_labels": [],
      "vlm_judge_raw_response": {},
      "generated_inventory": {},
      "errors": []
    }
  ],
  "best_candidate_path": "...",
  "best_candidate_index": 0,
  "best_raw_ensemble_score": 0.0,
  "best_penalized_ensemble_score": 0.0,
  "errors": []
}
```

## Baseline strategy

Implement this first.

Strategy name:

```text
baseline_dense_caption
```

Prompt the VLM with the target image and ask it to produce a dense image-generation prompt.

The VLM instruction should ask for:

- main subject
- subject count
- precise pose/action
- scene and background
- spatial layout
- camera angle
- lens/framing
- lighting
- color palette
- material and texture
- style or medium
- mood
- important small details
- text visible in the image
- constraints to avoid common generation mistakes

The output must be JSON:

```json
{
  "short_caption": "...",
  "dense_caption": "...",
  "generation_prompt": "...",
  "negative_prompt": "...",
  "important_details": ["...", "..."],
  "uncertain_details": ["...", "..."]
}
```

Use `generation_prompt` for image generation.

If the generator supports a negative prompt field, use `negative_prompt`. Otherwise append a short negative section to the main prompt.

## Candidate prompt strategies to explore

After the baseline works, improve `src/prompt_from_image.py`.

Try one strategy at a time and log it clearly.

### Strategy: structured_prompt_v1

Construct the final prompt from labeled sections:

```text
Medium:
Subject:
Composition:
Environment:
Lighting:
Camera:
Color palette:
Texture:
Style:
Fine details:
Negative constraints:
```

Then convert the structured fields into a natural image-generation prompt.

### Strategy: forensic_caption_v1

Ask the VLM to act like a visual forensic analyst.

Focus on exact layout, object relationships, colors, shapes, lighting direction, and camera geometry.

Avoid poetic interpretation.

### Strategy: generator_native_prompt_v1

Ask the VLM to write specifically for the image generation model.

The prompt should be direct, visual, and unambiguous.

Avoid references to “the image” or “the original.”

### Strategy: two_pass_v1

Pass 1:

- produce a neutral caption
- produce a dense visual inventory

Pass 2:

- convert both into a concise generation prompt
- preserve only visually actionable details

### Strategy: critique_and_rewrite_v1

Generate an initial prompt.

Then ask the VLM to critique the prompt for likely image-generation failure modes.

Then rewrite the prompt.

### Strategy: prompt_ensemble_v1

Generate multiple prompt variants for each target:

- literal prompt
- composition-heavy prompt
- style-heavy prompt
- subject-heavy prompt
- concise prompt

Generate images from each variant and score all candidates.

This is more expensive, so keep candidate count low.

### Strategy: metric_feedback_rewrite_v1

After a generated candidate is scored, feed the failure labels and VLM judge mismatches back into the VLM prompt constructor.

Ask it to produce a revised prompt that specifically addresses the mismatches.

Important: the VLM may see the target image, prior prompt, generated image, and score breakdown. The image generator still receives only the revised text prompt.

Limit to one rewrite round by default.

## Guardrails

Do not optimize by cheating.

Forbidden:

- passing the target image into the image generator
- using image editing instead of text-to-image generation
- manually selecting generated images
- altering target images
- tuning prompts for only one hand-picked target image
- deleting bad results
- changing the scorer to inflate scores without improving visual similarity
- using web search grounding during image generation unless a specific experiment explicitly enables it and logs it
- hiding failed API calls or filtered outputs

Allowed:

- changing prompt structure
- changing VLM instructions
- generating multiple candidates
- adding negative prompts
- adding self-critique
- adding structured JSON outputs
- caching embeddings
- improving error handling
- improving logging and reports
- adding non-semantic local visual metrics
- adding VLM pairwise judgment as one component of the evaluator

## Reproducibility

Each run must log:

- timestamp
- git commit hash
- model names
- strategy name
- scorer name
- target image filename
- prompt
- generated image paths
- all component scores
- ensemble score
- penalized score
- errors
- failure labels

Use deterministic naming for saved files.

If Gemini image generation exposes seed or deterministic controls, record them. If it does not, record that generation is stochastic and rely on multiple candidates.

## Cost control

Default settings:

```text
MAX_TARGET_IMAGES = 10
CANDIDATES_PER_TARGET = 4
MAX_PROMPT_CHARS = 4000
SCORER = ensemble_v1
```

For exploratory runs, use:

```text
MAX_TARGET_IMAGES = 3
CANDIDATES_PER_TARGET = 2
```

For benchmark runs, use:

```text
MAX_TARGET_IMAGES = all
CANDIDATES_PER_TARGET = 8
```

Cost-heavy metrics:

```text
vlm_pairwise_judge_v1
object_inventory_v1
caption_fact_score
metric_feedback_rewrite_v1
```

Do not increase cost-heavy settings unless the previous run shows a real improvement.

## Caching

Cache target embeddings by file hash:

```text
.cache/embeddings/<model>/<sha256>/full.npy
.cache/embeddings/<model>/<sha256>/center.npy
.cache/embeddings/<model>/<sha256>/grid_r0_c0.npy
```

Cache generated image embeddings by file hash too.

Cache VLM inventory extraction by:

```text
.cache/inventories/<model>/<image_sha256>.json
```

Cache local visual metrics by target hash plus generated hash:

```text
.cache/visual_metrics/<target_sha256>_<generated_sha256>.json
```

Never cache prompts by target alone, because prompt strategies change.

## Evaluation protocol

For each new strategy:

1. Run on the small exploratory subset.
2. Compare against the baseline using `ensemble_v1_score`.
3. Check whether at least one non-global metric improved.
4. If improved, run on the full benchmark set.
5. Log both exploratory and benchmark results.
6. Keep the code change only if the benchmark score improves or the failure analysis is valuable.

A strategy is considered promising if:

```text
mean(best_penalized_ensemble_score) improves by at least 0.01
```

and at least one of these also improves:

```text
regional_embedding_score
vlm_pairwise_judge_score
object_inventory_score
```

A strategy is considered suspicious if:

```text
global_embedding_score improves
```

but all visual/layout/detail metrics get worse.

## Human calibration set

Even though the loop is automated, maintain a small human calibration set.

Create:

```text
data/human_calibration_pairs.jsonl
```

Each row should contain:

```json
{
  "target_image": "...",
  "candidate_a": "...",
  "candidate_b": "...",
  "human_preference": "a|b|tie",
  "notes": "..."
}
```

Periodically compare automated metrics against human preference.

Report:

```text
agreement_rate_global_embedding
agreement_rate_ensemble_v1
agreement_rate_vlm_pairwise_judge
```

Do not block implementation on human calibration. The first milestone should run without it.

## Reporting

After each run, create:

```text
runs/<run_id>/summary.md
```

Include:

- overall score
- per-image best scores
- component score averages
- best and worst examples
- prompt examples
- generated image links or paths
- observed failure modes
- whether global embedding and ensemble disagreed
- next experiment proposal

Also update `notes.md` with a short entry:

```text
## <timestamp> <strategy_name>

Hypothesis:
Change:
Result:
Metric breakdown:
Failure modes:
Global-vs-ensemble disagreement:
Next:
```

## Implementation notes

### Gemini client

Create `src/gemini_client.py` with wrappers:

```python
caption_image_with_vlm(image_path, instruction) -> dict
generate_image_from_prompt(prompt, negative_prompt=None, n=1) -> list[Path]
embed_image(image_path) -> np.ndarray
embed_text(text) -> np.ndarray
judge_image_pair(target_image_path, generated_image_path, instruction) -> dict
extract_visual_inventory(image_path) -> dict
extract_visible_text(image_path) -> dict
```

The wrappers should:

- handle retries
- save raw responses
- fail gracefully
- use local caching where safe
- preserve original API errors in trace files
- validate that JSON responses parse
- retry with a JSON-repair instruction if the first VLM response is invalid JSON

### Image format

Use PNG or JPEG for embedding inputs.

If a generated image is returned in another format, convert it to PNG before scoring.

### JSON validation

For every VLM JSON response:

1. Parse JSON.
2. Check required keys.
3. Clamp numeric scores to `[0, 1]`.
4. Save raw response and parsed response.
5. Log parse failures.

### Error handling

If prompt construction fails:

- mark target as failed
- log error
- continue to next target

If one generated candidate fails:

- mark candidate as failed
- continue with other candidates

If all candidates fail:

- set target score to null
- log `safety_filter_or_generation_failure` or `api_failure`

## First milestone

Implement a working baseline end to end:

```bash
python -m src.run_experiment --strategy baseline_dense_caption --max-targets 3 --candidates 2 --scorer global_embedding_v1
```

Success criteria:

- target images load
- VLM produces valid JSON
- generator creates images
- embeddings are computed
- cosine scores are logged
- generated images are saved
- `results.tsv` is appended
- `summary.md` is created

## Second milestone

Implement `ensemble_v1`:

```bash
python -m src.run_experiment --strategy baseline_dense_caption --max-targets 3 --candidates 2 --scorer ensemble_v1
```

Success criteria:

- global embedding score works
- regional embedding score works
- VLM pairwise judge score works
- object inventory score works or cleanly logs unavailable status
- color palette score works
- edge layout score works
- text similarity score works or cleanly logs no visible text
- raw and penalized ensemble scores are logged
- trace JSON contains full score breakdown

## Third milestone

Run the full baseline:

```bash
python -m src.run_experiment --strategy baseline_dense_caption --max-targets all --candidates 4 --scorer ensemble_v1
```

Record the baseline score. Do not start optimizing prompt strategies until the baseline and ensemble scorer are stable.

## Fourth milestone

Try at least three prompt strategies:

```text
structured_prompt_v1
forensic_caption_v1
critique_and_rewrite_v1
```

Compare them against baseline using:

```text
best_penalized_ensemble_score
regional_embedding_score
vlm_pairwise_judge_score
object_inventory_score
```

Do not declare a winner based only on global embedding score.

## Research loop

Repeat:

1. Inspect worst-scoring target images.
2. Read their prompts, generated outputs, score breakdowns, and VLM mismatch explanations.
3. Identify the likely failure mode.
4. Modify the prompt-construction strategy.
5. Run a small experiment.
6. If promising, run a full benchmark.
7. Log results.
8. Keep changes only if they improve the benchmark or reveal useful failure analysis.

## Final deliverable

The final repo should contain:

- working experiment runner
- baseline prompt strategy
- ensemble visual scorer
- at least three explored prompt strategies
- `results.tsv`
- run summaries
- failure analysis
- recommended best strategy
- examples of target/generated image pairs
- comparison of global embedding score versus ensemble score
- clear README instructions

The final report should explain:

- which prompt strategy worked best
- how much it improved over baseline
- which metric components improved
- which metric components disagreed
- whether Gemini embedding similarity aligned with human visual judgment
- which failure modes remain
- what to try next
