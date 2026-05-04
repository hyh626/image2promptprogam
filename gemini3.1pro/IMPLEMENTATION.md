# Implementation Guide: Vision-Ratchet Harness

You are an AI coding agent tasked with building `main.py`, the core
evaluation harness for the Vision-Ratchet system.

Vision-Ratchet is an autonomous prompt-engineering system that iteratively
discovers the text prompt needed to reproduce a target image. The harness
orchestrates a Vision-Language Model, an image generator, and visual
similarity metrics through a strict keep-or-revert loop.

See `program.md` for the user-facing workflow.

## 1. Tech Stack And APIs

Use the Google GenAI SDK against Vertex AI for all Gemini model calls:

```python
client = genai.Client(
    vertexai=True,
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)
```

Default Gemini settings:

| Role | Model | Notes |
|---|---|---|
| Researcher / VLM | `gemini-3.1-flash-lite-preview` | Hypothesizes prompt edits from visual differences. |
| Image generator | `gemini-3.1-flash-image-preview` | Generates images from candidate prompts. Supports an `aspect_ratio` parameter. |
| Embedding scorer | `gemini-embedding-2` | Embeds target/generated images or crops for similarity scoring. |

Also use DINOv2 ViT-B/14 (`facebook/dinov2-base`) locally as a structural
similarity metric. DINOv2 should complement Gemini embeddings by catching pose,
layout, and appearance differences that semantic embeddings can miss.

## 2. Directory And File Structure

Your script must manage this topology:

```text
main.py             # Core execution loop
prompt.txt          # Current best prompt
eval_data/images/eval/target_image.png # Canonical target image location
target_image.png    # Optional compatibility symlink/copy to canonical target
workspace/          # Generated images, e.g. workspace/gen_001.png
history.log         # CSV log of iteration, score, prompt, and action
```

`EVAL_STORAGE_SCHEMA.md` is canonical for persisted eval inputs and run
artifacts. Put the target image under `eval_data/images/eval/`; if
`target_image.png` exists, it must be only a compatibility symlink/copy to that
canonical image. Treat `workspace/` and `history.log` as compatibility/debug
surfaces for the Vision-Ratchet loop, and bridge durable run outputs into
`experiments/` so `python check_eval_storage.py --root .` can pass after eval
artifacts exist.

## 3. Generation

Read `prompt.txt`, call `gemini-3.1-flash-image-preview` for each configured
repeat, and save the returned images under `workspace/`.

Each evaluation of an example must generate and score at least three images
from the current prompt. Add a configurable repeat count, for example
`--eval-repeats`, with default `3`, and validate that real eval runs reject
values below `3`. Use distinct generation seeds for the repeats, save each
generated image to a deterministic path such as
`workspace/gen_<iteration>_seed_<seed>.png`, score each generated image, and
aggregate the per-seed scores before the keep-or-revert decision. The harness
must not decide whether a prompt improved from a single generated image.

Do not force every generation to square. Inspect the target image dimensions,
choose the nearest supported aspect ratio, and pass it through the Gemini image
generation model's `aspect_ratio` parameter. For example, use `1:1` for square
targets, `4:3` or `16:9` for landscape targets, and `3:4` or `9:16` for
portrait targets.

## 4. Core Metric: Patch-Based Similarity

Whole-image multimodal embeddings can over-reward broad semantic similarity.
Implement patch-based scoring:

1. Load the configured target image from `eval_data/images/eval/` and the newly generated image.
2. Resize both to a common square for scoring, e.g. 512x512.
3. Slice both into a 2x2 grid.
4. Embed each corresponding crop with `gemini-embedding-2`.
5. Compute cosine similarity for each corresponding crop pair.
6. Average the four crop similarities.

Add a DINOv2 structural similarity score:

1. Extract DINOv2 features for the target and generated image.
2. L2-normalize the vectors.
3. Compute cosine similarity.

The primary score can be a weighted composite of patch embedding similarity and
DINOv2 structural similarity. Keep the weights explicit in code so future
experiments can adjust them deliberately.

## 5. The Ratchet Loop

Implement a loop driven by `--iterations`:

1. Generate `--eval-repeats` images from the current `prompt.txt`.
2. Score each image against the configured canonical target image.
3. Aggregate the repeated scores, using mean composite as the default.
4. If the aggregated score improves, keep the prompt and update `best_score`.
5. If the aggregated score does not improve, restore `prompt.txt` to the last best prompt.
6. Log the iteration and all per-seed scores to `history.log`.
7. Ask `gemini-3.1-flash-lite-preview` to propose one targeted prompt edit.
8. Write the VLM output back to `prompt.txt`.

Use this VLM instruction shape:

```text
You are an expert prompt engineer. Your goal is to make the generated image
visually identical to the target image.
Do not make the image beautiful or high quality if the target is blurry,
low-res, or has artifacts; reproduce the flaws.
The current similarity score is {best_score}.
Look at the differences between the target and the generation. Identify one
specific spatial, textural, color, or lighting element that is incorrect.
Rewrite the text prompt to fix this specific element. Output only the raw text
of the new prompt.
```

## 6. CLI And Error Handling

Add argparse support:

```text
--iterations  default 50
--target      default eval_data/images/eval/target_image.png
--eval-repeats default 3, must be >=3
```

Fail clearly if:

- the configured target image is missing,
- `prompt.txt` is missing,
- `GOOGLE_CLOUD_PROJECT` or Vertex AI auth is unavailable,
- a generation request is blocked or returns no image,
- an embedding/scoring call fails after retries.

Handle API rate limits and transient errors with retry/backoff.
