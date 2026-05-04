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
target_image.png    # Target image to reproduce
workspace/          # Generated images, e.g. workspace/gen_001.png
history.log         # CSV log of iteration, score, prompt, and action
```

## 3. Generation

Read `prompt.txt`, call `gemini-3.1-flash-image-preview`, and save the
returned image to `workspace/gen_<iteration>.png`.

Do not force every generation to square. Inspect the target image dimensions,
choose the nearest supported aspect ratio, and pass it through the Gemini image
generation model's `aspect_ratio` parameter. For example, use `1:1` for square
targets, `4:3` or `16:9` for landscape targets, and `3:4` or `9:16` for
portrait targets.

## 4. Core Metric: Patch-Based Similarity

Whole-image multimodal embeddings can over-reward broad semantic similarity.
Implement patch-based scoring:

1. Load `target_image.png` and the newly generated image.
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

1. Generate an image from the current `prompt.txt`.
2. Score it against `target_image.png`.
3. If the score improves, keep the prompt and update `best_score`.
4. If the score does not improve, restore `prompt.txt` to the last best prompt.
5. Log the iteration to `history.log`.
6. Ask `gemini-3.1-flash-lite-preview` to propose one targeted prompt edit.
7. Write the VLM output back to `prompt.txt`.

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
--target      default target_image.png
```

Fail clearly if:

- `target_image.png` is missing,
- `prompt.txt` is missing,
- `GOOGLE_CLOUD_PROJECT` or Vertex AI auth is unavailable,
- a generation request is blocked or returns no image,
- an embedding/scoring call fails after retries.

Handle API rate limits and transient errors with retry/backoff.
