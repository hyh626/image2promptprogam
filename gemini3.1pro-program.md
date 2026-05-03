# Project: Vision-Ratchet (Prompt-to-Image Reverse Engineering)

## 1. Objective
Build an autonomous, iterative Python system that discovers the exact text prompt required to mathematically reproduce a target image. The system will use a Multimodal LLM to iteratively edit a prompt, generate an image, and score the visual similarity against the target using patch-based multimodal embeddings. 

## 2. Tech Stack & APIs
You will interact with the Google Gemini API (via the official Python SDK). Use the following specific models:
* **The Researcher (VLM):** `gemini-3.1-flash-lite` (Used for hypothesizing prompt edits based on visual diffs).
* **The Generator:** `nano-banana-2` (Official API name might map to `gemini-3-flash-image`; verify in SDK docs. Used to generate the image from the text prompt).
* **The Scorer:** `text-embedding-004` or the latest Gemini Multimodal Embedding model (Used to embed images for cosine similarity).

## 3. Directory & File Structure
Create and maintain the following structure:
* `main.py`: The core execution loop (The Ratchet).
* `prompt.txt`: The current best text prompt. (This is the only file the VLM is allowed to modify conceptually, though `main.py` will handle the I/O).
* `target_image.png` or `target_image.jpg`: The baseline image we are trying to replicate.
* `workspace/`: A directory to save generations (e.g., `gen_001.png`, `gen_002.png`).
* `history.log`: A CSV tracking `Iteration`, `Cosine_Score`, `Prompt`, and `Action (Kept/Reverted)`.

## 4. The Patch-Based Scoring Metric (CRITICAL)
Do not embed the whole image at once. To prevent "semantic drift" and enforce spatial accuracy, implement the following scoring mechanism in `main.py`:
1.  Take `target_image` and `current_gen_image`.
2.  Resize both to a standard square (e.g., 512x512).
3.  Slice both images into a 2x2 grid (yielding 4 patches per image: Top-Left, Top-Right, Bottom-Left, Bottom-Right).
4.  Generate multimodal embeddings for all 8 patches.
5.  Calculate the Cosine Similarity for each corresponding pair (e.g., Target-TL vs Gen-TL).
6.  The final `Similarity_Score` is the average of these 4 cosine similarity values.

## 5. The Execution Loop (The Ratchet)
Implement a `while` loop in `main.py` that executes the following steps:

**Step A: Generate**
Read `prompt.txt`. Call the Image Generation API to produce `current_gen_image`. Save it to `workspace/`.

**Step B: Score**
Calculate the `Similarity_Score` using the Patch-Based method defined in Section 4.

**Step C: Decide (The Git-style Revert)**
* If `Iteration == 1`, set `best_score = Similarity_Score` and `best_prompt = prompt.txt`.
* If `Similarity_Score > best_score`:
    * Accept the new prompt. 
    * Update `best_score` and `best_prompt`.
    * Log "Kept".
* If `Similarity_Score <= best_score`:
    * Reject the new prompt.
    * Overwrite `prompt.txt` with `best_prompt` (Revert).
    * Log "Reverted".

**Step D: Hypothesize & Edit**
Construct a prompt for `gemini-3.1-flash-lite`. Pass it the `target_image`, the *last accepted* generated image, and the *last accepted* text prompt. 
Use this exact instruction block for the VLM:
> "You are an expert prompt engineer. Your goal is to make the generated image visually identical to the target image. 
> DO NOT try to make the image 'beautiful' or 'high quality' if the target is blurry, low-res, or has artifacts; you must reproduce the flaws.
> The current similarity score is {best_score}. 
> Look at the differences between the target and the generation. Identify ONE specific spatial, textural, or lighting element that is incorrect.
> Rewrite the text prompt to fix this specific element. Output ONLY the raw text of the new prompt, with no markdown or conversational filler."

Overwrite `prompt.txt` with the VLM's output.
Repeat loop.

## 6. Agent Instructions
1.  Begin by writing `main.py` based on the architecture above.
2.  Ensure graceful error handling for API rate limits and timeouts.
3.  Add a command-line argument `--iterations` to set the loop limit (default to 50).
4.  If the target image is not present, halt and prompt the user to place one in the directory.
