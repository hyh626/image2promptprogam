# Vision-Ratchet Autoresearch

Vision-Ratchet is an autonomous prompt-engineering loop for discovering the
text prompt needed to reproduce a target image with a fixed image generator.

The harness has already been implemented when this document is used by the
research/driver agent. The driver edits `prompt.txt`, runs the harness, reads
the score trajectory, and decides whether to continue, restart, or stop.

## Variance-Control Evaluation Rule

For every target image/task, future implementation agents should avoid drawing
conclusions from one VLM caption and one generated image. Recent variance
studies found that robust evaluation should use nested sampling:

1. Generate **M >= 3 independent VLM captions/decompositions** of the target
   image.
2. For **each** caption/prompt, generate **N >= 2 independent images** with
   distinct generation seeds.
3. Score all `M × N` generated images against the target. Aggregate per caption
   and across captions, and record both mean score and variance.
4. Promote or report a prompt only when the averaged M-by-N result is robust.

Single-caption or single-generation runs are acceptable only as smoke tests for
plumbing and must be labeled as such. Confirm any apparent winner with at least
3 VLM captions and 2 generations per caption before treating it as evidence.

## Canonical Data Storage

`EVAL_STORAGE_SCHEMA.md` is authoritative for on-disk data and run artifacts.
For this single-target workflow, place the target image under:

```text
eval_data/images/eval/
```

`target_image.png` is only a compatibility shortcut to the canonical target
file. Durable run outputs should be bridged into `experiments/` so:

```bash
python check_eval_storage.py --root .
```

can pass after eval artifacts exist.

## Workflow

1. Put the target image in `eval_data/images/eval/`.
2. Keep or create `target_image.png` only as a shortcut to that image if older
   commands require it.
3. Write a foundational prompt in `prompt.txt`.
4. Run the harness:

   ```bash
   python main.py --iterations 100
   ```

The harness should generate at least three independent VLM captions per target
and at least two generated images per caption, score every generation against
the canonical target image, keep prompt changes that improve the aggregate
M-by-N score, and revert prompt changes that regress.

## Artifacts

- `workspace/`: compatibility/debug generated images.
- `history.log`: compatibility/debug score trajectory.
- `experiments/`: canonical durable run artifacts.
- `prompt.txt`: final optimized prompt at the end of the run.

## Kickoff Prompt

Paste the contents of
[`autoresearch-kickoff-prompt.txt`](autoresearch-kickoff-prompt.txt)
as the first message to the driver agent.
