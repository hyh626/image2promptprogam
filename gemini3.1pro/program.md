# Vision-Ratchet Autoresearch

Vision-Ratchet is an autonomous prompt-engineering loop for discovering the
text prompt needed to reproduce a target image with a fixed image generator.

The harness has already been implemented when this document is used by the
research/driver agent. The driver edits `prompt.txt`, runs the harness, reads
the score trajectory, and decides whether to continue, restart, or stop.

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

The harness generates images, scores each generation against the canonical
target image, keeps prompt changes that improve the aggregate score, and
reverts prompt changes that regress.

## Artifacts

- `workspace/`: compatibility/debug generated images.
- `history.log`: compatibility/debug score trajectory.
- `experiments/`: canonical durable run artifacts.
- `prompt.txt`: final optimized prompt at the end of the run.

## Kickoff Prompt

Paste the contents of
[`autoresearch-kickoff-prompt.txt`](autoresearch-kickoff-prompt.txt)
as the first message to the driver agent.
