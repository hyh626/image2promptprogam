# Image2Prompt Autoresearch Specs

This repo contains self-contained specifications for testing whether coding
agents can build and run an image-to-prompt autoresearch harness.

The core experiment asks:

> Given a target image, can a vision model produce a text prompt that causes a
> fixed image generator to reproduce that image as closely as possible?

Each model-named subfolder is a different version of the task specification.
The repo also includes a helper script that packages one subfolder plus the
public train/eval data into a standalone folder for a test agent.

## Repository Layout

```text
opus4.7/
  program.md              # Instructions for the research/driver agent
  IMPLEMENTATION.md       # Instructions for the implementation agent
  kickoff-prompt.txt      # Shared implementation kickoff prompt
  autoresearch-kickoff-prompt.txt # Shared autoresearch kickoff prompt
  single-repro-prompt.txt # Source prompt notes; not copied into task folders

gpt5.5/
gemini3.1pro/
  ...                     # Same shape, different spec variants

prepare_agent_task.sh     # Creates a standalone task folder
EVAL_STORAGE_SCHEMA.md    # Canonical eval output storage schema
check_eval_storage.py     # Conformance checker for built task repos
prompts                   # Conversation notes used to create the specs
```

## Data

The train and eval data live in:

```text
gs://image2promptdata/data
```

The helper script copies only:

```text
gs://image2promptdata/data/train -> eval_data/images/eval
gs://image2promptdata/data/eval  -> eval_data/images/val
```

It intentionally does not copy holdout data. Keep holdout private so it can be
used later to evaluate whether an implementation or agent overfit the visible
data.

## Prepare A Standalone Agent Task

Use `prepare_agent_task.sh` to copy one spec subfolder into a local
experiment implementation folder:

```bash
./prepare_agent_task.sh opus4.7 opus4.7_gpt5.5_baseline
```

That creates:

```text
./exp_implementation/opus4.7_gpt5.5_baseline/
```

The standalone experiment folder contains:

- `program.md`
- `IMPLEMENTATION.md`
- `kickoff-prompt.txt`
- `autoresearch-kickoff-prompt.txt`
- `EVAL_STORAGE_SCHEMA.md`
- `check_eval_storage.py`
- implementation-phase agent context files: `AGENTS.md`, `CLAUDE.md`,
  `GEMINI.md`
- `eval_data/images/eval`
- `eval_data/images/val`
- `eval_data/images/manifest.json`

The bucket uses `train`, `eval`, and `holdout` split names. The prepared task
uses the canonical storage schema names: bucket `train` becomes the 20-image
evaluation split at `eval_data/images/eval`, and bucket `eval` becomes the
5-image validation split at `eval_data/images/val`. The schema's `train/` and
`holdout/` directories may stay empty for this harness.

Since some spec variants use older local names, the script may create symlink
compatibility shortcuts when a selected spec mentions them:

```text
target_image.png  # copied from the first image found in eval_data/images/eval
prompt.txt        # starter prompt for single-target specs
```

By default, the script only writes beneath `./exp_implementation/`. You can use
`--root <dir>` or the legacy three-argument form to choose a different root,
but the script still writes only beneath that root's `exp_implementation/`.
The generated experiment folder is overwritten by default. Use `--no-overwrite`
if you want the script to fail instead when the experiment folder already
exists.

## Script Options

```bash
./prepare_agent_task.sh <source-subfolder> <experiment-name> [options]
```

Examples:

```bash
./prepare_agent_task.sh opus4.7 opus4.7_gpt5.5_baseline
./prepare_agent_task.sh gpt5.5 gpt5.5_claude_run01
./prepare_agent_task.sh gemini3.1pro gemini3.1pro_gemini_run01 --skip-data
./prepare_agent_task.sh opus4.7 opus4.7_opus_run02 --root /tmp/custom-task-root
DATA_URI=gs://my-bucket/data ./prepare_agent_task.sh opus4.7 opus4.7_opus_run03
```

Options:

- `--skip-data`: copy only the spec files.
- `--root DIR`: choose a root other than the current directory.
- `--no-overwrite`: fail if the experiment folder exists and is not empty.
- `--overwrite`: replace the experiment folder if it exists. This is the default.

The script uses `gcloud storage rsync` when available, otherwise `gsutil -m
rsync`. Install one of those tools or use `--skip-data`.

## What The Script Excludes

The prepared task folder does not include:

- holdout data
- `single-repro-prompt.txt`

The script creates implementation-phase agent context files instead of
symlinking `program.md` immediately. This avoids auto-loaded driver-agent
instructions conflicting with the implementation kickoff. After the harness is
built, the implementation agent should replace those files with copies or
symlinks to `program.md` for the research/driver session. The script also
copies the canonical eval storage schema and checker from this repo root so the
test agent can implement and validate output persistence without seeing the
rest of the source repo.

## Recommended Workflow

1. Pick the spec variant you want to test, usually `opus4.7` for the most
   complete fixed-harness version.
2. Prepare a fresh task folder:

   ```bash
   ./prepare_agent_task.sh opus4.7 opus4.7_gpt5.5_baseline
   ```

3. Open the generated experiment folder in the coding agent you want to
   evaluate.
4. Paste `kickoff-prompt.txt` into the agent to start the implementation run.
5. After implementation, paste `autoresearch-kickoff-prompt.txt` into the
   research/driver agent to start the experiment loop.
6. Use the visible eval data for smoke tests and development checks.
7. Evaluate final generalization separately with holdout data outside the task
   folder.

Do not point a test agent at this source repo when you want a clean benchmark.
Use a prepared task folder so the agent cannot see source prompt notes or
private holdout data.
