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
prompts                   # Conversation notes used to create the specs
```

## Data

The train and eval data live in:

```text
gs://image2promptdata/data
```

The helper script copies only:

```text
gs://image2promptdata/data/train
gs://image2promptdata/data/eval
```

It intentionally does not copy holdout data. Keep holdout private so it can be
used later to evaluate whether an implementation or agent overfit the visible
data.

## Prepare A Standalone Agent Task

Use `prepare_agent_task.sh` to copy one spec subfolder into a target folder:

```bash
./prepare_agent_task.sh opus4.7 /tmp/image2prompt-opus-task
```

That creates a standalone folder containing:

- `program.md`
- `IMPLEMENTATION.md`
- `kickoff-prompt.txt`
- `autoresearch-kickoff-prompt.txt`
- copied agent context files: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`
- `data/train`
- `data/eval`

For specs that refer to `eval_images/` and `val_images/`, the script also
copies compatibility directories from the visible data:

```text
eval_images/  # copied from data/train
val_images/   # copied from data/eval
```

The generated folder is overwritten by default. Use `--no-overwrite` if you
want the script to fail instead when the target already exists.

## Script Options

```bash
./prepare_agent_task.sh <source-subfolder> <target-folder> [options]
```

Examples:

```bash
./prepare_agent_task.sh opus4.7 /tmp/image2prompt-opus-task
./prepare_agent_task.sh gpt5.5 /tmp/image2prompt-gpt-task
./prepare_agent_task.sh gemini3.1pro /tmp/image2prompt-gemini-task --skip-data
DATA_URI=gs://my-bucket/data ./prepare_agent_task.sh opus4.7 /tmp/custom-task
```

Options:

- `--skip-data`: copy only the spec files.
- `--no-overwrite`: fail if the target folder exists and is not empty.
- `--overwrite`: replace the target folder if it exists. This is the default.

The script uses `gcloud storage rsync` when available, otherwise `gsutil -m
rsync`. Install one of those tools or use `--skip-data`.

## What The Script Excludes

The prepared task folder does not include:

- holdout data
- `single-repro-prompt.txt`

The script also copies agent context files instead of symlinking them. This
keeps edits made by a test agent contained inside the prepared folder.

## Recommended Workflow

1. Pick the spec variant you want to test, usually `opus4.7` for the most
   complete fixed-harness version.
2. Prepare a fresh task folder:

   ```bash
   ./prepare_agent_task.sh opus4.7 /tmp/image2prompt-agent-task
   ```

3. Open the target folder in the coding agent you want to evaluate.
4. Paste `kickoff-prompt.txt` into the agent to start the implementation run.
5. After implementation, paste `autoresearch-kickoff-prompt.txt` into the
   research/driver agent to start the experiment loop.
6. Use the visible eval data for smoke tests and development checks.
7. Evaluate final generalization separately with holdout data outside the task
   folder.

Do not point a test agent at this source repo when you want a clean benchmark.
Use a prepared task folder so the agent cannot see source prompt notes or
private holdout data.
