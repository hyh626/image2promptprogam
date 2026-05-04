#!/usr/bin/env bash
set -euo pipefail

DATA_URI="${DATA_URI:-gs://image2promptdata/data}"

usage() {
  cat <<'EOF'
Usage:
  ./prepare_agent_task.sh <source-subfolder> <target-folder> [options]

Copies one spec subfolder into a standalone target folder and downloads the
visible train/eval image data from GCS. The holdout split is intentionally
not copied. The source subfolder's single-repro-prompt.txt is also excluded.

Arguments:
  source-subfolder   Repo subfolder to copy, e.g. opus4.7, gpt5.5, gemini3.1pro
  target-folder      Destination folder to create

Options:
  --skip-data        Copy only the spec files; do not download GCS data
  --no-overwrite     Fail if target-folder already exists and is not empty
  --overwrite        Replace target-folder if it already exists (default)
  -h, --help         Show this help

Environment:
  DATA_URI           GCS data root. Default: gs://image2promptdata/data

Examples:
  ./prepare_agent_task.sh opus4.7 /tmp/agent-harness-opus
  DATA_URI=gs://my-bucket/data ./prepare_agent_task.sh gpt5.5 ./agent-harness-gpt
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

copy_gcs_dir() {
  local remote="$1"
  local dest="$2"

  mkdir -p "$dest"

  if have_cmd gcloud; then
    gcloud storage rsync --recursive "$remote" "$dest"
  elif have_cmd gsutil; then
    gsutil -m rsync -r "$remote" "$dest"
  else
    die "install gcloud or gsutil to copy data from GCS, or rerun with --skip-data"
  fi
}

copy_source_dir() {
  local source="$1"
  local dest="$2"

  (
    shopt -s dotglob nullglob

    local item
    local base
    for item in "$source"/*; do
      base="$(basename -- "$item")"
      if [[ "$base" == "single-repro-prompt.txt" ]]; then
        continue
      fi
      cp -R "$item" "$dest"/
    done
  )
}

copy_file_if_missing() {
  local source_path="$1"
  local dest_path="$2"

  if [[ -e "$dest_path" || -L "$dest_path" ]]; then
    return 0
  fi

  cp "$source_path" "$dest_path"
}

copy_dir_if_missing() {
  local source_path="$1"
  local dest_path="$2"

  if [[ -e "$dest_path" || -L "$dest_path" ]]; then
    return 0
  fi

  cp -R "$source_path" "$dest_path"
}

SOURCE_ARG=""
TARGET_ARG=""
SKIP_DATA=0
OVERWRITE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --no-overwrite)
      OVERWRITE=0
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -z "$SOURCE_ARG" ]]; then
        SOURCE_ARG="$1"
      elif [[ -z "$TARGET_ARG" ]]; then
        TARGET_ARG="$1"
      else
        die "too many positional arguments"
      fi
      shift
      ;;
  esac
done

[[ -n "$SOURCE_ARG" ]] || { usage; die "missing source-subfolder"; }
[[ -n "$TARGET_ARG" ]] || { usage; die "missing target-folder"; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "$SOURCE_ARG" ]]; then
  SOURCE_DIR="$(cd -- "$SOURCE_ARG" && pwd)"
elif [[ -d "$SCRIPT_DIR/$SOURCE_ARG" ]]; then
  SOURCE_DIR="$(cd -- "$SCRIPT_DIR/$SOURCE_ARG" && pwd)"
else
  die "source subfolder does not exist: $SOURCE_ARG"
fi

[[ -f "$SOURCE_DIR/program.md" ]] || die "source must contain program.md: $SOURCE_DIR"
[[ -f "$SOURCE_DIR/IMPLEMENTATION.md" ]] || die "source must contain IMPLEMENTATION.md: $SOURCE_DIR"

TARGET_PARENT="$(mkdir -p -- "$(dirname -- "$TARGET_ARG")" && cd -- "$(dirname -- "$TARGET_ARG")" && pwd)"
TARGET_DIR="$TARGET_PARENT/$(basename -- "$TARGET_ARG")"

case "$TARGET_DIR/" in
  "$SOURCE_DIR"/*)
    die "target folder must not be inside the source folder"
    ;;
esac

[[ "$TARGET_DIR" != "$SOURCE_DIR" ]] || die "target folder must not be the source folder"
[[ "$TARGET_DIR" != "$SCRIPT_DIR" ]] || die "target folder must not be the repo root"

if [[ -e "$TARGET_DIR" ]]; then
  if [[ "$OVERWRITE" -eq 1 ]]; then
    info "Removing existing target: $TARGET_DIR"
    rm -rf -- "$TARGET_DIR"
  elif [[ -n "$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
    die "target exists and is not empty: $TARGET_DIR"
  fi
fi

mkdir -p "$TARGET_DIR"

info "Copying spec from $SOURCE_DIR"
copy_source_dir "$SOURCE_DIR" "$TARGET_DIR"

if [[ -f "$TARGET_DIR/program.md" ]]; then
  copy_file_if_missing "$TARGET_DIR/program.md" "$TARGET_DIR/AGENTS.md"
  copy_file_if_missing "$TARGET_DIR/program.md" "$TARGET_DIR/CLAUDE.md"
  copy_file_if_missing "$TARGET_DIR/program.md" "$TARGET_DIR/GEMINI.md"
fi

if [[ "$SKIP_DATA" -eq 0 ]]; then
  DATA_ROOT="$TARGET_DIR/data"
  mkdir -p "$DATA_ROOT"

  for split in train eval; do
    info "Copying $DATA_URI/$split -> $DATA_ROOT/$split"
    copy_gcs_dir "${DATA_URI%/}/$split" "$DATA_ROOT/$split"
  done

  cat > "$DATA_ROOT/README.md" <<EOF
# Data

This folder was prepared from:

- train: ${DATA_URI%/}/train
- eval: ${DATA_URI%/}/eval

The holdout split is intentionally not copied into this task folder.
EOF

  if grep -Eq "eval_images/|val_images/" "$TARGET_DIR/program.md" "$TARGET_DIR/IMPLEMENTATION.md" 2>/dev/null; then
    info "Creating opus-style image directory copies"
    copy_dir_if_missing "$DATA_ROOT/train" "$TARGET_DIR/eval_images"
    copy_dir_if_missing "$DATA_ROOT/eval" "$TARGET_DIR/val_images"
  fi
else
  info "Skipping GCS data copy"
fi

info "Prepared standalone task folder: $TARGET_DIR"
info "Holdout data was not copied."
