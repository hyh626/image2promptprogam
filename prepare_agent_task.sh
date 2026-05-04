#!/usr/bin/env bash
set -euo pipefail

DATA_URI="${DATA_URI:-gs://image2promptdata/data}"

usage() {
  cat <<'EOF'
Usage:
  ./prepare_agent_task.sh <source-subfolder> <experiment-name> [options]
  ./prepare_agent_task.sh <source-subfolder> <root-folder> <experiment-name> [options]

Copies one spec subfolder into a standalone experiment folder under
<root-folder>/exp_implementation/<experiment-name> and downloads the visible
GCS train/eval splits into the canonical eval storage folders:
eval_data/images/eval and eval_data/images/val. When <root-folder> is omitted,
it defaults to the current directory, so the usual output is
./exp_implementation/<experiment-name>. The holdout split is intentionally not
copied. The source subfolder's single-repro-prompt.txt is also excluded. The
canonical eval storage schema and checker are copied from the repo root.
Older image paths are created only as compatibility symlinks when the selected
spec still mentions them.

Arguments:
  source-subfolder   Repo subfolder to copy, e.g. opus4.7, gpt5.5, gemini3.1pro
  root-folder        Optional destination root; only exp_implementation/ is written under it
  experiment-name    Safe folder name, e.g. opus4.7_gpt5.5_baseline

Options:
  --root DIR         Destination root. Default: current directory
  --skip-data        Copy only the spec files; do not download GCS data
  --no-overwrite     Fail if the experiment folder already exists and is not empty
  --overwrite        Replace the experiment folder if it already exists (default)
  -h, --help         Show this help

Environment:
  DATA_URI           GCS or local data root. Default: gs://image2promptdata/data

Examples:
  ./prepare_agent_task.sh opus4.7 opus4.7_gpt5.5_baseline
  ./prepare_agent_task.sh gpt5.5 gpt5.5_claude_run01 --root ./agent-tasks
  DATA_URI=gs://my-bucket/data ./prepare_agent_task.sh gpt5.5 ./agent-tasks gpt5.5_claude_run01
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

  if [[ -d "$remote" ]]; then
    copy_dir_contents "$remote" "$dest"
    return 0
  fi

  if have_cmd gcloud; then
    gcloud storage rsync --recursive "$remote" "$dest"
  elif have_cmd gsutil; then
    gsutil -m rsync -r "$remote" "$dest"
  else
    die "install gcloud or gsutil to copy data from GCS, or rerun with --skip-data"
  fi
}

copy_dir_contents() {
  local source="$1"
  local dest="$2"

  (
    shopt -s dotglob nullglob

    local item
    for item in "$source"/*; do
      cp -R "$item" "$dest"/
    done
  )
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

link_dir_if_missing() {
  local source_path="$1"
  local dest_path="$2"

  if [[ -e "$dest_path" || -L "$dest_path" ]]; then
    return 0
  fi

  ln -s "$source_path" "$dest_path"
}

ensure_eval_storage_dirs() {
  local image_root="$1"

  mkdir -p "$image_root/train" "$image_root/eval" "$image_root/val" "$image_root/holdout"
}

generate_eval_manifest() {
  local image_root="$1"

  python3 - "$image_root" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SPLITS = ("train", "eval", "val", "holdout")


def image_id_for(path: Path) -> str:
    stem = path.stem.strip().lower()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^a-z0-9_.-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_.-")
    return stem or "image"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def png_size(data: bytes) -> tuple[int, int] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    return None


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    if not data.startswith(b"\xff\xd8"):
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        if marker in {0xD8, 0xD9}:
            continue
        if i + 2 > len(data):
            return None
        length = struct.unpack(">H", data[i:i + 2])[0]
        if length < 2 or i + length > len(data):
            return None
        if marker in {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }:
            height = struct.unpack(">H", data[i + 3:i + 5])[0]
            width = struct.unpack(">H", data[i + 5:i + 7])[0]
            return width, height
        i += length
    return None


def webp_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30 or not (data.startswith(b"RIFF") and data[8:12] == b"WEBP"):
        return None
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 " and len(data) >= 30:
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height
    if chunk == b"VP8L" and len(data) >= 25:
        b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
        width = 1 + (((b1 & 0x3F) << 8) | b0)
        height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
        return width, height
    return None


def image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    size = png_size(data) or jpeg_size(data) or webp_size(data)
    if size is None:
        # The checker requires positive dimensions. Unknown formats should be
        # rare here, but keep the manifest structurally valid.
        return 1, 1
    return size


image_root = Path(sys.argv[1])
manifest = {"schema_version": "1.0.0", "splits": {split: [] for split in SPLITS}}

for split in SPLITS:
    split_dir = image_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in split_dir.rglob("*") if p.is_file()):
        if path.name == ".gitkeep" or path.suffix.lower() not in IMAGE_EXTS:
            continue
        width, height = image_size(path)
        manifest["splits"][split].append(
            {
                "image_id": image_id_for(path),
                "filename": str(path.relative_to(split_dir)),
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
            }
        )

(image_root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

write_eval_data_readme() {
  local eval_data_root="$1"
  local data_status="$2"

  mkdir -p "$eval_data_root"

  cat > "$eval_data_root/README.md" <<EOF
# Eval Data

Data copy status: $data_status

Canonical image splits:

- eval_data/images/eval: 20-image evaluation split, copied from ${DATA_URI%/}/train when data copy is enabled
- eval_data/images/val: 5-image validation split, copied from ${DATA_URI%/}/eval when data copy is enabled

The schema also defines eval_data/images/train and eval_data/images/holdout.
They may stay empty for this harness. Holdout data is intentionally not copied
into prepared task folders.

Compatibility shortcuts may also exist, depending on the selected spec:

- eval_images -> eval_data/images/eval
- val_images -> eval_data/images/val
- target_image.png copied from the first image in eval_data/images/eval for single-target specs
EOF
}

write_agent_context_file() {
  local dest_path="$1"

  if [[ -e "$dest_path" || -L "$dest_path" ]]; then
    return 0
  fi

  cat > "$dest_path" <<'EOF'
# Implementation Agent Context

This prepared folder is currently in the implementation/bootstrap phase.

Read `kickoff-prompt.txt`, then follow `IMPLEMENTATION.md`. Also read
`program.md` for the user-facing research contract and
`EVAL_STORAGE_SCHEMA.md` for canonical eval artifact storage.

If `program.md` says the harness already exists, only `prompt_strategy.py` may
be edited, or the harness must not be modified, treat that as instructions for
the later research/driver phase. During this implementation phase,
`IMPLEMENTATION.md` is the implementation contract.

For eval storage layout and metadata files, `EVAL_STORAGE_SCHEMA.md` wins over
all other docs. Keep `check_eval_storage.py` available and run
`python check_eval_storage.py --root .` after producing eval artifacts.

After the harness is built and handed off, replace this context file with a
copy or symlink to `program.md` before starting the research/driver agent.
EOF
}

docs_mention() {
  local pattern="$1"

  grep -Eq "$pattern" "$TARGET_DIR/program.md" "$TARGET_DIR/IMPLEMENTATION.md" 2>/dev/null
}

first_image_in_dir() {
  local source="$1"
  local image

  while IFS= read -r image; do
    printf '%s\n' "$image"
    return 0
  done < <(
    find "$source" -type f \( \
      -iname '*.png' -o \
      -iname '*.jpg' -o \
      -iname '*.jpeg' -o \
      -iname '*.webp' \
    \) -print | sort
  )
}

create_single_target_files() {
  local first_image

  first_image="$(first_image_in_dir "$EVAL_IMAGE_ROOT/eval")"

  if [[ -n "$first_image" ]]; then
    copy_file_if_missing "$first_image" "$TARGET_DIR/target_image.png"
  else
    info "No eval image found for target_image.png compatibility copy"
  fi

  if [[ ! -e "$TARGET_DIR/prompt.txt" && ! -L "$TARGET_DIR/prompt.txt" ]]; then
    printf '%s\n' "A faithful, detailed reproduction of the target image." > "$TARGET_DIR/prompt.txt"
  fi
}

SOURCE_ARG=""
ROOT_ARG="."
EXPERIMENT_ARG=""
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
    --root)
      [[ $# -ge 2 ]] || die "--root requires a directory"
      ROOT_ARG="$2"
      shift 2
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
      elif [[ -z "$EXPERIMENT_ARG" ]]; then
        EXPERIMENT_ARG="$1"
      elif [[ "$ROOT_ARG" == "." ]]; then
        ROOT_ARG="$EXPERIMENT_ARG"
        EXPERIMENT_ARG="$1"
      else
        die "too many positional arguments"
      fi
      shift
      ;;
  esac
done

[[ -n "$SOURCE_ARG" ]] || { usage; die "missing source-subfolder"; }
[[ -n "$EXPERIMENT_ARG" ]] || { usage; die "missing experiment-name"; }

SLUG_RE='^[A-Za-z0-9][A-Za-z0-9._-]*$'
if [[ ! "$EXPERIMENT_ARG" =~ $SLUG_RE || "$EXPERIMENT_ARG" == *..* ]]; then
  die "experiment-name must be a safe slug using letters, numbers, dot, underscore, or hyphen: $EXPERIMENT_ARG"
fi

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

ROOT_DIR="$(mkdir -p -- "$ROOT_ARG" && cd -- "$ROOT_ARG" && pwd)"
EXPERIMENT_ROOT="$ROOT_DIR/exp_implementation"
mkdir -p "$EXPERIMENT_ROOT"
TARGET_DIR="$EXPERIMENT_ROOT/$EXPERIMENT_ARG"

case "$TARGET_DIR/" in
  "$SOURCE_DIR"/*)
    die "experiment folder must not be inside the source folder"
    ;;
esac

[[ "$TARGET_DIR" != "$SOURCE_DIR" ]] || die "experiment folder must not be the source folder"
[[ "$TARGET_DIR" != "$SCRIPT_DIR" ]] || die "experiment folder must not be the repo root"

if [[ -e "$TARGET_DIR" ]]; then
  if [[ "$OVERWRITE" -eq 1 ]]; then
    info "Removing existing experiment folder: $TARGET_DIR"
    rm -rf -- "$TARGET_DIR"
  elif [[ -n "$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
    die "experiment folder exists and is not empty: $TARGET_DIR"
  fi
fi

mkdir -p "$TARGET_DIR"

info "Copying spec from $SOURCE_DIR"
copy_source_dir "$SOURCE_DIR" "$TARGET_DIR"

info "Copying eval storage schema"
copy_file_if_missing "$SCRIPT_DIR/EVAL_STORAGE_SCHEMA.md" "$TARGET_DIR/EVAL_STORAGE_SCHEMA.md"
copy_file_if_missing "$SCRIPT_DIR/check_eval_storage.py" "$TARGET_DIR/check_eval_storage.py"

if [[ -f "$TARGET_DIR/program.md" ]]; then
  write_agent_context_file "$TARGET_DIR/AGENTS.md"
  write_agent_context_file "$TARGET_DIR/CLAUDE.md"
  write_agent_context_file "$TARGET_DIR/GEMINI.md"
fi

EVAL_DATA_ROOT="$TARGET_DIR/eval_data"
EVAL_IMAGE_ROOT="$EVAL_DATA_ROOT/images"
ensure_eval_storage_dirs "$EVAL_IMAGE_ROOT"

if [[ "$SKIP_DATA" -eq 0 ]]; then
  info "Copying ${DATA_URI%/}/train -> $EVAL_IMAGE_ROOT/eval"
  copy_gcs_dir "${DATA_URI%/}/train" "$EVAL_IMAGE_ROOT/eval"

  info "Copying ${DATA_URI%/}/eval -> $EVAL_IMAGE_ROOT/val"
  copy_gcs_dir "${DATA_URI%/}/eval" "$EVAL_IMAGE_ROOT/val"

  if docs_mention "target_image\\.png"; then
    info "Creating single-target compatibility files"
    create_single_target_files
  fi
else
  info "Skipping GCS data copy"
fi

if [[ "$SKIP_DATA" -eq 0 ]]; then
  write_eval_data_readme "$EVAL_DATA_ROOT" "copied visible GCS train/eval splits into canonical eval_data/images/{eval,val}"
else
  write_eval_data_readme "$EVAL_DATA_ROOT" "skipped by --skip-data; populate eval_data/images/eval and eval_data/images/val manually"
fi

generate_eval_manifest "$EVAL_IMAGE_ROOT"

if docs_mention "eval_images/|val_images/"; then
  info "Creating opus-style image directory symlinks"
  link_dir_if_missing "eval_data/images/eval" "$TARGET_DIR/eval_images"
  link_dir_if_missing "eval_data/images/val" "$TARGET_DIR/val_images"
fi

if docs_mention "data/targets"; then
  info "Creating data/targets compatibility symlinks"
  mkdir -p "$TARGET_DIR/data/targets"
  link_dir_if_missing "../../eval_data/images/eval" "$TARGET_DIR/data/targets/eval"
  link_dir_if_missing "../../eval_data/images/val" "$TARGET_DIR/data/targets/val"
fi

info "Prepared standalone task folder: $TARGET_DIR"
info "Holdout data was not copied."
