#!/usr/bin/env bash
set -euo pipefail

# Usage: remove.bak.sh [-n|--dry-run] [-f|--force] [-v|--verbose] [path]
DRY_RUN=0
FORCE=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DRY_RUN=1; shift;;
    -f|--force) FORCE=1; shift;;
    -v|--verbose) VERBOSE=1; shift;;
    -h|--help) echo "Usage: $0 [-n|--dry-run] [-f|--force] [-v|--verbose] [path]"; exit 0;;
    *) TARGET="$1"; shift;;
  esac
done

TARGET=${TARGET:-.}

# find files ending with .bak
readarray -d '' files < <(find "$TARGET" -type f -name '*.bak' -print0)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No .bak files found under '$TARGET'."
  exit 0
fi

echo "Found ${#files[@]} file(s) ending with .bak under '$TARGET'."
if [[ $DRY_RUN -eq 1 ]]; then
  printf '%s\n' "${files[@]}"
  exit 0
fi

if [[ $FORCE -eq 0 ]]; then
  read -r -p "Delete these ${#files[@]} files? [y/N] " ans
  case "$ans" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted."; exit 1;;
  esac
fi

for f in "${files[@]}"; do
  if [[ $VERBOSE -eq 1 ]]; then
    echo "Removing: $f"
  fi
  rm -- "$f"
done

echo "Done."
