#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-.}"

# Only touch common source/header files.
find "$ROOT_DIR" -type f \( \
  -name '*.h' -o -name '*.hpp' -o -name '*.hh' -o -name '*.hxx' -o \
  -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
  -name '*.inl' \
\) -print0 | while IFS= read -r -d '' file; do
  # Replace: #include "path/..." -> #include "dolphin/path/..."
  # But skip includes that already start with dolphin/, nlohmann, or spdlog
  perl -i -pe 's/#include\s+"(?!dolphin\/|nlohmann|spdlog)/#include "dolphin\//g' "$file"
  
  # Special case: remove dolphin/ prefix for dolphinbackend and cpu_backend
  perl -i -pe 's/#include\s+"dolphin\/(dolphinbackend|cpu_backend)\//#include "$1\//g' "$file"
done

echo "Updated includes under: $ROOT_DIR"
