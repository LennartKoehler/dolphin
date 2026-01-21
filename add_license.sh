#!/usr/bin/env bash
set -euo pipefail

# Usage: ./add_license_header.sh [--dry-run]
# Recursively prepend the license header to all .h and .cpp files that don't already contain it.
# --dry-run : only list files that would be changed

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

header='/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/'

# Find all .h and .cpp files below current directory
find . -type f \( -name "*.h" -o -name "*.cpp" \) -print0 |
while IFS= read -r -d '' file; do
  # Skip if file already contains the unique marker
  if grep -q "Copyright by Lennart Koehler" "$file"; then
    continue
  fi

  if $DRY_RUN; then
    printf "Would add header to: %s\n" "$file"
    continue
  fi

  # Preserve file permissions
  perms=$(stat -c "%a" "$file")

  # Create a temporary file in same directory (to keep same FS)
  tmp="$(mktemp "$(dirname "$file")/._tmp.XXXXXX")"
  # Write header + newline + original content
  {
    printf "%s\n\n" "$header"
    cat "$file"
  } > "$tmp"

  mv "$tmp" "$file"
  chmod "$perms" "$file"

  printf "Added header to: %s\n" "$file"
done