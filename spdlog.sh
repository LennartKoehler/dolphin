#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

# file extensions to process
exts=(cpp cc cxx c++ c m h hh hpp)

find_args=()
for e in "${exts[@]}"; do find_args+=( -o -iname "*.${e}" ); done
find_args=( "${find_args[@]:1}" )

echo "Scanning ${ROOT} ..."

# Insert spdlog include if missing
ensure_include() {
  local f="$1"
  if ! grep -qE '#include\s*<spdlog/spdlog.h>' "$f"; then
    # insert after last include or at top
    last=$(grep -n '^#include' "$f" | tail -n1 | cut -d: -f1 || true)
    if [ -n "$last" ]; then
      awk -v n="$last" 'NR==n{print; print "#include <spdlog/spdlog.h>"; next} {print}' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
    else
      printf '%s\n%s\n' '#include <spdlog/spdlog.h>' "$(cat "$f")" > "$f.tmp" && mv "$f.tmp" "$f"
    fi
  fi
}

# Transform expressions
# - Finds occurrences like: std::cout << "..." << var << "..." << std::endl;
# - Uses the first string literal's [LEVEL] token to pick spdlog level (INFO/ERROR/WARNING/WARN)
# - Removes the leading [LEVEL] token from the message text
# - Replaces subsequent expression insertions with {} placeholders and passes them as args
# - Removes std::endl/std::flush manipulators
# Caveat: heuristic; may match in comments/macros or complex raw-string delimiters.
find "$ROOT" -type f \( "${find_args[@]}" \) | while IFS= read -r file; do
  # skip binaries
  if file "$file" | grep -qiE 'executable|compressed|image|directory'; then continue; fi
  if ! grep -qE 'std::cout|std::cerr|std::clog' "$file"; then continue; fi

  echo "Processing: $file"
  # backup and edit in-place with perl
  perl -0777 -i.bak -pe '

    sub trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s }

    sub unquote_double {
      my $s = shift;
      if ($s =~ /^"(.*)"$/s) { my $inner = $1; $inner =~ s/\\"/"/g; return $inner }
      return undef;
    }

    sub unquote_raw_R {
      my $s = shift;
      if ($s =~ /^R"\((.*)\)"$/s) { return $1 }
      return undef;
    }

    sub escape_braces {
      my $t = shift;
      $t =~ s/\{/{{/g;
      $t =~ s/\}/}}/g;
      return $t;
    }

    sub build_spdlog {
      my ($stream, $body) = @_;
      # split on << (keep contents)
      my @parts = split(/\<\<\s*/, $body);
      # normalize and drop empty leading parts
      @parts = map { s/^\s+|\s+$//g; $_ } @parts;
      shift @parts if @parts && $parts[0] eq "";

      # find first string literal index
      my $first_idx = -1;
      for my $i (0..$#parts) {
        my $p = $parts[$i];
        my $lit = unquote_double($p);
        $lit = unquote_raw_R($p) unless defined $lit;
        if (defined $lit) { $first_idx = $i; last }
      }

      # determine level
      my $level = ($stream eq "cout" ? "info" : $stream eq "cerr" ? "error" : "debug");
      if ($first_idx >= 0) {
        my $first_text = unquote_double($parts[$first_idx]);
        $first_text = unquote_raw_R($parts[$first_idx]) unless defined $first_text;
        if ($first_text =~ /\[(INFO|ERROR|WARNING|WARN)\]/i) {
          my $tok = uc($1);
          $level = $tok eq "WARNING" || $tok eq "WARN" ? "warn" : lc($tok);
          # remove first [LEVEL] token from text only once
          $first_text =~ s/^\s*\[(?:INFO|ERROR|WARNING|WARN)\]\s*//i;
          $parts[$first_idx] = "\"".($first_text). "\"";
        }
      }

      # build fmt and args
      my $fmt = "";
      my @args = ();

      # start with first literal if exists, else empty
      if ($first_idx >= 0) {
        # append first literal content (after removal above)
        my $lit = unquote_double($parts[$first_idx]) // unquote_raw_R($parts[$first_idx]) // "";
        $fmt .= $lit;
      }

      # process remaining parts after the first literal (or all parts if no literal found):
      my @iter;
      if ($first_idx >= 0) { @iter = @parts[$first_idx+1 .. $#parts] } else { @iter = @parts }
      foreach my $p (@iter) {
        next unless defined $p;
        # skip manipulators
        next if $p =~ /\bstd::endl\b/ || $p =~ /\bendl\b/ || $p =~ /\bstd::flush\b/;
        # string literal?
        my $lit = unquote_double($p);
        $lit = unquote_raw_R($p) unless defined $lit;
        if (defined $lit) {
          $fmt .= $lit;
        } else {
          # expression -> placeholder (strip trailing semicolons)
          my $expr = $p;
          $expr =~ s/;\s*$//;
          $expr =~ s/^\s+|\s+$//g;
          next if $expr eq "";
          $fmt .= "{}";
          push @args, $expr;
        }
      }

      # escape braces and double quotes in format
      $fmt = escape_braces($fmt);
      $fmt =~ s/"/\\"/g;
      my $out = "spdlog::" . $level . "(\"" . $fmt . "\"";
      if (@args) { $out .= ", " . join(", ", @args) }
      $out .= ");";
      return $out;
    }

    # main replacement: match std::cout/cerr/clog followed by one or more << pieces until semicolon
    s{
      \bstd::(cout|cerr|clog)\b          # stream
      (                                 # body of << ... << ... ;
        (?:\s*<<\s* (?: R\"\(.*?\)\" | "(?:\\.|[^\"\\])*" | [^;"]+ ) )+  # non-greedy parts: raw-string, double-quoted, or expressions
      )
      \s*;                               # terminator
    }{ build_spdlog($1, $2) }gesx;

  ' "$file"

  # ensure include
  ensure_include "$file"
done

echo "Done. Backups saved as .bak. Please review changes and run build/tests."