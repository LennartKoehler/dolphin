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
    last=$(grep -n '^#include' "$f" | tail -n1 | cut -d: -f1 || true)
    if [ -n "$last" ]; then
      awk -v n="$last" 'NR==n{print; print "#include <spdlog/spdlog.h>"; next} {print}' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
    else
      printf '%s\n%s\n' '#include <spdlog/spdlog.h>' "$(cat "$f")" > "$f.tmp" && mv "$f.tmp" "$f"
    fi
  fi
}

find "$ROOT" -type f \( "${find_args[@]}" \) | while IFS= read -r file; do
  # skip binaries
  if file "$file" | grep -qiE 'executable|compressed|image|directory'; then continue; fi
  if ! grep -qE 'std::cout|std::cerr|std::clog' "$file"; then continue; fi

  echo "Processing: $file"

  perl -0777 -i.bak -pe '

    sub unquote_double {
      my $s = shift;
      if ($s =~ /^"(.*)"$/s) {
        my $inner = $1;
        $inner =~ s/\\"/"/g;
        return $inner;
      }
      return undef;
    }

    sub unquote_raw_R {
      my $s = shift;
      if ($s =~ /^R"\((.*)\)"$/s) {
        return $1;
      }
      return undef;
    }

    # Escape braces ONLY inside literal text and strip level tokens
    sub escape_literal {
      my $t = shift;
      $t =~ s/^\s*\[(?:INFO|ERROR|WARNING|WARN|STATUS)\]\s*//i;
      $t =~ s/\[(?:INFO|ERROR|WARNING|WARN|STATUS)\]\s*//ig;
      $t =~ s/\{/{{/g;
      $t =~ s/\}/}}/g;
      return $t;
    }

    sub build_spdlog {
      my ($stream, $body) = @_;

      my @parts = split(/\<\<\s*/, $body);
      @parts = map { s/^\s+|\s+$//g; $_ } @parts;
      shift @parts if @parts && $parts[0] eq "";

      # find first literal
      my $first_idx = -1;
      for my $i (0..$#parts) {
        my $lit = unquote_double($parts[$i]);
        $lit = unquote_raw_R($parts[$i]) unless defined $lit;
        if (defined $lit) { $first_idx = $i; last }
      }

      # determine level
      my $level = ($stream eq "cout" ? "info" : $stream eq "cerr" ? "error" : "debug");
      if ($first_idx >= 0) {
        my $txt = unquote_double($parts[$first_idx]) // unquote_raw_R($parts[$first_idx]) // "";
        if ($txt =~ /\[(INFO|ERROR|WARNING|WARN|STATUS)\]/i) {
          my $tok = uc($1);
          $level =
            $tok eq "ERROR"   ? "error" :
            $tok =~ /WARN/    ? "warn"  :
            $tok eq "STATUS"  ? "debug" :
                                "info";
        }
      }

      my $fmt = "";
      my @args;

      if ($first_idx >= 0) {
        my $lit = unquote_double($parts[$first_idx]) // unquote_raw_R($parts[$first_idx]) // "";
        $fmt .= escape_literal($lit);
      }

      my @iter = $first_idx >= 0 ? @parts[$first_idx+1 .. $#parts] : @parts;
      for my $p (@iter) {
        next unless defined $p;
        next if $p =~ /\bstd::endl\b|\bendl\b|\bstd::flush\b/;

        my $lit = unquote_double($p);
        $lit = unquote_raw_R($p) unless defined $lit;

        if (defined $lit) {
          $fmt .= escape_literal($lit);
        } else {
          my $expr = $p;
          $expr =~ s/;\s*$//;
          $expr =~ s/^\s+|\s+$//g;
          next if $expr eq "";
          $fmt .= "{}";
          push @args, $expr;
        }
      }

      $fmt =~ s/"/\\"/g;

      my $out = "spdlog::${level}(\"${fmt}\"";
      $out .= ", " . join(", ", @args) if @args;
      $out .= ");";

      return $out;
    }

    s{
      \bstd::(cout|cerr|clog)\b
      (
        (?:\s*<<\s*(?:R"\(.*?\)"|"(?:\\.|[^"\\])*"|[^;"]+))+
      )
      \s*;
    }{ build_spdlog($1, $2) }gesx;

  ' "$file"

  ensure_include "$file"
done

echo "Done. Backups saved as .bak. Please review changes and run build/tests."
