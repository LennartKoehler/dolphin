#!/usr/bin/env python3
"""
oras_publish_v2.py
Enhancements over oras_publish.py:
- Preflight validation (existence & type of sources; non-empty selection for files mode)
- Existence check in registry before push (skip if present) — can be disabled with --no-exists-check, or overridden with --overwrite
- Auto-adds 'org.opencontainers.image.source' for GHCR to help linking packages to a repo
- Utility flags: --require-source (fail if the source annotation would be missing), --print-ref-only

Coordinates supported:
- "group.artifact:tag"
- "group:artifact:version[-classifier]"
"""

import argparse
import json
import mimetypes
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

COORD_PATTERNS = [
    re.compile(r"^(?P<group>(?:[a-zA-Z0-9_]+\.)+[a-zA-Z0-9_]+)\.(?P<artifact>[a-zA-Z0-9._-]+):(?P<tag>[^:\s]+)$"),
    re.compile(r"^(?P<group>(?:[a-zA-Z0-9_]+\.)+[a-zA-Z0-9_]+):(?P<artifact>[a-zA-Z0-9._-]+):(?P<tag>[^:\s]+)$"),
]

EXTRA_MIME = {
    ".whl": "application/zip",
    ".onnx": "application/octet-stream",
    ".pt": "application/octet-stream",
    ".pth": "application/octet-stream",
    ".bin": "application/octet-stream",
    ".gz": "application/gzip",
    ".tgz": "application/gzip",
    ".tar": "application/x-tar",
    ".xz": "application/x-xz",
    ".zip": "application/zip",
    ".so": "application/octet-stream",
    ".dll": "application/octet-stream",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".sh": "text/x-shellscript",
    ".py": "text/x-python",
}

def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in EXTRA_MIME:
        return EXTRA_MIME[ext]
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"

def ensure_oras():
    try:
        subprocess.run(["oras", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("ERROR: ORAS CLI not found in PATH. Install from https://oras.land", file=sys.stderr)
        sys.exit(2)

def run(cmd: List[str], dry_run=False) -> int:
    if dry_run:
        print("[dry-run]", shlex.join(cmd))
        return 0
    print(">", shlex.join(cmd))
    return subprocess.call(cmd)

def run_capture(cmd: List[str]) -> Tuple[int, str]:
    try:
        cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return cp.returncode, (cp.stdout or "") + (cp.stderr or "")
    except Exception as e:
        return 1, str(e)

def login_if_requested(args) -> None:
    if args.login:
        parts = args.login.split(":", 2)
        if len(parts) != 3:
            raise SystemExit("--login must look like 'host:USER:TOKEN' or 'ghcr:USER:TOKEN'")
        host, user, token = parts
        if host == "ghcr":
            host = "ghcr.io"
        rc = run(["oras", "login", host, "-u", user, "-p", token], dry_run=args.dry_run)
        if rc != 0:
            raise SystemExit(rc)
    if args.login_env:
        ci_reg = os.environ.get("CI_REGISTRY")
        ci_user = os.environ.get("CI_REGISTRY_USER")
        ci_pass = os.environ.get("CI_REGISTRY_PASSWORD")
        if ci_reg and ci_user and ci_pass:
            rc = run(["oras", "login", ci_reg, "-u", ci_user, "-p", ci_pass], dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)
        gh_user = os.environ.get("GHCR_USERNAME")
        gh_pat  = os.environ.get("GHCR_PAT")
        if gh_user and gh_pat:
            rc = run(["oras", "login", "ghcr.io", "-u", gh_user, "-p", gh_pat], dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(rc)

def build_base_ref(args) -> str:
    prefix = args.prefix.strip("/") if args.prefix else None
    if args.target == "gitlab":
        base = args.gitlab_base.rstrip("/") if args.gitlab_base else os.environ.get("CI_REGISTRY_IMAGE", "").rstrip("/")
        if not base:
            raise SystemExit("For --target gitlab, provide --gitlab-base or set CI_REGISTRY_IMAGE.")
        return f"{base}/{prefix}" if prefix else base
    elif args.target == "ghcr":
        if not args.ghcr_owner or not args.ghcr_repo:
            raise SystemExit("For --target ghcr, provide --ghcr-owner and --ghcr-repo.")
        base = f"ghcr.io/{args.ghcr_owner}/{args.ghcr_repo}"
        return f"{base}/{prefix}" if prefix else base
    else:
        raise SystemExit(f"Unknown target {args.target}")

def make_ref(base: str, group_path: str, artifact: str, tag: str) -> str:
    return f"{base}/{group_path}/{artifact}:{tag}"

def parse_coord(coord: str, explicit_artifact_id: Optional[str]=None) -> Tuple[str, str, str]:
    for pat in COORD_PATTERNS:
        m = pat.match(coord)
        if m:
            group = m.group("group")
            artifact = m.group("artifact")
            tag = m.group("tag")
            if explicit_artifact_id:
                artifact = explicit_artifact_id
            return group.replace(".", "/"), artifact, tag
    if ":" in coord and "." in coord.split(":")[0]:
        group = coord.split(":")[0]
        tag = coord.split(":")[1]
        if explicit_artifact_id is None:
            raise ValueError(f"Could not parse artifact id from coord '{coord}'. Provide 'artifact_id' in JSON.")
        return group.replace(".", "/"), explicit_artifact_id, tag
    raise ValueError(f"Unrecognized coordinate format: '{coord}'")

def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def merge_defaults(defaults: Dict, item: Dict) -> Dict:
    out = dict(defaults or {})
    out.update(item or {})
    ann = dict(defaults.get("annotations", {})) if defaults else {}
    ann.update(item.get("annotations", {}))
    out["annotations"] = ann
    return out

def expand_globs(root: Path, include: List[str], exclude: List[str]) -> List[Path]:
    files = set()
    if not include:
        include = ["**/*"]
    for pattern in include:
        files.update(root.glob(pattern))
    files = {p for p in files if p.is_file()}
    for pattern in exclude or []:
        for p in list(files):
            if p.match(pattern):
                files.discard(p)
    return sorted(files)

def build_push_args_files(source_dir: Path, include: List[str], exclude: List[str]) -> List[str]:
    rels: List[str] = []
    root = source_dir.resolve()
    files = expand_globs(root, include, exclude)
    for f in files:
        rel = f.relative_to(root)
        rel_str = rel.as_posix()
        if rel_str.startswith("/") or ".." in rel.parts:
            continue
        rels.append(f"{rel_str}:{guess_mime(f)}")
    return rels

def build_archive(source_dir: Path, out_path: Path, fmt: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "tgz":
        import tarfile
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(source_dir, arcname=".")
    elif fmt == "zip":
        import zipfile
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in source_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(source_dir))
    else:
        raise ValueError(f"Unsupported archive_format '{fmt}', use 'tgz' or 'zip'.")
    return out_path

# --- NEW: preflight validation -----------------------------------------------
def preflight_validate(item: Dict) -> None:
    source = Path(item["source"]).expanduser()
    mode = item.get("mode", "archive")
    if not source.exists():
        raise SystemExit(f"[ERROR] Source does not exist: {source}")
    if mode == "files":
        if not source.is_dir():
            raise SystemExit(f"[ERROR] mode=files requires a directory source: {source}")
        include = item.get("include", ["**/*"])
        exclude = item.get("exclude", ["**/.git/**", "**/__pycache__/**"])
        files = expand_globs(source, include, exclude)
        if not files:
            raise SystemExit(f"[ERROR] No files matched include/exclude under: {source}")
    elif mode == "archive":
        if source.is_file():
            # ok
            pass
        elif source.is_dir():
            # ensure we can write temp archive
            tmp_dir = Path(".oras_tmp")
            tmp_dir.mkdir(exist_ok=True)
            if not os.access(tmp_dir, os.W_OK):
                raise SystemExit("[ERROR] Cannot write to .oras_tmp for archive creation")
        else:
            raise SystemExit(f"[ERROR] Invalid source for archive mode: {source}")

# --- NEW: existence checks ----------------------------------------------------
def ref_exists(ref: str) -> bool:
    # Fast path: try manifest fetch; returns 0 if ref/tag exists
    rc, out = run_capture(["oras", "manifest", "fetch", ref])
    return rc == 0

def ensure_source_annotation(args, annotations: Dict[str, str]) -> Dict[str, str]:
    ann = dict(annotations or {})
    if args.target == "ghcr":
        default_src = f"https://github.com/{args.ghcr_owner}/{args.ghcr_repo}"
        if "org.opencontainers.image.source" not in ann:
            ann["org.opencontainers.image.source"] = default_src
        elif args.require_source and not ann.get("org.opencontainers.image.source"):
            raise SystemExit("[ERROR] Missing org.opencontainers.image.source annotation for GHCR (use --require-source to enforce)")
    return ann

def push_one(ref: str, item: Dict, args) -> int:
    preflight_validate(item)
    if not args.no_exists_check and not args.overwrite and ref_exists(ref):
        print(f"[INFO] {ref} already exists — skipping (use --overwrite to force or --no-exists-check to disable test)")
        return 0

    mode = item.get("mode", "archive")
    artifact_type = item.get("artifact_type", "application/vnd.asb.bundle.v1")
    annotations = ensure_source_annotation(args, item.get("annotations", {}) or {})
    source = Path(item["source"]).expanduser()

    if mode not in ("archive", "files"):
        print(f"[WARN] Unsupported mode '{mode}', falling back to 'archive'.")
        mode = "archive"

    if mode == "files":
        include = item.get("include", ["**/*"])
        exclude = item.get("exclude", ["**/.git/**", "**/__pycache__/**"])
        file_args = build_push_args_files(source, include, exclude)
        cmd = ["oras", "push", ref] + file_args + ["--artifact-type", artifact_type]
        for k, v in (annotations or {}).items():
            cmd += ["--annotation", f"{k}={v}"]
        return run(cmd, dry_run=args.dry_run)

    # archive mode
    if source.is_dir():
        fmt = item.get("archive_format", "tgz")
        out_name = f"{item.get('artifact_id','payload')}-{os.getpid()}.{ 'tgz' if fmt=='tgz' else 'zip'}"
        out_path = Path(".oras_tmp") / out_name
        archive_path = build_archive(source, out_path, fmt)
        payload = f"{archive_path}:{guess_mime(archive_path)}"
        cleanup = True
    else:
        archive_path = source
        payload = f"{archive_path}:{guess_mime(archive_path)}"
        cleanup = False

    cmd = ["oras", "push", ref, payload, "--artifact-type", artifact_type]
    for k, v in (annotations or {}).items():
        cmd += ["--annotation", f"{k}={v}"]
    rc = run(cmd, dry_run=args.dry_run)
    if cleanup and not args.dry_run:
        try:
            archive_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            Path(".oras_tmp").rmdir()
        except Exception:
            pass
    return rc

def main():
    ensure_oras()
    ap = argparse.ArgumentParser(description="Publish artifacts to OCI registries via ORAS with Maven-style coordinates (enhanced).")
    ap.add_argument("--config", required=True, help="Path to JSON configuration file.")
    ap.add_argument("--target", required=True, choices=["gitlab", "ghcr"], help="Target registry kind.")
    ap.add_argument("--gitlab-base", help="Base repo for GitLab (e.g., registry.example.com/group/project). Defaults to CI_REGISTRY_IMAGE if unset.")
    ap.add_argument("--ghcr-owner", help="Owner (org/user) for GHCR.")
    ap.add_argument("--ghcr-repo", help="Repository name for GHCR.")
    ap.add_argument("--prefix", default="artifacts", help="Optional logical prefix path (default: artifacts).")
    ap.add_argument("--login", help="Perform a login: '<host>:USER:TOKEN' or 'ghcr:USER:TOKEN'.")
    ap.add_argument("--login-env", action="store_true", help="Login using CI env vars (GitLab: CI_REGISTRY_*, GHCR: GHCR_USERNAME/GHCR_PAT).")
    ap.add_argument("--dry-run", action="store_true", help="Print commands instead of executing.")
    # new toggles
    ap.add_argument("--no-exists-check", action="store_true", help="Skip pre-check for existing ref in registry.")
    ap.add_argument("--overwrite", action="store_true", help="Push even if ref exists (forces re-publish attempt).")
    ap.add_argument("--require-source", action="store_true", help="Error if org.opencontainers.image.source is missing for GHCR.")
    ap.add_argument("--print-ref-only", action="store_true", help="Resolve and print ORAS refs then exit.")
    args = ap.parse_args()

    login_if_requested(args)
    base = build_base_ref(args)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = load_config(cfg_path)
    defaults = cfg.get("defaults", {})
    items = cfg.get("artifacts", [])
    if not items:
        print("No artifacts in config.", file=sys.stderr)
        sys.exit(1)

    overall_rc = 0
    refs = []
    for raw in items:
        item = merge_defaults(defaults, raw)
        coord = item.get("coord")
        if not coord:
            print("[ERROR] Artifact entry missing 'coord'", file=sys.stderr)
            overall_rc = 2
            continue
        try:
            group_path, artifact, tag = parse_coord(coord, explicit_artifact_id=item.get("artifact_id"))
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            overall_rc = 2
            continue
        ref = make_ref(base, group_path, artifact, tag)
        refs.append((coord, ref))
        if args.print_ref_only:
            continue
        print(f"[INFO] -> {coord}  ==>  {ref}")
        try:
            rc = push_one(ref, {**item, "artifact_id": artifact}, args)
        except SystemExit as se:
            overall_rc = max(overall_rc, int(str(se) or 1))
            continue
        except Exception as e:
            print(f"[ERROR] Unhandled exception while pushing {coord}: {e}", file=sys.stderr)
            overall_rc = 1
            continue
        if rc != 0:
            overall_rc = rc

    if args.print_ref_only:
        for c, r in refs:
            print(f"{c} -> {r}")
        sys.exit(0)

    sys.exit(overall_rc)

if __name__ == "__main__":
    main()
