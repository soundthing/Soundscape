"""Patch validation, storage, and integrity checks for code experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from soundthing.harness import paths

PROTECTED_CONFIG_FILES = {"environments.yaml", "vocabulary.yaml", "priors.yaml"}


def _clean_diff_path(p: str) -> str | None:
    p = p.strip()
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1]
    p = p.replace("\\", "/")
    if p in ("/dev/null", "dev/null"):
        return None
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    p = p.lstrip("/")
    return p if p else None


def parse_patch_paths(patch_text: str) -> set[str]:
    """Parse all file paths modified, created, deleted, or moved in a patch."""
    touched: set[str] = set()
    for line in patch_text.splitlines():
        line = line.strip()
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                for p in parts[2:4]:
                    cleaned = _clean_diff_path(p)
                    if cleaned:
                        touched.add(cleaned)
        elif line.startswith("--- "):
            raw = line[4:].strip().split("\t")[0]
            cleaned = _clean_diff_path(raw)
            if cleaned:
                touched.add(cleaned)
        elif line.startswith("+++ "):
            raw = line[4:].strip().split("\t")[0]
            cleaned = _clean_diff_path(raw)
            if cleaned:
                touched.add(cleaned)
        elif line.startswith("rename from ") or line.startswith("copy from "):
            raw = line.split(maxsplit=2)[-1]
            cleaned = _clean_diff_path(raw)
            if cleaned:
                touched.add(cleaned)
        elif line.startswith("rename to ") or line.startswith("copy to "):
            raw = line.split(maxsplit=2)[-1]
            cleaned = _clean_diff_path(raw)
            if cleaned:
                touched.add(cleaned)
    return touched


def validate_patch(patch_text: str, code_paths: Sequence[str]) -> None:
    """Validate that a patch only touches files in code_paths or non-shared config/.

    Raises:
        ValueError: If any file in the patch is outside code_paths and config/,
                    or touches protected shared config files.
    """
    touched = parse_patch_paths(patch_text)
    if not touched:
        return

    normalized_code_paths = [cp.replace("\\", "/").strip("/") for cp in code_paths]

    for path in sorted(touched):
        parts = path.split("/")
        if parts[0] == "config":
            if len(parts) >= 2 and parts[1] in PROTECTED_CONFIG_FILES:
                raise ValueError(
                    f"Patch modifies protected shared config file {path!r}. "
                    f"Shared config files cannot be modified in an experiment."
                )
            continue

        is_allowed = False
        for cp in normalized_code_paths:
            if path == cp or path.startswith(cp + "/"):
                is_allowed = True
                break

        if not is_allowed:
            raise ValueError(
                f"Patch modifies file {path!r} outside component code_paths "
                f"{tuple(normalized_code_paths)} and config/."
            )


def patch_sha256(patch_text: str) -> str:
    """Compute sha256 hex digest of patch text."""
    return hashlib.sha256(patch_text.encode("utf-8")).hexdigest()


def save_patch(
    run_id: str,
    patch_text: str,
    target_dir: Path | None = None,
) -> tuple[Path, str]:
    """Save patch to patches_dir/<run_id>.patch and return (file_path, sha256)."""
    dest_dir = target_dir or paths.patches_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    patch_file = dest_dir / f"{run_id}.patch"
    patch_file.write_text(patch_text, encoding="utf-8")
    sha = patch_sha256(patch_text)
    return patch_file, sha


def load_and_verify_patch(
    run_id: str,
    expected_sha: str | None = None,
    target_dir: Path | None = None,
) -> str:
    """Read saved patch file and verify against expected sha256."""
    dest_dir = target_dir or paths.patches_dir()
    patch_file = dest_dir / f"{run_id}.patch"
    if not patch_file.is_file():
        raise FileNotFoundError(f"Patch file not found: {patch_file}")

    content = patch_file.read_text(encoding="utf-8")
    if expected_sha is not None:
        actual_sha = patch_sha256(content)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Patch sha mismatch for {run_id}: expected {expected_sha}, got {actual_sha}"
            )
    return content
