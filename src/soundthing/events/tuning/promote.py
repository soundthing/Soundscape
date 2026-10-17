"""Config writer for events tuning promotion.

Applies tuned overrides to `config/pipeline.yaml` (`events:` section only)
and `config/thresholds.yaml` with minimal line edits that preserve comments,
blank lines, key order, and every untouched line byte-for-byte.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from soundthing.config import load_config
from soundthing.events.tuning import bootstrap, holdout, space
from soundthing.events.tuning import ledger as ledger_module
from soundthing.events.tuning.score import score as score_func



WRITABLE_PREFIXES = ("events.", "thresholds.")


class PromoteError(ValueError):
    """Raised when an override or promotion operation is invalid."""


def split_target(key: str) -> tuple[str, list[str]]:
    """Split an override dotted key into target filename and YAML path list.

    'events.a.b' -> ('pipeline.yaml', ['events', 'a', 'b'])
    'thresholds.default' -> ('thresholds.yaml', ['default'])
    'thresholds.safety_default' -> ('thresholds.yaml', ['safety_default'])
    'thresholds.classes.<id>' -> ('thresholds.yaml', ['classes', '<id>'])

    Anything else raises PromoteError. 'events.backend' is refused.
    """
    if not isinstance(key, str) or not key:
        raise PromoteError(f"Invalid key: {key!r}")

    parts = key.split(".")
    if any(p == "" for p in parts):
        raise PromoteError(f"Invalid key with empty component: {key!r}")

    if parts[0] == "events":
        if len(parts) < 2:
            raise PromoteError(f"Key too short: {key!r}")
        if parts[1] == "backend":
            raise PromoteError(f"Refusing to override events.backend: {key!r}")
        return ("pipeline.yaml", parts)

    if parts[0] == "thresholds":
        if len(parts) == 2 and parts[1] == "default":
            return ("thresholds.yaml", ["default"])
        if len(parts) == 2 and parts[1] == "safety_default":
            return ("thresholds.yaml", ["safety_default"])
        if len(parts) == 3 and parts[1] == "classes":
            return ("thresholds.yaml", ["classes", parts[2]])
        raise PromoteError(f"Invalid thresholds key: {key!r}")

    raise PromoteError(f"Non-writable key prefix: {key!r} (must start with {WRITABLE_PREFIXES})")


def format_scalar(value: Any) -> str:
    """Format a Python scalar value into its YAML representation.

    bool -> 'true' / 'false'
    int -> str(int)
    float -> repr(round(value, 6))
    str -> plain if yaml.safe_load(s) == s else JSON-quoted
    Other types -> PromoteError
    """
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return repr(round(value, 6))

    if isinstance(value, str):
        try:
            loaded = yaml.safe_load(value)
            if loaded == value:
                return value
        except Exception:
            pass
        return json.dumps(value)

    raise PromoteError(f"Unsupported value type for YAML scalar: {type(value).__name__} ({value!r})")


MAPPING_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_.-]+):(.*)$")


def _is_blank(line: str) -> bool:
    return line.strip() == ""


def _is_full_comment(line: str) -> bool:
    return line.strip().startswith("#")


def _get_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _has_trailing_newline(text: str) -> bool:
    return text.endswith("\r\n") or text.endswith("\n")


def _parse_rest(rest: str) -> tuple[str | None, str | None, str, int | None]:
    """Parse the remainder of a mapping key line after the colon.

    Returns:
        (scalar, comment, leading_space, comment_col_offset)
        where comment_col_offset is the 0-indexed character offset in rest where '#' starts.
    """
    pos = 0
    while pos < len(rest) and rest[pos].isspace():
        pos += 1
    leading_space = rest[:pos]

    if pos >= len(rest):
        return None, None, leading_space, None

    if rest[pos] == "#":
        return None, rest[pos:], leading_space, pos

    in_quote = None
    escaped = False
    comment_start = None
    val_end = len(rest)

    for i in range(pos, len(rest)):
        ch = rest[i]
        if in_quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\" and in_quote == '"':
                escaped = True
            elif ch == in_quote:
                in_quote = None
        else:
            if ch in ('"', "'"):
                in_quote = ch
            elif ch == "#" and i > pos and rest[i - 1].isspace():
                comment_start = i
                val_end = i - 1
                while val_end > pos and rest[val_end - 1].isspace():
                    val_end -= 1
                break

    if comment_start is None:
        val_end = len(rest)
        while val_end > pos and rest[val_end - 1].isspace():
            val_end -= 1
        return rest[pos:val_end], None, leading_space, None

    return rest[pos:val_end], rest[comment_start:], leading_space, comment_start


def _replace_scalar_on_line(line: str, new_val_str: str, path: list[str]) -> str:
    m = MAPPING_KEY_RE.match(line)
    if not m:
        raise PromoteError(f"Line does not match mapping key: {line!r}")

    indent_str = m.group(1)
    key_name = m.group(2)
    rest = m.group(3)

    scalar, comment, leading_space, comment_col = _parse_rest(rest)
    if scalar is None:
        raise PromoteError(f"Target key {'.'.join(path)} has no scalar value to replace")

    if not leading_space:
        leading_space = " "

    prefix = f"{indent_str}{key_name}:{leading_space}"
    new_before = f"{prefix}{new_val_str}"

    if comment is None:
        return new_before

    orig_comment_col = len(indent_str) + len(key_name) + 1 + (comment_col if comment_col is not None else 0)
    current_len = len(new_before)
    if orig_comment_col > current_len:
        gap = " " * (orig_comment_col - current_len)
    else:
        gap = "  "
    return f"{new_before}{gap}{comment}"


def _get_block_range(lines: list[str], parent_idx: int) -> tuple[int, int]:
    """Find the start and end line indices of a parent key's block.

    The block starts at parent_idx + 1 and ends at the next non-blank,
    non-comment line whose indent <= parent_indent (or EOF).
    Returns (start, end) where end is exclusive.
    """
    parent_indent = _get_indent(lines[parent_idx])
    start = parent_idx + 1
    end = len(lines)
    for i in range(start, len(lines)):
        if _is_blank(lines[i]) or _is_full_comment(lines[i]):
            continue
        if _get_indent(lines[i]) <= parent_indent:
            end = i
            break
    return start, end


def _get_child_indent(lines: list[str], block_start: int, block_end: int, parent_indent: int) -> int:
    """Determine the child indentation for a block.

    Child indent is the indent of the first non-blank, non-comment line inside
    the block, or parent_indent + 2 when the parent has no children.
    """
    for i in range(block_start, block_end):
        if not _is_blank(lines[i]) and not _is_full_comment(lines[i]):
            return _get_indent(lines[i])
    return parent_indent + 2


def _find_direct_child(lines: list[str], block_start: int, block_end: int, child_indent: int, key: str) -> int | None:
    """Find a direct child with the given key at child_indent."""
    for i in range(block_start, block_end):
        if _is_blank(lines[i]) or _is_full_comment(lines[i]):
            continue
        if _get_indent(lines[i]) == child_indent:
            m = MAPPING_KEY_RE.match(lines[i])
            if m and m.group(2) == key:
                return i
    return None


def set_yaml_value(text: str, path: list[str], value: Any) -> str:
    """Set a scalar value at path in YAML text, preserving structure and comments.

    Lines are parsed by indentation (spaces only). A mapping key line matches
    `^(\\s*)([A-Za-z0-9_.-]+):(.*)$`. Blank lines and full-line comments are ignored
    when determining structure.
    """
    if not path:
        raise PromoteError("Empty path")

    val_str = format_scalar(value)

    newline = _detect_newline(text)
    ends_with_newline = _has_trailing_newline(text)
    lines = text.splitlines()

    if not lines:
        new_lines = []
        for i, k in enumerate(path):
            ind = " " * (i * 2)
            if i == len(path) - 1:
                new_lines.append(f"{ind}{k}: {val_str}")
            else:
                new_lines.append(f"{ind}{k}:")
        res = newline.join(new_lines) + newline
        return res

    curr_parent_idx: int | None = None
    curr_parent_indent = -1
    curr_block_start = 0
    curr_block_end = len(lines)
    curr_child_indent = 0

    for depth, key in enumerate(path):
        is_final = depth == len(path) - 1

        if curr_parent_idx is None:
            found_idx = _find_direct_child(lines, 0, len(lines), 0, key)
        else:
            found_idx = _find_direct_child(lines, curr_block_start, curr_block_end, curr_child_indent, key)

        if found_idx is not None:
            if is_final:
                b_start, b_end = _get_block_range(lines, found_idx)
                has_children = any(
                    not _is_blank(lines[j]) and not _is_full_comment(lines[j])
                    for j in range(b_start, b_end)
                )
                if has_children:
                    raise PromoteError(f"Target key {'.'.join(path)} is a mapping, not a scalar")

                lines[found_idx] = _replace_scalar_on_line(lines[found_idx], val_str, path)
                result = newline.join(lines)
                if ends_with_newline:
                    result += newline
                return result
            else:
                m = MAPPING_KEY_RE.match(lines[found_idx])
                if m:
                    rest_scalar, _, _, _ = _parse_rest(m.group(3))
                    if rest_scalar is not None:
                        raise PromoteError(
                            f"Intermediate key {key} along path {'.'.join(path)} has a scalar value, cannot have children"
                        )
                curr_parent_idx = found_idx
                curr_parent_indent = _get_indent(lines[found_idx])
                curr_block_start, curr_block_end = _get_block_range(lines, found_idx)
                curr_child_indent = _get_child_indent(lines, curr_block_start, curr_block_end, curr_parent_indent)
        else:
            missing_keys = path[depth:]

            if curr_parent_idx is None:
                insert_idx = len(lines)
                base_indent = 0
            else:
                base_indent = curr_child_indent
                insert_idx = None
                for j in range(curr_block_end - 1, curr_block_start - 1, -1):
                    if not _is_blank(lines[j]) and not _is_full_comment(lines[j]):
                        insert_idx = j + 1
                        break
                if insert_idx is None:
                    insert_idx = curr_parent_idx + 1

            lines_to_insert = []
            for k_idx, m_key in enumerate(missing_keys):
                indent_level = base_indent + k_idx * 2
                ind_str = " " * indent_level
                if k_idx == len(missing_keys) - 1:
                    lines_to_insert.append(f"{ind_str}{m_key}: {val_str}")
                else:
                    lines_to_insert.append(f"{ind_str}{m_key}:")

            lines[insert_idx:insert_idx] = lines_to_insert
            result = newline.join(lines)
            if ends_with_newline:
                result += newline
            return result

    result = newline.join(lines)
    if ends_with_newline:
        result += newline
    return result


def write_overrides(overrides: dict, config_dir: Path) -> list[Path]:
    """Apply tuned overrides to YAML files in config_dir.

    Only config/pipeline.yaml (events: section) and config/thresholds.yaml
    may be written.
    """
    if not isinstance(overrides, dict):
        raise PromoteError(f"overrides must be a dict, got {type(overrides).__name__}")
    if not overrides:
        return []

    config_dir = Path(config_dir)

    targets: list[tuple[str, str, list[str], Any]] = []
    needed_files: set[str] = set()
    for key in sorted(overrides.keys()):
        filename, path = split_target(key)
        targets.append((key, filename, path, overrides[key]))
        needed_files.add(filename)

    forbidden = {"vocabulary.yaml", "priors.yaml", "environments.yaml"}
    if needed_files & forbidden:
        raise PromoteError(f"Refusing to write to protected files: {needed_files & forbidden}")

    original_texts: dict[str, str] = {}
    current_texts: dict[str, str] = {}
    for filename in sorted(needed_files):
        file_path = config_dir / filename
        if not file_path.exists():
            raise PromoteError(f"Config file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            content = f.read()
        original_texts[filename] = content
        current_texts[filename] = content

    for key, filename, path, val in targets:
        current_texts[filename] = set_yaml_value(current_texts[filename], path, val)

    for key, filename, path, val in targets:
        loaded = yaml.safe_load(current_texts[filename])
        cur = loaded
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                raise PromoteError(f"Verification failed: key {key!r} not found after edit in {filename}")
            cur = cur[p]

        match = False
        if cur == val:
            match = True
        elif isinstance(val, (int, float)) and isinstance(cur, (int, float)):
            if abs(cur - val) < 1e-5:
                match = True
        if not match:
            raise PromoteError(
                f"Verification failed for {key!r}: expected {val!r}, parsed {cur!r} from {filename}"
            )

    changed_files: list[Path] = []
    for filename in sorted(needed_files):
        if current_texts[filename] != original_texts[filename]:
            file_path = config_dir / filename
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(current_texts[filename])
            changed_files.append(file_path)

    if not changed_files:
        return []

    try:
        load_config(config_dir)
    except Exception as e:
        for file_path in changed_files:
            orig = original_texts[file_path.name]
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(orig)
        raise PromoteError(f"Config validation failed after editing: {e}") from e

    return changed_files


def _git_state(ledger_mod: Any, root: Path) -> dict[str, Any]:
    return ledger_mod.git_state(root)


def _fmt_val(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, int):
        return str(val)
    return str(val)


def _get_old_value(
    key: str,
    space_idx: dict[str, dict[str, Any]],
    config_dir: Path,
) -> Any:
    if key in space_idx and space_idx[key].get("current") is not None:
        return space_idx[key]["current"]
    try:
        filename, y_path = split_target(key)
        file_path = config_dir / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                data = yaml.safe_load(f)
            cur = data
            for p in y_path:
                cur = cur[p]
            return cur
    except Exception:
        pass
    return "missing"


def holdout_check(
    overrides: dict,
    score_fn: Any = None,
    cfg: Any = None,
) -> bool:
    """Check a candidate configuration against the hidden holdout set.

    Uses score_fn (soundthing.events.tuning.score.score) and holdout labels
    directory (soundthing.events.tuning.holdout.holdout_labels_dir()).
    Pass iff candidate objective, safety_recall, and fp_per_min are at least as
    good as the current config on holdout.

    Returns ONLY a bool. Never prints, logs, returns, or stores holdout numbers.
    """
    if score_fn is None:
        score_fn = score_func

    try:
        hdir = holdout.holdout_labels_dir()
    except Exception as e:
        raise PromoteError(f"holdout set not available: {e}") from e

    hdir_path = Path(hdir) if hdir is not None else None
    if hdir_path is None or not hdir_path.is_dir() or not any(hdir_path.glob("*.json")):
        raise PromoteError(f"holdout set not available: directory {hdir} missing or contains no JSON files")

    cand = score_fn(overrides=overrides, split="holdout", cfg=cfg, best=None, labels_dir=hdir_path)
    cur = score_fn(overrides={}, split="holdout", cfg=cfg, best=None, labels_dir=hdir_path)

    def _metric(res: Any, name: str) -> float:
        if isinstance(res, dict):
            return float(res[name])
        return float(getattr(res, name))

    cand_obj = _metric(cand, "objective")
    cur_obj = _metric(cur, "objective")
    cand_safety = _metric(cand, "safety_recall")
    cur_safety = _metric(cur, "safety_recall")
    cand_fp = _metric(cand, "fp_per_min")
    cur_fp = _metric(cur, "fp_per_min")

    passed = (cand_obj >= cur_obj and cand_safety >= cur_safety and cand_fp <= cur_fp)
    return bool(passed)



def _format_tuning_log_entry(
    entry: dict[str, Any],
    old: dict[str, Any],
    ci: dict[str, float],
    before: float | None,
    git_sha: str,
) -> list[str]:
    run_id = entry.get("run_id", "norun")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sha_str = str(git_sha)[:10]
    n_clips = len(entry.get("per_clip", {}))

    overrides = entry.get("overrides", {})
    param_parts = []
    for k in sorted(overrides.keys()):
        old_v = _fmt_val(old.get(k, "missing"))
        new_v = _fmt_val(overrides[k])
        param_parts.append(f"{k}: {old_v} -> {new_v}")
    params_str = "; ".join(param_parts)

    before_str = f"{float(before):.4f}" if before is not None else "n/a"
    after_str = f"{float(entry['objective']):.4f}"
    base_str = f"{float(entry.get('baseline', 0.0)):.4f}"
    wrong_str = f"{float(entry.get('wrong_env', 0.0)):.4f}"

    gain_str = f"{float(ci['gain']):.4f}"
    lo_str = f"{float(ci['lo']):.4f}"
    hi_str = f"{float(ci['hi']):.4f}"

    return [
        f"## {date_str} — promote {run_id}",
        f"Git: {sha_str}    Split: dev ({n_clips} clips)",
        f"Params: {params_str}",
        f"Dev F0.5: {before_str} -> {after_str}   (baseline {base_str}, wrong-env {wrong_str})",
        f"Bootstrap gain: {gain_str} [{lo_str}, {hi_str}]",
        "Holdout: pass",
        "Outcome: promoted",
    ]


def append_tuning_log(
    path: Path,
    entry: dict[str, Any],
    old: dict[str, Any],
    ci: dict[str, float],
    before: float | None,
    git_sha: str,
) -> None:
    """Insert a promote entry into tuning-log.md newest first.

    Inserts directly before the first '## ' entry line after the header,
    or replacing '_No runs yet._', or at EOF. Preserves newline style.
    """
    path = Path(path)
    if not path.exists():
        content = ""
        newline = "\n"
        ends_with_newline = False
        lines: list[str] = []
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            content = f.read()
        newline = "\r\n" if "\r\n" in content else "\n"
        ends_with_newline = content.endswith("\r\n") or content.endswith("\n")
        lines = content.splitlines()

    entry_lines = _format_tuning_log_entry(entry, old, ci, before, git_sha)

    in_code_block = False
    first_entry_idx: int | None = None
    no_runs_idx: int | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            if stripped == "_No runs yet._" and no_runs_idx is None:
                no_runs_idx = i
            elif line.startswith("## ") and first_entry_idx is None:
                first_entry_idx = i

    if first_entry_idx is not None:
        lines[first_entry_idx:first_entry_idx] = entry_lines + [""]
    elif no_runs_idx is not None:
        lines[no_runs_idx:no_runs_idx + 1] = entry_lines
    else:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.extend(entry_lines)

    new_content = newline.join(lines)
    if ends_with_newline or not content:
        new_content += newline

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(new_content)


def _refuse(
    ledger: Any,
    run_id: str,
    reason: str,
    git_info: dict[str, Any],
    holdout_pass: bool | None = None,
    ci: dict[str, float] | None = None,
) -> dict[str, Any]:
    ledger.append({
        "kind": "promote",
        "run_id": run_id,
        "promoted": False,
        "reason": reason,
        "holdout_pass": holdout_pass,
        "bootstrap": ci,
        "git": git_info,
    })
    return {
        "run_id": run_id,
        "promoted": False,
        "reason": reason,
    }


def promote(
    run_id: str,
    *,
    ledger: Any = None,
    score_fn: Any = None,
    holdout_fn: Any = None,
    cfg: Any = None,
    root: Path | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Take a ledger candidate, verify limits, bootstrap and holdout, then promote.

    Args:
        run_id: Candidate identifier in ledger.
        ledger: Ledger module or fake (lazy loaded if None).
        score_fn: Score function or fake (lazy loaded if None).
        holdout_fn: Holdout check callable (defaults to holdout_check).
        cfg: Config instance (defaults to load_config(root / "config")).
        root: Repo root Path (defaults to Path.cwd()).
        n_boot: Bootstrap iterations for paired bootstrap test.
        seed: Random seed for bootstrap.

    Returns:
        Dict with "promoted": True/False and associated metadata.
    """
    root_path = Path(root) if root is not None else Path.cwd()
    config_dir = root_path / "config"
    tuning_log_path = root_path / "src" / "soundthing" / "events" / "references" / "tuning-log.md"

    if cfg is None:
        cfg = load_config(config_dir)

    if ledger is None:
        ledger = ledger_module


    if holdout_fn is None:
        holdout_fn = lambda ov: holdout_check(ov, score_fn=score_fn, cfg=cfg)

    entries = ledger.read()
    candidate: dict[str, Any] | None = None
    for e in entries:
        if isinstance(e, dict) and e.get("run_id") == run_id and e.get("kind", "score") == "score":
            candidate = e

    if candidate is None:
        raise PromoteError(f"Candidate run_id {run_id!r} not found in ledger")

    overrides = candidate.get("overrides", {})
    if overrides is None:
        overrides = {}

    git_info = _git_state(ledger, root_path)

    # 1. overrides empty -> reason "no overrides"
    if not overrides:
        return _refuse(ledger, run_id, "no overrides", git_info)

    # 2. Any shared key -> reason mentions propose_shared_change.
    # Any key failing split_target -> reason with PromoteError message.
    reason = None
    for k in sorted(overrides.keys()):
        if space.is_shared(k):
            reason = f"key {k!r} is shared; use propose_shared_change to propose changes to shared files"
            break

    if reason is None:
        for k in sorted(overrides.keys()):
            try:
                split_target(k)
            except PromoteError as e:
                reason = str(e)
                break

    if reason is not None:
        return _refuse(ledger, run_id, reason, git_info)

    # 3. entry["passed"] not True -> "limits not passed"
    if not candidate.get("passed", False):
        return _refuse(ledger, run_id, "limits not passed", git_info)

    # 4. Bootstrap: comparator per_clip = ledger.best()["per_clip"] if a best exists,
    # else score_fn(overrides={}, split="dev", cfg=cfg).per_clip (current config).
    # ci = bootstrap.paired_bootstrap(entry["per_clip"], comparator, n=n_boot, seed=seed);
    # not bootstrap.accepts(ci) -> "bootstrap lower bound <= 0".
    best_entry = ledger.best()
    if best_entry is not None and "per_clip" in best_entry:
        comparator = best_entry["per_clip"]
        before = best_entry.get("objective")
    else:
        if score_fn is None:
            score_fn = score_func

        cur_res = score_fn(overrides={}, split="dev", cfg=cfg)
        if isinstance(cur_res, dict):
            comparator = cur_res.get("per_clip")
            before = cur_res.get("objective")
        else:
            comparator = getattr(cur_res, "per_clip")
            before = getattr(cur_res, "objective")

    try:
        ci = bootstrap.paired_bootstrap(candidate["per_clip"], comparator, n=n_boot, seed=seed)
    except ValueError:
        return _refuse(ledger, run_id, "bootstrap: no common clips with comparator", git_info)

    if not bootstrap.accepts(ci):
        return _refuse(ledger, run_id, "bootstrap lower bound <= 0", git_info, ci=ci)

    # 5. holdout_fn(overrides) False -> "holdout failed".
    # If it raises PromoteError -> reason with its message.
    try:
        passed_holdout = holdout_fn(overrides)
    except PromoteError as e:
        return _refuse(ledger, run_id, str(e), git_info, ci=ci)

    if not passed_holdout:
        return _refuse(ledger, run_id, "holdout failed", git_info, holdout_pass=False, ci=ci)

    # On success:
    space_idx = space.space_index(space.describe_space(cfg, None))
    old: dict[str, Any] = {}
    for k in overrides:
        old[k] = _get_old_value(k, space_idx, config_dir)

    try:
        changed = write_overrides(overrides, config_dir)
    except PromoteError as e:
        return _refuse(ledger, run_id, f"write failed: {e}", git_info, holdout_pass=True, ci=ci)

    git_sha = str(git_info.get("sha", ""))
    ledger.set_best({**candidate, "git_sha": git_sha})
    append_tuning_log(tuning_log_path, candidate, old, ci, before, git_sha)

    ledger.append({
        "kind": "promote",
        "run_id": run_id,
        "promoted": True,
        "reason": None,
        "holdout_pass": True,
        "bootstrap": ci,
        "git": git_info,
    })

    return {
        "run_id": run_id,
        "promoted": True,
        "holdout": "pass",
        "files": [p.relative_to(root_path).as_posix() for p in changed],
        "bootstrap": ci,
        "objective": candidate["objective"],
    }
