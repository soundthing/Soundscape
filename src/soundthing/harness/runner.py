"""Experiment runner and code promotion for the evaluation harness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import uuid

import yaml

from soundthing.harness import (
    bootstrap as bootstrap_module,
    ledger as ledger_module,
    paths,
)
from soundthing.harness.patches import patch_sha256, save_patch, validate_patch
from soundthing.harness.trusted import _get_component, trusted_score


def _remove_worktree(main: Path, wt_path: Path) -> None:
    try:
        subprocess.run(
            ["git", "-C", str(main), "worktree", "remove", "--force", str(wt_path)],
            capture_output=True,
            text=True,
        )
    except Exception:
        pass
    if wt_path.exists():
        shutil.rmtree(wt_path, ignore_errors=True)


def _delete_branch(main: Path, branch_name: str) -> None:
    try:
        subprocess.run(
            ["git", "-C", str(main), "branch", "-D", branch_name],
            capture_output=True,
            text=True,
        )
    except Exception:
        pass


def _run_bootstrap(
    boot_fn: Any,
    cand_per_clip: dict,
    best_per_clip: dict,
    component: str,
) -> dict:
    sig = inspect.signature(boot_fn)
    kwargs: dict[str, Any] = {}
    if "component" in sig.parameters:
        kwargs["component"] = component
    if "n_boot" in sig.parameters:
        kwargs["n_boot"] = 1000
    elif "n" in sig.parameters:
        kwargs["n"] = 1000
    return boot_fn(cand_per_clip, best_per_clip, **kwargs)


def _check_accepts(ci: dict | None) -> bool:
    if ci is None:
        return False
    if hasattr(bootstrap_module, "accepts"):
        return bool(bootstrap_module.accepts(ci))
    return float(ci.get("lo", 0.0)) > 0


def _get_best_entry(ledger_mod: Any, component: str) -> dict | None:
    try:
        return ledger_mod.best(component=component)
    except TypeError:
        return ledger_mod.best()


def run_experiment(
    component: str,
    hypothesis: str,
    *,
    patch: str | Path | None = None,
    overrides: dict | None = None,
    base: str | None = None,
    keep_worktree: bool = False,
    run_id: str | None = None,
    trusted_score_fn: Any = None,
    ledger: Any = None,
    bootstrap_fn: Any = None,
    main_repo: Path | None = None,
) -> dict:
    """Run an isolated code experiment in a git worktree.

    Args:
        component: Target component name (e.g. 'events').
        hypothesis: Clear hypothesis for this experiment.
        patch: Patch content string or path to a patch file.
        overrides: Parameter overrides dict.
        base: Git commit/ref to base the experiment on (default HEAD).
        keep_worktree: Keep experiment worktree even after scoring.
        run_id: Optional run ID (generated if None).
        trusted_score_fn: Injected trusted score function for testing.
        ledger: Injected ledger module for testing.
        bootstrap_fn: Injected paired_bootstrap function for testing.
        main_repo: Injected main repository root for testing.

    Returns:
        Ledger entry dictionary.
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    main = Path(main_repo).resolve() if main_repo is not None else paths.main_checkout()
    ledger_mod = ledger or ledger_module

    if base is None:
        base_sha = subprocess.run(
            ["git", "-C", str(main), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    else:
        base_sha = subprocess.run(
            ["git", "-C", str(main), "rev-parse", base],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    patch_text: str | None = None
    if patch is not None:
        if isinstance(patch, Path) or (isinstance(patch, str) and "\n" not in patch and Path(patch).is_file()):
            patch_text = Path(patch).read_text(encoding="utf-8")
        else:
            patch_text = str(patch)

    comp = _get_component(component)

    # Validate patch BEFORE creating any worktree
    if patch_text:
        validate_patch(patch_text, comp.code_paths)

    wt_root = paths.experiments_root()
    wt_root.mkdir(parents=True, exist_ok=True)
    wt_path = wt_root / f"soundthing-exp-{run_id}"
    branch_name = f"harness/exp-{run_id}"

    # Create worktree
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-b", branch_name, str(wt_path), base_sha],
        capture_output=True,
        text=True,
        check=True,
    )

    try:
        if patch_text:
            subprocess.run(
                ["git", "-C", str(wt_path), "apply", "--whitespace=nowarn"],
                input=patch_text,
                text=True,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(wt_path), "add", "-A"],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=soundthing-runner",
                    "-c",
                    "user.email=runner@soundthing.local",
                    "-C",
                    str(wt_path),
                    "commit",
                    "-m",
                    f"exp({component}): {hypothesis}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

        score_fn = trusted_score_fn or trusted_score
        score_res = score_fn(wt_path, component, overrides, pinned=base_sha)

        passed = bool(score_res.get("passed", False))
        best_entry = _get_best_entry(ledger_mod, component)

        if best_entry is None or "per_clip" not in best_entry:
            accept = passed
            bootstrap_ci = None
        else:
            cand_per_clip = score_res.get("per_clip") or {}
            best_per_clip = best_entry.get("per_clip") or {}
            boot_fn = bootstrap_fn or bootstrap_module.paired_bootstrap
            bootstrap_ci = _run_bootstrap(boot_fn, cand_per_clip, best_per_clip, component)
            accept = bool(passed and _check_accepts(bootstrap_ci))

        patch_hash: str | None = None
        if patch_text:
            _patch_file, patch_hash = save_patch(run_id, patch_text)

        entry = {
            "kind": "experiment",
            "component": component,
            "run_id": run_id,
            "hypothesis": hypothesis,
            "overrides": overrides or {},
            "patch": patch_text,
            "patch_sha256": patch_hash,
            "objective": score_res.get("objective"),
            "metrics": score_res.get("metrics", {}),
            "limits": score_res.get("limits", {}),
            "passed": passed,
            "bootstrap": bootstrap_ci,
            "accept": accept,
            "branch": branch_name,
            "base": base_sha,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        ledger_mod.append(entry)

        if accept:
            if not keep_worktree:
                _remove_worktree(main, wt_path)
        else:
            _remove_worktree(main, wt_path)
            _delete_branch(main, branch_name)

        return entry

    except Exception:
        if wt_path.exists():
            _remove_worktree(main, wt_path)
        _delete_branch(main, branch_name)
        raise


def _refuse_promote(
    ledger_mod: Any,
    run_id: str,
    component: str | None,
    reason: str,
    holdout_pass: bool | None = None,
) -> None:
    ledger_mod.append({
        "kind": "promote_code",
        "run_id": run_id,
        "component": component,
        "promoted": False,
        "reason": reason,
        "holdout": "fail" if holdout_pass is False else "n/a",
        "branch": None,
        "ts": datetime.now(timezone.utc).isoformat(),
    })


def promote_code(
    run_id: str,
    *,
    ledger: Any = None,
    holdout_fn: Any = None,
    main_repo: Path | None = None,
) -> dict:
    """Promote an accepted code experiment to a human-review branch."""
    main = Path(main_repo).resolve() if main_repo is not None else paths.main_checkout()
    ledger_mod = ledger or ledger_module

    entries = ledger_mod.read()
    candidate: dict | None = None
    for e in entries:
        if isinstance(e, dict) and e.get("run_id") == run_id and e.get("kind") == "experiment":
            candidate = e

    if candidate is None:
        reason = f"Candidate run_id {run_id!r} not found in ledger"
        _refuse_promote(ledger_mod, run_id, None, reason)
        return {"promoted": False, "branch": None, "run_id": run_id, "reason": reason}

    component = candidate.get("component", "events")

    if not candidate.get("accept", False):
        reason = "experiment not accepted"
        _refuse_promote(ledger_mod, run_id, component, reason)
        return {"promoted": False, "branch": None, "run_id": run_id, "reason": reason}

    expected_sha = candidate.get("patch_sha256")
    if expected_sha:
        patch_file = paths.patches_dir() / f"{run_id}.patch"
        if not patch_file.is_file():
            reason = f"patch file {patch_file} missing"
            _refuse_promote(ledger_mod, run_id, component, reason)
            return {"promoted": False, "branch": None, "run_id": run_id, "reason": reason}
        actual_sha = hashlib.sha256(patch_file.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            reason = f"patch sha mismatch: expected {expected_sha}, got {actual_sha} (tampered patch)"
            _refuse_promote(ledger_mod, run_id, component, reason)
            return {"promoted": False, "branch": None, "run_id": run_id, "reason": reason}

    if component == "events":
        if holdout_fn is not None:
            try:
                holdout_ok = holdout_fn(candidate.get("overrides") or {})
            except Exception:
                holdout_ok = False
        else:
            from soundthing.harness import promote as promote_module

            try:
                holdout_ok = promote_module.holdout_check(candidate.get("overrides") or {})
            except Exception:
                holdout_ok = False

        if not holdout_ok:
            reason = "holdout failed"
            _refuse_promote(ledger_mod, run_id, component, reason, holdout_pass=False)
            return {"promoted": False, "branch": None, "run_id": run_id, "reason": reason}
        holdout_val = "pass"
    else:
        holdout_val = "n/a"

    review_branch = f"harness/review/{run_id}"
    exp_branch = candidate.get("branch") or f"harness/exp-{run_id}"

    subprocess.run(
        ["git", "-C", str(main), "branch", "-f", review_branch, exp_branch],
        capture_output=True,
        text=True,
        check=True,
    )

    promote_entry = {
        "kind": "promote_code",
        "run_id": run_id,
        "component": component,
        "promoted": True,
        "branch": review_branch,
        "experiment_branch": exp_branch,
        "patch_sha256": expected_sha,
        "holdout": holdout_val,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    ledger_mod.append(promote_entry)

    return {
        "promoted": True,
        "branch": review_branch,
        "run_id": run_id,
    }


def mcp_run_experiment(
    component: str,
    hypothesis: str,
    patch: str | None = None,
    overrides: dict | None = None,
    base: str | None = None,
    keep_worktree: bool = False,
    **kwargs: Any,
) -> dict:
    """Run an isolated code experiment with trusted scoring and paired bootstrap."""
    return run_experiment(
        component=component,
        hypothesis=hypothesis,
        patch=patch,
        overrides=overrides,
        base=base,
        keep_worktree=keep_worktree,
        **kwargs,
    )


def mcp_promote_code(run_id: str, **kwargs: Any) -> dict:
    """Promote an accepted code experiment to a human-review branch."""
    return promote_code(run_id, **kwargs)


def cli_register(subparsers: Any) -> None:
    """Register run-experiment and promote-code subcommands."""
    p_exp = subparsers.add_parser(
        "run-experiment",
        help="Run code experiment in an isolated worktree with trusted scoring",
    )
    p_exp.add_argument("--component", required=True, type=str, help="Component name")
    p_exp.add_argument("--hypothesis", required=True, type=str, help="Hypothesis for this run")
    p_exp.add_argument("--patch", default=None, type=str, help="Path to patch file")
    p_exp.add_argument("--set", dest="set_items", action="append", default=[], help="Override parameter KEY=VALUE")
    p_exp.add_argument("--overrides", default=None, type=str, help="JSON object of overrides")
    p_exp.add_argument("--base", default=None, type=str, help="Base commit/branch (default HEAD)")
    p_exp.add_argument("--keep", action="store_true", help="Keep experiment worktree after run")
    p_exp.set_defaults(func=cli_handle)

    p_promote = subparsers.add_parser(
        "promote-code",
        help="Promote an accepted code experiment to a review branch",
    )
    p_promote.add_argument("run_id", type=str, help="Experiment run ID")
    p_promote.set_defaults(func=cli_handle)


def cli_handle(args: argparse.Namespace) -> int:
    """Handle CLI execution for run-experiment and promote-code."""
    sub = getattr(args, "subcommand", None)
    if sub == "run-experiment":
        patch_text = None
        if args.patch:
            patch_file = Path(args.patch)
            if not patch_file.is_file():
                raise FileNotFoundError(f"Patch file not found: {args.patch}")
            patch_text = patch_file.read_text(encoding="utf-8")

        overrides: dict[str, Any] = {}
        if args.overrides:
            try:
                parsed = json.loads(args.overrides)
                if not isinstance(parsed, dict):
                    raise ValueError("--overrides must be a JSON object")
                overrides.update(parsed)
            except Exception as e:
                raise ValueError(f"Invalid JSON for --overrides: {e}") from e

        for item in args.set_items:
            if "=" not in item:
                raise ValueError(f"Invalid --set format: {item!r}, expected KEY=VALUE")
            k, v_str = item.split("=", 1)
            overrides[k.strip()] = yaml.safe_load(v_str)

        res = run_experiment(
            component=args.component,
            hypothesis=args.hypothesis,
            patch=patch_text,
            overrides=overrides,
            base=args.base,
            keep_worktree=args.keep,
        )
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("accept") else 1

    elif sub == "promote-code":
        res = promote_code(args.run_id)
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("promoted") else 1

    return 0
