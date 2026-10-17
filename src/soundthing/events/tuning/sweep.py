"""Automated search over tunable parameters for events tuning.

Evaluates candidates using coordinate descent or Optuna TPE search, logging
every score result to the ledger.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from soundthing.config import load_config
from soundthing.events.tuning import ledger as ledger_module, space
from soundthing.events.tuning.score import score as score_func


def result_as_dict(result: Any) -> dict[str, Any]:
    """Convert a score result to a dictionary.

    Uses dataclasses.asdict if result is a dataclass instance,
    else dict(vars(result)) if it has __dict__,
    else dict(result) for a dict or mapping.
    """
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    if hasattr(result, "__dict__"):
        return dict(vars(result))
    return dict(result)



def log_result(result: Any, hypothesis: str, ledger: Any) -> dict[str, Any]:
    """Build and append a score result entry to the ledger."""
    entry = {
        **result_as_dict(result),
        "kind": "score",
        "hypothesis": hypothesis,
        "git": ledger.git_state(),
    }
    ledger.append(entry)
    return entry


def sweep(
    params: list[str],
    strategy: str = "coordinate",
    score_fn: Any = None,
    budget: int = 20,
    *,
    base_overrides: dict[str, Any] | None = None,
    cfg: Any = None,
    ledger: Any = None,
    seed: int = 0,
    n_values: int = 5,
) -> dict[str, Any]:
    """Automated search over tunable parameters with ledger logging.

    Args:
        params: List of parameter names in search space to tune.
        strategy: 'coordinate' for coordinate descent, 'optuna' for TPE search.
        score_fn: Scoring function(overrides, split, cfg). Defaults to
            soundthing.events.tuning.score.score.
        budget: Total number of score_fn calls permitted (including initial evaluation).
        base_overrides: Starting overrides mapping. Default is empty dict.
        cfg: Config instance. Defaults to loading repo config.
        ledger: Ledger module or object. Defaults to soundthing.events.tuning.ledger.
        seed: Random seed for samplers.
        n_values: Number of grid values per parameter in coordinate search.

    Returns:
        Dict with keys: strategy, evaluations, best_overrides, best_objective,
        best_run_id, improved, history.
    """
    if not params:
        raise ValueError("params must not be empty")

    if strategy not in ("coordinate", "optuna"):
        raise ValueError(f"Unknown strategy {strategy!r}. Must be 'coordinate' or 'optuna'.")

    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")

    if score_fn is None:
        score_fn = score_func
    if ledger is None:
        ledger = ledger_module
    if cfg is None:
        cfg = load_config()

    sp = space.describe_space(cfg, space.dev_event_counts())
    s_idx = space.space_index(sp)

    if base_overrides is None:
        base_overrides = {}
    for k in base_overrides:
        if space.is_shared(k):
            raise ValueError(f"base_overrides contains shared key {k!r}")
    space.validate_overrides(base_overrides, sp)

    for p in params:
        if p not in s_idx:
            raise ValueError(f"Parameter {p!r} not in search space")
        entry = s_idx[p]
        if entry.get("shared") or space.is_shared(p):
            raise ValueError(f"Parameter {p!r} is shared and cannot be tuned in sweep")
        if entry.get("type") not in ("float", "int"):
            raise ValueError(
                f"Parameter {p!r} must have type 'float' or 'int', got {entry.get('type')!r}"
            )

    # Initial evaluation of base_overrides
    init_overrides = dict(base_overrides)
    init_result = score_fn(overrides=init_overrides, split="dev", cfg=cfg)
    evaluations = 1
    log_result(init_result, "sweep", ledger)

    init_run_id = init_result.run_id
    init_obj = float(init_result.objective)
    init_passed = bool(init_result.passed)

    history: list[dict[str, Any]] = [
        {
            "run_id": init_run_id,
            "overrides": init_overrides,
            "objective": init_obj,
            "passed": init_passed,
        }
    ]

    incumbent_overrides = dict(init_overrides)
    incumbent_objective = init_obj
    incumbent_run_id = init_run_id

    if strategy == "coordinate":
        while evaluations < budget:
            improved_in_this_pass = False
            for param in params:
                if evaluations >= budget:
                    break
                entry = s_idx[param]
                p_type = entry["type"]
                lo, hi = entry["bounds"]

                if param in incumbent_overrides:
                    incumbent_val = incumbent_overrides[param]
                else:
                    incumbent_val = entry["current"]

                raw_vals = np.linspace(float(lo), float(hi), n_values)
                if p_type == "int":
                    all_vals = list(dict.fromkeys(int(round(float(v))) for v in raw_vals))
                    cand_vals = [v for v in all_vals if v != incumbent_val]
                else:
                    all_vals = [float(v) for v in raw_vals]
                    cand_vals = [
                        v
                        for v in all_vals
                        if not math.isclose(v, float(incumbent_val), abs_tol=1e-9)
                    ]

                best_candidate_for_param: tuple[dict[str, Any], Any] | None = None
                for val in cand_vals:
                    if evaluations >= budget:
                        break
                    cand_overrides = {**incumbent_overrides, param: val}
                    result = score_fn(overrides=cand_overrides, split="dev", cfg=cfg)
                    evaluations += 1
                    log_result(result, "sweep", ledger)

                    r_run_id = result.run_id
                    r_obj = float(result.objective)
                    r_passed = bool(result.passed)

                    history.append({
                        "run_id": r_run_id,
                        "overrides": cand_overrides,
                        "objective": r_obj,
                        "passed": r_passed,
                    })

                    if r_passed and r_obj > incumbent_objective:
                        if (
                            best_candidate_for_param is None
                            or r_obj > float(best_candidate_for_param[1].objective)
                        ):
                            best_candidate_for_param = (cand_overrides, result)

                if best_candidate_for_param is not None:
                    cand_overrides, cand_result = best_candidate_for_param
                    incumbent_overrides = cand_overrides
                    incumbent_objective = float(cand_result.objective)
                    incumbent_run_id = cand_result.run_id
                    improved_in_this_pass = True

            if not improved_in_this_pass:
                break

    elif strategy == "optuna":
        import optuna  # optional `tuning` extra: import only when used

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective_fn(trial: Any) -> float:
            nonlocal evaluations, incumbent_overrides, incumbent_objective, incumbent_run_id
            suggested: dict[str, Any] = {}
            for param in params:
                entry = s_idx[param]
                p_type = entry["type"]
                lo, hi = entry["bounds"]
                if p_type == "int":
                    suggested[param] = int(trial.suggest_int(param, int(lo), int(hi)))
                else:
                    suggested[param] = float(trial.suggest_float(param, float(lo), float(hi)))

            cand_overrides = {**base_overrides, **suggested}
            result = score_fn(overrides=cand_overrides, split="dev", cfg=cfg)
            evaluations += 1
            log_result(result, "sweep", ledger)

            r_run_id = result.run_id
            r_obj = float(result.objective)
            r_passed = bool(result.passed)

            history.append({
                "run_id": r_run_id,
                "overrides": cand_overrides,
                "objective": r_obj,
                "passed": r_passed,
            })

            if r_passed and r_obj > incumbent_objective:
                incumbent_overrides = cand_overrides
                incumbent_objective = r_obj
                incumbent_run_id = r_run_id

            return r_obj if r_passed else -1.0

        if budget > 1:
            study.optimize(objective_fn, n_trials=budget - 1)


    return {
        "strategy": strategy,
        "evaluations": evaluations,
        "best_overrides": dict(incumbent_overrides),
        "best_objective": incumbent_objective,
        "best_run_id": incumbent_run_id,
        "improved": incumbent_run_id != init_run_id,
        "history": history,
    }
