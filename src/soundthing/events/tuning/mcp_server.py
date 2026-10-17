"""MCP server for events tuning harness.

Exposes tools for tuning agents to explore search space, score configurations,
run sweeps, check promotion, propose shared config changes, and request recaching.
Enforces holdout confidentiality, ledger logging, and immutable contracts.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np

from soundthing.config import load_config
from soundthing.events.tuning import bootstrap, objective, space, sweep
from soundthing.events.tuning import ledger as ledger_module
from soundthing.events.tuning.score import score as score_func
from soundthing.events.tuning.sweep import sweep as sweep_func
from soundthing.events.tuning.promote import promote as promote_func
from soundthing.events.tuning.proposals import propose_shared_change as propose_func

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

OBJECTIVE_NAME: str = objective.OBJECTIVE

INSTRUCTIONS: str = (
    "Tuning agent loop:\n"
    "1. Read the ledger and get_best to inspect past experiments and baseline metrics.\n"
    "2. State one clear hypothesis, adjust parameters within describe_space, and score on dev.\n"
    "3. Keep candidates only if limit checks pass and bootstrap CI accepts; every attempt is logged.\n"
    "4. Propose vocabulary or priors changes via propose_shared_change; promote for holdout validation."
)


def _make_json_safe(obj: Any) -> Any:
    """Convert numpy scalars and non-standard types to JSON-serialisable Python types."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    return obj


class TuningTools:
    """Encapsulates the 8 MCP tuning tools with injected or lazy dependencies."""

    def __init__(
        self,
        score_fn: Any = None,
        ledger: Any = None,
        cfg_loader: Callable[[], Any] | None = None,
        sweep_fn: Any = None,
        promote_fn: Any = None,
        propose_fn: Any = None,
        recache_fn: Any = None,
        max_sweep_budget: int = 200,
    ) -> None:
        self._score_fn = score_fn
        self._ledger = ledger
        self._cfg_loader = cfg_loader
        self._sweep_fn = sweep_fn
        self._promote_fn = promote_fn
        self._propose_fn = propose_fn
        self._recache_fn = recache_fn
        self.max_sweep_budget = max_sweep_budget

    @property
    def score_fn(self) -> Any:
        if self._score_fn is not None:
            return self._score_fn
        return score_func

    @score_fn.setter
    def score_fn(self, value: Any) -> None:
        self._score_fn = value

    @property
    def ledger(self) -> Any:
        if self._ledger is not None:
            return self._ledger
        return ledger_module

    @ledger.setter
    def ledger(self, value: Any) -> None:
        self._ledger = value

    @property
    def cfg_loader(self) -> Callable[[], Any]:
        if self._cfg_loader is not None:
            return self._cfg_loader
        return load_config

    @cfg_loader.setter
    def cfg_loader(self, value: Callable[[], Any]) -> None:
        self._cfg_loader = value

    @property
    def sweep_fn(self) -> Any:
        if self._sweep_fn is not None:
            return self._sweep_fn
        return sweep_func

    @sweep_fn.setter
    def sweep_fn(self, value: Any) -> None:
        self._sweep_fn = value

    @property
    def promote_fn(self) -> Any:
        if self._promote_fn is not None:
            return self._promote_fn
        return promote_func

    @promote_fn.setter
    def promote_fn(self, value: Any) -> None:
        self._promote_fn = value

    @property
    def propose_fn(self) -> Any:
        if self._propose_fn is not None:
            return self._propose_fn
        return propose_func

    @propose_fn.setter
    def propose_fn(self, value: Any) -> None:
        self._propose_fn = value

    @property
    def recache_fn(self) -> Any:
        if self._recache_fn is not None:
            return self._recache_fn
        try:
            from soundthing.events.tuning.kaggle_io import request_recache
            return request_recache
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "request_recache needs soundthing.events.tuning.kaggle_io (Track B), which is not available in this tree"
            ) from exc

    @recache_fn.setter
    def recache_fn(self, value: Any) -> None:
        self._recache_fn = value

    @contextmanager
    def _log_call(self, tool_name: str, args: dict[str, Any]) -> Iterator[None]:
        ok = False
        error: str | None = None
        try:
            yield
            ok = True
        except BaseException as exc:
            error = str(exc)
            raise
        finally:
            self.ledger.append({
                "kind": "mcp_call",
                "tool": tool_name,
                "args": _make_json_safe(args),
                "ok": ok,
                "error": error,
            })

    def describe_space(self) -> dict[str, Any]:
        """Describe the tunable search space and bounds."""
        with self._log_call("describe_space", {}):
            cfg = self.cfg_loader()
            counts = space.dev_event_counts()
            params = space.describe_space(cfg, counts)
            return {
                "objective": OBJECTIVE_NAME,
                "min_class_events": space.MIN_CLASS_EVENTS,
                "params": _make_json_safe(params),
                "note": (
                    "Search space for soundthing events tuning. Per-class thresholds require >= "
                    f"{space.MIN_CLASS_EVENTS} dev events. Shared vocabulary and priors are for proposals only."
                ),
            }

    def get_best(self) -> dict[str, Any] | None:
        """Get the current best score result from ledger without per_clip details."""
        with self._log_call("get_best", {}):
            best = self.ledger.best()
            if best is None:
                return None
            res = {k: v for k, v in best.items() if k != "per_clip"}
            return _make_json_safe(res)

    def get_ledger(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent ledger entries up to limit without per_clip or diagnostics."""
        with self._log_call("get_ledger", {"limit": limit}):
            if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 200):
                raise ValueError(f"limit must be between 1 and 200, got {limit}")
            entries = self.ledger.read()
            slice_entries = entries[-limit:] if limit < len(entries) else entries
            res: list[dict[str, Any]] = []
            for entry in slice_entries:
                cleaned = {k: v for k, v in entry.items() if k not in ("per_clip", "diagnostics")}
                res.append(_make_json_safe(cleaned))
            return res

    def score(self, overrides: dict[str, Any] | None = None, hypothesis: str = "") -> dict[str, Any]:
        """Score candidate overrides against the dev split and evaluate limits."""
        args: dict[str, Any] = {"overrides": overrides, "hypothesis": hypothesis}
        with self._log_call("score", args):
            cfg = self.cfg_loader()
            ovr = dict(overrides or {})
            sp = space.describe_space(cfg, space.dev_event_counts())
            space.validate_overrides(ovr, sp)

            is_shared = any(space.is_shared(k) for k in ovr)
            result = self.score_fn(overrides=ovr, split="dev", cfg=cfg)
            sweep.log_result(result, hypothesis or "mcp score", self.ledger)

            best = self.ledger.best()
            bootstrap_dict: dict[str, Any] | None = None
            if best is not None:
                best_per_clip = best.get("per_clip") if isinstance(best, dict) else getattr(best, "per_clip", None)
                cand_per_clip = result.per_clip
                if best_per_clip and cand_per_clip:
                    boot = bootstrap.paired_bootstrap(cand_per_clip, best_per_clip)
                    bootstrap_dict = {
                        "gain": float(boot["gain"]),
                        "lo": float(boot["lo"]),
                        "hi": float(boot["hi"]),
                        "accept": bootstrap.accepts(boot),
                    }

            diag = result.diagnostics or {}
            per_class = diag.get("per_class", {})
            top_fps = diag.get("top_false_positives", [])

            reply: dict[str, Any] = {
                "run_id": str(result.run_id),
                "objective": float(result.objective),
                "baseline": float(result.baseline),
                "wrong_env": float(result.wrong_env),
                "safety_recall": float(result.safety_recall),
                "fp_per_min": float(result.fp_per_min),
                "grid_f1": float(result.grid_f1),
                "limits": _make_json_safe(result.limits),
                "passed": bool(result.passed),
                "shared": bool(is_shared),
                "bootstrap": bootstrap_dict,
                "per_class": _make_json_safe(per_class),
                "top_false_positives": _make_json_safe(top_fps),
            }
            if is_shared:
                reply["note"] = "use propose_shared_change; promote will refuse"
            return reply

    def sweep(
        self,
        params: list[str],
        strategy: str = "coordinate",
        budget: int = 20,
    ) -> dict[str, Any]:
        """Automated search over tunable parameters within budget."""
        args: dict[str, Any] = {"params": params, "strategy": strategy, "budget": budget}
        with self._log_call("sweep", args):
            if isinstance(budget, bool) or not isinstance(budget, int) or not (1 <= budget <= self.max_sweep_budget):
                raise ValueError(f"budget must be between 1 and {self.max_sweep_budget}, got {budget}")
            res = self.sweep_fn(
                params,
                strategy=strategy,
                score_fn=self.score_fn,
                budget=budget,
                cfg=self.cfg_loader(),
                ledger=self.ledger,
            )
            return _make_json_safe(res)

    def promote(self, run_id: str) -> dict[str, Any]:
        """Promote candidate to config; hidden holdout reported as pass/fail only."""
        args: dict[str, Any] = {"run_id": run_id}
        with self._log_call("promote", args):
            raw_res = self.promote_fn(
                run_id,
                ledger=self.ledger,
                score_fn=self.score_fn,
                cfg=self.cfg_loader(),
            )
            res = dict(raw_res)

            # Strip any unexpected holdout_* numeric field
            for k in list(res.keys()):
                if k.startswith("holdout_") and isinstance(res[k], (int, float, np.number)) and not isinstance(res[k], bool):
                    del res[k]

            # Assert defensively that no key other than "holdout"/"holdout_pass" mentions holdout
            for k in res:
                if "holdout" in k and k not in ("holdout", "holdout_pass"):
                    raise AssertionError(f"Unexpected holdout key in promote reply: {k!r}")

            return _make_json_safe(res)

    def propose_shared_change(self, overrides: dict[str, Any], note: str) -> dict[str, Any]:
        """Draft a proposal markdown file for changes to shared vocabulary or priors."""
        args: dict[str, Any] = {"overrides": overrides, "note": note}
        with self._log_call("propose_shared_change", args):
            cfg = self.cfg_loader()
            result = self.score_fn(overrides=overrides, split="dev", cfg=cfg)
            sweep.log_result(result, "proposal", self.ledger)

            path = self.propose_fn(overrides, result, note, cfg=cfg)

            run_id = str(result.run_id)
            path_str = str(path)
            self.ledger.append({"kind": "proposal", "path": path_str, "run_id": run_id})
            return {"path": path_str, "run_id": run_id}

    def request_recache(self, window: float, hop: float, model: str) -> dict[str, Any]:
        """Request audio feature cache rebuilding via Kaggle GPU kernel."""
        args: dict[str, Any] = {"window": window, "hop": hop, "model": model}
        with self._log_call("request_recache", args):
            if isinstance(window, bool) or not isinstance(window, (int, float)) or window <= 0:
                raise ValueError(f"window must be > 0, got {window}")
            if isinstance(hop, bool) or not isinstance(hop, (int, float)) or not (0 < hop <= window):
                raise ValueError(f"hop must satisfy 0 < hop <= window, got hop={hop}, window={window}")
            if not isinstance(model, str) or not model:
                raise ValueError(f"model must be non-empty string, got {model!r}")
            res = self.recache_fn(window, hop, model)
            return _make_json_safe(dict(res) if hasattr(res, "items") else res)


def build_server(tools: TuningTools | None = None) -> MCPServer:
    """Build and configure the soundthing tuning MCP server."""
    if tools is None:
        tools = TuningTools()

    from mcp.server.mcpserver import MCPServer  # optional `tuning` extra

    server = MCPServer(name="soundthing-tuning", instructions=INSTRUCTIONS)

    server.tool(
        name="describe_space",
        description="Describe the tunable search space and bounds; per-class thresholds require >= 20 dev events.",
    )(tools.describe_space)

    server.tool(
        name="get_best",
        description="Get the current best score result from the ledger without per_clip details.",
    )(tools.get_best)

    server.tool(
        name="get_ledger",
        description="Get recent ledger entries up to limit without per_clip details or diagnostics.",
    )(tools.get_ledger)

    server.tool(
        name="score",
        description="Score candidate overrides against the dev split and evaluate limit checks against current best.",
    )(tools.score)

    server.tool(
        name="sweep",
        description="Automated coordinate descent or Optuna search over specified parameters within budget.",
    )(tools.sweep)

    server.tool(
        name="promote",
        description="Promote candidate to config; hidden holdout reported as pass/fail only.",
    )(tools.promote)

    server.tool(
        name="propose_shared_change",
        description="Draft a proposal markdown file for changes to shared vocabulary or priors configuration.",
    )(tools.propose_shared_change)

    server.tool(
        name="request_recache",
        description="Request cache rebuilding for window, hop, and model via Kaggle GPU kernel.",
    )(tools.request_recache)

    return server


def main() -> None:
    """Run the MCP server over stdio."""
    build_server().run("stdio")


if __name__ == "__main__":
    main()
