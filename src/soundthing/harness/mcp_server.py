"""MCP server for tuning harness.

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
from soundthing.harness import bootstrap, components, paths, space, sweep
from soundthing.harness import ledger as ledger_module
from soundthing.harness.score import score as score_func
from soundthing.harness.sweep import sweep as sweep_func
from soundthing.harness.promote import promote as promote_func
from soundthing.harness.proposals import propose_shared_change as propose_func

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

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


def _extract(res: Any, key: str, default: Any = None) -> Any:
    """Safely extract key/attribute from a dict or object."""
    if isinstance(res, dict):
        return res.get(key, default)
    return getattr(res, key, default)


class TuningTools:
    """Encapsulates MCP tuning tools with injected or lazy dependencies."""

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
            from soundthing.harness.kaggle_io import request_recache
            return request_recache
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "request_recache needs soundthing.harness.kaggle_io (Track B), which is not available in this tree"
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

    def describe_space(self, component: str = "events") -> dict[str, Any]:
        """Describe the tunable search space and bounds for a component."""
        args: dict[str, Any] = {}
        if component != "events":
            args["component"] = component
        with self._log_call("describe_space", args):
            comp = components.get(component)
            cfg = self.cfg_loader()
            params = comp.space(cfg, paths.dev_labels_dir())
            return {
                "component": component,
                "min_class_events": space.MIN_CLASS_EVENTS,
                "params": _make_json_safe(params),
                "note": (
                    f"Search space for soundthing {component} tuning. "
                    "Shared vocabulary and priors are for proposals only."
                ),
            }

    def get_best(self, component: str = "events") -> dict[str, Any] | None:
        """Get the current best score result from ledger without per_clip details."""
        args: dict[str, Any] = {}
        if component != "events":
            args["component"] = component
        with self._log_call("get_best", args):
            best = self.ledger.best(component)
            if best is None:
                return None
            res = {k: v for k, v in best.items() if k != "per_clip"}
            return _make_json_safe(res)

    def get_ledger(self, limit: int = 20, component: str = "events") -> list[dict[str, Any]]:
        """Get recent ledger entries up to limit without per_clip or diagnostics."""
        args: dict[str, Any] = {"limit": limit}
        if component != "events":
            args["component"] = component
        with self._log_call("get_ledger", args):
            if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= 200):
                raise ValueError(f"limit must be between 1 and 200, got {limit}")
            entries = self.ledger.read(component)
            slice_entries = entries[-limit:] if limit < len(entries) else entries
            res: list[dict[str, Any]] = []
            for entry in slice_entries:
                cleaned = {k: v for k, v in entry.items() if k not in ("per_clip", "diagnostics")}
                res.append(_make_json_safe(cleaned))
            return res

    def score(
        self,
        overrides: dict[str, Any] | None = None,
        hypothesis: str = "",
        component: str = "events",
        labels_dir: str | None = None,
    ) -> dict[str, Any]:
        """Score candidate overrides against the dev split (or labels_dir) and evaluate limits."""
        args: dict[str, Any] = {
            "overrides": overrides,
            "hypothesis": hypothesis,
        }
        if component != "events":
            args["component"] = component
        with self._log_call("score", args):
            comp = components.get(component)
            cfg = self.cfg_loader()
            ovr = dict(overrides or {})
            sp = comp.space(cfg, Path(labels_dir) if labels_dir else paths.dev_labels_dir())
            space.validate_overrides(ovr, sp)

            is_shared = any(space.is_shared(k) for k in ovr)
            result = self.score_fn(
                overrides=ovr, split="dev", cfg=cfg, component=component, labels_dir=labels_dir
            )

            sweep.log_result(result, hypothesis or "mcp score", self.ledger)

            best = self.ledger.best(component)
            bootstrap_dict: dict[str, Any] | None = None
            if best is not None:
                best_per_clip = _extract(best, "per_clip", None)
                cand_per_clip = _extract(result, "per_clip", None)
                if best_per_clip and cand_per_clip:
                    boot = bootstrap.paired_bootstrap(cand_per_clip, best_per_clip, component=component)
                    bootstrap_dict = {
                        "gain": float(boot["gain"]),
                        "lo": float(boot["lo"]),
                        "hi": float(boot["hi"]),
                        "accept": bootstrap.accepts(boot),
                    }

            diag = _extract(result, "diagnostics", {}) or {}
            limits = _extract(result, "limits", {})
            passed = _extract(result, "passed", True)
            run_id = _extract(result, "run_id", "")
            objective = _extract(result, "objective", 0.0)

            reply: dict[str, Any] = {
                "run_id": str(run_id),
                "component": component,
                "objective": float(objective),
                "limits": _make_json_safe(limits),
                "passed": bool(passed),
                "shared": bool(is_shared),
                "bootstrap": bootstrap_dict,
            }

            metrics = _extract(result, "metrics", {}) or {}
            if metrics:
                reply["metrics"] = _make_json_safe(metrics)

            for k in ("baseline", "wrong_env", "safety_recall", "fp_per_min", "grid_f1"):
                val = _extract(result, k, None)
                if val is not None:
                    reply[k] = float(val)
                elif k in metrics:
                    reply[k] = float(metrics[k])

            per_class = diag.get("per_class", {})
            top_fps = diag.get("top_false_positives", [])
            if per_class:
                reply["per_class"] = _make_json_safe(per_class)
            if top_fps:
                reply["top_false_positives"] = _make_json_safe(top_fps)

            if is_shared:
                reply["note"] = "use propose_shared_change; promote will refuse"
            return reply

    def sweep(
        self,
        params: list[str],
        strategy: str = "coordinate",
        budget: int = 20,
        component: str = "events",
    ) -> dict[str, Any]:
        """Automated search over tunable parameters within budget."""
        args: dict[str, Any] = {
            "params": params,
            "strategy": strategy,
            "budget": budget,
        }
        if component != "events":
            args["component"] = component
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
                component=component,
            )
            return _make_json_safe(res)

    def promote(self, run_id: str, component: str = "events") -> dict[str, Any]:
        """Promote candidate to config; hidden holdout reported as pass/fail only."""
        args: dict[str, Any] = {"run_id": run_id}
        if component != "events":
            args["component"] = component
        with self._log_call("promote", args):
            raw_res = self.promote_fn(
                run_id,
                ledger=self.ledger,
                score_fn=self.score_fn,
                cfg=self.cfg_loader(),
                component=component,
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

    def list_components(self) -> list[dict[str, Any]]:
        """List registered tuning harness components and their status."""
        with self._log_call("list_components", {}):
            output: list[dict[str, Any]] = []
            for name in components.names():
                try:
                    comp = components.get(name)
                    output.append({
                        "name": comp.name,
                        "owner": comp.owner,
                        "sections": list(comp.sections),
                        "code_paths": list(comp.code_paths),
                        "available": True,
                    })
                except (ImportError, ModuleNotFoundError, AttributeError):
                    output.append({
                        "name": name,
                        "available": False,
                    })
            return _make_json_safe(output)

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

    def run_experiment(
        self,
        component: str,
        hypothesis: str,
        patch: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply a code patch in a fresh experiment worktree, score it with the trusted
        scorer, and keep its branch only if it passes limits and the bootstrap."""
        args = {"component": component, "hypothesis": hypothesis, "overrides": overrides,
                "has_patch": patch is not None}
        with self._log_call("run_experiment", args):
            from soundthing.harness import runner

            return _make_json_safe(runner.mcp_run_experiment(component, hypothesis, patch=patch, overrides=overrides))

    def promote_code(self, run_id: str) -> dict[str, Any]:
        """Turn an accepted code experiment into a review branch for a human to merge."""
        with self._log_call("promote_code", {"run_id": run_id}):
            from soundthing.harness import runner

            return _make_json_safe(runner.mcp_promote_code(run_id))



def build_server(tools: TuningTools | None = None) -> MCPServer:
    """Build and configure the soundthing tuning MCP server."""
    if tools is None:
        tools = TuningTools()

    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="soundthing-harness", instructions=INSTRUCTIONS)

    server.tool(
        name="describe_space",
        description="Describe the tunable search space and bounds for a component.",
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
        name="list_components",
        description="List registered tuning harness components and their status.",
    )(tools.list_components)

    server.tool(
        name="request_recache",
        description="Request cache rebuilding for window, hop, and model via Kaggle GPU kernel.",
    )(tools.request_recache)
    server.tool(name="run_experiment")(tools.run_experiment)
    server.tool(name="promote_code")(tools.promote_code)

    return server


def main() -> None:
    """Run the MCP server over stdio."""
    build_server().run("stdio")


if __name__ == "__main__":
    main()
