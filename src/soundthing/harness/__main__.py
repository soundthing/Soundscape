"""CLI entry point for soundthing tuning harness."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from soundthing.config import load_config
from soundthing.harness import paths
from soundthing.harness.promote import PromoteError


def _parse_overrides_and_sets(
    overrides_json: str | None,
    set_items: list[str],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if overrides_json:
        try:
            parsed = json.loads(overrides_json)
        except Exception as exc:
            raise ValueError(f"Invalid JSON for --overrides: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("--overrides must be a JSON object")
        overrides.update(parsed)

    for item in set_items:
        if "=" not in item:
            raise ValueError(f"Invalid --set format: {item!r}, expected KEY=VALUE")
        key, val_str = item.split("=", 1)
        key = key.strip()
        val = yaml.safe_load(val_str)
        overrides[key] = val

    return overrides


def _handle_score(args: argparse.Namespace, tools: Any) -> int:
    overrides = _parse_overrides_and_sets(args.overrides, args.set_items)
    if args.no_log:
        from soundthing.harness.score import score as raw_score

        cfg = load_config()
        res = raw_score(overrides, split=args.split, cfg=cfg, component=args.component, labels_dir=args.labels)
        res_dict = res.to_dict()
    else:
        if args.split != "dev":
            raise ValueError("only dev scores are logged; use --no-log for other splits")
        res_dict = tools.score(overrides, args.hypothesis or "", component=args.component, labels_dir=args.labels
        )

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res_dict, indent=2, default=str), encoding="utf-8")

    print(json.dumps(res_dict, indent=2, default=str))
    return 0


def _handle_sweep(args: argparse.Namespace, tools: Any) -> int:
    params = [p.strip() for p in args.params.split(",") if p.strip()]
    if not params:
        raise ValueError("--params cannot be empty")
    res = tools.sweep(params, args.strategy, args.budget, component=args.component)
    print(json.dumps(res, indent=2, default=str))
    return 0


def _handle_promote(args: argparse.Namespace, tools: Any) -> int:
    res = tools.promote(args.run_id, component=args.component)
    print(json.dumps(res, indent=2, default=str))
    promoted = res.get("promoted", False) if isinstance(res, dict) else False
    return 0 if promoted else 1


def _handle_propose(args: argparse.Namespace, tools: Any) -> int:
    overrides = _parse_overrides_and_sets(args.overrides, args.set_items)
    if not overrides:
        raise ValueError("propose requires either --set or --overrides")
    res = tools.propose_shared_change(overrides, args.note)
    print(json.dumps(res, indent=2, default=str))
    return 0


def _handle_space(args: argparse.Namespace, tools: Any) -> int:
    res = tools.describe_space(component=args.component)
    print(json.dumps(res, indent=2, default=str))
    return 0


def _handle_best(args: argparse.Namespace, tools: Any) -> int:
    res = tools.get_best(component=args.component)
    print(json.dumps(res, indent=2, default=str))
    return 0


def _handle_ledger(args: argparse.Namespace, tools: Any) -> int:
    res = tools.get_ledger(args.limit, component=args.component)
    print(json.dumps(res, indent=2, default=str))
    return 0


def _handle_list_components(args: argparse.Namespace, tools: Any) -> int:
    from soundthing.harness import components

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
    print(json.dumps(output, indent=2))
    return 0


def _handle_recache(args: argparse.Namespace, tools: Any) -> int:
    res = tools.request_recache(args.window, args.hop, args.model)
    print(json.dumps(res, indent=2, default=str))
    return 0


def _handle_serve_mcp(args: argparse.Namespace, tools: Any) -> int:
    from soundthing.harness.mcp_server import main as serve

    serve()
    return 0


def _handle_build_dev(args: argparse.Namespace, tools: Any) -> int:
    try:
        datasets = importlib.import_module("soundthing.harness.datasets")
        if datasets is None:
            raise ImportError("soundthing.harness.datasets is None")
    except (ImportError, ModuleNotFoundError, AttributeError):
        print(
            "error: build-dev needs soundthing.harness.datasets (Track B), not available in this tree",
            file=sys.stderr,
        )
        return 2

    cfg = load_config()
    out = Path(args.out) if args.out is not None else (paths.harness_home() / "dev")
    mix_paths = datasets.esc50_mixes(Path(args.esc50), cfg, args.n_per_env, args.seed, out)
    cache_paths = datasets.build_dev_caches(out / "audio", out / "labels", cfg)
    counts = {
        "mixes": len(mix_paths) if mix_paths is not None else 0,
        "caches": len(cache_paths) if cache_paths is not None else 0,
    }
    print(json.dumps(counts, indent=2, default=str))
    return 0


def _handle_import_audioset(args: argparse.Namespace, tools: Any) -> int:
    import tempfile

    from soundthing.harness import datasets

    cfg = load_config()
    with tempfile.TemporaryDirectory() as tmp:
        datasets.audioset_strong_labels(Path(args.tsv), cfg, Path(tmp), Path(args.class_csv))
        split = datasets.split_audioset(
            Path(tmp) / "labels", Path(args.audio), holdout_frac=args.holdout_frac
        )
    counts: dict[str, Any] = {"dev": len(split["dev"]), "holdout": len(split["holdout"])}
    if args.build_caches:
        counts["dev_caches"] = len(
            datasets.build_dev_caches(paths.dev_audio_dir(), paths.dev_labels_dir(), cfg)
        )
        counts["holdout_caches"] = len(datasets.build_holdout_caches(cfg))
    print(json.dumps(counts, indent=2))
    return 0



def _handle_build_cache(args: argparse.Namespace, tools: Any) -> int:
    cfg = load_config()
    labels, audio = Path(args.labels), Path(args.audio)
    if args.component == "environment":
        from soundthing.harness.components.environment import build_env_caches

        n = build_env_caches(labels, audio, cfg)
    elif args.component == "events":
        from soundthing.harness import datasets

        n = len(datasets.build_dev_caches(audio, labels, cfg) or [])
    else:
        print(f"{args.component} reuses the events and environment caches; nothing to build")
        return 0
    print(json.dumps({"component": args.component, "caches": n}))
    return 0


def main(argv: list[str] | None = None, tools: Any = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="soundthing.harness",
        description="Soundthing tuning harness CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # score
    p_score = subparsers.add_parser("score", help="Score candidate overrides on dev split")
    p_score.add_argument("--set", dest="set_items", action="append", default=[], help="Override parameter KEY=VALUE")
    p_score.add_argument("--overrides", type=str, default=None, help="JSON object of overrides")
    p_score.add_argument("--hypothesis", type=str, default="", help="Hypothesis for this run")
    p_score.add_argument("--component", type=str, default="events", help="Component to score")
    p_score.add_argument("--json-out", type=str, default=None, help="Path to write ScoreResult JSON")
    p_score.add_argument("--no-log", action="store_true", help="Skip writing to experiment ledger")
    p_score.add_argument("--split", type=str, default="dev", choices=["dev", "holdout"], help="Split to score")
    p_score.add_argument("--labels", type=str, default=None, help="Score this labels dir instead of the split's")
    p_score.set_defaults(func=_handle_score)

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Automated search over parameters")
    p_sweep.add_argument("--params", required=True, type=str, help="Comma-separated parameter names")
    p_sweep.add_argument(
        "--strategy",
        type=str,
        default="coordinate",
        choices=["coordinate", "optuna"],
        help="Search strategy",
    )
    p_sweep.add_argument("--budget", type=int, default=20, help="Search budget")
    p_sweep.add_argument("--component", type=str, default="events", help="Component to sweep")
    p_sweep.set_defaults(func=_handle_sweep)

    # promote
    p_promote = subparsers.add_parser("promote", help="Promote candidate run to config")
    p_promote.add_argument("run_id", type=str, help="Ledger run_id to promote")
    p_promote.add_argument("--component", type=str, default="events", help="Component to promote")
    p_promote.set_defaults(func=_handle_promote)

    # propose
    p_propose = subparsers.add_parser("propose", help="Propose changes to shared config")
    p_propose.add_argument("--set", dest="set_items", action="append", default=[], help="Override parameter KEY=VALUE")
    p_propose.add_argument("--overrides", type=str, default=None, help="JSON object of overrides")
    p_propose.add_argument("--note", required=True, type=str, help="Rationale note for proposal")
    p_propose.set_defaults(func=_handle_propose)

    # space
    p_space = subparsers.add_parser("space", help="Describe tunable parameter space")
    p_space.add_argument("--component", type=str, default="events", help="Component to describe")
    p_space.set_defaults(func=_handle_space)

    # best
    p_best = subparsers.add_parser("best", help="Get current best result from ledger")
    p_best.add_argument("--component", type=str, default="events", help="Component to get best")
    p_best.set_defaults(func=_handle_best)

    # ledger
    p_ledger = subparsers.add_parser("ledger", help="Get recent ledger entries")
    p_ledger.add_argument("--limit", type=int, default=20, help="Number of entries to return")
    p_ledger.add_argument("--component", type=str, default="events", help="Component to filter ledger")
    p_ledger.set_defaults(func=_handle_ledger)

    # list-components
    p_list = subparsers.add_parser("list-components", help="List registered harness components")
    p_list.set_defaults(func=_handle_list_components)

    # recache
    p_recache = subparsers.add_parser("recache", help="Request cache build on Kaggle")
    p_recache.add_argument("--window", required=True, type=float, help="Window size in seconds")
    p_recache.add_argument("--hop", required=True, type=float, help="Hop size in seconds")
    p_recache.add_argument("--model", required=True, type=str, help="Tagger model name")
    p_recache.set_defaults(func=_handle_recache)

    # serve-mcp
    p_serve = subparsers.add_parser("serve-mcp", help="Run MCP server over stdio")
    p_serve.set_defaults(func=_handle_serve_mcp)

    # build-dev
    p_build = subparsers.add_parser("build-dev", help="Build synthetic dev dataset and feature caches")
    p_build.add_argument("--esc50", required=True, type=str, help="Directory containing ESC-50 dataset")
    p_build.add_argument("--n-per-env", type=int, default=20, help="Clips per environment")
    p_build.add_argument("--seed", type=int, default=0, help="Random seed")
    p_build.add_argument("--out", type=str, default=None, help="Output directory")
    p_build.set_defaults(func=_handle_build_dev)

    # import-audioset
    p_as = subparsers.add_parser(
        "import-audioset", help="Install AudioSet strong-label eval clips into dev/holdout"
    )
    p_as.add_argument("--tsv", required=True, type=str, help="audioset_eval_strong.tsv")
    p_as.add_argument("--class-csv", required=True, type=str, help="AudioSet class_labels_indices.csv")
    p_as.add_argument("--audio", required=True, type=str, help="Directory of <segment_id>.wav/.flac")
    p_as.add_argument("--holdout-frac", type=float, default=0.2, help="Share of clips held out")
    p_as.add_argument("--build-caches", action="store_true", help="Also build feature caches (slow on CPU)")
    p_as.set_defaults(func=_handle_import_audioset)

    from soundthing.harness import runner

    runner.cli_register(subparsers)
    for name in ("run-experiment", "promote-code"):
        subparsers.choices[name].set_defaults(func=lambda a, _tools: runner.cli_handle(a))

    p_bc = subparsers.add_parser("build-cache", help="Build a component's caches for a labels dir")
    p_bc.add_argument("--component", type=str, default="events", help="Component")
    p_bc.add_argument("--labels", required=True, type=str, help="Labels directory")
    p_bc.add_argument("--audio", required=True, type=str, help="Audio directory (<video_id>.wav)")
    p_bc.set_defaults(func=_handle_build_cache)

    if not argv:
        parser.print_help(sys.stderr)
        return 2

    if tools is None:
        from soundthing.harness.mcp_server import TuningTools

        tools = TuningTools()

    try:
        args = parser.parse_args(argv)
        return args.func(args, tools)
    except (ValueError, RuntimeError, KeyError, PromoteError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
