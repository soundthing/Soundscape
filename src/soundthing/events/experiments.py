"""Wrong-environment experiments for sound event detection (roadmap item 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..contracts import EnvSegment
from .conditioning import prior_weights


def wrong_env_segments(segments: list[EnvSegment], cfg: Config) -> list[EnvSegment]:
    """Return segments with the same spans, where each segment's environment is
    replaced with the one in cfg.env_ids (excluding the top one) having the lowest
    cosine similarity in prior weight vectors over cfg.class_ids.
    """
    if not segments:
        return []

    env_vectors: dict[str, np.ndarray] = {
        env_id: prior_weights(cfg, {env_id: 1.0}, cfg.class_ids)
        for env_id in cfg.env_ids
    }
    env_norms: dict[str, float] = {
        env_id: float(np.linalg.norm(vec))
        for env_id, vec in env_vectors.items()
    }

    def _cosine_sim(env1: str, env2: str) -> float:
        n1 = env_norms[env1]
        n2 = env_norms[env2]
        if n1 <= 1e-12 or n2 <= 1e-12:
            return 0.0
        return float(np.dot(env_vectors[env1], env_vectors[env2]) / (n1 * n2))

    wrong_segs: list[EnvSegment] = []
    for s in segments:
        top_env = max(sorted(s.probs.keys()), key=lambda k: s.probs[k])
        candidates = [e for e in cfg.env_ids if e != top_env]
        if not candidates:
            chosen_env = top_env
        else:
            chosen_env = min(candidates, key=lambda cand: (_cosine_sim(top_env, cand), cand))

        wrong_segs.append(
            EnvSegment(
                start=s.start,
                end=s.end,
                env=chosen_env,
                confidence=1.0,
                probs={chosen_env: 1.0},
            )
        )

    return wrong_segs


def wrong_env_report(
    runs_dir: str | Path,
    labels_dir: str | Path,
    cfg: Config,
) -> dict[str, Any]:
    """Score baseline, conditioned, and wrong-environment detections against hand labels."""
    from ..contracts import load_events, load_inputs, load_labels, load_segments
    from ..evaluate import EventCounts, check_labels, score_events
    from . import detector

    runs_p = Path(runs_dir)
    labels_p = Path(labels_dir)
    tol = float(cfg.pipeline.get("eval", {}).get("onset_tolerance_sec", 0.5))

    report: dict[str, Any] = {
        "onset_tolerance_sec": tol,
        "clips": {},
        "totals": {},
    }

    modes = ("baseline", "conditioned", "wrong_env")
    levels = ("fine", "coarse")
    totals = {(m, lvl): EventCounts() for m in modes for lvl in levels}

    for label_path in sorted(labels_p.glob("*.json")):
        labels = load_labels(label_path)
        problems = check_labels(labels, cfg)
        if problems:
            raise ValueError("\n".join(problems))

        run_dir = runs_p / labels.video_id
        inputs_path = run_dir / "inputs.json"
        env_path = run_dir / "env_segments.json"

        if not (inputs_path.exists() and env_path.exists()):
            continue

        inputs = load_inputs(inputs_path)
        orig_segments = load_segments(env_path)

        clip: dict[str, Any] = {}

        # Run detector with wrong_env_segments
        bad_segments = wrong_env_segments(orig_segments, cfg)
        wrong_preds = detector.detect(inputs, bad_segments, cfg)
        for lvl in levels:
            c = score_events(wrong_preds, labels, cfg, tol, coarse=(lvl == "coarse"))
            totals[("wrong_env", lvl)] += c
            clip[f"wrong_env.{lvl}"] = c.summary()

        # Score stored events.baseline.json / events.conditioned.json if present
        for mode in ("baseline", "conditioned"):
            ev_file = run_dir / f"events.{mode}.json"
            if ev_file.exists():
                stored_preds = load_events(ev_file)
                for lvl in levels:
                    c = score_events(stored_preds, labels, cfg, tol, coarse=(lvl == "coarse"))
                    totals[(mode, lvl)] += c
                    clip[f"{mode}.{lvl}"] = c.summary()

        report["clips"][labels.video_id] = clip

    for (mode, lvl), counts in totals.items():
        if counts.minutes > 0:
            report["totals"][f"{mode}.{lvl}"] = counts.summary()

    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys
    from ..config import load_config

    parser = argparse.ArgumentParser(description="Wrong-environment event detection experiment")
    parser.add_argument("--config", default=None, help="config directory (default: ./config)")
    parser.add_argument("--runs", default="runs", help="runs directory (default: runs)")
    parser.add_argument("--labels", default="labels", help="labels directory (default: labels)")
    parser.add_argument("--out", default=None, help="write JSON report to this file")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    report = wrong_env_report(args.runs, args.labels, cfg)
    text = json.dumps(report, indent=2)

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
