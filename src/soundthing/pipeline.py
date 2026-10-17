"""End-to-end orchestration and CLI — OWNER: Person 3.

Run layout (one directory per video):

    runs/<video_id>/
      inputs.json                ExtractedInputs
      audio.wav, audio_stereo.wav, frames/
      env_segments.json          list[EnvSegment]
      events.baseline.json       list[CaptionEvent]  (no environment conditioning)
      events.conditioned.json    list[CaptionEvent]
      captions.baseline.vtt
      captions.conditioned.vtt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import contracts as C
from .captions import write_captions
from .config import Config, ConfigError, load_config, read_audioset_labels
from .evaluate import EventCounts, check_labels, env_accuracy, score_events
from .extract import extract, video_id_for
from .stages import get_environment_classifier, get_event_detector

MODES = ("baseline", "conditioned")


def run_video(source: str | Path, runs_dir: str | Path, cfg: Config, modes=MODES) -> Path:
    """`source` is a video file, or an existing inputs.json to skip extraction."""
    source = Path(source)
    if source.suffix == ".json":
        inputs = C.load_inputs(source)
        run_dir = source.parent
    else:
        run_dir = Path(runs_dir) / video_id_for(source)
        inputs = extract(source, run_dir, cfg)
        C.save_inputs(inputs, run_dir / "inputs.json")

    segments = None
    if "conditioned" in modes:
        segments = get_environment_classifier(cfg)(inputs, cfg)
        C.validate_segments(segments, inputs.duration)
        C.save_segments(segments, run_dir / "env_segments.json")

    detect = get_event_detector(cfg)
    ccfg = cfg.pipeline["captions"]
    for mode in modes:
        events = detect(inputs, segments if mode == "conditioned" else None, cfg)
        C.save_events(events, run_dir / f"events.{mode}.json")
        write_captions(events, run_dir / f"captions.{mode}.vtt", ccfg["min_cue_sec"], ccfg["max_cue_sec"])
    return run_dir


def evaluate_runs(runs_dir: str | Path, labels_dir: str | Path, cfg: Config) -> dict:
    tol = cfg.pipeline["eval"]["onset_tolerance_sec"]
    report: dict = {"onset_tolerance_sec": tol, "clips": {}, "totals": {}}
    totals = {(m, lvl): EventCounts() for m in MODES for lvl in ("fine", "coarse")}
    env_scores = []
    for label_path in sorted(Path(labels_dir).glob("*.json")):
        labels = C.load_labels(label_path)
        problems = check_labels(labels, cfg)
        if problems:
            raise ValueError("\n".join(problems))
        run_dir = Path(runs_dir) / labels.video_id
        clip: dict = {}
        for mode in MODES:
            ev_path = run_dir / f"events.{mode}.json"
            if not ev_path.exists():
                continue
            preds = C.load_events(ev_path)
            for lvl in ("fine", "coarse"):
                counts = score_events(preds, labels, cfg, tol, coarse=(lvl == "coarse"))
                totals[(mode, lvl)] += counts
                clip[f"{mode}.{lvl}"] = counts.summary()
        seg_path = run_dir / "env_segments.json"
        if seg_path.exists() and labels.segments:
            acc = env_accuracy(C.load_segments(seg_path), labels)
            env_scores.append(acc)
            clip["env_accuracy"] = acc
        report["clips"][labels.video_id] = clip
    for (mode, lvl), counts in totals.items():
        if counts.minutes:
            report["totals"][f"{mode}.{lvl}"] = counts.summary()
    if env_scores:
        report["totals"]["env_accuracy"] = sum(env_scores) / len(env_scores)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="soundthing")
    ap.add_argument("--config", default=None, help="config directory (default: ./config)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="process a video (or inputs.json) end to end")
    p.add_argument("source", nargs="+")
    p.add_argument("--runs", default="runs")
    p.add_argument("--mode", choices=[*MODES, "both"], default="both")

    p = sub.add_parser("eval", help="score runs against hand labels")
    p.add_argument("--runs", default="runs")
    p.add_argument("--labels", default="labels")
    p.add_argument("--out", default=None, help="write JSON report here")

    p = sub.add_parser("check-config", help="validate config/ cross-references")
    p.add_argument("--audioset", default=None, help="class_labels_indices.csv to verify AudioSet names")

    p = sub.add_parser("serve", help="run the web demo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)

    if args.cmd == "check-config":
        cfg = load_config(args.config, validate=False)
        try:
            labels = read_audioset_labels(args.audioset) if args.audioset else None
            warnings = cfg.validate(labels)
        except ConfigError as e:
            print(e, file=sys.stderr)
            return 1
        for w in warnings:
            print("warning:", w)
        print(f"ok: {len(cfg.environments)} environments, {len(cfg.classes)} classes "
              f"({len(cfg.safety_ids())} safety)")
        return 0

    cfg = load_config(args.config)
    if args.cmd == "run":
        modes = MODES if args.mode == "both" else (args.mode,)
        for src in args.source:
            print(run_video(src, args.runs, cfg, modes))
    elif args.cmd == "eval":
        report = evaluate_runs(args.runs, args.labels, cfg)
        text = json.dumps(report, indent=2)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        print(json.dumps(report["totals"], indent=2))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("soundthing.server.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
