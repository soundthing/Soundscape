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
import logging
import sys
from pathlib import Path

from . import contracts as C
from .captions import merge_subtitle_files, read_subtitles, write_captions
from .config import Config, ConfigError, load_config, read_audioset_labels
from .evaluate import EventCounts, check_labels, env_accuracy, score_events
from .extract import extract, video_id_for, prepare_analysis_video
from .direction import ensure_direction
from .stages import get_environment_classifier, get_event_detector

MODES = ("baseline", "conditioned")
LOG = logging.getLogger(__name__)


def run_video(source: str | Path, runs_dir: str | Path, cfg: Config, modes=MODES,
              speech_subtitles: str | Path | None = None) -> Path:
    """`source` is a video file, or an existing inputs.json to skip extraction."""
    source = Path(source)
    if speech_subtitles is not None:
        planned_run = source.parent if source.suffix == ".json" else Path(runs_dir) / video_id_for(source)
        generated = {planned_run / f"captions.{mode}{merged}.vtt" for mode in MODES for merged in ("", ".merged")}
        if Path(speech_subtitles).resolve() in {path.resolve() for path in generated}:
            raise ValueError("Speech input must use a different filename from this run's generated caption tracks")
        read_subtitles(speech_subtitles)  # Validate before extraction/model work.
    if source.suffix == ".json":
        inputs = C.load_inputs(source)
        run_dir = source.parent
    else:
        run_dir = Path(runs_dir) / video_id_for(source)
        LOG.info("Extracting audio and frames: %s", source)
        inputs = extract(source, run_dir, cfg)
        C.save_inputs(inputs, run_dir / "inputs.json")

    # Soundtrack direction is independent of event confidence/model success.
    # Make it available to the player before the slower caption stages.
    ensure_direction(inputs.video_path, run_dir, inputs.duration)
    if cfg.pipeline["environment"]["backend"] == "real":
        compatible = prepare_analysis_video(inputs.video_path, run_dir)
        if str(compatible) != inputs.video_path:
            inputs.video_path = str(compatible)
            C.save_inputs(inputs, run_dir / "inputs.json")

    segments = None
    if "conditioned" in modes:
        LOG.info("Classifying environment (%s backend): %s", cfg.pipeline["environment"]["backend"], inputs.video_id)
        segments = get_environment_classifier(cfg)(inputs, cfg)
        C.validate_segments(segments, inputs.duration)
        C.save_segments(segments, run_dir / "env_segments.json")

    detect = get_event_detector(cfg)
    ccfg = cfg.pipeline["captions"]
    for mode in modes:
        LOG.info("Detecting %s events (%s backend): %s", mode, cfg.pipeline["events"]["backend"], inputs.video_id)
        events = detect(inputs, segments if mode == "conditioned" else None, cfg)
        C.save_events(events, run_dir / f"events.{mode}.json")
        caption_path = run_dir / f"captions.{mode}.vtt"
        write_captions(events, caption_path, ccfg["min_cue_sec"], ccfg["max_cue_sec"], hierarchy=cfg.hierarchy())
        LOG.info("Wrote %s (%d events)", caption_path, len(events))
        if speech_subtitles is not None:
            # Keep canonical sound-only captions intact for the YouTube overlay.
            merge_subtitle_files(caption_path, speech_subtitles, run_dir / f"captions.{mode}.merged.vtt")
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
    p.add_argument("--speech", help="existing .srt/.vtt speech subtitles (one source only); also writes .merged.vtt tracks")

    p = sub.add_parser("merge-captions", help="merge existing sound and speech subtitle files without running models")
    p.add_argument("sounds", help="sound caption .vtt/.srt")
    p.add_argument("speech", help="speech subtitle .vtt/.srt")
    p.add_argument("--out", required=True, help="combined output .vtt/.srt; different from the inputs")

    sub.add_parser("doctor", help="check local server, downloader, and cookie-file setup")
    p = sub.add_parser("sync-extension", help="copy extension runtime files into a Windows/Chrome folder")
    p.add_argument("--destination", default=None, help="target folder; WSL defaults to Windows Downloads/soundthing-extension")

    p = sub.add_parser("eval", help="score runs against hand labels")
    p.add_argument("--runs", default="runs")
    p.add_argument("--labels", default="labels")
    p.add_argument("--out", default=None, help="write JSON report here")

    p = sub.add_parser("check-config", help="validate config/ cross-references")
    p.add_argument("--audioset", default=None, help="class_labels_indices.csv to verify AudioSet names")

    p = sub.add_parser("serve", help="run the web demo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cookies", help="exported YouTube cookie file; overrides SOUNDTHING_YOUTUBE_COOKIES for this process")

    args = ap.parse_args(argv)

    if args.cmd == "merge-captions":
        try:
            merge_subtitle_files(args.sounds, args.speech, args.out)
        except (OSError, ValueError) as exc:
            print(f"Cannot merge captions: {exc}", file=sys.stderr)
            return 1
        print(args.out)
        return 0
    if args.cmd in {"doctor", "sync-extension", "serve"}:
        from .server.development import prepare_local_path, doctor, sync_extension
        prepare_local_path()
        if args.cmd == "doctor":
            return doctor()
        if args.cmd == "sync-extension":
            try:
                print(sync_extension(args.destination))
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"Cannot sync extension: {exc}", file=sys.stderr)
                return 1
            return 0
        if args.cookies is not None:
            import os
            os.environ["SOUNDTHING_YOUTUBE_COOKIES"] = str(Path(args.cookies).expanduser().resolve())
            from .server.youtube import _cookie_arguments
            try:
                with _cookie_arguments():
                    pass
            except (OSError, RuntimeError) as exc:
                print(f"Cannot start server: {exc}", file=sys.stderr)
                return 1
    if args.cmd == "run" and args.speech and len(args.source) != 1:
        ap.error("--speech requires exactly one video or inputs.json source")

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
            try:
                print(run_video(src, args.runs, cfg, modes, speech_subtitles=args.speech))
            except (OSError, ValueError) as exc:
                print(f"Cannot process {src}: {exc}", file=sys.stderr)
                return 1
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
