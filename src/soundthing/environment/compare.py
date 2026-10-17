"""Compare audio, visual, and fused environment accuracy on hand-labeled clips.

Run with: python -m soundthing.environment.compare --runs runs --labels labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..contracts import load_inputs, load_labels
from ..evaluate import check_labels, env_accuracy
from .classifier import classify


def compare(runs_dir: Path, labels_dir: Path, config_dir: Path | None = None) -> dict:
    cfg = load_config(config_dir)
    report: dict = {"clips": {}, "mean_accuracy": {}, "overall_accuracy": {},
                    "confusion_seconds": {}}
    for mode in ("audio", "visual", "fused"):
        report["confusion_seconds"][mode] = {}
    for path in sorted(labels_dir.glob("*.json")):
        labels = load_labels(path)
        problems = check_labels(labels, cfg)
        if problems:
            raise ValueError("\n".join(problems))
        if not labels.segments:
            continue
        inputs = load_inputs(runs_dir / labels.video_id / "inputs.json")
        predictions = {
            "audio": classify(inputs, cfg, use_visual=False),
            "visual": classify(inputs, cfg, use_audio=False),
            "fused": classify(inputs, cfg),
        }
        report["clips"][labels.video_id] = {}
        for mode, segments in predictions.items():
            report["clips"][labels.video_id][mode] = env_accuracy(segments, labels)
            confusion = report["confusion_seconds"][mode]
            for truth in labels.segments:
                row = confusion.setdefault(truth.env, {})
                for pred in segments:
                    overlap = max(0.0, min(truth.end, pred.end) - max(truth.start, pred.start))
                    if overlap:
                        row[pred.env] = row.get(pred.env, 0.0) + overlap
    if not report["clips"]:
        raise ValueError(f"no clips with environment labels found in {labels_dir}")
    for mode in ("audio", "visual", "fused"):
        values = [row[mode] for row in report["clips"].values()]
        report["mean_accuracy"][mode] = sum(values) / len(values)
        confusion = report["confusion_seconds"][mode]
        total = sum(seconds for row in confusion.values() for seconds in row.values())
        correct = sum(row.get(truth, 0.0) for truth, row in confusion.items())
        report["overall_accuracy"][mode] = correct / total if total else None
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--labels", type=Path, default=Path("labels"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.runs, args.labels, args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
