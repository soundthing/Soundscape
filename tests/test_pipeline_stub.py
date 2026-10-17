from soundthing import contracts as C
from soundthing.captions import Cue, to_srt, to_vtt
from soundthing.evaluate import match_events
from soundthing.pipeline import evaluate_runs, main, run_video


def test_end_to_end_with_stubs(cfg, fake_inputs, tmp_path):
    run_dir = run_video(fake_inputs, tmp_path / "runs", cfg)
    inputs = C.load_inputs(run_dir / "inputs.json")
    C.validate_segments(C.load_segments(run_dir / "env_segments.json"), inputs.duration)
    for mode in ("baseline", "conditioned"):
        C.load_events(run_dir / f"events.{mode}.json")
        assert (run_dir / f"captions.{mode}.vtt").read_text().startswith("WEBVTT")

    # Use the baseline output as fake ground truth: eval must give a perfect score.
    preds = C.load_events(run_dir / "events.baseline.json")
    labels = C.ClipLabels(
        video_id="clip01", duration=30.0, labeler="test",
        segments=[C.LabeledSegment(0, 30, "kitchen")],
        events=[C.LabeledEvent(e.start, e.end, e.class_id) for e in preds],
    )
    C.save_labels(labels, tmp_path / "labels" / "clip01.json")
    report = evaluate_runs(run_dir.parent, tmp_path / "labels", cfg)
    if preds:
        assert report["totals"]["baseline.fine"]["f1"] == 1.0
    assert "env_accuracy" in report["totals"]


def test_cli_check_config():
    assert main(["check-config"]) == 0


def test_match_tolerance():
    truth = [(1.0, "a"), (5.0, "b")]
    assert match_events([(1.4, "a"), (5.0, "a")], truth, 0.5) == (1, 1, 1)
    assert match_events([(1.6, "a")], truth, 0.5) == (0, 1, 2)


def test_caption_formats():
    cues = [Cue(3661.5, 3663.25, "[door slams]")]
    assert "01:01:01.500 --> 01:01:03.250" in to_vtt(cues)
    assert to_srt(cues).startswith("1\n01:01:01,500 --> 01:01:03,250\n[door slams]")
