import json

import pytest

from soundthing import contracts as C
from soundthing.evaluate import check_labels

from .conftest import EXAMPLES


def test_examples_load(cfg):
    inputs = C.load_inputs(EXAMPLES / "inputs.json")
    segs = C.load_segments(EXAMPLES / "env_segments.json")
    C.validate_segments(segs, inputs.duration)
    for s in segs:
        assert set(s.probs) <= set(cfg.env_ids)
    for e in C.load_events(EXAMPLES / "events.json"):
        assert e.class_id in cfg.class_ids or e.class_id in cfg.coarse_groups
    assert check_labels(C.load_labels(EXAMPLES / "labels.json"), cfg) == []


def test_roundtrip(tmp_path):
    ev = C.CaptionEvent(1.0, 2.0, "door slams", "door_slam", 0.7, ["right"])
    C.save_events([ev], tmp_path / "e.json")
    assert C.load_events(tmp_path / "e.json") == [ev]
    assert ev.caption_text() == "[door slams, right]"


@pytest.mark.parametrize(
    "bad",
    [
        dict(start=2.0, end=1.0),
        dict(confidence=1.5),
        dict(modifiers=["sideways"]),
        dict(class_id=""),
    ],
)
def test_event_validation(bad):
    base = dict(start=1.0, end=2.0, label="x", class_id="x", confidence=0.5, modifiers=[])
    with pytest.raises(C.ContractError):
        C.CaptionEvent(**{**base, **bad}).validate()


def test_segments_must_cover_clip():
    s = C.EnvSegment(0.0, 5.0, "kitchen", 0.9, {"kitchen": 1.0})
    with pytest.raises(C.ContractError):
        C.validate_segments([s], duration=10.0)


def test_wrong_kind_rejected(tmp_path):
    C.save_events([], tmp_path / "e.json")
    with pytest.raises(C.ContractError):
        C.load_segments(tmp_path / "e.json")


@pytest.mark.parametrize("name,loader", [
    ("inputs.json", C.load_inputs),
    ("env_segments.json", C.load_segments),
    ("events.json", C.load_events),
    ("direction.json", C.load_direction),
    ("labels.json", C.load_labels),
])
def test_previous_schema_requires_regeneration(tmp_path, name, loader):
    payload = json.loads((EXAMPLES / name).read_text())
    assert payload["schema_version"] == C.SCHEMA_VERSION == 2
    payload["schema_version"] = 1
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    with pytest.raises(C.ContractError, match="schema_version 1 != 2"):
        loader(path)
