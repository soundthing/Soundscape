"""Approved additive direction contract and deterministic model-free fake."""
from pathlib import Path

import pytest

from soundthing import contracts as C


def test_direction_example_and_roundtrip(tmp_path):
    track = C.load_direction(Path(__file__).parents[1] / "docs/examples/direction.json")
    assert track.mode == "surround51"
    assert track.frames[2].levels["SR"] == .9
    C.save_direction(track, tmp_path / "track.json")
    assert C.load_direction(tmp_path / "track.json") == track
    unavailable = C.DirectionTrack(2, "unavailable", [], [])
    unavailable.validate()


@pytest.mark.parametrize("track", [
    C.DirectionTrack(float("nan"), "unavailable", [], []),
    C.DirectionTrack(1, "unknown", [], []),
    C.DirectionTrack(1, "unavailable", [], [C.DirectionFrame(0, {})]),
    C.DirectionTrack(1, "stereo", ["FL", "FR"], []),
    C.DirectionTrack(1, "stereo", ["FR", "FL"], [C.DirectionFrame(0, {"FL": 0, "FR": 0})]),
    C.DirectionTrack(1, "stereo", ["FL", "FR"], [C.DirectionFrame(0, {"LFE": 1})]),
    C.DirectionTrack(1, "stereo", ["FL", "FR"], [C.DirectionFrame(0, {"FL": float("nan"), "FR": 0})]),
    C.DirectionTrack(1, "stereo", ["FL", "FR"], [C.DirectionFrame(0, {"FL": 2, "FR": 0})]),
    C.DirectionTrack(1, "stereo", ["FL", "FR"], [C.DirectionFrame(0, {"FL": 0, "FR": 0}), C.DirectionFrame(0, {"FL": 0, "FR": 0})]),
    C.DirectionTrack(1, "stereo", ["FL", "FR"], [C.DirectionFrame(0, {"FL": 0, "FR": 0}), C.DirectionFrame(1, {"FL": 0, "FR": 0})]),
])
def test_direction_rejects_invalid_tracks(track):
    with pytest.raises(C.ContractError):
        track.validate()
