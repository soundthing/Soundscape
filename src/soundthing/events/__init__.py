"""Sound event detection — OWNER: Person 1.

Input:  ExtractedInputs (+ list[EnvSegment] from Person 2, or None for baseline)
Output: list[CaptionEvent]  ->  events.json

    core.score_window        audio window -> (C,) raw class scores   (pure; reusable for streaming)
    tagger                   AudioSet tagger mapped to vocabulary
    vad.speech_regions       speech mask
    conditioning.apply_priors  FrameScores x EnvSegments -> FrameScores   (skipped for baseline)
    postprocess.*            smoothing, thresholds, merging, coarse fallback, priority filter
"""
