"""Visual branch: CLIP/SigLIP zero-shot similarity of sampled frames to env prompts.

Visuals are used for environment context ONLY; sound events come from audio.
"""

from __future__ import annotations

import numpy as np

from ..config import Config, Environment
from ..contracts import ExtractedInputs


def visual_env_logits(
    inputs: ExtractedInputs,
    spans: list[tuple[float, float]],
    envs: list[Environment],
    cfg: Config,
) -> np.ndarray:
    """Return (S, E) similarity logits: for each span, mean-pool image embeddings of
    frames with start <= inputs.frame_time(i) < end, compare to the mean text
    embedding of each env's visual_prompts. Spans with no frames: use nearest frame."""
    raise NotImplementedError("TODO(Person 2): CLIP/SigLIP visual branch")
