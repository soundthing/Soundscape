"""Audio branch: CLAP zero-shot similarity of segment audio to env prompts."""

from __future__ import annotations

import numpy as np

from ..config import Config, Environment
from ..contracts import ExtractedInputs


def audio_env_logits(
    inputs: ExtractedInputs,
    spans: list[tuple[float, float]],
    envs: list[Environment],
    cfg: Config,
) -> np.ndarray:
    """Return (S, E) similarity logits: CLAP audio embeddings over ~10 s chunks of
    each span (mean-pooled), compared to the mean text embedding of each env's
    audio_prompts. Consider masking speech regions so dialogue doesn't dominate."""
    raise NotImplementedError("TODO(Person 2): CLAP audio branch")
