"""Visual branch: CLIP/SigLIP zero-shot similarity of sampled frames to env prompts.

Visuals are used for environment context ONLY; sound events come from audio.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import Config, Environment
from ..contracts import ExtractedInputs
from .embeddings import as_array, normalize, prompt_vectors


@lru_cache(maxsize=2)
def _model(model_id: str):
    from transformers import AutoProcessor, SiglipModel

    return SiglipModel.from_pretrained(model_id).eval(), AutoProcessor.from_pretrained(model_id)


@lru_cache(maxsize=8)
def _text_vectors(model_id: str, groups: tuple[tuple[str, ...], ...]) -> np.ndarray:
    model, processor = _model(model_id)
    return prompt_vectors(model, processor, groups, pad_to_max=True)


def visual_env_logits(
    inputs: ExtractedInputs,
    spans: list[tuple[float, float]],
    envs: list[Environment],
    cfg: Config,
) -> np.ndarray:
    """Return (S, E) similarity logits: for each span, mean-pool image embeddings of
    frames with start <= inputs.frame_time(i) < end, compare to the mean text
    embedding of each env's visual_prompts. Spans with no frames: use nearest frame."""
    import torch
    from PIL import Image

    paths = inputs.frame_paths()
    if not paths:
        raise FileNotFoundError(f"no extracted frames in {inputs.frames_dir}")
    model_id = cfg.pipeline["environment"]["visual"]["model"]
    model, processor = _model(model_id)
    texts = _text_vectors(model_id, tuple(tuple(e.visual_prompts) for e in envs))
    times = np.array([int(path.stem.rsplit("_", 1)[1]) / inputs.fps for path in paths])

    features = []
    for offset in range(0, len(paths), 8):
        images = []
        for path in paths[offset : offset + 8]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        batch = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            features.append(normalize(as_array(model.get_image_features(**batch))))
    features = np.concatenate(features)

    rows = []
    for start, end in spans:
        selected = np.flatnonzero((times >= start) & (times < end))
        if not len(selected):
            selected = np.array([np.argmin(np.abs(times - (start + end) / 2))])
        rows.append(normalize(features[selected].mean(axis=0)))
    return np.stack(rows) @ texts.T
