"""Audio branch: CLAP zero-shot similarity of segment audio to env prompts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil
import subprocess

import numpy as np

from ..config import Config, Environment
from ..contracts import ExtractedInputs
from .embeddings import as_array, normalize, prompt_vectors


@lru_cache(maxsize=2)
def _model(model_id: str):
    from transformers import AutoProcessor, ClapModel

    return ClapModel.from_pretrained(model_id).eval(), AutoProcessor.from_pretrained(model_id)


@lru_cache(maxsize=8)
def _text_vectors(model_id: str, groups: tuple[tuple[str, ...], ...]) -> np.ndarray:
    model, processor = _model(model_id)
    return prompt_vectors(model, processor, groups)


def _audio_at_clap_rate(inputs: ExtractedInputs, sample_rate: int) -> np.ndarray:
    """Decode the best available track directly to mono float PCM, without side files."""
    source = inputs.audio_stereo_path if inputs.audio_stereo_path and Path(inputs.audio_stereo_path).is_file() else inputs.audio_path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to prepare CLAP audio")
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"],
        check=True, capture_output=True,
    )
    audio = np.frombuffer(result.stdout, dtype="<f4")
    if len(audio) == 0:
        raise ValueError(f"no decodable audio in {source}")
    return audio


def audio_env_logits(
    inputs: ExtractedInputs,
    spans: list[tuple[float, float]],
    envs: list[Environment],
    cfg: Config,
) -> np.ndarray:
    """Return (S, E) similarity logits: CLAP audio embeddings over ~10 s chunks of
    each span (mean-pooled), compared to the mean text embedding of each env's
    audio_prompts. Consider masking speech regions so dialogue doesn't dominate."""
    import torch

    model_id = cfg.pipeline["environment"]["audio"]["model"]
    model, processor = _model(model_id)
    sample_rate = processor.feature_extractor.sampling_rate
    audio = _audio_at_clap_rate(inputs, sample_rate)
    texts = _text_vectors(model_id, tuple(tuple(e.audio_prompts) for e in envs))
    chunk_samples = 10 * sample_rate
    chunks = []
    owners = []
    for index, (start, end) in enumerate(spans):
        first = max(0, min(len(audio), int(round(start * sample_rate))))
        last = max(first, min(len(audio), int(round(end * sample_rate))))
        if last == first:
            raise ValueError(f"empty audio span {start}–{end}")
        for offset in range(first, last, chunk_samples):
            chunks.append(audio[offset : min(offset + chunk_samples, last)])
            owners.append(index)

    embeddings = []
    for offset in range(0, len(chunks), 2):
        batch = processor(audio=chunks[offset : offset + 2], sampling_rate=sample_rate,
                          return_tensors="pt", padding=True)
        with torch.inference_mode():
            embeddings.append(normalize(as_array(model.get_audio_features(**batch))))
    embeddings = np.concatenate(embeddings)
    rows = [normalize(embeddings[np.array(owners) == i].mean(axis=0)) for i in range(len(spans))]
    return np.stack(rows) @ texts.T
