"""Small embedding helpers shared by the environment model branches."""

from __future__ import annotations

import numpy as np


def as_array(features) -> np.ndarray:
    """Accept tensor outputs from supported Transformers versions."""
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    return features.detach().cpu().numpy().astype(np.float32)


def normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-12)


def prompt_vectors(model, processor, groups: tuple[tuple[str, ...], ...],
                   pad_to_max: bool = False) -> np.ndarray:
    """Average normalized prompt embeddings within each environment."""
    import torch

    prompts = [prompt for group in groups for prompt in group]
    if any(not group for group in groups):
        raise ValueError("every environment needs prompts for this modality")
    tokens = processor(text=prompts, padding="max_length" if pad_to_max else True,
                       return_tensors="pt")
    with torch.inference_mode():
        features = as_array(model.get_text_features(**tokens))
    features = normalize(features)
    vectors = []
    offset = 0
    for group in groups:
        count = len(group)
        vectors.append(normalize(features[offset : offset + count].mean(axis=0)))
        offset += count
    return np.stack(vectors)
