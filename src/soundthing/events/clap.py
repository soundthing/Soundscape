"""CLAP zero-shot scoring for vocabulary classes AudioSet doesn't cover well."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Config


class Clap:
    """Wrapper around Hugging Face ClapModel + ClapProcessor."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: str | Any = "cpu",
        center: float = 0.2,
        scale: float = 20.0,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.center = center
        self.scale = scale

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings (n, D) for a list of text prompts."""
        if not prompts:
            return np.zeros((0, 512), dtype=np.float32)

        import torch

        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            norm = torch.norm(features, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            features = features / norm
        return features.cpu().numpy().astype(np.float32)

    def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return L2-normalised float32 embedding (D,) for mono audio (N,), resampling to 48 kHz."""
        import torch
        import torchaudio

        if sr != 48000:
            t = torch.as_tensor(audio, dtype=torch.float32)
            t = torchaudio.functional.resample(t, sr, 48000)
            audio = t.cpu().numpy()

        inputs = self.processor(audios=audio, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_audio_features(**inputs)
            norm = torch.norm(features, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            features = features / norm
        emb = features[0].cpu().numpy().astype(np.float32)
        return emb


def load_clap(cfg: Config) -> Clap:
    """Load Hugging Face CLAP model and processor."""
    clap_cfg = cfg.pipeline.get("events", {}).get("clap", {})
    model_name = clap_cfg.get("model", "laion/clap-htsat-unfused")
    calib = clap_cfg.get("calibration", {})
    center = float(calib.get("center", 0.2))
    scale = float(calib.get("scale", 20.0))

    import torch
    from transformers import ClapModel, ClapProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClapModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = ClapProcessor.from_pretrained(model_name)
    return Clap(model=model, processor=processor, device=device, center=center, scale=scale)


def text_embeddings(clap: Any, cfg: Config) -> np.ndarray:
    """(C, D) in cfg.classes order: mean of the class's `clap` prompt embeddings,
    re-normalised; zero row for classes without prompts.
    """
    class_embs: list[np.ndarray | None] = []
    dim: int | None = None

    for c in cfg.classes:
        if c.clap:
            emb = clap.embed_text(c.clap)  # (n, D)
            if dim is None and emb.shape[0] > 0:
                dim = emb.shape[1]
            mean_vec = emb.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-12:
                mean_vec = mean_vec / norm
            else:
                mean_vec = np.zeros(dim if dim is not None else 512, dtype=np.float32)
            class_embs.append(mean_vec.astype(np.float32))
        else:
            class_embs.append(None)

    if dim is None:
        dim = 512

    out = np.zeros((len(cfg.classes), dim), dtype=np.float32)
    for i, vec in enumerate(class_embs):
        if vec is not None:
            out[i] = vec
    return out


def calibrate(
    sims: np.ndarray,
    text_emb: np.ndarray,
    center: float = 0.2,
    scale: float = 20.0,
) -> np.ndarray:
    """Calibrate cosine similarities to [0, 1] scores via sigmoid.

    Zero rows in text_emb (no prompts) score exactly 0.
    """
    has_prompts = np.any(text_emb != 0, axis=-1)
    x = (sims - center) * scale
    x_clipped = np.clip(x, -50.0, 50.0)
    calibrated = 1.0 / (1.0 + np.exp(-x_clipped))
    scores = np.where(has_prompts, calibrated, 0.0)
    return scores.astype(np.float32)


def clap_scores(
    clap: Any,
    audio: np.ndarray,
    sr: int,
    text_emb: np.ndarray,
    center: float | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """(C,) calibrated scores in [0, 1] for one window.

    Cosine similarity between the audio embedding and each class text embedding row,
    calibrated as sigmoid((sim - center) * scale).
    Rows that are all zero (no prompts) score exactly 0.
    """
    if center is None:
        center = float(getattr(clap, "center", 0.2))
    if scale is None:
        scale = float(getattr(clap, "scale", 20.0))

    audio_emb = clap.embed_audio(audio, sr)  # (D,)
    sim = text_emb @ audio_emb  # (C,)
    return calibrate(sim, text_emb, center, scale)
