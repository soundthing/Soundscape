"""ESC-50 and FSD50K synthetic dev mixes for tuning.

Loads dataset mappings and synthesizes deterministic multi-event mixes with
exact ground truth timestamps and prior-weighted event sampling.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from ...config import Config
from ...contracts import ClipLabels, LabeledEvent, LabeledSegment, save_labels
from .. import conditioning


@dataclass(frozen=True)
class _SourceRecord:
    path: Path
    category: str
    class_id: str
    fold: int


def load_datasets_yaml() -> dict[str, Any]:
    """Load the datasets configuration YAML."""
    path = Path(__file__).with_name("datasets.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def prior_weights(cfg: Config, env_probs: dict[str, float], class_ids: list[str]) -> np.ndarray:
    """Wrapper around conditioning.prior_weights allowing monkeypatching."""
    return conditioning.prior_weights(cfg, env_probs, class_ids)


def _load_mono_16k(path: Path | str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file as 16 kHz mono float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        resampled = F.resample(tensor, sr, target_sr)
        data = resampled.squeeze(0).numpy()
    return data.astype(np.float32)


def _trim_silence(audio: np.ndarray, sr: int, mix_cfg: dict[str, Any]) -> tuple[int, int]:
    """Find start and end sample of active audio using an energy gate."""
    trim_frame_sec = float(mix_cfg.get("trim_frame_sec", 0.02))
    trim_rel_db = float(mix_cfg.get("trim_rel_db", -35.0))
    trim_floor_dbfs = float(mix_cfg.get("trim_floor_dbfs", -60.0))
    max_event_sec = float(mix_cfg.get("max_event_sec", 8.0))

    frame_len = max(1, int(round(sr * trim_frame_sec)))
    if len(audio) == 0:
        return (0, 0)

    n_frames = (len(audio) + frame_len - 1) // frame_len
    frame_dbs = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = audio[i * frame_len : min((i + 1) * frame_len, len(audio))]
        rms = float(np.sqrt(np.mean(chunk**2)))
        frame_dbs[i] = 20.0 * np.log10(rms) if rms > 1e-12 else -120.0

    peak_frame_db = float(np.max(frame_dbs))
    gate = max(peak_frame_db + trim_rel_db, trim_floor_dbfs)
    active_indices = np.where(frame_dbs >= gate)[0]

    if len(active_indices) == 0:
        return (0, 0)

    first_idx = int(active_indices[0])
    last_idx = int(active_indices[-1])
    start_sample = first_idx * frame_len
    end_sample = min((last_idx + 1) * frame_len, len(audio))

    max_samples = int(round(sr * max_event_sec))
    if end_sample - start_sample > max_samples:
        end_sample = start_sample + max_samples

    return (start_sample, end_sample)


def _make_bed(
    bg_sources: list[_SourceRecord],
    fold: int,
    total_samples: int,
    bed_dbfs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Concatenate background clips from the fold and scale to bed_dbfs RMS."""
    fold_bg = [s for s in bg_sources if s.fold == fold]
    target_rms = 10.0 ** (bed_dbfs / 20.0)
    if not fold_bg:
        return rng.normal(0.0, target_rms, size=total_samples).astype(np.float32)

    chunks: list[np.ndarray] = []
    collected = 0
    while collected < total_samples:
        idx = int(rng.integers(0, len(fold_bg)))
        clip_audio = _load_mono_16k(fold_bg[idx].path)
        if len(clip_audio) == 0:
            continue
        chunks.append(clip_audio)
        collected += len(clip_audio)

    if not chunks:
        return rng.normal(0.0, target_rms, size=total_samples).astype(np.float32)

    bed = np.concatenate(chunks)[:total_samples].astype(np.float32)
    bed_rms = float(np.sqrt(np.mean(bed**2)))
    if bed_rms > 1e-12:
        bed = bed * (target_rms / bed_rms)
    else:
        bed = rng.normal(0.0, target_rms, size=total_samples).astype(np.float32)
    return bed


def _synthesize_mix(
    env: str,
    fold: int,
    sources: list[_SourceRecord],
    bg_sources: list[_SourceRecord],
    cfg: Config,
    mix_cfg: dict[str, Any],
    rng: np.random.Generator,
    video_id: str,
) -> tuple[np.ndarray, ClipLabels, dict[str, Any]]:
    """Synthesize audio, ClipLabels, and manifest metadata for one mix."""
    clip_sec = float(mix_cfg["clip_sec"])
    sr = int(mix_cfg["sample_rate"])
    total_samples = int(round(clip_sec * sr))
    min_gap_sec = float(mix_cfg["min_gap_sec"])

    fold_sources = [s for s in sources if s.fold == fold]
    candidate_classes = sorted({s.class_id for s in fold_sources})

    chosen_sources: list[_SourceRecord] = []
    if candidate_classes:
        weights = prior_weights(cfg, {env: 1.0}, candidate_classes)
        w_sum = float(np.sum(weights))
        if w_sum > 0:
            p = weights / w_sum
        else:
            p = np.full(len(candidate_classes), 1.0 / len(candidate_classes))
        p = p / np.sum(p)

        min_k, max_k = mix_cfg["events_per_mix"]
        k = int(rng.integers(min_k, max_k + 1))
        chosen_classes = rng.choice(candidate_classes, size=k, replace=True, p=p)
        for cls in chosen_classes:
            matching = [s for s in fold_sources if s.class_id == cls]
            src_idx = int(rng.integers(0, len(matching)))
            chosen_sources.append(matching[src_idx])

    loaded_events: list[tuple[np.ndarray, _SourceRecord]] = []
    for src in chosen_sources:
        audio = _load_mono_16k(src.path, sr)
        s_start, s_end = _trim_silence(audio, sr, mix_cfg)
        if s_end <= s_start:
            continue
        trimmed = audio[s_start:s_end]
        loaded_events.append((trimmed, src))

    if loaded_events:
        perm = rng.permutation(len(loaded_events))
        loaded_events = [loaded_events[idx] for idx in perm]

    cursor_sec = float(rng.uniform(0.0, min_gap_sec))
    placed_events: list[dict[str, Any]] = []

    for trimmed_audio, src in loaded_events:
        gap_sec = float(rng.uniform(min_gap_sec, 2.0 * min_gap_sec))
        start_sec = cursor_sec + gap_sec
        start_sample = int(round(start_sec * sr))
        end_sample = start_sample + len(trimmed_audio)
        end_sec = end_sample / sr

        if end_sample > total_samples or end_sec > clip_sec:
            break

        onset = round(start_sample / sr, 3)
        end = round(end_sample / sr, 3)

        placed_events.append({
            "audio": trimmed_audio,
            "src": src,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start": onset,
            "end": end,
        })
        cursor_sec = end_sec

    bed_dbfs = float(mix_cfg["bed_dbfs"])
    bed = _make_bed(bg_sources, fold, total_samples, bed_dbfs, rng)
    bed_rms = 10.0 ** (bed_dbfs / 20.0)

    min_snr, max_snr = mix_cfg["snr_db"]
    mix = bed.copy()
    manifest_sources: list[dict[str, Any]] = []

    for ev in placed_events:
        snr_db = float(rng.uniform(min_snr, max_snr))
        target_ev_rms = bed_rms * (10.0 ** (snr_db / 20.0))
        ev_audio = ev["audio"]
        ev_rms = float(np.sqrt(np.mean(ev_audio**2)))
        if ev_rms > 1e-12:
            scaled_ev = ev_audio * (target_ev_rms / ev_rms)
        else:
            scaled_ev = ev_audio

        mix[ev["start_sample"] : ev["end_sample"]] += scaled_ev

        manifest_sources.append({
            "file": Path(ev["src"].path).name,
            "category": ev["src"].category,
            "class_id": ev["src"].class_id,
            "fold": ev["src"].fold,
            "start": ev["start"],
            "end": ev["end"],
            "snr_db": round(snr_db, 2),
        })

    peak = float(np.max(np.abs(mix)))
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    mix = np.clip(mix, -1.0, 1.0)

    labeled_events = [
        LabeledEvent(
            start=ev["start"],
            end=ev["end"],
            class_id=ev["src"].class_id,
            note=Path(ev["src"].path).name,
        )
        for ev in placed_events
    ]
    labeled_events.sort(key=lambda e: (e.start, e.end))

    labels = ClipLabels(
        video_id=video_id,
        duration=float(clip_sec),
        labeler="esc50-synthetic",
        segments=[LabeledSegment(start=0.0, end=float(clip_sec), env=env)],
        events=labeled_events,
    )

    clip_manifest = {
        "env": env,
        "fold": fold,
        "sources": manifest_sources,
    }

    return (mix, labels, clip_manifest)


def esc50_mixes(
    esc50_dir: Path,
    cfg: Config,
    n_per_env: int,
    seed: int,
    out_dir: Path,
) -> list[Path]:
    """Generate synthetic dev mixes from ESC-50 sources for each environment."""
    esc50_dir = Path(esc50_dir)
    out_dir = Path(out_dir)
    datasets_cfg = load_datasets_yaml()
    mix_cfg = datasets_cfg.get("mix", {})

    valid_class_ids = set(cfg.class_ids)
    for section_name, section_val in datasets_cfg.items():
        if isinstance(section_val, dict) and "map" in section_val:
            for cat, cid in section_val["map"].items():
                if cid not in valid_class_ids:
                    raise ValueError(
                        f"Mapped class id {cid!r} (for category {cat!r}) is not in cfg.class_ids"
                    )

    esc_section = datasets_cfg.get("esc50", {})
    esc_map = esc_section.get("map", {})
    esc_bg = set(esc_section.get("background", []))

    csv_path = esc50_dir / "esc50.csv"
    if not csv_path.is_file():
        csv_path = esc50_dir / "meta" / "esc50.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"ESC-50 CSV not found at {esc50_dir / 'esc50.csv'} or {esc50_dir / 'meta' / 'esc50.csv'}"
        )

    mapped_sources: list[_SourceRecord] = []
    bg_sources: list[_SourceRecord] = []

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            fold = int(row["fold"])
            category = row["category"]

            wav_path = esc50_dir / "audio" / filename
            if not wav_path.is_file():
                wav_path = esc50_dir / filename
            if not wav_path.is_file():
                continue

            if category in esc_map:
                mapped_sources.append(
                    _SourceRecord(
                        path=wav_path,
                        category=category,
                        class_id=esc_map[category],
                        fold=fold,
                    )
                )
            if category in esc_bg:
                bg_sources.append(
                    _SourceRecord(
                        path=wav_path,
                        category=category,
                        class_id="",
                        fold=fold,
                    )
                )

    folds_present = sorted({s.fold for s in mapped_sources})

    audio_dir = out_dir / "audio"
    labels_dir = out_dir / "labels"
    audio_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    sr = int(mix_cfg.get("sample_rate", 16000))
    label_paths: list[Path] = []
    new_clips: dict[str, Any] = {}

    for env_index, env in enumerate(cfg.env_ids):
        for i in range(n_per_env):
            rng = np.random.default_rng([seed, env_index, i])
            fold = folds_present[i % len(folds_present)] if folds_present else 1
            video_id = f"esc50mix_{env}_s{seed}_{i:03d}"

            mix_audio, labels, clip_manifest = _synthesize_mix(
                env=env,
                fold=fold,
                sources=mapped_sources,
                bg_sources=bg_sources,
                cfg=cfg,
                mix_cfg=mix_cfg,
                rng=rng,
                video_id=video_id,
            )

            audio_path = audio_dir / f"{video_id}.wav"
            label_path = labels_dir / f"{video_id}.json"

            sf.write(str(audio_path), mix_audio, sr, subtype="PCM_16")
            save_labels(labels, label_path)

            label_paths.append(label_path)
            new_clips[video_id] = clip_manifest

    manifest_path = out_dir / "manifest.json"
    manifest_data: dict[str, Any] = {
        "dataset": "esc50",
        "seed": seed,
        "n_per_env": n_per_env,
        "clips": {},
    }
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest_data = loaded
        except Exception:
            pass

    if "clips" not in manifest_data or not isinstance(manifest_data["clips"], dict):
        manifest_data["clips"] = {}

    manifest_data["dataset"] = "esc50"
    manifest_data["seed"] = seed
    manifest_data["n_per_env"] = n_per_env
    manifest_data["clips"].update(new_clips)

    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return sorted(label_paths)
