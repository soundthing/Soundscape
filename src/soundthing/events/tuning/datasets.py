"""ESC-50 and FSD50K synthetic dev mixes for tuning.

Loads dataset mappings and synthesizes deterministic multi-event mixes with
exact ground truth timestamps and prior-weighted event sampling.
"""

from __future__ import annotations

import csv
import json
import warnings
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
    fold: str | int


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
    fold: str | int,
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
    available = list(fold_bg)
    while collected < total_samples and available:
        idx = int(rng.integers(0, len(available)))
        clip_audio = _load_mono_16k(available[idx].path)
        if len(clip_audio) == 0:
            available.pop(idx)
            continue
        chunks.append(clip_audio)
        collected += len(clip_audio)

    if not chunks:
        return rng.normal(0.0, target_rms, size=total_samples).astype(np.float32)

    bed = np.concatenate(chunks)
    if len(bed) < total_samples:
        bed = np.pad(bed, (0, total_samples - len(bed)))
    else:
        bed = bed[:total_samples]
    bed = bed.astype(np.float32)
    bed_rms = float(np.sqrt(np.mean(bed**2)))
    if bed_rms > 1e-12:
        bed = bed * (target_rms / bed_rms)
    else:
        bed = rng.normal(0.0, target_rms, size=total_samples).astype(np.float32)
    return bed


def _synthesize_mix(
    env: str,
    fold: str | int,
    sources: list[_SourceRecord],
    bg_sources: list[_SourceRecord],
    cfg: Config,
    mix_cfg: dict[str, Any],
    rng: np.random.Generator,
    video_id: str,
    labeler: str = "esc50-synthetic",
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
        labeler=labeler,
        segments=[LabeledSegment(start=0.0, end=float(clip_sec), env=env)],
        events=labeled_events,
    )

    clip_manifest = {
        "env": env,
        "fold": fold,
        "sources": manifest_sources,
    }

    return (mix, labels, clip_manifest)


def _merge_manifest(
    out_dir: Path,
    dataset: str,
    seed: int,
    n_per_env: int,
    new_clips: dict[str, Any],
) -> Path:
    """Merge newly synthesized clip records into the output directory manifest."""
    manifest_path = out_dir / "manifest.json"
    manifest_data: dict[str, Any] = {
        "dataset": dataset,
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

    manifest_data["dataset"] = dataset
    manifest_data["seed"] = seed
    manifest_data["n_per_env"] = n_per_env
    manifest_data["clips"].update(new_clips)

    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_path


def _validate_maps(datasets_cfg: dict[str, Any], cfg: Config) -> None:
    """Every class id a dataset label maps to must exist in the vocabulary."""
    valid_class_ids = set(cfg.class_ids)
    for section_val in datasets_cfg.values():
        if isinstance(section_val, dict) and "map" in section_val:
            for cat, cid in (section_val["map"] or {}).items():
                if cid not in valid_class_ids:
                    raise ValueError(
                        f"Mapped class id {cid!r} (for category {cat!r}) is not in cfg.class_ids"
                    )


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

    _validate_maps(datasets_cfg, cfg)

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

    _merge_manifest(out_dir, "esc50", seed, n_per_env, new_clips)

    return sorted(label_paths)


def fsd50k_mixes(
    fsd_dir: Path,
    cfg: Config,
    n_per_env: int,
    seed: int,
    out_dir: Path,
    split: str = "dev",
) -> list[Path]:
    """Generate synthetic dev mixes from FSD50K sources for each environment."""
    fsd_dir = Path(fsd_dir)
    out_dir = Path(out_dir)
    datasets_cfg = load_datasets_yaml()
    mix_cfg = datasets_cfg.get("mix", {})

    _validate_maps(datasets_cfg, cfg)

    fsd_section = datasets_cfg.get("fsd50k", {})
    fsd_map = fsd_section.get("map", {})
    fsd_bg = set(fsd_section.get("background", []))

    csv_path = fsd_dir / "FSD50K.ground_truth" / f"{split}.csv"
    if not csv_path.is_file():
        csv_path = fsd_dir / "ground_truth" / f"{split}.csv"
    if not csv_path.is_file():
        csv_path = fsd_dir / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"FSD50K {split}.csv not found at {fsd_dir / 'FSD50K.ground_truth' / f'{split}.csv'} or {fsd_dir / f'{split}.csv'}"
        )

    cand_dirs = [
        fsd_dir / f"FSD50K.{split}_audio",
        fsd_dir / f"{split}_audio",
        fsd_dir / "audio",
        fsd_dir,
    ]

    mapped_sources: list[_SourceRecord] = []
    bg_sources: list[_SourceRecord] = []

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("fname", "").strip()
            if not fname:
                continue
            wav_name = fname if fname.endswith(".wav") else f"{fname}.wav"
            wav_path: Path | None = None
            for d in cand_dirs:
                p = d / wav_name
                if p.is_file():
                    wav_path = p
                    break
            if wav_path is None:
                continue

            labels_str = row.get("labels", "").strip()
            tokens = [tok.strip() for tok in labels_str.split(",") if tok.strip()]
            mapped_classes = {fsd_map[tok] for tok in tokens if tok in fsd_map}
            has_bg = any(tok in fsd_bg for tok in tokens)

            if split == "dev":
                fold = (row.get("split") or split).strip()
            else:
                fold = "eval"

            if len(mapped_classes) == 1:
                cid = next(iter(mapped_classes))
                mapped_sources.append(
                    _SourceRecord(
                        path=wav_path,
                        category=labels_str,
                        class_id=cid,
                        fold=fold,
                    )
                )
            elif len(mapped_classes) == 0 and has_bg:
                bg_sources.append(
                    _SourceRecord(
                        path=wav_path,
                        category=labels_str,
                        class_id="",
                        fold=fold,
                    )
                )

    folds_present = sorted({s.fold for s in mapped_sources})

    out_audio_dir = out_dir / "audio"
    out_labels_dir = out_dir / "labels"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    sr = int(mix_cfg.get("sample_rate", 16000))
    label_paths: list[Path] = []
    new_clips: dict[str, Any] = {}

    for env_index, env in enumerate(cfg.env_ids):
        for i in range(n_per_env):
            rng = np.random.default_rng([seed, env_index, i])
            fold = (
                folds_present[i % len(folds_present)]
                if folds_present
                else split
            )
            video_id = f"fsd50kmix_{env}_s{seed}_{i:03d}"

            mix_audio, labels, clip_manifest = _synthesize_mix(
                env=env,
                fold=fold,
                sources=mapped_sources,
                bg_sources=bg_sources,
                cfg=cfg,
                mix_cfg=mix_cfg,
                rng=rng,
                video_id=video_id,
                labeler="fsd50k-synthetic",
            )

            audio_path = out_audio_dir / f"{video_id}.wav"
            label_path = out_labels_dir / f"{video_id}.json"

            sf.write(str(audio_path), mix_audio, sr, subtype="PCM_16")
            save_labels(labels, label_path)

            label_paths.append(label_path)
            new_clips[video_id] = clip_manifest

    _merge_manifest(out_dir, "fsd50k", seed, n_per_env, new_clips)

    return sorted(label_paths)


def _merge_same_class_events(events: list[LabeledEvent]) -> list[LabeledEvent]:
    """Merge overlapping intervals that share the same class_id."""
    if not events:
        return []
    by_class: dict[str, list[LabeledEvent]] = {}
    for ev in events:
        by_class.setdefault(ev.class_id, []).append(ev)

    merged: list[LabeledEvent] = []
    for cid, ev_list in by_class.items():
        ev_list.sort(key=lambda e: (e.start, e.end))
        cur_start = ev_list[0].start
        cur_end = ev_list[0].end
        for e in ev_list[1:]:
            if e.start <= cur_end:
                cur_end = max(cur_end, e.end)
            else:
                merged.append(LabeledEvent(start=cur_start, end=cur_end, class_id=cid))
                cur_start = e.start
                cur_end = e.end
        merged.append(LabeledEvent(start=cur_start, end=cur_end, class_id=cid))

    merged.sort(key=lambda e: (e.start, e.end))
    return merged


def audioset_strong_labels(
    tsv: Path,
    cfg: Config,
    out_dir: Path,
    class_labels_csv: Path | None = None,
) -> list[Path]:
    """Convert AudioSet strong-label eval annotations to ClipLabels format.

    Note: This is for baseline mode only, as AudioSet clips have no known environment.
    """
    if class_labels_csv is None:
        raise ValueError(
            "class_labels_csv is required: path to AudioSet class_labels_indices.csv "
            "(available from AudioSet download or PANNs checkpoint)"
        )

    tsv = Path(tsv)
    class_labels_csv = Path(class_labels_csv)
    out_dir = Path(out_dir)

    if not tsv.is_file():
        raise FileNotFoundError(f"AudioSet TSV file not found at {tsv}")
    if not class_labels_csv.is_file():
        raise FileNotFoundError(f"class_labels_csv not found at {class_labels_csv}")

    mid_to_display: dict[str, str] = {}
    with open(class_labels_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("mid", "").strip()
            display_name = row.get("display_name", "").strip()
            if mid and display_name:
                mid_to_display[mid] = display_name

    display_to_class: dict[str, str] = {}
    ambiguous_names: set[str] = set()
    for cid in cfg.class_ids:
        sc = cfg.sound_class(cid)
        for name in sc.audioset:
            if name in ambiguous_names:
                continue
            if name in display_to_class and display_to_class[name] != cid:
                warnings.warn(
                    f"AudioSet display name {name!r} mapped to multiple classes "
                    f"({display_to_class[name]!r}, {cid!r}); skipping as ambiguous",
                    UserWarning,
                )
                ambiguous_names.add(name)
                del display_to_class[name]
            else:
                display_to_class[name] = cid

    mid_to_class: dict[str, str] = {}
    for mid, dname in mid_to_display.items():
        if dname in display_to_class:
            mid_to_class[mid] = display_to_class[dname]

    datasets_cfg = load_datasets_yaml()
    duration = float(datasets_cfg.get("audioset", {}).get("clip_sec", 10.0))

    segments_events: dict[str, list[tuple[float, float, str]]] = {}
    with open(tsv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is not None:
            col_names = [h.lstrip("#").strip() for h in header]
            col_idx = {name: idx for idx, name in enumerate(col_names)}
            seg_idx = col_idx.get("segment_id", 0)
            start_idx = col_idx.get("start_time_seconds", 1)
            end_idx = col_idx.get("end_time_seconds", 2)
            label_idx = col_idx.get("label", 3)

            for row in reader:
                if not row or len(row) <= max(seg_idx, start_idx, end_idx, label_idx):
                    continue
                seg_id = row[seg_idx].strip()
                if not seg_id:
                    continue
                if seg_id not in segments_events:
                    segments_events[seg_id] = []
                try:
                    start_t = float(row[start_idx].strip())
                    end_t = float(row[end_idx].strip())
                except ValueError:
                    continue
                label_mid = row[label_idx].strip()
                segments_events[seg_id].append((start_t, end_t, label_mid))

    out_labels_dir = out_dir / "labels"
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    label_paths: list[Path] = []

    for seg_id, raw_events in segments_events.items():
        valid_events: list[LabeledEvent] = []
        for start_t, end_t, label_mid in raw_events:
            if label_mid not in mid_to_class:
                continue
            cid = mid_to_class[label_mid]
            s = max(0.0, min(start_t, duration))
            e = max(0.0, min(end_t, duration))
            if e > s:
                valid_events.append(LabeledEvent(start=s, end=e, class_id=cid))

        merged_events = _merge_same_class_events(valid_events)
        labels = ClipLabels(
            video_id=seg_id,
            duration=duration,
            labeler="audioset-strong",
            segments=[],
            events=merged_events,
        )
        label_path = out_labels_dir / f"{seg_id}.json"
        save_labels(labels, label_path)
        label_paths.append(label_path)

    return sorted(label_paths)


def build_dev_caches(
    audio_dir: Path,
    labels_dir: Path,
    cfg: Config,
    *,
    models: Any = None,
    overwrite: bool = False,
    root: Path | None = None,
) -> list[Path]:
    """Build or verify feature caches for development clips."""
    from . import cache as cache_mod
    from . import holdout

    audio_dir = Path(audio_dir)
    labels_dir = Path(labels_dir)

    all_json = (
        sorted(p for p in labels_dir.glob("*.json") if p.is_file())
        if labels_dir.is_dir()
        else []
    )
    ids = sorted(p.stem for p in all_json)
    holdout.assert_disjoint(ids)

    key = cache_mod.cache_key(cfg)

    missing_audio_count = 0
    clips_with_audio: list[tuple[str, Path]] = []
    for clip_id in ids:
        found_audio: Path | None = None
        for ext in (".wav", ".flac"):
            cand = audio_dir / f"{clip_id}{ext}"
            if cand.is_file():
                found_audio = cand
                break
        if found_audio is None:
            missing_audio_count += 1
        else:
            clips_with_audio.append((clip_id, found_audio))

    if missing_audio_count > 0:
        warnings.warn(
            f"{missing_audio_count} labels have no matching audio in {audio_dir} and were skipped",
            UserWarning,
        )

    to_build: list[tuple[str, Path, Path]] = []
    cache_paths: list[Path] = []
    for clip_id, audio_path in clips_with_audio:
        out_path = cache_mod.cache_path(clip_id, key, root)
        cache_paths.append(out_path)
        if not out_path.exists() or overwrite:
            to_build.append((clip_id, audio_path, out_path))

    if to_build and models is None:
        from ..detector import get_models

        models = get_models(cfg)

    for clip_id, audio_path, out_path in to_build:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        c = cache_mod.build_cache(data, sr, clip_id, cfg, models=models)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache_mod.save_cache(c, out_path)

    return sorted(cache_paths)
