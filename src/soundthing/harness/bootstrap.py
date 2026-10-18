"""Paired bootstrap acceptance test for candidate configurations.

Computes pooled F_beta objective and runs paired bootstrap resampling over clips
to determine if a candidate's gain over the best known result is statistically
significant (lower confidence bound > 0).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from soundthing.harness.objective import ClipMatch, f_beta


def pooled_f_beta(
    tp_fine: float, tp_coarse: float, fp: float, fn: float, beta: float = 0.5
) -> float:
    """Compute pooled micro F_beta score."""
    m = ClipMatch(
        video_id="",
        minutes=0.0,
        tp_fine=tp_fine,
        tp_coarse=tp_coarse,
        fp=fp,
        fn=fn,
    )
    return f_beta([m], beta=beta)


def _get_counts(entry: Any, mode: str) -> tuple[float, float, float, float]:
    mode_data = entry[mode] if isinstance(entry, dict) else getattr(entry, mode)
    if isinstance(mode_data, dict):
        return (
            float(mode_data["tp_fine"]),
            float(mode_data["tp_coarse"]),
            float(mode_data["fp"]),
            float(mode_data["fn"]),
        )
    return (
        float(mode_data.tp_fine),
        float(mode_data.tp_coarse),
        float(mode_data.fp),
        float(mode_data.fn),
    )


def _vectorized_f_beta(totals: np.ndarray, beta: float) -> np.ndarray:
    tp_fine = totals[:, 0]
    tp_coarse = totals[:, 1]
    fp = totals[:, 2]
    fn = totals[:, 3]

    credit = tp_fine + 0.5 * tp_coarse
    fp_prime = fp + 0.5 * tp_coarse
    fn_prime = fn + 0.5 * tp_coarse

    prec_denom = credit + fp_prime
    rec_denom = credit + fn_prime

    p = np.zeros_like(credit, dtype=np.float64)
    valid_p = prec_denom > 0
    p[valid_p] = credit[valid_p] / prec_denom[valid_p]

    r = np.zeros_like(credit, dtype=np.float64)
    valid_r = rec_denom > 0
    r[valid_r] = credit[valid_r] / rec_denom[valid_r]

    b2 = beta ** 2
    f_denom = b2 * p + r
    f = np.zeros_like(p, dtype=np.float64)
    valid_f = (p + r > 0) & (f_denom > 0)
    f[valid_f] = (1.0 + b2) * p[valid_f] * r[valid_f] / f_denom[valid_f]
    return f


def paired_bootstrap(
    cand_per_clip: dict[str, Any],
    best_per_clip: dict[str, Any],
    n: int = 1000,
    seed: int = 0,
    beta: float = 0.5,
    mode: str = "conditioned",
) -> dict[str, float]:
    """Paired bootstrap test over common clips.

    Resamples clips with replacement n times for both candidate and best,
    computing the pooled F_beta difference for each resample.
    Returns: {"gain": float, "lo": float, "hi": float}
    """
    clips = sorted(set(cand_per_clip.keys()) & set(best_per_clip.keys()))
    if not clips:
        raise ValueError("No common clips between candidate and best")

    c_count = len(clips)
    cand_arr = np.array([_get_counts(cand_per_clip[vid], mode) for vid in clips], dtype=np.float64)
    best_arr = np.array([_get_counts(best_per_clip[vid], mode) for vid in clips], dtype=np.float64)

    cand_sum = cand_arr.sum(axis=0)
    best_sum = best_arr.sum(axis=0)
    f_cand = pooled_f_beta(cand_sum[0], cand_sum[1], cand_sum[2], cand_sum[3], beta=beta)
    f_best = pooled_f_beta(best_sum[0], best_sum[1], best_sum[2], best_sum[3], beta=beta)
    gain = float(f_cand - f_best)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c_count, size=(n, c_count))

    cand_samples = cand_arr[idx]
    best_samples = best_arr[idx]

    cand_totals = cand_samples.sum(axis=1)
    best_totals = best_samples.sum(axis=1)

    f_cand_resampled = _vectorized_f_beta(cand_totals, beta=beta)
    f_best_resampled = _vectorized_f_beta(best_totals, beta=beta)

    diffs = f_cand_resampled - f_best_resampled

    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))

    return {"gain": gain, "lo": lo, "hi": hi}


def accepts(ci: dict[str, float]) -> bool:
    """Return True if lower confidence bound is strictly positive."""
    return bool(ci["lo"] > 0)
