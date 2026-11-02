# -*- coding: utf-8 -*-
"""DENS7: seven peak-density descriptors for PT/non-PT MS/MS screening.

The feature definition is frozen for the delivery checkpoints.  It follows the
preprocessing used to build the training matrices:

1. drop non-finite peaks and peaks with intensity <= 0;
2. merge exactly equal m/z values by summing intensity;
3. remove the precursor and its +1/+2 isotope peaks within +/-10 ppm;
4. keep the 50 most intense peaks (ties: lower m/z first);
5. divide by the total intensity of the cleaned peaks, then take sqrt;
6. compute the seven descriptors below.

Only the 50 selected peaks enter the descriptors.  ``n_clean`` is the number of
peaks left after step 3, i.e. before the top-50 truncation.

Feature order is part of the model contract and must not be changed.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

FEATURE_NAMES = [
    "n_clean",
    "n_above_1pct_top50",
    "gini_top50",
    "effective_peaks_entropy",
    "median_mz_gap",
    "top1_share_top50",
    "top5_share_top50",
]

FEATURE_DESCRIPTIONS = {
    "n_clean": "Number of peaks after precursor/isotope removal (before top-50); log1p is used by the model.",
    "n_above_1pct_top50": "Number of top-50 peaks whose share of top-50 intensity is above 1%.",
    "gini_top50": "Gini coefficient of the top-50 intensity distribution.",
    "effective_peaks_entropy": "exp(Shannon entropy) of the top-50 intensity distribution.",
    "median_mz_gap": "Median adjacent m/z gap among the top-50 peaks, in Da.",
    "top1_share_top50": "Share of the most intense peak in the top-50 intensity sum.",
    "top5_share_top50": "Share of the five most intense peaks in the top-50 intensity sum.",
}

PRECURSOR_PPM = 10.0
ISOTOPE_OFFSETS = (0.0, 1.00335, 2.00671)
MIN_PEAKS = 50
TOP_N = 50


def _merge_duplicate_mz(mz: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mz.size <= 1:
        return mz, intensity
    order = np.argsort(mz, kind="mergesort")
    mz = mz[order]
    intensity = intensity[order]
    uniq, inv = np.unique(mz, return_inverse=True)
    if uniq.size == mz.size:
        return mz, intensity
    summed = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(summed, inv, intensity)
    return uniq, summed


def clean_peaks(
    mz: Sequence[float],
    intensity: Sequence[float],
    precursor_mz: float | None,
    ppm: float = PRECURSOR_PPM,
    isotope_offsets: Iterable[float] = ISOTOPE_OFFSETS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return peaks after zero/finite filtering, duplicate merging and precursor removal."""
    mz_arr = np.asarray(mz, dtype=np.float64).ravel()
    int_arr = np.asarray(intensity, dtype=np.float64).ravel()
    if mz_arr.size != int_arr.size:
        raise ValueError("mz and intensity must have the same length")
    if mz_arr.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    finite = np.isfinite(mz_arr) & np.isfinite(int_arr)
    mz_arr = mz_arr[finite]
    int_arr = int_arr[finite]
    positive = int_arr > 0
    mz_arr = mz_arr[positive]
    int_arr = int_arr[positive]
    if mz_arr.size == 0:
        return mz_arr, int_arr

    mz_arr, int_arr = _merge_duplicate_mz(mz_arr, int_arr)
    if precursor_mz is None or not np.isfinite(precursor_mz):
        return mz_arr, int_arr

    keep = np.ones(mz_arr.size, dtype=bool)
    for offset in isotope_offsets:
        center = float(precursor_mz) + float(offset)
        if center <= 0:
            continue
        ppm_error = np.abs(mz_arr - center) / center * 1e6
        keep &= ppm_error > float(ppm)
    return mz_arr[keep], int_arr[keep]


def top50(
    mz: Sequence[float],
    intensity: Sequence[float],
    precursor_mz: float | None,
    min_peaks: int = MIN_PEAKS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (top_mz, top_sqrt_norm, top_share, n_clean).

    ``top_share`` is the intensity share of each selected peak in the top-50
    sum.  ``top_sqrt_norm`` is the sqrt-transformed intensity used by the
    original feature matrix; it is returned for auditability.
    """
    mz_clean, int_clean = clean_peaks(mz, intensity, precursor_mz)
    n_clean = int(mz_clean.size)
    if n_clean < min_peaks:
        raise ValueError(f"need at least {min_peaks} cleaned peaks, got {n_clean}")

    order = np.lexsort((mz_clean, -int_clean))
    selected = order[:TOP_N]
    top_mz = mz_clean[selected]
    top_int = int_clean[selected]

    total = float(int_clean.sum())
    if total <= 0:
        raise ValueError("total cleaned intensity must be positive")
    top_norm = top_int / total
    top_sqrt = np.sqrt(top_norm)

    top_sum = float(top_int.sum())
    top_share = top_int / top_sum if top_sum > 0 else np.zeros_like(top_int)
    return top_mz, top_sqrt, top_share, n_clean


def _gini(values: np.ndarray) -> float:
    s = np.sort(np.asarray(values, dtype=np.float64))
    n = s.size
    if n == 0 or s.sum() <= 0:
        return float("nan")
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum()))


def dens7_from_top50(
    top_mz: Sequence[float],
    top_sqrt_norm: Sequence[float],
    n_clean: int | None = None,
) -> np.ndarray:
    """Compute DENS7 from an already selected top-50 peak list.

    ``top_mz`` and ``top_sqrt_norm`` must be in intensity-descending order,
    exactly as stored in the project feature matrices.  Use
    :func:`dens7_from_peaks` when starting from raw peak lists.
    """
    mz = np.asarray(top_mz, dtype=np.float64).ravel()
    norm = np.asarray(top_sqrt_norm, dtype=np.float64).ravel()
    if mz.size != TOP_N or norm.size != TOP_N:
        raise ValueError(f"expected exactly {TOP_N} top peaks, got mz={mz.size}, intensity={norm.size}")
    intensity = np.square(norm)
    total = float(intensity.sum())
    if total <= 0:
        raise ValueError("top-50 intensity sum must be positive")
    share = intensity / total

    n_clean_value = float(len(mz) if n_clean is None else n_clean)
    return np.asarray([
        np.log1p(n_clean_value),
        float((share > 0.01).sum()),
        _gini(share),
        float(np.exp(-(share * np.log(share + 1e-12)).sum())),
        float(np.median(np.diff(np.sort(mz)))),
        float(share[0]),
        float(share[:5].sum()),
    ], dtype=np.float64)


def dens7_from_peaks(
    mz: Sequence[float],
    intensity: Sequence[float],
    precursor_mz: float | None,
    min_peaks: int = MIN_PEAKS,
) -> np.ndarray:
    """Compute DENS7 from a raw centroided MS/MS peak list."""
    top_mz, top_sqrt, _, n_clean = top50(mz, intensity, precursor_mz, min_peaks=min_peaks)
    return dens7_from_top50(top_mz, top_sqrt, n_clean=n_clean)
