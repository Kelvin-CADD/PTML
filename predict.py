# -*- coding: utf-8 -*-
"""Batch-score MGF spectra with a frozen DENS7 checkpoint.

Example
-------
python predict.py \
  --checkpoint checkpoints/dens7_rf_nist.joblib \
  --mgf spectra.mgf \
  --out scores.tsv

The output contains one row per scored spectrum, with P(PT) and the 0.5
threshold decision.  Spectra with fewer than 50 cleaned peaks are skipped and
reported in the console summary.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np

from dens7 import FEATURE_NAMES, dens7_from_top50, top50


def parse_pepmass(raw: str) -> float | None:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return float(parts[-1])
    except ValueError:
        return None


def iter_mgf(path: Path) -> Iterator[dict]:
    """Yield {'title', 'pepmass', 'peaks'} blocks from an MGF file."""
    title = ""
    pepmass = None
    peaks: list[tuple[float, float]] = []
    in_block = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text == "BEGIN IONS":
                title, pepmass, peaks, in_block = "", None, [], True
                continue
            if text == "END IONS":
                if in_block:
                    yield {"title": title, "pepmass": pepmass, "peaks": peaks}
                in_block = False
                continue
            if not in_block or not text:
                continue
            if "=" in text and not text[0].isdigit():
                key, value = text.split("=", 1)
                key = key.strip().upper()
                if key == "TITLE":
                    title = value.strip()
                elif key == "PEPMASS":
                    pepmass = parse_pepmass(value)
                continue
            parts = text.replace("\t", " ").split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    if in_block:
        yield {"title": title, "pepmass": pepmass, "peaks": peaks}


def load_checkpoint(path: Path):
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError("checkpoint must be a dict saved by train_delivery_models.py")
    if list(obj.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("checkpoint feature order does not match this dens7.py")
    return obj


def score_peaks(model, peaks: list[tuple[float, float]], precursor_mz: float) -> tuple[float, int]:
    mz = np.asarray([p[0] for p in peaks], dtype=float)
    intensity = np.asarray([p[1] for p in peaks], dtype=float)
    top_mz, top_sqrt, _, n_clean = top50(mz, intensity, precursor_mz)
    z = dens7_from_top50(top_mz, top_sqrt, n_clean=n_clean)
    p = float(model.predict_proba(z.reshape(1, -1))[0, 1])
    return p, n_clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score MGF spectra with a DENS7 checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to .joblib checkpoint")
    parser.add_argument("--mgf", required=True, help="input MGF file")
    parser.add_argument("--out", required=True, help="output TSV file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="decision threshold; default is the value stored in the checkpoint")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ckpt = load_checkpoint(Path(args.checkpoint))
    model = ckpt["model"]
    threshold = float(ckpt.get("decision_threshold", 0.5) if args.threshold is None else args.threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_scored = n_skipped = n_no_precursor = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["title", "precursor_mz", "n_clean", "p_pt", "predicted_pt"])
        for block in iter_mgf(Path(args.mgf)):
            if block["pepmass"] is None:
                n_no_precursor += 1
                continue
            try:
                p, n_clean = score_peaks(model, block["peaks"], block["pepmass"])
            except ValueError:
                n_skipped += 1
                continue
            writer.writerow([
                block["title"],
                f"{block['pepmass']:.6f}",
                n_clean,
                f"{p:.6f}",
                int(p >= threshold),
            ])
            n_scored += 1

    if not args.quiet:
        print(f"checkpoint: {args.checkpoint}")
        print(f"threshold : {threshold:.3f}")
        print(f"scored    : {n_scored}")
        print(f"skipped (<50 cleaned peaks): {n_skipped}")
        print(f"skipped (no PEPMASS)      : {n_no_precursor}")
        print(f"output    : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
