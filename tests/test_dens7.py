# -*- coding: utf-8 -*-
"""Contract tests for the DENS7 feature module and frozen checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dens7 import FEATURE_NAMES, dens7_from_peaks  # noqa: E402


def test_feature_contract():
    mz = np.linspace(100.0, 800.0, 60)
    intensity = np.linspace(1000.0, 100.0, 60)
    z = dens7_from_peaks(mz, intensity, precursor_mz=9999.0)
    assert z.shape == (7,)
    assert np.isfinite(z).all()
    assert z[0] == pytest.approx(np.log1p(60))


def test_precursor_and_isotope_removed():
    mz = np.concatenate([np.linspace(100.0, 800.0, 60), [500.0, 501.00335]])
    intensity = np.concatenate([np.linspace(1000.0, 100.0, 60), [5000.0, 4000.0]])
    z = dens7_from_peaks(mz, intensity, precursor_mz=500.0)
    assert z[0] == pytest.approx(np.log1p(60))


def test_too_few_peaks():
    with pytest.raises(ValueError):
        dens7_from_peaks(np.linspace(100.0, 200.0, 49), np.ones(49), 9999.0)


def test_checkpoint_loads_and_predicts():
    ckpt = joblib.load(ROOT / "checkpoints" / "dens7_rf_nist.joblib")
    assert ckpt["feature_names"] == FEATURE_NAMES
    assert ckpt["decision_threshold"] == 0.5
    model = ckpt["model"]
    z = np.zeros((1, 7))
    p = model.predict_proba(z)[0, 1]
    assert 0.0 <= p <= 1.0
