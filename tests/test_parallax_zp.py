"""Tests for `crosscat.parallax_zp` — Lindegren+2021b Z5/Z6 zero-point.

The wrapper delegates to the official ``gaiadr3-zeropoint`` package
(Ramos & Brown, code released with Lindegren+2021b). These tests cross-
check our project-style API against ``zpt.get_zpt`` directly and verify
correct dispatch by ``astrometric_params_solved``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crosscat.parallax_zp import apply_zp, z5, z6


# ---------------------------------------------------------------------------
# z5: 5-parameter solutions
# ---------------------------------------------------------------------------


def test_z5_quasar_global():
    """Typical QSO parameters (G≈19, ν_eff≈1.55, sin β=0.5) → roughly
    the −17 μas global QSO median noted in Lindegren+2021b §1.

    The single-source value depends on (G, ν_eff, β); a ±5 μas tolerance
    covers the spread for points close to the calibration centre.
    """
    val = z5(19.0, 1.55, 0.5).item()
    assert np.isfinite(val)
    assert -22.0 < val < -12.0, (
        f"z5(QSO-typical) = {val:.2f} μas — outside expected (-22, -12)"
    )


def test_z5_bright_blue_star():
    """Cross-check z5 against the underlying ``zpt.get_zpt`` to <0.5 μas.

    For G=10, ν_eff=1.7 μm⁻¹, sin β=0 (i.e. ecliptic equator), the
    official package returns a single mas value; our wrapper must agree
    to 0.5 μas after unit conversion.
    """
    from zero_point import zpt
    zpt.load_tables()

    g = 10.0
    nu = 1.7
    sin_b = 0.0
    ecl_lat_deg = float(np.degrees(np.arcsin(sin_b)))

    # Official package: pass arrays to avoid the numpy-2 scalar bug.
    ref_mas = zpt.get_zpt(
        np.array([g]),
        np.array([nu]),
        np.array([np.nan]),
        np.array([ecl_lat_deg]),
        np.array([31]),
        _warnings=False,
    )[0]
    ref_uas = float(ref_mas) * 1000.0

    ours = z5(g, nu, sin_b).item()
    assert abs(ours - ref_uas) < 0.5, (
        f"z5={ours:.3f} μas vs reference={ref_uas:.3f} μas (Δ>{0.5})"
    )


# ---------------------------------------------------------------------------
# z6: 6-parameter solutions
# ---------------------------------------------------------------------------


def test_z6_basic():
    """For G=14, pseudocolour=1.5, sin β=0.2, agree with ``zpt.get_zpt`` <0.5 μas."""
    from zero_point import zpt
    zpt.load_tables()

    g = 14.0
    pc = 1.5
    sin_b = 0.2
    ecl_lat_deg = float(np.degrees(np.arcsin(sin_b)))

    ref_mas = zpt.get_zpt(
        np.array([g]),
        np.array([np.nan]),
        np.array([pc]),
        np.array([ecl_lat_deg]),
        np.array([95]),
        _warnings=False,
    )[0]
    ref_uas = float(ref_mas) * 1000.0

    ours = z6(g, pc, sin_b).item()
    assert abs(ours - ref_uas) < 0.5, (
        f"z6={ours:.3f} μas vs reference={ref_uas:.3f} μas (Δ>{0.5})"
    )


# ---------------------------------------------------------------------------
# apply_zp: dispatch + 5p/6p/2p handling + out-of-range NaN
# ---------------------------------------------------------------------------


def test_apply_zp_dispatch():
    """5p uses z5; 6p uses z6; 2p passes parallax unchanged; OOR returns NaN."""
    df = pd.DataFrame(
        {
            "parallax": [10.0, 5.0, np.nan, 8.0],
            "phot_g_mean_mag": [12.0, 15.0, 19.0, 12.0],
            "colour": [1.55, 1.45, np.nan, 0.5],  # last row: ν_eff out of [1.1,1.9]
            "sin_ecl_lat": [0.0, 0.2, 0.0, 0.0],
            "astrometric_params_solved": [31, 95, 3, 31],
        }
    )

    out = apply_zp(
        df["parallax"].to_numpy(),
        df["phot_g_mean_mag"].to_numpy(),
        df["colour"].to_numpy(),
        df["sin_ecl_lat"].to_numpy(),
        df["astrometric_params_solved"].to_numpy(),
    )

    # 5p row: parallax should change (Z != 0)
    assert np.isfinite(out[0])
    assert out[0] != df["parallax"].iloc[0], "5p row parallax was not corrected"
    # Magnitude of correction should be < 0.1 mas (=100 μas) for sane inputs
    assert abs(out[0] - df["parallax"].iloc[0]) < 0.1

    # 6p row: same
    assert np.isfinite(out[1])
    assert out[1] != df["parallax"].iloc[1]
    assert abs(out[1] - df["parallax"].iloc[1]) < 0.1

    # 2p row: parallax unchanged (NaN stays NaN)
    assert np.isnan(out[2])

    # 5p row with ν_eff out of range: parallax → NaN
    assert np.isnan(out[3]), (
        f"out-of-range 5p produced {out[3]}, expected NaN"
    )

    # Cross-check rows 0 and 1 against direct z5/z6 calls
    z5_expected = z5(12.0, 1.55, 0.0).item()
    z6_expected = z6(15.0, 1.45, 0.2).item()
    assert abs(out[0] - (10.0 - z5_expected / 1000.0)) < 1e-6
    assert abs(out[1] - (5.0 - z6_expected / 1000.0)) < 1e-6


def test_apply_zp_array_broadcast(rng):
    """1000-row array: shape preserved, dtype float64, no NaN in valid range."""
    n = 1000
    g = rng.uniform(8.0, 20.0, size=n)
    nu = rng.uniform(1.2, 1.8, size=n)
    sin_b = rng.uniform(-0.9, 0.9, size=n)
    params = np.full(n, 31, dtype=int)
    plx = rng.uniform(1.0, 20.0, size=n)

    out = apply_zp(plx, g, nu, sin_b, params)

    assert out.shape == (n,), f"shape {out.shape} != ({n},)"
    assert out.dtype == np.float64, f"dtype {out.dtype} != float64"
    assert np.all(np.isfinite(out)), (
        f"unexpected NaN(s) in valid-range output: "
        f"{np.sum(~np.isfinite(out))} non-finite"
    )
    # Corrections are small (sub-mas), so corrected parallax should sit within
    # 1 mas of input.
    assert np.all(np.abs(out - plx) < 1.0)
