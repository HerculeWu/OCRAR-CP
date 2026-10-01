"""Tests for `crosscat.gaia_quality` — DR3 quality boolean masks.

Fixture-driven unit tests for each Lindegren+2018 / Fabricius+2021 /
Riello+2021 cut, plus a small DR3-stack integration check.
"""

from __future__ import annotations

import numpy as np
import pytest

from crosscat.gaia_quality import (
    chromaticity_mask,
    combine_and,
    excess_noise_mask,
    fabricius2021_binary_mask,
    lindegren2018_ruwe_mask,
    parallax_snr_mask,
    riello2021_flux_excess_mask,
    solution_type_mask,
)


# ---------------------------------------------------------------------------
# lindegren2018_ruwe_mask
# ---------------------------------------------------------------------------


def test_lindegren2018_ruwe():
    """RUWE strictly < 1.4 passes; threshold itself and above fail; NaN fails."""
    ruwe = np.array([1.0, 1.39, 1.4, 1.5, 10.0, np.nan])
    out = lindegren2018_ruwe_mask(ruwe)
    expected = np.array([True, True, False, False, False, False])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# fabricius2021_binary_mask
# ---------------------------------------------------------------------------


def test_fabricius2021_binary():
    """ipd_frac_multi_peak ≤ 2 AND ipd_gof_harmonic_amplitude < 0.1.

    Tests the 4×4 grid of boundary values plus NaN handling on either column.
    """
    # Grid: ipd_frac_multi_peak ∈ {0, 1, 2, 3} × ipd_gof ∈ {0.05, 0.099, 0.1, 0.5}
    fracs = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    gofs = np.array(
        [0.05, 0.05, 0.05, 0.05,
         0.099, 0.099, 0.099, 0.099,
         0.10, 0.10, 0.10, 0.10,
         0.50, 0.50, 0.50, 0.50]
    )
    out = fabricius2021_binary_mask(fracs, gofs)
    expected = np.array(
        [True, True, True, False,
         True, True, True, False,
         False, False, False, False,
         False, False, False, False]
    )
    np.testing.assert_array_equal(out, expected)

    # NaN handling: NaN in either column → reject
    fracs_nan = np.array([0, 1, np.nan, 0])
    gofs_nan = np.array([0.05, np.nan, 0.05, 0.05])
    out_nan = fabricius2021_binary_mask(fracs_nan, gofs_nan)
    expected_nan = np.array([True, False, False, True])
    np.testing.assert_array_equal(out_nan, expected_nan)


# ---------------------------------------------------------------------------
# excess_noise_mask
# ---------------------------------------------------------------------------


def test_excess_noise():
    """astrometric_excess_noise < 1.0 passes; threshold itself fails; NaN fails."""
    en = np.array([0.0, 0.5, 1.0, 1.5, np.nan])
    out = excess_noise_mask(en, threshold=1.0)
    expected = np.array([True, True, False, False, False])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# riello2021_flux_excess_mask
# ---------------------------------------------------------------------------


def test_riello2021_flux_excess():
    """1 + 0.015 C² ≤ excess_factor ≤ 1.3 + 0.06 C², C = bp_rp."""
    # Blue case: bp_rp = 0.2 → C² = 0.04 → lo = 1.0006, hi = 1.3024
    # Red case:  bp_rp = 2.0 → C² = 4.00 → lo = 1.060,  hi = 1.540
    bp_rp = np.array([
        0.2,   # blue, in-window
        0.2,   # blue, below lo
        0.2,   # blue, above hi
        2.0,   # red, in-window
        2.0,   # red, below lo
        2.0,   # red, above hi
        np.nan,
        1.0,
    ])
    excess = np.array([
        1.15,  # blue in
        0.99,  # blue too small
        1.40,  # blue too large
        1.20,  # red in
        1.00,  # red too small
        1.60,  # red too large
        1.10,  # NaN bp_rp
        np.nan,  # NaN excess
    ])
    out = riello2021_flux_excess_mask(bp_rp, excess)
    expected = np.array([True, False, False, True, False, False, False, False])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# parallax_snr_mask
# ---------------------------------------------------------------------------


def test_parallax_snr():
    """Positive parallax AND SNR ≥ snr_min."""
    # ϖ=1.0, σ=0.05 → SNR=20 PASS
    # ϖ=1.0, σ=0.15 → SNR≈6.7 FAIL
    # ϖ=0.5, σ=0.04 → SNR=12.5 PASS (small but positive parallax)
    # ϖ=-0.1     → negative FAIL
    # ϖ=NaN      → FAIL
    plx = np.array([1.0, 1.0, 0.5, -0.1, np.nan])
    plx_err = np.array([0.05, 0.15, 0.04, 0.05, 0.05])
    out = parallax_snr_mask(plx, plx_err, snr_min=10.0)
    expected = np.array([True, False, True, False, False])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# solution_type_mask
# ---------------------------------------------------------------------------


def test_solution_type():
    """31 (5p) accepted by default; 95 (6p) accepted only when accept_6p=True."""
    codes = np.array([31, 95, 3, 5, 7, 31])
    out_strict = solution_type_mask(codes, accept_6p=False)
    expected_strict = np.array([True, False, False, False, False, True])
    np.testing.assert_array_equal(out_strict, expected_strict)

    out_lax = solution_type_mask(codes, accept_6p=True)
    expected_lax = np.array([True, True, False, False, False, True])
    np.testing.assert_array_equal(out_lax, expected_lax)


# ---------------------------------------------------------------------------
# chromaticity_mask
# ---------------------------------------------------------------------------


def test_chromaticity():
    """5p sources need ν_eff ∈ [1.24, 1.72]; non-5p sources pass unconditionally."""
    nu_eff = np.array([1.20, 1.24, 1.50, 1.72, 1.75, np.nan])
    params_5p = np.full(6, 31)
    out_5p = chromaticity_mask(nu_eff, params_5p)
    expected_5p = np.array([False, True, True, True, False, False])
    np.testing.assert_array_equal(out_5p, expected_5p)

    # Non-5p sources: ν_eff value (including NaN) is irrelevant — always True.
    params_6p = np.full(6, 95)
    out_6p = chromaticity_mask(nu_eff, params_6p)
    expected_6p = np.full(6, True)
    np.testing.assert_array_equal(out_6p, expected_6p)


# ---------------------------------------------------------------------------
# combine_and
# ---------------------------------------------------------------------------


def test_combine_and():
    """Element-wise AND across N boolean masks; shape mismatch / empty input raise."""
    m1 = np.array([True, True, True, False])
    m2 = np.array([True, True, False, True])
    m3 = np.array([True, False, True, True])
    out = combine_and(m1, m2, m3)
    expected = np.array([True, False, False, False])
    np.testing.assert_array_equal(out, expected)

    # Two-mask call
    out2 = combine_and(m1, m2)
    np.testing.assert_array_equal(out2, np.array([True, True, False, False]))

    # Single-mask call returns the mask itself
    out1 = combine_and(m1)
    np.testing.assert_array_equal(out1, m1)

    # Shape mismatch → ValueError
    with pytest.raises(ValueError):
        combine_and(m1, np.array([True, False]))

    # Empty *masks → ValueError
    with pytest.raises(ValueError):
        combine_and()


# ---------------------------------------------------------------------------
# Integration: DR3 quality stack on a 100-row fixture
# ---------------------------------------------------------------------------


def test_pipeline_dr3_stack(rng):
    """100-row fixture passed through the full DR3 stack retains 30-70% of rows.

    Distributions roughly match Gaia DR3 cone-search expectations:
    most stars are well-behaved 5p sources but the tails of each diagnostic
    fail a few percent.
    """
    n = 100
    # Distributions chosen so most stars pass each cut individually, but
    # the AND-stack still removes a non-trivial tail (typical Gaia DR3
    # cone result: ~50% retention).
    ruwe = rng.normal(1.05, 0.12, size=n).clip(0.8, None)
    ipd_frac = rng.choice([0, 1, 2, 3], size=n, p=[0.7, 0.15, 0.1, 0.05])
    ipd_gof = rng.uniform(0.0, 0.15, size=n)
    excess_noise = np.abs(rng.normal(0.2, 0.3, size=n))
    bp_rp = rng.uniform(0.3, 2.5, size=n)
    C2 = bp_rp ** 2
    flux_lo = 1.0 + 0.015 * C2
    flux_hi = 1.3 + 0.06 * C2
    # 90% inside the Riello window, 10% in its outer skirt
    inside = rng.uniform(flux_lo, flux_hi, size=n)
    outside = rng.uniform(flux_lo - 0.05, flux_hi + 0.05, size=n)
    flux_excess = np.where(rng.uniform(size=n) < 0.9, inside, outside)
    plx = rng.uniform(0.5, 20.0, size=n)
    plx_err = plx / rng.uniform(8.0, 30.0, size=n)
    params = rng.choice([31, 95], size=n, p=[0.9, 0.1])
    # ν_eff well inside window for most stars, with a few stragglers
    nu_eff = rng.normal(1.5, 0.12, size=n).clip(1.0, 2.0)

    m1 = lindegren2018_ruwe_mask(ruwe)
    m2 = fabricius2021_binary_mask(ipd_frac, ipd_gof)
    m3 = excess_noise_mask(excess_noise)
    m4 = riello2021_flux_excess_mask(bp_rp, flux_excess)
    m5 = parallax_snr_mask(plx, plx_err)
    m6 = solution_type_mask(params, accept_6p=False)
    m7 = chromaticity_mask(nu_eff, params)
    final = combine_and(m1, m2, m3, m4, m5, m6, m7)

    kept = int(final.sum())
    assert 30 <= kept <= 70, (
        f"DR3 quality stack retained {kept}/{n} — outside reasonable [30, 70]"
    )
    assert final.dtype == np.bool_
    assert final.shape == (n,)
