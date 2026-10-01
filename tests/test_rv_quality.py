"""Tests for `crosscat.rv_quality` — Katz+2023 DR3 RV quality gate.

Verifies the four Katz+2023 (arXiv:2206.05902) cuts and the magnitude-
dependent zero-point correction (Eq. vrcor / §6.1, "Magnitude trend"):

- ``katz2023_teff_mask``        — rv_template_teff ∈ [3900, 8000] K
- ``katz2023_error_mask``       — radial_velocity_error < 2 km/s
- ``katz2023_variability_mask`` — constant-star criterion from §10.2
- ``katz2023_rv_zp``            — magnitude-dependent ZP offset
- ``apply_rv_quality``          — pipeline that ANDs the three masks
                                  and adds ``radial_velocity_corrected``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crosscat.rv_quality import (
    apply_rv_quality,
    katz2023_error_mask,
    katz2023_rv_zp,
    katz2023_teff_mask,
    katz2023_variability_mask,
)


# ---------------------------------------------------------------------------
# katz2023_teff_mask
# ---------------------------------------------------------------------------


def test_teff_mask():
    """rv_template_teff ∈ [3900, 8000] inclusive at lo, inclusive at hi.

    Katz+2023 §10.2 (line 110) gives the criterion as
    ``rv_template_teff ∈ [3900, 8000]`` — closed interval. We adopt the
    closed-interval convention: both 3900 and 8000 pass; values outside
    fail; NaN fails. (The cool-end systematics in §6.2 and the hot-star
    template mismatch in §6.3 motivate excluding *both* tails.)
    """
    teff = np.array([3500, 3899.99, 3900.0, 5000.0, 8000.0, 8000.01, 9000.0, np.nan])
    out = katz2023_teff_mask(teff)
    expected = np.array([False, False, True, True, True, False, False, False])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# katz2023_error_mask
# ---------------------------------------------------------------------------


def test_error_mask():
    """radial_velocity_error strictly < 2 km/s passes; 2.0 fails; NaN fails."""
    err = np.array([0.5, 1.9, 2.0, 5.0, np.nan])
    out = katz2023_error_mask(err, max_kms=2.0)
    expected = np.array([True, True, False, False, False])
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# katz2023_variability_mask
# ---------------------------------------------------------------------------


def test_variability_mask():
    """Constant-star: rv_nb_transits ≥ 10 AND gof ≤ 4 AND pvalue > 0.01.

    Katz+2023 §10.2 gives the *variable*-star criterion
    (``gof > 4 & pvalue ≤ 0.01``); we accept the complement plus the
    ``nb_transits ≥ 10`` reliability gate from §3.1 line 110.

    Cases (one per row):

    0. Constant star (gof=1, p=0.5, transits=15)              → True
    1. Variable: high gof and tiny p (10, 0.001, 15)          → False
    2. High gof only (5 > 4): still flagged variable          → False
    3. Tiny p only (0.005 ≤ 0.01): still flagged variable     → False
    4. Constant but only 5 transits                           → False (too few)
    5. NaN in rv_nb_transits                                  → False
    6. NaN in gof                                             → False
    7. NaN in pvalue                                          → False
    """
    nb_transits = np.array([15, 15, 15, 15, 5, np.nan, 15, 15], dtype=float)
    gof = np.array([1.0, 10.0, 5.0, 1.0, 1.0, 1.0, np.nan, 1.0])
    pval = np.array([0.5, 0.001, 0.5, 0.005, 0.5, 0.5, 0.5, np.nan])
    out = katz2023_variability_mask(gof, pval, nb_transits)
    expected = np.array(
        [True, False, False, False, False, False, False, False]
    )
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# katz2023_rv_zp — magnitude-dependent zero-point
# ---------------------------------------------------------------------------


def test_rv_zp_bright():
    """G_RVS < 11.0 → no correction (Katz+2023 Eq. vrcor branch 1)."""
    grvs = np.array([5.0, 9.0, 10.5, 10.999])
    out = katz2023_rv_zp(grvs)
    expected = np.array([0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_rv_zp_faint():
    """G_RVS ≥ 11 → Eq. vrcor quadratic (0.02755·g² − 0.55863·g + 2.81129).

    Katz+2023 documents a positive bias (catalog v_r biased to *larger*
    values at faint mag, because CCD trapping skews lines redward).
    The returned offset is SUBTRACTED from v_r:
        v_r_corrected = v_r − katz2023_rv_zp(G_RVS)
    so the function returns POSITIVE values (≈ +0.4 km/s at G_RVS = 14).

    Spot-check the exact paper formula at three faint magnitudes.
    """
    # G_RVS = 14.0 → 0.02755·196 − 0.55863·14 + 2.81129
    #              = 5.3998 − 7.82082 + 2.81129 = 0.39027 km/s
    # G_RVS = 12.5 → 0.02755·156.25 − 0.55863·12.5 + 2.81129
    #              = 4.30469 − 6.98288 + 2.81129 = 0.13310 km/s
    # G_RVS = 13.0 → 0.02755·169 − 0.55863·13 + 2.81129
    #              = 4.65595 − 7.26219 + 2.81129 = 0.20505 km/s
    grvs = np.array([12.5, 13.0, 14.0])
    out = katz2023_rv_zp(grvs)
    np.testing.assert_allclose(
        out, np.array([0.13310, 0.20505, 0.39027]), atol=1e-4
    )

    # The published abstract says the magnitude trend "reaches about 400 m/s
    # at G_RVS = 14", which is exactly what Eq. vrcor evaluates to: 0.39 km/s.
    assert 0.3 < out[-1] < 0.5

    # NaN G_RVS → NaN offset
    assert np.isnan(katz2023_rv_zp(np.array([np.nan]))[0])


# ---------------------------------------------------------------------------
# apply_rv_quality — pipeline
# ---------------------------------------------------------------------------


def _baseline_rv_row(**overrides) -> dict:
    """A passing-by-default RV row; override one field to break it."""
    row = dict(
        radial_velocity=20.0,
        radial_velocity_error=0.8,
        rv_template_teff=5500.0,
        rv_renormalised_gof=1.0,
        rv_chisq_pvalue=0.5,
        rv_nb_transits=20,
        grvs_mag=11.5,
    )
    row.update(overrides)
    return row


def test_apply_rv_quality_pipeline():
    """20-row mixed DataFrame: only passing rows survive; ZP correction applied."""
    rows = [
        _baseline_rv_row(),                                       # 0 PASS
        _baseline_rv_row(grvs_mag=14.0),                          # 1 PASS (faint, large ZP)
        _baseline_rv_row(grvs_mag=8.0),                           # 2 PASS (bright, ZP=0)
        _baseline_rv_row(rv_template_teff=3500.0),                # 3 FAIL teff lo
        _baseline_rv_row(rv_template_teff=9000.0),                # 4 FAIL teff hi
        _baseline_rv_row(rv_template_teff=np.nan),                # 5 FAIL teff NaN
        _baseline_rv_row(radial_velocity_error=2.5),              # 6 FAIL err >= 2
        _baseline_rv_row(radial_velocity_error=2.0),              # 7 FAIL err == 2
        _baseline_rv_row(radial_velocity_error=np.nan),           # 8 FAIL err NaN
        _baseline_rv_row(rv_renormalised_gof=10.0),               # 9 FAIL gof > 4
        _baseline_rv_row(rv_chisq_pvalue=0.001),                  # 10 FAIL pvalue too small
        _baseline_rv_row(rv_nb_transits=5),                       # 11 FAIL too few transits
        _baseline_rv_row(rv_nb_transits=np.nan),                  # 12 FAIL NaN transits
        _baseline_rv_row(rv_renormalised_gof=np.nan),             # 13 FAIL NaN gof
        _baseline_rv_row(rv_chisq_pvalue=np.nan),                 # 14 FAIL NaN pval
        _baseline_rv_row(),                                       # 15 PASS
        _baseline_rv_row(rv_template_teff=3900.0),                # 16 PASS (lo boundary)
        _baseline_rv_row(rv_template_teff=8000.0),                # 17 PASS (hi boundary)
        _baseline_rv_row(radial_velocity_error=1.99),             # 18 PASS (just under)
        _baseline_rv_row(rv_nb_transits=10),                      # 19 PASS (boundary)
    ]
    df = pd.DataFrame(rows)
    # Use a non-default index to verify it is preserved through the join.
    df.index = pd.Index([100 + i for i in range(len(df))], name="source_id")

    cleaned, n_rv_qual = apply_rv_quality(df)

    # (a) Only passing rows survive.
    expected_passing_positions = [0, 1, 2, 15, 16, 17, 18, 19]
    expected_index = [100 + i for i in expected_passing_positions]
    assert list(cleaned.index) == expected_index
    assert n_rv_qual == len(expected_passing_positions)
    assert n_rv_qual == len(cleaned)

    # (b) radial_velocity_corrected = radial_velocity − katz2023_rv_zp(grvs_mag)
    # Row 0 (G_RVS=11.5): Z = 0.02755*132.25 - 0.55863*11.5 + 2.81129 = 0.03763 km/s
    # Row 1 (G_RVS=14.0): Z = 0.39027 km/s
    # Row 2 (G_RVS=8.0): Z = 0.0 km/s
    expected_zp = katz2023_rv_zp(cleaned["grvs_mag"].to_numpy())
    np.testing.assert_allclose(
        cleaned["radial_velocity_corrected"].to_numpy(),
        cleaned["radial_velocity"].to_numpy() - expected_zp,
        atol=1e-9,
    )

    # (c) Original columns preserved.
    for col in [
        "radial_velocity",
        "radial_velocity_error",
        "rv_template_teff",
        "rv_renormalised_gof",
        "rv_chisq_pvalue",
        "rv_nb_transits",
        "grvs_mag",
    ]:
        assert col in cleaned.columns


def test_apply_rv_quality_empty():
    """Empty DataFrame in → empty DataFrame out, N_rv_qual = 0."""
    df = pd.DataFrame(
        {
            "radial_velocity": [],
            "radial_velocity_error": [],
            "rv_template_teff": [],
            "rv_renormalised_gof": [],
            "rv_chisq_pvalue": [],
            "rv_nb_transits": [],
            "grvs_mag": [],
        }
    )
    cleaned, n_rv_qual = apply_rv_quality(df)
    assert n_rv_qual == 0
    assert len(cleaned) == 0
    assert "radial_velocity_corrected" in cleaned.columns
