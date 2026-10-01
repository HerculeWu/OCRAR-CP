"""Katz+2023 DR3 RV quality gate.

Stateless cuts and the magnitude-dependent zero-point correction from
Katz et al. 2023 (arXiv:2206.05902), "Gaia DR3 Radial velocities":

- Template-Teff window  (§3.1 line 110; §6.2 cool / §6.3 hot systematics)
- Formal-error ceiling  (cluster σ_RV ≈ 0.3 km/s drowns under DR3 floor)
- Variability filter    (§10.2 conservative constant-star criterion)
- Magnitude-trend ZP    (§6.1 / Eq. vrcor, applicable for
                         rv_template_teff < 8500 K)

The header docstring of each function gives the exact Katz+2023
reference. NaN in any required column is treated as failing the cut.

Why ``radial_velocity_error < 2 km/s`` (much tighter than the 6.4 km/s
formal floor at G_RVS = 14):

    A cluster has σ_RV ~ 0.3 km/s (Hyades; Röser+2011). Even a single
    star with σ_v ~ 6 km/s would inflate the σ-estimator by ≈ √(0.3² +
    6²/N) for membership of size N — drowning the signal unless N ≫ 400.
    Cf. ocrar §3 reliance on a σ-channel from DR3 RVs that cannot
    resolve a Hyades-like cluster without a tight err cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual boolean masks
# ---------------------------------------------------------------------------


def katz2023_teff_mask(
    rv_template_teff, lo: float = 3900.0, hi: float = 8000.0
) -> np.ndarray:
    """rv_template_teff ∈ [lo, hi] (Katz+2023 §10.2, closed interval).

    Outside this window, Katz+2023 §6.2 (cool end, Teff < 3900 K) and
    §6.3 (hot end, Teff > 8000 K) document template-mismatch systematics
    on both the RV measurement and the variability indices. NaN → False.
    """
    t = np.asarray(rv_template_teff, dtype=float)
    return np.isfinite(t) & (t >= lo) & (t <= hi)


def katz2023_error_mask(radial_velocity_error, max_kms: float = 2.0) -> np.ndarray:
    """Finite radial_velocity_error strictly < ``max_kms``.

    Default 2 km/s is much tighter than the 6.4 km/s formal floor at
    G_RVS = 14 reported in Katz+2023 §11, and is essential for cluster-
    σ recovery (expected σ_RV ~ 0.3 km/s). NaN → False.
    """
    e = np.asarray(radial_velocity_error, dtype=float)
    return np.isfinite(e) & (e < max_kms)


def katz2023_variability_mask(
    rv_renormalised_gof, rv_chisq_pvalue, rv_nb_transits
) -> np.ndarray:
    """Constant-star criterion from Katz+2023 §10.2.

    The paper defines the *variable* criterion as
        ``rv_chisq_pvalue ≤ 0.01 AND rv_renormalised_gof > 4``
    and requires ``rv_nb_transits ≥ 10`` and ``rv_template_teff ∈
    [3900, 8000]`` for reliability of the two indices (line 110). We
    apply a stricter constant-star conjunction with the reliability gate:

        rv_nb_transits ≥ 10 AND rv_renormalised_gof ≤ 4 AND rv_chisq_pvalue > 0.01

    (The Teff reliability sub-gate is enforced separately by
    :func:`katz2023_teff_mask`, so it is not duplicated here.) NaN in
    any of the three columns → False.
    """
    gof = np.asarray(rv_renormalised_gof, dtype=float)
    pval = np.asarray(rv_chisq_pvalue, dtype=float)
    nt = np.asarray(rv_nb_transits, dtype=float)
    finite = np.isfinite(gof) & np.isfinite(pval) & np.isfinite(nt)
    return finite & (nt >= 10.0) & (gof <= 4.0) & (pval > 0.01)


# ---------------------------------------------------------------------------
# Magnitude-dependent zero-point — Katz+2023 §6.1 / Eq. vrcor
# ---------------------------------------------------------------------------


# Coefficients of the quadratic fit (Eq. vrcor / §6.1, "Magnitude trend").
# Derived by Katz+2023 from the APOGEE-B (red giant) sample residuals.
# Applicable to rv_template_teff < 8500 K only — for hot stars use the
# correction in Blomme+2023 (DR3-DPACP-151) instead. The teff_mask above
# (≤8000 K) pre-removes hot stars from the apply_rv_quality pipeline, so
# this restriction is enforced by the upstream filter.
_KATZ_ZP_A2 = 0.02755   # km/s / mag²
_KATZ_ZP_A1 = -0.55863  # km/s / mag
_KATZ_ZP_A0 = 2.81129   # km/s
_KATZ_ZP_GRVS_MIN = 11.0  # No measured trend below this magnitude


def katz2023_rv_zp(grvs_mag) -> np.ndarray:
    """Magnitude-dependent RV zero-point offset in km/s (Katz+2023 Eq. vrcor).

    Returns the value that should be **subtracted** from
    ``radial_velocity``::

        v_r_corrected = v_r - katz2023_rv_zp(grvs_mag)

    Following Katz+2023 §6.1 ("Magnitude trend") and the Eq. tagged
    ``vrcor`` in the source text:

    * G_RVS < 11.0     : V_R^corr = 0 km/s  (no measured trend)
    * G_RVS ≥ 11.0     : V_R^corr = 0.02755·g² − 0.55863·g + 2.81129  km/s

    The bias is **positive** (catalog v_r is biased to *larger* values
    at faint mag, because CCD trapping skews the spectral lines toward
    longer wavelengths — see Katz+2023 §6.1 paragraph 1). Subtracting
    the returned offset removes the bias.

    Reaches ≈ +0.4 km/s at G_RVS = 14, matching the abstract claim.

    Important: the correction is calibrated for ``rv_template_teff <
    8500 K`` only. Hot stars need the Blomme+2023 (DR3-DPACP-151)
    correction instead. The temperature gate in :func:`katz2023_teff_mask`
    (≤ 8000 K) pre-removes hot stars from the
    :func:`apply_rv_quality` pipeline, so this restriction does not
    leak into the cleaned output.

    NaN G_RVS → NaN offset.
    """
    g = np.asarray(grvs_mag, dtype=float)
    out = np.where(
        np.isnan(g),
        np.nan,
        np.where(
            g >= _KATZ_ZP_GRVS_MIN,
            _KATZ_ZP_A2 * g * g + _KATZ_ZP_A1 * g + _KATZ_ZP_A0,
            0.0,
        ),
    )
    return out


# ---------------------------------------------------------------------------
# Pipeline dispatcher
# ---------------------------------------------------------------------------


def apply_rv_quality(stars_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply all Katz+2023 RV quality gates and the ZP correction.

    Required columns
    ----------------
    radial_velocity, radial_velocity_error, rv_template_teff,
    rv_renormalised_gof, rv_chisq_pvalue, rv_nb_transits, grvs_mag

    Returns
    -------
    (cleaned, N_rv_qual) : tuple
        - cleaned : pd.DataFrame containing only rows passing all three
          masks, with ``radial_velocity_corrected = radial_velocity −
          katz2023_rv_zp(grvs_mag)`` added. Original columns and index
          are preserved.
        - N_rv_qual : int, equal to ``len(cleaned)``.

    Notes
    -----
    The teff cut (≤ 8000 K) ensures the Katz+2023 zero-point formula
    (calibrated for Teff < 8500 K) is applied only to in-domain stars
    in ``cleaned``. Hot stars are removed *before* the ZP is applied.
    """
    mask_teff = katz2023_teff_mask(stars_df["rv_template_teff"].to_numpy())
    mask_err = katz2023_error_mask(stars_df["radial_velocity_error"].to_numpy())
    mask_var = katz2023_variability_mask(
        stars_df["rv_renormalised_gof"].to_numpy(),
        stars_df["rv_chisq_pvalue"].to_numpy(),
        stars_df["rv_nb_transits"].to_numpy(),
    )

    keep = mask_teff & mask_err & mask_var
    cleaned = stars_df.loc[keep].copy()
    cleaned["radial_velocity_corrected"] = (
        cleaned["radial_velocity"].to_numpy()
        - katz2023_rv_zp(cleaned["grvs_mag"].to_numpy())
    )
    return cleaned, len(cleaned)
