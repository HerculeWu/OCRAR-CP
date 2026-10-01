"""σ_∥, σ_⊥, σ_RV with measurement-error deconvolution (plan §5).

The CP method's kinematics layer. For each cluster we compute:

- ``sigma_tangential`` — σ_∥ from CP-frame V∥ residuals (V∥_obs − V∥_pred)
  and σ_⊥ from V⊥_obs, both deconvolved against the per-star V error
  budget (PM covariance + finite-distance error) and clamped to 0.
- ``sigma_rv`` — σ_RV from a Katz+2023-cleaned RV-bearing subset
  (``rv_quality.apply_rv_quality``), deconvolved against
  ``radial_velocity_error²`` and clamped to 0. Returns NaN if the subset
  has fewer than ``min_N`` rows.
- ``sigma_1d_combined`` — dispatch between the tangential-only and the
  tangential+RV recipes based on ``N_rv,qual`` and propagate NaNs cleanly.

References
----------
Röser+2019 §2.2 (Hyades), §2.3 (Praesepe); Van Leeuwen 2009 §3.2;
Katz+2023 §6.1 (RV zero-point); plan §5 (error-deconvolved σ formulas).

Sign convention for the V error propagation
-------------------------------------------
ψ is the predicted-PM position angle at the star (Van Leeuwen 2009 §3.2 =
``cp_method.cp_transform`` ``psi_rad``). The CP rotation is::

    μ∥ =  cos ψ · μ_α* + sin ψ · μ_δ
    μ⊥ = −sin ψ · μ_α* + cos ψ · μ_δ

Propagating the (μ_α*, μ_δ) covariance through the rotation::

    Var(μ∥)  = cos²ψ · σ²_α + sin²ψ · σ²_δ
               + 2 cos ψ sin ψ · σ_α σ_δ ρ
    Var(μ⊥)  = sin²ψ · σ²_α + cos²ψ · σ²_δ
               − 2 cos ψ sin ψ · σ_α σ_δ ρ

Multiplying by (κ d / 1000)² gives the PM-only term in σ²_V∥, σ²_V⊥; the
distance-error term (V∥_obs · σ_d / d)² adds in quadrature.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .constants import MASYR_KPC_TO_KMS
from .cp_method import cp_observed, cp_predict, cp_transform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deconvolve_variance(
    raw_var: float, mean_err_var: float
) -> tuple[float, bool]:
    """``σ² = raw - ⟨σ²_err⟩``; clamp negative to 0 and flag the clamp."""
    val = float(raw_var) - float(mean_err_var)
    clamped = val <= 0.0
    return (max(val, 0.0), clamped)


def _per_star_sigma_V(
    psi_rad: np.ndarray,
    d_pc: np.ndarray,
    d_err_pc: np.ndarray,
    v_par_obs: np.ndarray,
    v_perp_obs: np.ndarray,
    pmra_err: np.ndarray,
    pmdec_err: np.ndarray,
    corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-star σ_V∥, σ_V⊥ (km/s) from PM covariance + distance error.

    See module docstring for the rotation algebra. ``d_pc`` is in pc;
    the (κ · d_kpc) prefactor uses d/1000.
    """
    kappa = MASYR_KPC_TO_KMS
    d_kpc = d_pc / 1000.0
    s = np.sin(psi_rad)
    c = np.cos(psi_rad)

    # PM-rotation variance terms.
    var_mu_par = (
        (c ** 2) * pmra_err ** 2
        + (s ** 2) * pmdec_err ** 2
        + 2.0 * s * c * pmra_err * pmdec_err * corr
    )
    var_mu_perp = (
        (s ** 2) * pmra_err ** 2
        + (c ** 2) * pmdec_err ** 2
        - 2.0 * s * c * pmra_err * pmdec_err * corr
    )
    # Numerical safety: small-negative due to floating point.
    var_mu_par = np.clip(var_mu_par, 0.0, None)
    var_mu_perp = np.clip(var_mu_perp, 0.0, None)

    pm_factor = (kappa * d_kpc) ** 2

    # Distance term: (V · σ_d / d)²; safe against d → 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_d = np.where(d_pc > 0.0, d_err_pc / d_pc, 0.0)

    var_V_par = pm_factor * var_mu_par + (v_par_obs * rel_d) ** 2
    var_V_perp = pm_factor * var_mu_perp + (v_perp_obs * rel_d) ** 2

    var_V_par = np.clip(var_V_par, 0.0, None)
    var_V_perp = np.clip(var_V_perp, 0.0, None)
    return np.sqrt(var_V_par), np.sqrt(var_V_perp)


# ---------------------------------------------------------------------------
# sigma_tangential
# ---------------------------------------------------------------------------


def sigma_tangential(
    stars_df: pd.DataFrame,
    *,
    vc_dict: dict,
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    parallax_col: str = "parallax_corrected",
    pmra_err_col: str = "pmra_error",
    pmdec_err_col: str = "pmdec_error",
    pmra_pmdec_corr_col: str = "pmra_pmdec_corr",
    d_pc_col: Optional[str] = "d_pc",
    d_err_col: Optional[str] = None,
    clip_sigma: Optional[float] = None,
    clip_max_iters: int = 5,
) -> dict:
    """σ_∥, σ_⊥ from CP-frame residuals with error deconvolution (plan §5).

    Reduces (V∥_obs, V⊥_obs) via ``cp_method.cp_observed``; predicts
    V∥_pred = |V_c| · sin λ via ``cp_method.cp_predict``. Then::

        σ²_∥  = ⟨(V∥_obs − V∥_pred)²⟩ − ⟨σ²_V∥,err⟩   (clamp to 0)
        σ²_⊥  = ⟨V⊥_obs²⟩            − ⟨σ²_V⊥,err⟩   (clamp to 0)

    Per-star σ_V follows the rotation algebra in the module docstring:
    PM-covariance term scaled by (κ · d / 1000)² + distance term
    (V · σ_d / d)². For ϖ-inverse distance, σ_d / d = σ_ϖ / ϖ'. For
    Bailer-Jones distance, pass ``d_err_col=`` the half-bracket
    (d_hi − d_lo) / 2.

    The standard error on σ is ``σ / √(2 (N − 1))`` (Gaussian variance).

    Parameters
    ----------
    stars_df : DataFrame
        Required columns: ``ra``, ``dec``, ``pmra``, ``pmdec``,
        ``parallax_corrected`` (already ZP-corrected; positive),
        ``pmra_error``, ``pmdec_error``, ``pmra_pmdec_corr``. If
        ``d_pc_col`` is provided and present, uses that distance for the
        (κ · d / 1000) factor and the σ_d term; otherwise falls back to
        ϖ-inverse: d_pc = 1000 / parallax_corrected.
    vc_dict : dict
        Must contain ``ra_cp_deg``, ``dec_cp_deg``, ``vc_mag_kms``.
    d_err_col : str | None
        If provided and present in ``stars_df``, use that column as the
        per-star distance error σ_d (pc) — recommended for Bailer-Jones
        distances where σ_d / d ≠ σ_ϖ / ϖ. If None or absent, σ_d / d
        falls back to ``parallax_error / parallax_corrected``.
    clip_sigma : float | None
        If not None, iteratively σ-clip on ΔV∥ = V∥_obs − V∥_pred before
        the variance / deconvolution step. Each iteration drops rows
        with ``|ΔV∥ − mean(ΔV∥)| > clip_sigma · std(ΔV∥)`` (sample
        statistics). Iteration stops on no further drops or after
        ``clip_max_iters``. The same surviving subset is used for σ_⊥
        so both axes share an identical member sample. Plan §5 adapts
        Röser+2019 §2.2's implicit iterate-and-reselect by promoting it
        to an explicit numerical clip; default ``None`` preserves the
        plain plan §5 formula. Stage 0b (Hyades) uses ``clip_sigma=3.0``
        because the (-5, +3) × (-0.8, +1.1) CP rectangle deliberately
        admits ~2 % tail-leakage stars that inflate σ_∥.
    clip_max_iters : int
        Hard upper bound on σ-clip iterations. Only consulted when
        ``clip_sigma`` is not None. Defaults to 5 — convergence usually
        in 2-3 iterations.

    Returns
    -------
    dict with keys
        sigma_par_kms, sigma_perp_kms (km/s, clamped ≥ 0)
        sigma_par_err_kms, sigma_perp_err_kms (km/s, Gaussian SE)
        sigma_par_clamped, sigma_perp_clamped (bool)
        N (int, rows used in the final variance estimate — post-clip
        when ``clip_sigma`` is engaged)
        n_clipped (int, rows dropped by σ-clip; 0 when ``clip_sigma`` is None)
        n_clip_iters (int, number of clip iterations actually run;
        0 when ``clip_sigma`` is None)
    """
    ra = stars_df[ra_col].to_numpy(dtype=float)
    dec = stars_df[dec_col].to_numpy(dtype=float)
    pmra = stars_df[pmra_col].to_numpy(dtype=float)
    pmdec = stars_df[pmdec_col].to_numpy(dtype=float)
    plx = stars_df[parallax_col].to_numpy(dtype=float)
    pmra_err = stars_df[pmra_err_col].to_numpy(dtype=float)
    pmdec_err = stars_df[pmdec_err_col].to_numpy(dtype=float)
    if pmra_pmdec_corr_col in stars_df.columns:
        corr = stars_df[pmra_pmdec_corr_col].to_numpy(dtype=float)
    else:
        corr = np.zeros_like(pmra_err)

    ra_cp = float(vc_dict["ra_cp_deg"])
    dec_cp = float(vc_dict["dec_cp_deg"])
    vc_mag = float(vc_dict["vc_mag_kms"])

    # CP-frame observed and predicted velocities.
    v_par_obs, v_perp_obs = cp_observed(pmra, pmdec, plx, ra, dec, ra_cp, dec_cp)
    v_par_pred, _ = cp_predict(ra, dec, ra_cp, dec_cp, vc_mag)
    d_v_par = v_par_obs - v_par_pred
    d_v_perp = v_perp_obs  # V⊥_pred ≡ 0

    # ψ for the per-star σ_V rotation algebra.
    tr = cp_transform(ra, dec, pmra, pmdec, ra_cp, dec_cp)
    psi = tr["psi_rad"]

    # Distance and per-star σ_d.
    if d_pc_col is not None and d_pc_col in stars_df.columns:
        d_pc = stars_df[d_pc_col].to_numpy(dtype=float)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            d_pc = np.where(plx > 0, 1000.0 / plx, np.nan)

    if d_err_col is not None and d_err_col in stars_df.columns:
        d_err_pc = stars_df[d_err_col].to_numpy(dtype=float)
    else:
        # Fall back to ϖ-inverse σ_d/d = σ_ϖ / ϖ' → σ_d = d · (σ_ϖ / ϖ').
        if "parallax_error" in stars_df.columns:
            plx_err = stars_df["parallax_error"].to_numpy(dtype=float)
        else:
            plx_err = np.zeros_like(plx)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_err_pc = np.where(plx > 0, d_pc * (plx_err / plx), 0.0)

    sigma_V_par, sigma_V_perp = _per_star_sigma_V(
        psi, d_pc, d_err_pc, v_par_obs, v_perp_obs, pmra_err, pmdec_err, corr
    )

    # Restrict to finite stars for the moment estimators.
    finite = (
        np.isfinite(d_v_par)
        & np.isfinite(d_v_perp)
        & np.isfinite(sigma_V_par)
        & np.isfinite(sigma_V_perp)
    )
    N_finite = int(finite.sum())
    if N_finite < 2:
        return {
            "sigma_par_kms": float("nan"),
            "sigma_perp_kms": float("nan"),
            "sigma_par_err_kms": float("nan"),
            "sigma_perp_err_kms": float("nan"),
            "sigma_par_clamped": False,
            "sigma_perp_clamped": False,
            "N": N_finite,
            "n_clipped": 0,
            "n_clip_iters": 0,
        }

    # Optional iterative σ-clip on ΔV∥. The kept mask is initialised to
    # ``finite`` so non-finite rows are excluded from clip statistics
    # too. See docstring for the rationale.
    keep = finite.copy()
    n_iter = 0
    if clip_sigma is not None and N_finite >= 3:
        threshold = float(clip_sigma)
        for n_iter in range(1, int(clip_max_iters) + 1):
            sub = d_v_par[keep]
            m = float(np.mean(sub))
            s = float(np.std(sub, ddof=1))
            if s <= 0.0:
                break
            new_keep = keep & (np.abs(d_v_par - m) < threshold * s)
            if int(new_keep.sum()) == int(keep.sum()):
                break
            keep = new_keep
            if int(keep.sum()) < 2:
                break

    N = int(keep.sum())
    n_clipped = N_finite - N
    if N < 2:
        return {
            "sigma_par_kms": float("nan"),
            "sigma_perp_kms": float("nan"),
            "sigma_par_err_kms": float("nan"),
            "sigma_perp_err_kms": float("nan"),
            "sigma_par_clamped": False,
            "sigma_perp_clamped": False,
            "N": N,
            "n_clipped": n_clipped,
            "n_clip_iters": n_iter,
        }

    # Raw variance and mean per-star error variance.
    raw_var_par = float(np.var(d_v_par[keep], ddof=1))
    raw_var_perp = float(np.var(d_v_perp[keep], ddof=1))
    mean_err_var_par = float(np.mean(sigma_V_par[keep] ** 2))
    mean_err_var_perp = float(np.mean(sigma_V_perp[keep] ** 2))

    var_par, clamp_par = _deconvolve_variance(raw_var_par, mean_err_var_par)
    var_perp, clamp_perp = _deconvolve_variance(raw_var_perp, mean_err_var_perp)
    sig_par = float(np.sqrt(var_par))
    sig_perp = float(np.sqrt(var_perp))

    # Gaussian-σ standard error: σ_σ ≈ σ / √(2(N−1)) for a normal sample.
    denom = np.sqrt(2.0 * (N - 1))
    sig_par_err = float(sig_par / denom) if denom > 0 else float("nan")
    sig_perp_err = float(sig_perp / denom) if denom > 0 else float("nan")

    return {
        "sigma_par_kms": sig_par,
        "sigma_perp_kms": sig_perp,
        "sigma_par_err_kms": sig_par_err,
        "sigma_perp_err_kms": sig_perp_err,
        "sigma_par_clamped": bool(clamp_par),
        "sigma_perp_clamped": bool(clamp_perp),
        "N": N,
        "n_clipped": n_clipped,
        "n_clip_iters": n_iter,
    }


# ---------------------------------------------------------------------------
# sigma_rv
# ---------------------------------------------------------------------------


def sigma_rv(
    rv_clean_df: pd.DataFrame,
    *,
    rv_col: str = "radial_velocity_corrected",
    rv_err_col: str = "radial_velocity_error",
    min_N: int = 10,
) -> dict:
    """σ_RV from a Katz+2023-gated cleaned subset (plan §5).

    Computes::

        σ²_RV = ⟨(v_r' − ⟨v_r'⟩)²⟩ − ⟨σ²_v_r,err⟩   (clamp to 0)

    Assumes ``rv_col`` already has the Katz+2023 zero-point applied (e.g.
    ``rv_quality.apply_rv_quality`` writes ``radial_velocity_corrected``).
    Returns NaN with ``N_rv = len(rv_clean_df)`` if the subset has fewer
    than ``min_N`` rows.

    Returns
    -------
    dict with keys: ``sigma_rv_kms``, ``sigma_rv_err_kms``,
    ``sigma_rv_clamped``, ``N_rv``.
    """
    N = int(len(rv_clean_df))
    if N < int(min_N):
        return {
            "sigma_rv_kms": float("nan"),
            "sigma_rv_err_kms": float("nan"),
            "sigma_rv_clamped": False,
            "N_rv": N,
        }
    v = rv_clean_df[rv_col].to_numpy(dtype=float)
    e = rv_clean_df[rv_err_col].to_numpy(dtype=float)
    finite = np.isfinite(v) & np.isfinite(e)
    N_fin = int(finite.sum())
    if N_fin < int(min_N):
        return {
            "sigma_rv_kms": float("nan"),
            "sigma_rv_err_kms": float("nan"),
            "sigma_rv_clamped": False,
            "N_rv": N_fin,
        }
    raw_var = float(np.var(v[finite], ddof=1))
    mean_err_var = float(np.mean(e[finite] ** 2))
    var_rv, clamped = _deconvolve_variance(raw_var, mean_err_var)
    sig_rv = float(np.sqrt(var_rv))
    denom = np.sqrt(2.0 * (N_fin - 1))
    sig_rv_err = float(sig_rv / denom) if denom > 0 else float("nan")
    return {
        "sigma_rv_kms": sig_rv,
        "sigma_rv_err_kms": sig_rv_err,
        "sigma_rv_clamped": bool(clamped),
        "N_rv": N_fin,
    }


# ---------------------------------------------------------------------------
# sigma_1d_combined
# ---------------------------------------------------------------------------


def sigma_1d_combined(
    sigma_par_kms: float,
    sigma_perp_kms: float,
    sigma_rv_kms: float,
    N_rv_qual: int,
    *,
    min_N_rv: int = 10,
) -> dict:
    """σ_1D dispatched between tangential-only and tang+RV (plan §5).

    - ``N_rv_qual < min_N_rv``: σ_1D = √((σ²_∥ + σ²_⊥) / 2), method
      ``"CP_tangential"``.
    - ``N_rv_qual ≥ min_N_rv``: σ_1D = √((σ²_∥ + σ²_⊥ + σ²_RV) / 3),
      method ``"CP_tang+rv"``.

    Both ``sigma_1d_tangential_kms`` and ``sigma_1d_combined_kms`` are
    reported for transparency; when the RV channel is unused they are
    equal. NaN ``σ_par`` or ``σ_perp`` short-circuits to NaN outputs and
    ``sigma_method = "CP_failed"``. NaN ``σ_rv`` when the RV channel
    would be admitted also yields a failed combined value but preserves
    the tangential-only number.
    """
    sp = float(sigma_par_kms)
    sq = float(sigma_perp_kms)
    sr = float(sigma_rv_kms)

    if not (np.isfinite(sp) and np.isfinite(sq)):
        return {
            "sigma_1d_tangential_kms": float("nan"),
            "sigma_1d_combined_kms": float("nan"),
            "sigma_method": "CP_failed",
        }

    sig_tang = float(np.sqrt((sp * sp + sq * sq) / 2.0))

    if int(N_rv_qual) < int(min_N_rv):
        return {
            "sigma_1d_tangential_kms": sig_tang,
            "sigma_1d_combined_kms": sig_tang,
            "sigma_method": "CP_tangential",
        }
    if not np.isfinite(sr):
        # RV admission promised but σ_RV unavailable — preserve transparency.
        return {
            "sigma_1d_tangential_kms": sig_tang,
            "sigma_1d_combined_kms": float("nan"),
            "sigma_method": "CP_failed",
        }
    sig_comb = float(np.sqrt((sp * sp + sq * sq + sr * sr) / 3.0))
    return {
        "sigma_1d_tangential_kms": sig_tang,
        "sigma_1d_combined_kms": sig_comb,
        "sigma_method": "CP_tang+rv",
    }
