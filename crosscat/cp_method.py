"""Convergent-point (CP) transforms, V_c fit, and iterate loop.

Implements Röser, Schilbach & Goldman 2019 §2.2 (Hyades) plus the
iterative LSQ generalisation of van Leeuwen 2009 §3.2. The CP recipe
projects each star's proper motion into a frame aligned with the
cluster's bulk space velocity V_c, subtracting the perspective effect
that biases HR-pipeline σ_pm for spatially-extended clusters.

Notation::

    α, δ          star sky position (ICRS, deg)
    α_CP, δ_CP    convergent point (ICRS, deg) -- direction of V_c
    λ             angular distance star → CP (great circle)
    ψ             position angle of the predicted PM vector at the star,
                   measured from the α*-axis (van Leeuwen 2009 §3.2)
    μ_α*, μ_δ     proper motion (mas/yr, μ_α* = pmra already cos δ-corrected)
    μ∥, μ⊥        PM rotated into CP frame (mas/yr); μ⊥ ≡ 0 for a
                   strictly co-moving star
    ϖ             parallax (mas)
    V∥, V⊥        per-star tangential velocity (km/s)
                   V∥_obs = κ μ∥ / ϖ,    V∥_pred = |V_c| sin λ
    V_c           bulk space velocity vector (km/s, Galactic Cartesian UVW)

The Galactic Cartesian convention follows Röser+2019: U → Galactic
centre, V → Galactic rotation, W → Galactic north pole.

All inputs are ndarray-coercible. Functions are stateless. Equation
numbers refer to Röser+2019 (Hyades) §2.2 and van Leeuwen 2009 §3.2.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation,
    Galactic,
    ICRS,
    SkyCoord,
    SphericalRepresentation,
)

from scipy.optimize import minimize

from .constants import MASYR_KPC_TO_KMS


class CPNotConverged(Exception):
    """Raised when the iterative CP fit fails to converge in max_iter steps."""


# ---------------------------------------------------------------------------
# Vector-to-CP and CP-to-vector helpers (Galactic Cartesian ↔ ICRS spherical)
# ---------------------------------------------------------------------------


def uvw_to_cp(U_kms: float, V_kms: float, W_kms: float) -> tuple[float, float, float]:
    """(U, V, W) Galactic Cartesian → (α_CP, δ_CP, |V_c|) ICRS spherical.

    The convergent point is the ICRS sky direction of the velocity vector.
    """
    U = float(U_kms)
    V = float(V_kms)
    W = float(W_kms)
    mag = float(np.sqrt(U * U + V * V + W * W))
    if mag == 0.0:
        return 0.0, 0.0, 0.0
    rep = CartesianRepresentation(U * u.km / u.s, V * u.km / u.s, W * u.km / u.s)
    gal = Galactic(rep, representation_type="cartesian")
    icrs = gal.transform_to(ICRS())
    sph = icrs.represent_as(SphericalRepresentation)
    return float(np.rad2deg(sph.lon.rad)), float(np.rad2deg(sph.lat.rad)), mag


def cp_to_uvw(
    ra_cp_deg: float, dec_cp_deg: float, vc_mag_kms: float
) -> tuple[float, float, float]:
    """(α_CP, δ_CP, |V_c|) ICRS → (U, V, W) Galactic Cartesian (km/s)."""
    mag = float(vc_mag_kms)
    cos_d = np.cos(np.deg2rad(dec_cp_deg))
    sin_d = np.sin(np.deg2rad(dec_cp_deg))
    cos_r = np.cos(np.deg2rad(ra_cp_deg))
    sin_r = np.sin(np.deg2rad(ra_cp_deg))
    x = mag * cos_d * cos_r
    y = mag * cos_d * sin_r
    z = mag * sin_d
    rep = CartesianRepresentation(x * u.km / u.s, y * u.km / u.s, z * u.km / u.s)
    icrs = ICRS(rep, representation_type="cartesian")
    gal = icrs.transform_to(Galactic())
    gal.representation_type = "cartesian"
    return float(gal.u.to(u.km / u.s).value), float(gal.v.to(u.km / u.s).value), float(
        gal.w.to(u.km / u.s).value
    )


# ---------------------------------------------------------------------------
# Core CP geometry: (λ, ψ) and PM rotation
# ---------------------------------------------------------------------------
#
# Sign convention note (READ ME):
#   The plan section §5 gives a θ-based formula `tan θ = sin Δα /
#   (cos δ tan δ_CP − sin δ cos Δα)` and then `μ∥ = μ_α* sin θ + μ_δ cos θ`.
#   That θ does NOT match the angle van Leeuwen 2009 §3.2 (and Röser+2019
#   §2.2 which cites vL2009 verbatim) uses for the PM rotation.
#
#   The correct quantity is ψ = atan2(v_δ_pred, v_α*_pred), the
#   on-sky position angle of the *predicted* PM vector measured from the
#   α*-axis. With this ψ, the rotation that vL2009 §3.2 specifies is
#       μ∥ =  cos ψ · μ_α* + sin ψ · μ_δ
#       μ⊥ = −sin ψ · μ_α* + cos ψ · μ_δ
#   which makes a perfectly co-moving star (μ ∝ predicted) yield
#   μ⊥ = 0 and μ∥ > 0, as required.
#
#   The closed-form for ψ follows from projecting V_c onto the local
#   tangent frame at (α, δ):
#       v_α*_pred ∝ cos δ_CP · sin(α_CP − α)
#       v_δ_pred  ∝ cos δ · sin δ_CP − sin δ · cos δ_CP · cos(α_CP − α)
#   (no |V_c| prefactor needed since atan2 only cares about the ratio).
#
#   We verified this against a closed-loop synthetic Hyades mock in
#   tests/test_cp_method.py — the plan's θ-formula gives <μ⊥> ≈ −37
#   mas/yr at the true CP (i.e. it's wrong), while this ψ-formula gives
#   <μ⊥> ≈ 0 mas/yr and recovers |V_c| to <0.5%.


def _lambda_psi(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra_cp_deg: float,
    dec_cp_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle distance λ and predicted-PM position angle ψ at the star.

    See block comment above for sign-convention rationale.
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    ra_cp = np.deg2rad(float(ra_cp_deg))
    dec_cp = np.deg2rad(float(dec_cp_deg))

    d_ra = ra_cp - ra  # NOTE: CP − star, opposite sign from the plan text
    cos_lambda = np.sin(dec) * np.sin(dec_cp) + np.cos(dec) * np.cos(dec_cp) * np.cos(
        d_ra
    )
    cos_lambda = np.clip(cos_lambda, -1.0, 1.0)
    lam = np.arccos(cos_lambda)

    # v_α*_pred ∝ cos δ_CP · sin(α_CP − α)
    # v_δ_pred  ∝ cos δ · sin δ_CP − sin δ · cos δ_CP · cos(α_CP − α)
    v_alpha_pred = np.cos(dec_cp) * np.sin(d_ra)
    v_delta_pred = np.cos(dec) * np.sin(dec_cp) - np.sin(dec) * np.cos(dec_cp) * np.cos(
        d_ra
    )
    psi = np.arctan2(v_delta_pred, v_alpha_pred)
    return lam, psi


def cp_transform(
    ra_deg,
    dec_deg,
    pmra,
    pmdec,
    ra_cp_deg: float,
    dec_cp_deg: float,
    pmra_err=None,
    pmdec_err=None,
    pmra_pmdec_corr=None,
) -> dict:
    """Per-star CP geometry and PM rotation.

    pmra is assumed to be Gaia ``pmra`` (i.e. μ_α* = μ_α cos δ); no further
    cos δ multiplication is applied. Returns a dict with arrays:

    - ``mu_par`` (mas/yr): μ∥ =  cos ψ · μ_α* + sin ψ · μ_δ
    - ``mu_perp`` (mas/yr): μ⊥ = −sin ψ · μ_α* + cos ψ · μ_δ
    - ``lambda_rad`` (rad): great-circle distance star → CP
    - ``psi_rad`` (rad): predicted-PM position angle at the star
      (van Leeuwen 2009 §3.2)
    - ``sigma_mu_perp`` (mas/yr): error on μ⊥ via covariance propagation;
      ``None`` if errors are not provided.

    Sign convention: see block comment above ``_lambda_psi``. We follow
    van Leeuwen 2009 §3.2 (= Röser+2019 §2.2), not the plan §5's θ.
    """
    pmra = np.asarray(pmra, dtype=float)
    pmdec = np.asarray(pmdec, dtype=float)
    lam, psi = _lambda_psi(ra_deg, dec_deg, ra_cp_deg, dec_cp_deg)
    s, c = np.sin(psi), np.cos(psi)
    mu_par = c * pmra + s * pmdec
    mu_perp = -s * pmra + c * pmdec

    sigma_mu_perp: Optional[np.ndarray]
    if pmra_err is None or pmdec_err is None:
        sigma_mu_perp = None
    else:
        pmra_err = np.asarray(pmra_err, dtype=float)
        pmdec_err = np.asarray(pmdec_err, dtype=float)
        if pmra_pmdec_corr is None:
            corr = np.zeros_like(pmra_err)
        else:
            corr = np.asarray(pmra_pmdec_corr, dtype=float)
        # Var(μ⊥) for 2D rotation: row [-sin ψ, cos ψ] applied to Σ_pm
        var = (
            (s ** 2) * pmra_err ** 2
            + (c ** 2) * pmdec_err ** 2
            - 2.0 * s * c * pmra_err * pmdec_err * corr
        )
        var = np.clip(var, 0.0, None)
        sigma_mu_perp = np.sqrt(var)

    return {
        "mu_par": mu_par,
        "mu_perp": mu_perp,
        "lambda_rad": lam,
        "psi_rad": psi,
        "sigma_mu_perp": sigma_mu_perp,
    }


def cp_predict(
    ra_deg,
    dec_deg,
    ra_cp_deg: float,
    dec_cp_deg: float,
    V_c_kms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted (V∥, V⊥) for a perfectly co-moving star (V⊥_pred ≡ 0)."""
    lam, _ = _lambda_psi(ra_deg, dec_deg, ra_cp_deg, dec_cp_deg)
    v_par = float(V_c_kms) * np.sin(lam)
    v_perp = np.zeros_like(v_par)
    return v_par, v_perp


def cp_observed(
    pmra,
    pmdec,
    parallax_mas,
    ra_deg,
    dec_deg,
    ra_cp_deg: float,
    dec_cp_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-star observed (V∥, V⊥) in km/s = κ · μ / ϖ_mas.

    Returns NaN for stars with parallax ≤ 0; caller is responsible for
    filtering.
    """
    tr = cp_transform(ra_deg, dec_deg, pmra, pmdec, ra_cp_deg, dec_cp_deg)
    kappa = MASYR_KPC_TO_KMS
    plx = np.asarray(parallax_mas, dtype=float)
    bad = ~(plx > 0)
    plx_safe = np.where(bad, 1.0, plx)
    v_par = np.where(bad, np.nan, kappa * tr["mu_par"] / plx_safe)
    v_perp = np.where(bad, np.nan, kappa * tr["mu_perp"] / plx_safe)
    return v_par, v_perp


# ---------------------------------------------------------------------------
# V_c from external RV-bearing prior members (Stage 0b path)
# ---------------------------------------------------------------------------


def vc_from_prior_members(prior_members_df: pd.DataFrame) -> dict:
    """Mean (U, V, W) from RV-bearing prior members → CP dict.

    Columns expected: ``ra``, ``dec``, ``pmra``, ``pmdec``, ``parallax``,
    ``radial_velocity``. Only rows with finite radial_velocity contribute.

    Computes per-star (U, V, W) via the astropy ICRS→Galactic transform
    with the star's full 6D phase-space vector and averages the components.
    """
    df = prior_members_df
    rv = pd.to_numeric(df["radial_velocity"], errors="coerce").to_numpy()
    mask = np.isfinite(rv)
    if mask.sum() < 1:
        raise ValueError("vc_from_prior_members: no RV-bearing members")

    ra = df.loc[mask, "ra"].to_numpy(dtype=float)
    dec = df.loc[mask, "dec"].to_numpy(dtype=float)
    pmra = df.loc[mask, "pmra"].to_numpy(dtype=float)
    pmdec = df.loc[mask, "pmdec"].to_numpy(dtype=float)
    plx = df.loc[mask, "parallax"].to_numpy(dtype=float)
    rv_kms = rv[mask]

    dist_pc = 1000.0 / plx
    sc = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=dist_pc * u.pc,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=rv_kms * u.km / u.s,
        frame="icrs",
    )
    gal = sc.transform_to(Galactic())
    vel = gal.cartesian.differentials["s"]
    U = float(np.mean(vel.d_x.to(u.km / u.s).value))
    V = float(np.mean(vel.d_y.to(u.km / u.s).value))
    W = float(np.mean(vel.d_z.to(u.km / u.s).value))

    ra_cp, dec_cp, mag = uvw_to_cp(U, V, W)
    return {
        "U_kms": U,
        "V_kms": V,
        "W_kms": W,
        "vc_mag_kms": mag,
        "ra_cp_deg": ra_cp,
        "dec_cp_deg": dec_cp,
    }


# ---------------------------------------------------------------------------
# fit_cp_lsq — pure-PM LSQ fit of (α_CP, δ_CP) + |V_c|
# ---------------------------------------------------------------------------


def _mean_pm_pointing(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pmra: np.ndarray,
    pmdec: np.ndarray,
) -> tuple[float, float]:
    """Seed (α_CP, δ_CP) from the unweighted mean PM and centre position.

    Each star's PM vector defines a great circle through its position; the
    cluster mean PM points in the local sky direction the cluster is
    moving. We take the centre position, displace it along (μ_α*, μ_δ)
    by a fixed angular distance (so we land in the cluster's direction of
    motion), and return that as the seed CP.
    """
    ra_c = float(np.mean(ra_deg))
    dec_c = float(np.mean(dec_deg))
    mu_ra_mean = float(np.mean(pmra))
    mu_dec_mean = float(np.mean(pmdec))
    norm = float(np.hypot(mu_ra_mean, mu_dec_mean))
    if norm == 0.0:
        return ra_c, dec_c

    # Step ~10 deg along the local tangent in the mean-PM direction.
    step_rad = np.deg2rad(10.0)
    dec_c_rad = np.deg2rad(dec_c)
    d_ra_rad = (mu_ra_mean / norm) * step_rad / np.cos(dec_c_rad)
    d_dec_rad = (mu_dec_mean / norm) * step_rad
    ra_cp = ra_c + np.rad2deg(d_ra_rad)
    dec_cp = dec_c + np.rad2deg(d_dec_rad)
    # Clamp dec to (-89, 89) to keep tan(δ_CP) finite in subsequent ops.
    dec_cp = float(np.clip(dec_cp, -89.0, 89.0))
    return float(ra_cp % 360.0), dec_cp


def _vc_mag_from_mu_par(
    pmra: np.ndarray,
    pmdec: np.ndarray,
    parallax_mas: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra_cp_deg: float,
    dec_cp_deg: float,
    sigma_mu_par: Optional[np.ndarray] = None,
    sin_lambda_min: float = 0.05,
) -> float:
    """Weighted mean of κ μ∥ / (ϖ sin λ), excluding stars with sin λ < min.

    Per-star variance of the V_c estimator is (κ / (ϖ sin λ))² σ²_μ∥, so
    optimal weight ∝ (ϖ sin λ / σ_μ∥)². If σ_μ∥ not provided, weights are
    (ϖ sin λ)² which still down-weights small-λ outliers.
    """
    tr = cp_transform(ra_deg, dec_deg, pmra, pmdec, ra_cp_deg, dec_cp_deg)
    lam = tr["lambda_rad"]
    sin_l = np.sin(lam)
    mu_par = tr["mu_par"]
    plx = np.asarray(parallax_mas, dtype=float)
    keep = np.isfinite(sin_l) & np.isfinite(mu_par) & np.isfinite(plx) & (
        sin_l > sin_lambda_min
    ) & (plx > 0.0)
    if keep.sum() < 1:
        return float("nan")
    per_star = MASYR_KPC_TO_KMS * mu_par[keep] / (plx[keep] * sin_l[keep])
    if sigma_mu_par is None:
        w = (plx[keep] * sin_l[keep]) ** 2
    else:
        sig = np.asarray(sigma_mu_par, dtype=float)[keep]
        # Replace non-positive σ with the median of the positives; if no
        # positives exist, fall back to unweighted (σ = 1).
        positive = sig > 0.0
        if positive.any():
            sig = np.where(positive, sig, np.median(sig[positive]))
        else:
            sig = np.ones_like(sig)
        w = (plx[keep] * sin_l[keep] / sig) ** 2
    w_sum = float(np.sum(w))
    if w_sum == 0.0:
        return float(np.mean(per_star))
    return float(np.sum(w * per_star) / w_sum)


def fit_cp_lsq(
    stars_df: pd.DataFrame,
    *,
    weights=None,
    ra_cp_seed: Optional[float] = None,
    dec_cp_seed: Optional[float] = None,
) -> dict:
    """Minimise Σ (μ⊥ / σ_μ⊥)² over (α_CP, δ_CP); then estimate |V_c|.

    stars_df columns: ra, dec, pmra, pmdec, parallax, pmra_error,
    pmdec_error, pmra_pmdec_corr. Initial seed defaults to the mean-PM
    pointing if not given. Returns the same dict shape as
    ``vc_from_prior_members``.
    """
    ra = stars_df["ra"].to_numpy(dtype=float)
    dec = stars_df["dec"].to_numpy(dtype=float)
    pmra = stars_df["pmra"].to_numpy(dtype=float)
    pmdec = stars_df["pmdec"].to_numpy(dtype=float)
    plx = stars_df["parallax"].to_numpy(dtype=float)
    pmra_err = stars_df["pmra_error"].to_numpy(dtype=float)
    pmdec_err = stars_df["pmdec_error"].to_numpy(dtype=float)
    if "pmra_pmdec_corr" in stars_df.columns:
        corr = stars_df["pmra_pmdec_corr"].to_numpy(dtype=float)
    else:
        corr = np.zeros_like(pmra_err)

    if ra_cp_seed is None or dec_cp_seed is None:
        seed_ra, seed_dec = _mean_pm_pointing(ra, dec, pmra, pmdec)
        if ra_cp_seed is None:
            ra_cp_seed = seed_ra
        if dec_cp_seed is None:
            dec_cp_seed = seed_dec

    def chi2(params: np.ndarray) -> float:
        ra_cp, dec_cp = float(params[0]), float(params[1])
        # Clamp dec to keep tan(δ_CP) finite.
        if not (-89.5 < dec_cp < 89.5):
            return 1e30
        tr = cp_transform(
            ra,
            dec,
            pmra,
            pmdec,
            ra_cp,
            dec_cp,
            pmra_err=pmra_err,
            pmdec_err=pmdec_err,
            pmra_pmdec_corr=corr,
        )
        mu_perp = tr["mu_perp"]
        sig = tr["sigma_mu_perp"]
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            sig_eff = sig / np.sqrt(np.clip(w, 1e-12, None))
        else:
            sig_eff = sig
        # Zero/non-finite σ_μ⊥ falls back to 1 mas/yr to avoid divide-by-zero
        # in chi²; this should not normally occur on quality-cleaned data.
        sig_eff = np.where(sig_eff > 0.0, sig_eff, 1.0)
        return float(np.sum((mu_perp / sig_eff) ** 2))

    x0 = np.array([ra_cp_seed, dec_cp_seed], dtype=float)
    res_nm = minimize(
        chi2,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000},
    )
    res_bfgs = minimize(
        chi2,
        res_nm.x,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 500},
    )
    ra_cp_fit, dec_cp_fit = float(res_bfgs.x[0] % 360.0), float(res_bfgs.x[1])

    # σ_μ_par for V_c weighting (vL2009 §3.2):
    #   μ∥ =  cos ψ · μ_α* + sin ψ · μ_δ
    #   Var(μ∥) = cos²ψ σ²_α + sin²ψ σ²_δ + 2 cos ψ sin ψ · corr · σ_α σ_δ
    _, psi = _lambda_psi(ra, dec, ra_cp_fit, dec_cp_fit)
    s_p = np.sin(psi)
    c_p = np.cos(psi)
    var_par = (
        (c_p ** 2) * pmra_err ** 2
        + (s_p ** 2) * pmdec_err ** 2
        + 2.0 * s_p * c_p * pmra_err * pmdec_err * corr
    )
    var_par = np.clip(var_par, 0.0, None)
    sigma_mu_par = np.sqrt(var_par)

    vc_mag = _vc_mag_from_mu_par(
        pmra, pmdec, plx, ra, dec, ra_cp_fit, dec_cp_fit, sigma_mu_par=sigma_mu_par
    )
    # V_c can come out negative when CP is on the antipode; if so flip sign and CP.
    if not np.isfinite(vc_mag):
        raise CPNotConverged(
            f"fit_cp_lsq: |V_c| not finite at (ra_cp, dec_cp) = ({ra_cp_fit}, {dec_cp_fit})"
        )
    if vc_mag < 0.0:
        vc_mag = -vc_mag
        ra_cp_fit = (ra_cp_fit + 180.0) % 360.0
        dec_cp_fit = -dec_cp_fit

    U, V, W = cp_to_uvw(ra_cp_fit, dec_cp_fit, vc_mag)
    return {
        "U_kms": U,
        "V_kms": V,
        "W_kms": W,
        "vc_mag_kms": vc_mag,
        "ra_cp_deg": ra_cp_fit,
        "dec_cp_deg": dec_cp_fit,
    }


# ---------------------------------------------------------------------------
# cp_iterate — drive (α_CP, δ_CP, V_c, member_mask) to convergence
# ---------------------------------------------------------------------------


def _angular_sep_deg(ra1, dec1, ra2, dec2) -> float:
    cos_d = np.sin(np.deg2rad(dec1)) * np.sin(np.deg2rad(dec2)) + np.cos(
        np.deg2rad(dec1)
    ) * np.cos(np.deg2rad(dec2)) * np.cos(np.deg2rad(ra1 - ra2))
    cos_d = float(np.clip(cos_d, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_d)))


def cp_iterate(
    stars_df: pd.DataFrame,
    *,
    vc_seed_dict: dict,
    membership_fn: Optional[Callable[[pd.DataFrame, float, float, float], np.ndarray]] = None,
    max_iter: int = 10,
    tol_deg: float = 0.01,
    tol_vc_kms: float = 0.1,
) -> dict:
    """Iterate (membership → fit (α_CP, δ_CP, V_c)) until convergence.

    Starts from ``vc_seed_dict``'s ``ra_cp_deg`` / ``dec_cp_deg`` /
    ``vc_mag_kms``. On each iteration:

    1. Compute membership mask via ``membership_fn(stars_df, ra_cp,
       dec_cp, V_c)`` (default: all True).
    2. Re-fit (α_CP, δ_CP, V_c) via ``fit_cp_lsq`` on the selected stars.
    3. If both Δ(α_CP, δ_CP) < ``tol_deg`` AND Δ|V_c| < ``tol_vc_kms``,
       declare convergence.

    Raises ``CPNotConverged`` on failure with the last iterate.
    Returns the final Vc dict plus ``selected_mask``, ``n_iter``,
    ``converged``.
    """
    ra_cp = float(vc_seed_dict["ra_cp_deg"])
    dec_cp = float(vc_seed_dict["dec_cp_deg"])
    vc_mag = float(vc_seed_dict["vc_mag_kms"])

    if membership_fn is None:
        def membership_fn(df, _ra, _dec, _vc):
            return np.ones(len(df), dtype=bool)

    converged = False
    last_mask: Optional[np.ndarray] = None
    out_dict: Optional[dict] = None
    n_iter = 0
    d_ang = float("inf")
    d_vc = float("inf")

    for it in range(1, max_iter + 1):
        mask = np.asarray(
            membership_fn(stars_df, ra_cp, dec_cp, vc_mag), dtype=bool
        )
        if mask.sum() < 3:
            raise CPNotConverged(
                f"cp_iterate: only {mask.sum()} members at iter {it}; "
                f"(ra_cp, dec_cp, vc_mag) = ({ra_cp:.3f}, {dec_cp:.3f}, {vc_mag:.3f})"
            )
        sub = stars_df.loc[mask].reset_index(drop=True)
        fit = fit_cp_lsq(sub, ra_cp_seed=ra_cp, dec_cp_seed=dec_cp)

        d_ang = _angular_sep_deg(ra_cp, dec_cp, fit["ra_cp_deg"], fit["dec_cp_deg"])
        d_vc = abs(fit["vc_mag_kms"] - vc_mag)

        ra_cp = fit["ra_cp_deg"]
        dec_cp = fit["dec_cp_deg"]
        vc_mag = fit["vc_mag_kms"]
        last_mask = mask
        out_dict = fit
        n_iter = it

        if d_ang < tol_deg and d_vc < tol_vc_kms:
            converged = True
            break

    if not converged or out_dict is None or last_mask is None:
        raise CPNotConverged(
            f"cp_iterate: failed after {max_iter} iterations. "
            f"last (ra_cp, dec_cp, vc_mag) = ({ra_cp:.4f}, {dec_cp:.4f}, {vc_mag:.4f}). "
            f"last Δang = {d_ang:.4f} deg, ΔV_c = {d_vc:.4f} km/s"
        )

    return {
        **out_dict,
        "selected_mask": last_mask,
        "n_iter": n_iter,
        "converged": converged,
    }
