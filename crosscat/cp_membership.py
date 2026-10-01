"""CP membership selectors: rectangle cut + 5-D Poisson neighbourhood.

Two stateless selectors operating on CP-frame coordinates produced by
``crosscat.cp_method.cp_observed`` / ``cp_predict``:

- ``rectangle_select`` — Röser+2019 Hyades §2.2 axis-aligned cut in
  (ΔV∥, V⊥_obs) space, defaults to the published Hyades box
  (-5, +3) × (-0.8, +1.1) km/s.

- ``poisson_5d_neighbourhood`` — Röser+2019 Praesepe §2.3 5-D method:
  count neighbours of each star inside a Cartesian-spatial sphere
  (radius ``r_lim_pc``) × velocity ellipse (semi-axes ``a_kms`` along
  ΔV∥, ``b_kms`` along ΔV⊥), then estimate the Galactic-field
  background as a Poisson distribution with mean ``λ`` taken from the
  basic-sample (BS) cases where ``k ≤ 5``. Per-star contamination
  probability ``p_cont(k) = N_exp(k) / N_obs(k)`` is clipped to [0, 1];
  ``selected_mask = (k ≥ k_min)``.

Spatial coordinates are cluster-centric Galactic Cartesian (X_P, Y_P,
Z_P) — pre-supplied as DataFrame columns or computed on the fly via the
ICRS→Galactic transform and re-centred at the mean position of the
input stars (Praesepe §2.3 convention: cluster centre = barycentre of
the input sample).

Implementation note on the 5-D scaling: pre-scale spatial → r_lim_pc
and velocity → (a_kms, b_kms); then the neighbourhood is exactly a unit
ball in 5-D scaled space. The 5-D Euclidean distance² then equals
``(Δr_spatial/r_lim)² + (ΔV∥/a)² + (ΔV⊥/b)²`` which is stricter than the intersection of a spatial sphere and a
velocity ellipse. This is an adaptation, not the identical published selector.

Edge cases:
- If λ_BS = 0 (no neighbours anywhere) the Poisson background pmf is a
  Dirac at k=0; we treat p_cont(0) = 1 and p_cont(k>0) = 0 so that
  every k ≥ 1 star is "uncontaminated signal". This is a degenerate case
  in practice — it requires r_lim, a, b small enough that no star has
  any neighbours.
- If two stars share identical spatial+velocity coordinates the
  cKDTree counts both (each is the other's neighbour); ``self`` is
  excluded explicitly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Galactic, SkyCoord
from scipy.spatial import cKDTree
from scipy.stats import poisson

from .cp_method import cp_observed, cp_predict


# ---------------------------------------------------------------------------
# rectangle_select — Röser+2019 Hyades §2.2 box cut
# ---------------------------------------------------------------------------


def rectangle_select(
    stars_df: pd.DataFrame,
    *,
    dV_par_box: tuple[float, float] = (-5.0, 3.0),
    dV_perp_box: tuple[float, float] = (-0.8, 1.1),
    vc_dict: dict,
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    parallax_col: str = "parallax_corrected",
) -> np.ndarray:
    """Axis-aligned rectangle cut on (ΔV∥, V⊥_obs).

    For each star compute V∥_obs, V⊥_obs via ``cp_method.cp_observed``;
    V∥_pred = |V_c| · sin λ via ``cp_method.cp_predict`` (V⊥_pred ≡ 0).
    A star passes if both:

    - ``dV_par_box[0] ≤ V∥_obs − V∥_pred ≤ dV_par_box[1]``
    - ``dV_perp_box[0] ≤ V⊥_obs ≤ dV_perp_box[1]``

    Defaults reproduce the Hyades §2.2 box (-5, +3) × (-0.8, +1.1) km/s
    (Röser+2019). Bounds are interpreted **inclusive on both ends**; if
    you need a half-open convention apply a small epsilon to the input.

    Parameters
    ----------
    stars_df : DataFrame
        Must contain ``ra_col``, ``dec_col``, ``pmra_col``, ``pmdec_col``,
        ``parallax_col`` (parallax already zero-point-corrected; sign
        positive).
    vc_dict : dict
        Must contain ``ra_cp_deg``, ``dec_cp_deg``, ``vc_mag_kms``.

    Returns
    -------
    mask : ndarray[bool]
        Length matches ``len(stars_df)``. Stars with NaN V∥_obs or V⊥_obs
        (negative parallax) automatically fail (their values won't satisfy
        either bound).
    """
    ra = stars_df[ra_col].to_numpy(dtype=float)
    dec = stars_df[dec_col].to_numpy(dtype=float)
    pmra = stars_df[pmra_col].to_numpy(dtype=float)
    pmdec = stars_df[pmdec_col].to_numpy(dtype=float)
    plx = stars_df[parallax_col].to_numpy(dtype=float)

    ra_cp = float(vc_dict["ra_cp_deg"])
    dec_cp = float(vc_dict["dec_cp_deg"])
    vc_mag = float(vc_dict["vc_mag_kms"])

    v_par_obs, v_perp_obs = cp_observed(pmra, pmdec, plx, ra, dec, ra_cp, dec_cp)
    v_par_pred, _ = cp_predict(ra, dec, ra_cp, dec_cp, vc_mag)
    d_v_par = v_par_obs - v_par_pred

    in_par = (d_v_par >= dV_par_box[0]) & (d_v_par <= dV_par_box[1])
    in_perp = (v_perp_obs >= dV_perp_box[0]) & (v_perp_obs <= dV_perp_box[1])
    # NaN comparisons return False on both sides, so stars with bad ϖ
    # naturally fail.
    return np.asarray(in_par & in_perp, dtype=bool)


# ---------------------------------------------------------------------------
# poisson_5d_neighbourhood — Röser+2019 Praesepe §2.3 5-D density method
# ---------------------------------------------------------------------------


def _galactic_xp_yp_zp(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    d_pc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ICRS→Galactic Cartesian (pc), recentred at the mean position.

    Cluster centre = mean (X, Y, Z) of the input stars (Praesepe §2.3 has
    the cluster centre coincide with the barycentre of the over-density;
    on field-only or single-cluster input the mean is the most defensible
    centre.)
    """
    sc = SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        distance=np.asarray(d_pc, dtype=float) * u.pc,
        frame="icrs",
    )
    cart = sc.transform_to(Galactic()).cartesian
    X = cart.x.to(u.pc).value
    Y = cart.y.to(u.pc).value
    Z = cart.z.to(u.pc).value
    return X - X.mean(), Y - Y.mean(), Z - Z.mean()


def poisson_5d_neighbourhood(
    stars_df: pd.DataFrame,
    *,
    vc_dict: dict,
    a_kms: float = 1.2,
    b_kms: float = 0.5,
    r_lim_pc: float = 15.0,
    k_min: int = 3,
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: str = "pmra",
    pmdec_col: str = "pmdec",
    parallax_col: str = "parallax_corrected",
    d_pc_col: Optional[str] = "d_pc",
    X_P_col: Optional[str] = None,
    Y_P_col: Optional[str] = None,
    Z_P_col: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """5-D Poisson-density CP membership (Röser+2019 Praesepe §2.3).

    For each star compute:

    1. Cluster-centric Galactic Cartesian (X_P, Y_P, Z_P) — either taken
       from ``stars_df[X_P_col / Y_P_col / Z_P_col]`` if all three are
       provided, or derived from (ra, dec, ``d_pc_col``) via the
       ICRS→Galactic transform recentred at the mean position.
    2. CP-frame velocity coordinates (ΔV∥, ΔV⊥) where
       ΔV∥ = V∥_obs − V∥_pred and ΔV⊥ = V⊥_obs − V⊥_pred = V⊥_obs.

    Stars are pre-scaled — spatial by ``r_lim_pc``, ΔV∥ by ``a_kms``,
    ΔV⊥ by ``b_kms`` — and a ``scipy.spatial.cKDTree`` returns 5-D unit-
    radius neighbour counts (stricter than the intersection of the spatial sphere and
    velocity ellipse — see module docstring). Self-counts are excluded.

    Background model: mean λ taken from the ``k ≤ 5`` slice of the
    sample (Praesepe §2.3 "basic-sample" approximation); N_BS = number
    of stars in that slice; N_exp(k) = poisson.pmf(k, λ) · N_BS;
    N_obs(k) = histogram of observed k-values across the full input.
    p_cont(k) = N_exp(k) / N_obs(k) clipped to [0, 1]. p_cont(k=0)
    is set to 1.0 by convention (a zero-neighbour star is field by
    construction).

    Parameters
    ----------
    stars_df : DataFrame
        Required columns depend on whether X_P/Y_P/Z_P are pre-supplied:
        always (``ra_col``, ``dec_col``, ``pmra_col``, ``pmdec_col``,
        ``parallax_col``); if spatial columns are not provided also
        ``d_pc_col``.
    vc_dict : dict
        Must contain ``ra_cp_deg``, ``dec_cp_deg``, ``vc_mag_kms``.

    Returns
    -------
    selected_mask : ndarray[bool]
        ``k_neighbours >= k_min``.
    p_cont : ndarray[float]
        Per-star contamination probability in [0, 1]. Stars with bad
        astrometry (NaN ΔV) are conservatively assigned p_cont = 1 and
        excluded from selection.
    """
    n = len(stars_df)
    ra = stars_df[ra_col].to_numpy(dtype=float)
    dec = stars_df[dec_col].to_numpy(dtype=float)
    pmra = stars_df[pmra_col].to_numpy(dtype=float)
    pmdec = stars_df[pmdec_col].to_numpy(dtype=float)
    plx = stars_df[parallax_col].to_numpy(dtype=float)

    ra_cp = float(vc_dict["ra_cp_deg"])
    dec_cp = float(vc_dict["dec_cp_deg"])
    vc_mag = float(vc_dict["vc_mag_kms"])

    # ---- spatial coordinates ----
    use_supplied_xyz = (
        X_P_col is not None
        and Y_P_col is not None
        and Z_P_col is not None
        and X_P_col in stars_df.columns
        and Y_P_col in stars_df.columns
        and Z_P_col in stars_df.columns
    )
    if use_supplied_xyz:
        Xp = stars_df[X_P_col].to_numpy(dtype=float)
        Yp = stars_df[Y_P_col].to_numpy(dtype=float)
        Zp = stars_df[Z_P_col].to_numpy(dtype=float)
    else:
        if d_pc_col is None or d_pc_col not in stars_df.columns:
            raise ValueError(
                "poisson_5d_neighbourhood: need either (X_P_col, Y_P_col, Z_P_col) "
                f"or d_pc_col='{d_pc_col}' present in stars_df.columns"
            )
        d_pc = stars_df[d_pc_col].to_numpy(dtype=float)
        Xp, Yp, Zp = _galactic_xp_yp_zp(ra, dec, d_pc)

    # ---- velocity coordinates: ΔV∥ = V∥_obs − V∥_pred, ΔV⊥ = V⊥_obs ----
    v_par_obs, v_perp_obs = cp_observed(pmra, pmdec, plx, ra, dec, ra_cp, dec_cp)
    v_par_pred, _ = cp_predict(ra, dec, ra_cp, dec_cp, vc_mag)
    d_v_par = v_par_obs - v_par_pred
    d_v_perp = v_perp_obs

    # ---- finite-input filter ----
    finite = (
        np.isfinite(Xp)
        & np.isfinite(Yp)
        & np.isfinite(Zp)
        & np.isfinite(d_v_par)
        & np.isfinite(d_v_perp)
    )

    # ---- 5-D unit-ball KDTree on scaled coordinates ----
    # Stars with non-finite coordinates contribute neither as queries nor
    # as neighbours.
    if finite.sum() == 0:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=float)

    pts5 = np.column_stack(
        [
            Xp[finite] / r_lim_pc,
            Yp[finite] / r_lim_pc,
            Zp[finite] / r_lim_pc,
            d_v_par[finite] / a_kms,
            d_v_perp[finite] / b_kms,
        ]
    )
    tree = cKDTree(pts5)
    # query_ball_point with r=1 returns neighbours within unit 5-D radius
    # (inclusive of self → subtract 1).
    raw = tree.query_ball_point(pts5, r=1.0, return_length=True)
    k_finite = np.asarray(raw, dtype=int) - 1  # exclude self

    # Pad back into full-length k array (NaN-coord stars get k = -1 sentinel)
    k_all = np.full(n, -1, dtype=int)
    k_all[finite] = k_finite

    # ---- Poisson background model from the basic-sample (k ≤ 5) ----
    bs_mask = (k_all >= 0) & (k_all <= 5)
    if bs_mask.sum() == 0:
        # Pathological — no BS stars; treat everything as signal of unknown
        # contamination (p_cont = 0.5).
        p_cont_all = np.full(n, 0.5, dtype=float)
        selected = np.zeros(n, dtype=bool)
        return selected, p_cont_all
    lam = float(k_all[bs_mask].mean())
    N_BS = int(bs_mask.sum())

    # ---- N_exp(k) and N_obs(k) and p_cont(k) ----
    k_max = int(k_all[k_all >= 0].max()) if (k_all >= 0).any() else 0
    k_range = np.arange(0, k_max + 1)
    if lam <= 0.0:
        # Degenerate: Poisson collapses to a Dirac at k=0.
        # Standard convention: p_cont(0) = 1, p_cont(k>0) = 0.
        p_cont_per_k = np.where(k_range == 0, 1.0, 0.0)
    else:
        N_exp = poisson.pmf(k_range, lam) * N_BS
        # N_obs(k): histogram across all finite stars.
        N_obs = np.bincount(k_all[k_all >= 0], minlength=k_max + 1).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_cont_per_k = np.where(N_obs > 0, N_exp / N_obs, 0.0)
        p_cont_per_k = np.clip(p_cont_per_k, 0.0, 1.0)
        # Force p_cont(k=0) = 1 — zero-neighbour stars are pure field.
        p_cont_per_k[0] = 1.0

    # ---- per-star p_cont ----
    p_cont = np.ones(n, dtype=float)
    valid = k_all >= 0
    p_cont[valid] = p_cont_per_k[k_all[valid]]

    selected = (k_all >= int(k_min)) & valid
    return np.asarray(selected, dtype=bool), p_cont
