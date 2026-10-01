"""Tests for `crosscat.cp_membership`.

Stage 0b + Stage 1 CP membership selectors:

- `rectangle_select`: Röser+2019 Hyades §2.2 axis-aligned cut in
  (ΔV∥, V⊥) frame.
- `poisson_5d_neighbourhood`: Röser+2019 Praesepe §2.3 5-D
  spatial + velocity neighbour count + Poisson background model.

The synthesis fixture builds a cluster (3D Gaussian σ_spatial = 2 pc,
isotropic 0.3 km/s intrinsic) embedded in a uniform Galactic-field box,
and verifies that the selector recovers ≥95% of cluster members while
rejecting ≥98% of field stars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    Galactic,
    ICRS,
    SphericalCosLatDifferential,
    SphericalRepresentation,
)

from crosscat.cp_membership import poisson_5d_neighbourhood, rectangle_select
from crosscat.cp_method import cp_observed, cp_predict, uvw_to_cp


# ---------------------------------------------------------------------------
# Hand-computed fixture for rectangle_select
# ---------------------------------------------------------------------------


def _make_hyades_like_vc_dict() -> dict:
    """Hyades-like CP from a known (U, V, W) = (-42.24, -19.00, -1.48)."""
    U, V, W = -42.24, -19.00, -1.48
    ra_cp, dec_cp, vc_mag = uvw_to_cp(U, V, W)
    return {
        "U_kms": U,
        "V_kms": V,
        "W_kms": W,
        "vc_mag_kms": vc_mag,
        "ra_cp_deg": ra_cp,
        "dec_cp_deg": dec_cp,
    }


def _make_synthetic_cluster_in_galactic(
    rng: np.random.Generator,
    *,
    n_cluster: int,
    n_field: int,
    centre_xyz_pc: tuple[float, float, float],
    sigma_spatial_pc: float,
    field_half_size_pc: float,
    vc_uvw_kms: tuple[float, float, float],
    sigma_intr_kms: float,
    field_v_half_size_kms: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build cluster + field, return ICRS observables + cluster-membership mask.

    Cluster: 3D Gaussian spatial, isotropic Gaussian intrinsic velocity around V_c.
    Field: uniform inside a box of half-size ``field_half_size_pc`` (positions)
    and ``field_v_half_size_kms`` (per-axis velocity, around 0 — but in the
    cluster's barycentric frame the field is offset by V_c).

    Returns: (df, is_cluster_truth_mask). df has columns
    (ra, dec, parallax_corrected, pmra, pmdec, d_pc).
    """
    n_cluster = int(n_cluster)
    n_field = int(n_field)

    # Cluster spatial: 3D Gaussian centred at centre_xyz_pc.
    cx, cy, cz = centre_xyz_pc
    cluster_xyz = rng.normal(
        loc=(cx, cy, cz),
        scale=sigma_spatial_pc,
        size=(n_cluster, 3),
    )

    # Field spatial: uniform cube of half-size, centred on cluster.
    field_xyz = rng.uniform(
        low=(cx - field_half_size_pc, cy - field_half_size_pc, cz - field_half_size_pc),
        high=(cx + field_half_size_pc, cy + field_half_size_pc, cz + field_half_size_pc),
        size=(n_field, 3),
    )

    # Cluster velocity: V_c plus isotropic Gaussian intrinsic dispersion.
    Ucl = vc_uvw_kms[0] + rng.normal(0.0, sigma_intr_kms, n_cluster)
    Vcl = vc_uvw_kms[1] + rng.normal(0.0, sigma_intr_kms, n_cluster)
    Wcl = vc_uvw_kms[2] + rng.normal(0.0, sigma_intr_kms, n_cluster)

    # Field velocity: uniform box in each axis around 0 (i.e. field stars have
    # arbitrary UVW relative to the cluster's V_c).
    Ufd = rng.uniform(-field_v_half_size_kms, field_v_half_size_kms, n_field)
    Vfd = rng.uniform(-field_v_half_size_kms, field_v_half_size_kms, n_field)
    Wfd = rng.uniform(-field_v_half_size_kms, field_v_half_size_kms, n_field)

    X = np.concatenate([cluster_xyz[:, 0], field_xyz[:, 0]])
    Y = np.concatenate([cluster_xyz[:, 1], field_xyz[:, 1]])
    Z = np.concatenate([cluster_xyz[:, 2], field_xyz[:, 2]])
    U = np.concatenate([Ucl, Ufd])
    V = np.concatenate([Vcl, Vfd])
    W = np.concatenate([Wcl, Wfd])

    pos = CartesianRepresentation(X * u.pc, Y * u.pc, Z * u.pc)
    vel = CartesianDifferential(U * u.km / u.s, V * u.km / u.s, W * u.km / u.s)
    gal = Galactic(pos.with_differentials(vel))
    icrs = gal.transform_to(ICRS())
    sph = icrs.represent_as(SphericalRepresentation, SphericalCosLatDifferential)

    ra_deg = np.rad2deg(sph.lon.rad)
    dec_deg = np.rad2deg(sph.lat.rad)
    dist_pc = sph.distance.to(u.pc).value
    parallax_mas = 1000.0 / dist_pc
    diff = sph.differentials["s"]
    pmra = diff.d_lon_coslat.to(u.mas / u.yr).value
    pmdec = diff.d_lat.to(u.mas / u.yr).value

    df = pd.DataFrame(
        {
            "ra": ra_deg,
            "dec": dec_deg,
            "parallax_corrected": parallax_mas,
            "pmra": pmra,
            "pmdec": pmdec,
            "d_pc": dist_pc,
        }
    )
    is_cluster = np.concatenate(
        [np.ones(n_cluster, dtype=bool), np.zeros(n_field, dtype=bool)]
    )
    return df, is_cluster


# ---------------------------------------------------------------------------
# Test 1 — rectangle_select basic pass/fail with hand-computed positions
# ---------------------------------------------------------------------------


def test_rectangle_select_basic(rng):
    """6-star fixture, hand-verified pass/fail in the (ΔV∥, V⊥) box."""
    vc = _make_hyades_like_vc_dict()
    ra_cp, dec_cp, vc_mag = vc["ra_cp_deg"], vc["dec_cp_deg"], vc["vc_mag_kms"]

    # Place six stars at a single sky position 5° from CP so V∥_pred ≈ vc_mag·sin(5°)
    # ≈ 4 km/s. Then dial each star's PM so the resulting (ΔV∥, V⊥_obs) takes
    # six known values bracketing the default rectangle (-5, +3) × (-0.8, +1.1).
    # We approach this by constructing PMs that produce target V∥_obs, V⊥_obs
    # directly via the inverse CP rotation.
    from crosscat.cp_method import _lambda_psi
    from crosscat.constants import MASYR_KPC_TO_KMS

    # Star sky position: 5° offset from CP, so sin λ ≠ 0.
    star_ra = float((ra_cp + 5.0) % 360.0)
    star_dec = float(dec_cp)
    parallax = 20.0  # 50 pc (mas)

    # Target (ΔV∥, V⊥_obs) values to test the box (-5, +3) × (-0.8, +1.1):
    # - star 0: (0, 0) — clearly inside
    # - star 1: (-4.9, 0) — inside, near low-V∥ edge
    # - star 2: (2.9, 0) — inside, near high-V∥ edge
    # - star 3: (-5.1, 0) — just outside low-V∥
    # - star 4: (0, 1.2) — just outside high-V⊥
    # - star 5: (0, -1.0) — just outside low-V⊥
    targets_dv_par_vperp = [
        (0.0, 0.0),
        (-4.9, 0.0),
        (2.9, 0.0),
        (-5.1, 0.0),
        (0.0, 1.2),
        (0.0, -1.0),
    ]
    expected = np.array([True, True, True, False, False, False])

    ra_arr = np.array([star_ra] * 6)
    dec_arr = np.array([star_dec] * 6)
    plx_arr = np.array([parallax] * 6)

    lam, psi = _lambda_psi(ra_arr, dec_arr, ra_cp, dec_cp)
    v_par_pred = vc_mag * np.sin(lam)

    # For each star, set V∥_obs = v_par_pred + ΔV∥, V⊥_obs as given, then
    # invert μ∥ = ϖ V∥ / κ, μ⊥ = ϖ V⊥ / κ, and rotate back to (μ_α*, μ_δ):
    #   μ_α* =  cos ψ · μ∥ − sin ψ · μ⊥
    #   μ_δ  =  sin ψ · μ∥ + cos ψ · μ⊥
    pmra = np.zeros(6)
    pmdec = np.zeros(6)
    for i, (dv_par, v_perp) in enumerate(targets_dv_par_vperp):
        v_par_obs = v_par_pred[i] + dv_par
        mu_par = plx_arr[i] * v_par_obs / MASYR_KPC_TO_KMS
        mu_perp = plx_arr[i] * v_perp / MASYR_KPC_TO_KMS
        cpsi, spsi = np.cos(psi[i]), np.sin(psi[i])
        pmra[i] = cpsi * mu_par - spsi * mu_perp
        pmdec[i] = spsi * mu_par + cpsi * mu_perp

    df = pd.DataFrame(
        {
            "ra": ra_arr,
            "dec": dec_arr,
            "parallax_corrected": plx_arr,
            "pmra": pmra,
            "pmdec": pmdec,
        }
    )

    mask = rectangle_select(df, vc_dict=vc)
    assert mask.dtype == bool
    assert mask.shape == (6,)
    np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# Test 2 — default rectangle bounds equal Röser+2019 Hyades §2.2 values
# ---------------------------------------------------------------------------


def test_rectangle_select_default_box():
    """The default ``dV_par_box`` and ``dV_perp_box`` defaults must
    equal (-5, +3) × (-0.8, +1.1) km/s exactly (Röser+2019 Hyades §2.2).
    """
    import inspect

    sig = inspect.signature(rectangle_select)
    default_par = sig.parameters["dV_par_box"].default
    default_perp = sig.parameters["dV_perp_box"].default
    assert tuple(default_par) == (-5.0, 3.0)
    assert tuple(default_perp) == (-0.8, 1.1)


# ---------------------------------------------------------------------------
# Test 3 — Poisson 5-D: signal recovery on synthetic cluster+field mix
# ---------------------------------------------------------------------------


def test_poisson_5d_signal_recovery(rng, capsys):
    """≥95% cluster recovery + ≥98% field rejection on synthetic mix."""
    vc = _make_hyades_like_vc_dict()
    # Choose a cluster centre 100 pc away on the +X (Galactic centre) axis.
    df, is_cluster = _make_synthetic_cluster_in_galactic(
        rng,
        n_cluster=500,
        n_field=50_000,
        centre_xyz_pc=(100.0, 0.0, 0.0),
        sigma_spatial_pc=2.0,
        field_half_size_pc=100.0,
        vc_uvw_kms=(vc["U_kms"], vc["V_kms"], vc["W_kms"]),
        sigma_intr_kms=0.3,
        field_v_half_size_kms=14.0,
    )

    mask, p_cont = poisson_5d_neighbourhood(df, vc_dict=vc)
    assert mask.dtype == bool
    assert mask.shape == (len(df),)
    assert p_cont.shape == (len(df),)

    cluster_recovered = mask[is_cluster].sum() / is_cluster.sum()
    field_rejected = (~mask[~is_cluster]).sum() / (~is_cluster).sum()
    print(
        f"\n[test_poisson_5d_signal_recovery] cluster_recovered = "
        f"{cluster_recovered:.4f}, field_rejected = {field_rejected:.4f}"
    )
    assert cluster_recovered >= 0.95
    assert field_rejected >= 0.98


# ---------------------------------------------------------------------------
# Test 4 — Poisson 5-D: p_cont distribution
# ---------------------------------------------------------------------------


def test_poisson_5d_p_cont_distribution(rng, capsys):
    """Cluster median p_cont < 0.1; field median p_cont > 0.8."""
    vc = _make_hyades_like_vc_dict()
    df, is_cluster = _make_synthetic_cluster_in_galactic(
        rng,
        n_cluster=500,
        n_field=50_000,
        centre_xyz_pc=(100.0, 0.0, 0.0),
        sigma_spatial_pc=2.0,
        field_half_size_pc=100.0,
        vc_uvw_kms=(vc["U_kms"], vc["V_kms"], vc["W_kms"]),
        sigma_intr_kms=0.3,
        field_v_half_size_kms=14.0,
    )
    _, p_cont = poisson_5d_neighbourhood(df, vc_dict=vc)
    cl_med = float(np.median(p_cont[is_cluster]))
    fd_med = float(np.median(p_cont[~is_cluster]))
    print(
        f"\n[test_poisson_5d_p_cont_distribution] cluster_p_cont_median = "
        f"{cl_med:.4f}, field_p_cont_median = {fd_med:.4f}"
    )
    assert cl_med < 0.1
    assert fd_med > 0.8


# ---------------------------------------------------------------------------
# Test 5 — Poisson 5-D: provided X_P/Y_P/Z_P bypass spatial recomputation
# ---------------------------------------------------------------------------


def test_poisson_5d_uses_provided_xyz(rng):
    """If X_P/Y_P/Z_P columns are provided, the function uses them
    (not the ra/dec/d_pc spatial recomputation path).

    Verified two ways:
    (a) supplying the canonical X_P/Y_P/Z_P (computed from ra/dec/d_pc the
        same way the function would) yields ≥95% recovery — i.e. the
        supplied-XYZ branch is functional.
    (b) supplying mangled X_P/Y_P/Z_P alongside correct ra/dec/d_pc tanks
        recovery — proving the function does NOT silently re-derive spatial
        coords from sky position when the X_P/Y_P/Z_P kwargs are passed.
    """
    vc = _make_hyades_like_vc_dict()
    df, is_cluster = _make_synthetic_cluster_in_galactic(
        rng,
        n_cluster=500,
        n_field=50_000,
        centre_xyz_pc=(100.0, 0.0, 0.0),
        sigma_spatial_pc=2.0,
        field_half_size_pc=100.0,
        vc_uvw_kms=(vc["U_kms"], vc["V_kms"], vc["W_kms"]),
        sigma_intr_kms=0.3,
        field_v_half_size_kms=14.0,
    )

    # Compute the canonical X_P/Y_P/Z_P from the synthetic galactic positions.
    from astropy.coordinates import SkyCoord

    sc = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        distance=df["d_pc"].values * u.pc,
        frame="icrs",
    )
    gal_cart = sc.transform_to(Galactic()).cartesian
    Xg = gal_cart.x.to(u.pc).value
    Yg = gal_cart.y.to(u.pc).value
    Zg = gal_cart.z.to(u.pc).value
    Xp_true = Xg - Xg.mean()
    Yp_true = Yg - Yg.mean()
    Zp_true = Zg - Zg.mean()

    # (a) supplied correct XYZ — should recover signal.
    df_a = df.copy()
    df_a["X_P"] = Xp_true
    df_a["Y_P"] = Yp_true
    df_a["Z_P"] = Zp_true
    mask_a, _ = poisson_5d_neighbourhood(
        df_a, vc_dict=vc, X_P_col="X_P", Y_P_col="Y_P", Z_P_col="Z_P"
    )
    recovered_a = mask_a[is_cluster].sum() / is_cluster.sum()
    assert recovered_a >= 0.95, (
        f"supplying correct X_P/Y_P/Z_P: expected ≥95% recovery, got {recovered_a:.3f}"
    )

    # (b) mangled XYZ alongside correct ra/dec — if the function ignored the
    # kwargs and re-derived from ra/dec, recovery would still be high.
    # If it uses the supplied (mangled) XYZ, recovery drops sharply.
    df_b = df.copy()
    df_b["X_P"] = rng.uniform(-100.0, 100.0, len(df_b))
    df_b["Y_P"] = rng.uniform(-100.0, 100.0, len(df_b))
    df_b["Z_P"] = rng.uniform(-100.0, 100.0, len(df_b))
    mask_b, _ = poisson_5d_neighbourhood(
        df_b, vc_dict=vc, X_P_col="X_P", Y_P_col="Y_P", Z_P_col="Z_P"
    )
    recovered_b = mask_b[is_cluster].sum() / is_cluster.sum()
    assert recovered_b < 0.5, (
        f"mangled X_P/Y_P/Z_P should tank recovery (proving kwargs honoured); "
        f"got {recovered_b:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Poisson 5-D: pure field, no cluster, sees mostly nothing
# ---------------------------------------------------------------------------


def test_poisson_5d_no_cluster(rng, capsys):
    """5000 uniform-field stars, no cluster. Expect very few false positives."""
    vc = _make_hyades_like_vc_dict()
    df, _ = _make_synthetic_cluster_in_galactic(
        rng,
        n_cluster=0,
        n_field=5_000,
        centre_xyz_pc=(100.0, 0.0, 0.0),
        sigma_spatial_pc=2.0,
        field_half_size_pc=100.0,
        vc_uvw_kms=(vc["U_kms"], vc["V_kms"], vc["W_kms"]),
        sigma_intr_kms=0.3,
        field_v_half_size_kms=14.0,
    )

    mask, _ = poisson_5d_neighbourhood(df, vc_dict=vc)
    n_selected = int(mask.sum())
    print(
        f"\n[test_poisson_5d_no_cluster] selected = {n_selected}/5000 = "
        f"{n_selected / 5000:.4f}"
    )
    assert n_selected < 100, (
        f"expected <100 false positives on pure field; got {n_selected}"
    )
