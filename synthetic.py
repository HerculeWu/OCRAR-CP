"""Seeded synthetic CP validation; numerical routines retained from the manuscript generator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    Galactic,
    ICRS,
    SphericalCosLatDifferential,
    SphericalRepresentation,
)

PROJECT_ROOT = Path(__file__).resolve().parent


from crosscat.constants import MASYR_KPC_TO_KMS  # noqa: E402
from crosscat.cp_membership import poisson_5d_neighbourhood  # noqa: E402
from crosscat.cp_method import (  # noqa: E402
    cp_iterate,
    cp_observed,
    cp_transform,
    uvw_to_cp,
    vc_from_prior_members,
)
from crosscat.plotting import (  # noqa: E402
    SINGLE_COLUMN_WIDTH_INCH,
    configure_matplotlib,
)



# Seed: same value used by tests/conftest.py:rng so the script behaves
# identically to the unit tests.
SEED = 20260512


# ---------------------------------------------------------------------------
# Self-contained mock-cluster factory
# ---------------------------------------------------------------------------


def make_mock_cluster(
    rng: np.random.Generator,
    *,
    n_stars: int,
    centre_xyz_pc: tuple[float, float, float],
    radius_pc: float,
    vc_uvw_kms: tuple[float, float, float],
    sigma_intr_kms: float,
    pm_err_masyr: float,
) -> pd.DataFrame:
    """Generate a synthetic cluster as it would be observed by Gaia.

    Duplicate of ``tests/conftest.py:make_mock_cluster`` — kept in the
    script so it stays runnable without depending on the tests package.

    Stars are uniformly distributed inside a sphere of ``radius_pc`` at
    ``centre_xyz_pc`` in Galactic Cartesian coordinates; per-star
    velocity = V_c (Galactic Cartesian UVW) + 3D isotropic Gaussian
    ``sigma_intr_kms``. Returns observable columns: ra, dec, parallax,
    pmra (=μ_α cos δ), pmdec, pmra_error, pmdec_error, pmra_pmdec_corr,
    radial_velocity, plus pmra_true / pmdec_true for diagnostics.
    """
    n = int(n_stars)
    u_rad = rng.uniform(size=n)
    r = float(radius_pc) * u_rad ** (1.0 / 3.0)
    cos_theta = rng.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    dx = r * sin_theta * np.cos(phi)
    dy = r * sin_theta * np.sin(phi)
    dz = r * cos_theta

    X = centre_xyz_pc[0] + dx
    Y = centre_xyz_pc[1] + dy
    Z = centre_xyz_pc[2] + dz

    U = vc_uvw_kms[0] + rng.normal(0.0, sigma_intr_kms, size=n)
    V = vc_uvw_kms[1] + rng.normal(0.0, sigma_intr_kms, size=n)
    W = vc_uvw_kms[2] + rng.normal(0.0, sigma_intr_kms, size=n)

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
    pmra_true = diff.d_lon_coslat.to(u.mas / u.yr).value  # already cos(dec)-corrected
    pmdec_true = diff.d_lat.to(u.mas / u.yr).value
    rv_true = diff.d_distance.to(u.km / u.s).value

    pmra_obs = pmra_true + rng.normal(0.0, pm_err_masyr, size=n)
    pmdec_obs = pmdec_true + rng.normal(0.0, pm_err_masyr, size=n)

    return pd.DataFrame(
        {
            "ra": ra_deg,
            "dec": dec_deg,
            "parallax": parallax_mas,
            "pmra": pmra_obs,
            "pmdec": pmdec_obs,
            "pmra_error": np.full(n, pm_err_masyr),
            "pmdec_error": np.full(n, pm_err_masyr),
            "pmra_pmdec_corr": np.zeros(n),
            "radial_velocity": rv_true,
            "pmra_true": pmra_true,
            "pmdec_true": pmdec_true,
        }
    )


def _make_synthetic_cluster_field(
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
    """Cluster (3D-Gaussian spatial, V_c+σ velocity) embedded in uniform field.

    Duplicate of the test-side helper. Returns DataFrame with the
    columns ``poisson_5d_neighbourhood`` expects plus a truth mask.
    """
    n_cluster = int(n_cluster)
    n_field = int(n_field)

    cx, cy, cz = centre_xyz_pc
    cluster_xyz = rng.normal(
        loc=(cx, cy, cz),
        scale=sigma_spatial_pc,
        size=(n_cluster, 3),
    )
    field_xyz = rng.uniform(
        low=(cx - field_half_size_pc, cy - field_half_size_pc, cz - field_half_size_pc),
        high=(cx + field_half_size_pc, cy + field_half_size_pc, cz + field_half_size_pc),
        size=(n_field, 3),
    )
    Ucl = vc_uvw_kms[0] + rng.normal(0.0, sigma_intr_kms, n_cluster)
    Vcl = vc_uvw_kms[1] + rng.normal(0.0, sigma_intr_kms, n_cluster)
    Wcl = vc_uvw_kms[2] + rng.normal(0.0, sigma_intr_kms, n_cluster)
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
# Tests
# ---------------------------------------------------------------------------


def run_test_0a_1(rng: np.random.Generator) -> dict:
    """Narrow-angle (3°-extent) recovery of V_c and CP."""
    vc_true = (-40.0, -20.0, 0.0)
    df = make_mock_cluster(
        rng,
        n_stars=500,
        centre_xyz_pc=(100.0, 0.0, 0.0),
        radius_pc=5.0,
        vc_uvw_kms=vc_true,
        sigma_intr_kms=0.5,
        pm_err_masyr=0.05,
    )

    # RV-based seed
    seed = vc_from_prior_members(df)
    # Pure-PM iterate
    out = cp_iterate(df, vc_seed_dict=seed)

    ra_cp_true, dec_cp_true, vc_mag_true = uvw_to_cp(*vc_true)
    cos_d = np.sin(np.deg2rad(dec_cp_true)) * np.sin(
        np.deg2rad(out["dec_cp_deg"])
    ) + np.cos(np.deg2rad(dec_cp_true)) * np.cos(
        np.deg2rad(out["dec_cp_deg"])
    ) * np.cos(np.deg2rad(out["ra_cp_deg"] - ra_cp_true))
    cos_d = float(np.clip(cos_d, -1.0, 1.0))
    sep_deg = float(np.rad2deg(np.arccos(cos_d)))

    cp_ok = sep_deg < 0.5
    u_ok = abs(out["U_kms"] - vc_true[0]) < 1.0
    v_ok = abs(out["V_kms"] - vc_true[1]) < 1.0
    w_ok = abs(out["W_kms"] - vc_true[2]) < 1.0
    converged_ok = bool(out["converged"])
    passed = bool(converged_ok and cp_ok and u_ok and v_ok and w_ok)

    return {
        "test_id": "0a.1",
        "name": "narrow_angle_recovery",
        "passed": passed,
        "converged": converged_ok,
        "u_recovered_kms": float(out["U_kms"]),
        "v_recovered_kms": float(out["V_kms"]),
        "w_recovered_kms": float(out["W_kms"]),
        "cp_separation_deg": sep_deg,
        "vc_mag_recovered_kms": float(out["vc_mag_kms"]),
        "vc_mag_true_kms": float(vc_mag_true),
        "n_iter": int(out["n_iter"]),
        "n_stars": int(len(df)),
    }


def run_test_0a_2(rng: np.random.Generator, *, extras: dict) -> dict:
    """Perspective subtraction (10°-extent Hyades mock). CORE FALSIFICATION TEST.

    ``extras`` is populated with per-star arrays so the figure can be
    drawn afterwards.
    """
    centre = (-44.77, 0.40, -16.24)  # Hyades-like Galactic Cartesian pc
    vc_uvw = (-42.24, -19.00, -1.48)  # Röser+2019 Hyades V_c
    sigma_intr = 0.3
    pm_err = 0.05

    df = make_mock_cluster(
        rng,
        n_stars=500,
        centre_xyz_pc=centre,
        radius_pc=10.0,
        vc_uvw_kms=vc_uvw,
        sigma_intr_kms=sigma_intr,
        pm_err_masyr=pm_err,
    )

    # --- σ_raw: HR-pipeline style (subtract cluster-mean PM, scale by κ/ϖ) ---
    kappa = MASYR_KPC_TO_KMS
    plx = df["parallax"].to_numpy()
    v_pmra_kms = kappa * df["pmra"].to_numpy() / plx
    v_pmdec_kms = kappa * df["pmdec"].to_numpy() / plx
    mean_pmra = float(np.mean(df["pmra"].to_numpy()))
    mean_pmdec = float(np.mean(df["pmdec"].to_numpy()))
    v_pmra_mean = kappa * mean_pmra / plx
    v_pmdec_mean = kappa * mean_pmdec / plx
    res_ra = v_pmra_kms - v_pmra_mean
    res_dec = v_pmdec_kms - v_pmdec_mean
    sigma_raw_ra = float(np.std(res_ra, ddof=1))
    sigma_raw_dec = float(np.std(res_dec, ddof=1))
    sigma_raw = float(np.sqrt(0.5 * (sigma_raw_ra ** 2 + sigma_raw_dec ** 2)))

    # --- σ_CP: CP-frame residuals after iterative fit ---
    seed = vc_from_prior_members(df)
    out = cp_iterate(df, vc_seed_dict=seed)
    ra_cp = float(out["ra_cp_deg"])
    dec_cp = float(out["dec_cp_deg"])
    vc_mag = float(out["vc_mag_kms"])

    tr = cp_transform(
        df["ra"].to_numpy(),
        df["dec"].to_numpy(),
        df["pmra"].to_numpy(),
        df["pmdec"].to_numpy(),
        ra_cp,
        dec_cp,
        pmra_err=df["pmra_error"].to_numpy(),
        pmdec_err=df["pmdec_error"].to_numpy(),
        pmra_pmdec_corr=df["pmra_pmdec_corr"].to_numpy(),
    )
    lam = tr["lambda_rad"]
    v_par_obs, v_perp_obs = cp_observed(
        df["pmra"].to_numpy(),
        df["pmdec"].to_numpy(),
        plx,
        df["ra"].to_numpy(),
        df["dec"].to_numpy(),
        ra_cp,
        dec_cp,
    )
    v_par_pred = vc_mag * np.sin(lam)
    sigma_par = float(np.std(v_par_obs - v_par_pred, ddof=1))
    sigma_perp = float(np.std(v_perp_obs, ddof=1))
    sigma_cp = float(np.sqrt(0.5 * (sigma_par ** 2 + sigma_perp ** 2)))

    # Populate extras for the figure.
    extras["res_ra_kms"] = res_ra
    extras["res_dec_kms"] = res_dec
    extras["v_par_residual_kms"] = v_par_obs - v_par_pred
    extras["v_perp_obs_kms"] = v_perp_obs
    extras["sigma_raw"] = sigma_raw
    extras["sigma_cp"] = sigma_cp
    extras["sigma_par"] = sigma_par
    extras["sigma_perp"] = sigma_perp
    extras["sigma_intr_true"] = sigma_intr

    raw_ok = sigma_raw > 2.0
    cp_ok = 0.20 < sigma_cp < 0.45
    converged_ok = bool(out["converged"])
    passed = bool(converged_ok and raw_ok and cp_ok)

    return {
        "test_id": "0a.2",
        "name": "perspective_subtraction",
        "passed": passed,
        "converged": converged_ok,
        "sigma_raw_kms": sigma_raw,
        "sigma_cp_kms": sigma_cp,
        "sigma_par_kms": sigma_par,
        "sigma_perp_kms": sigma_perp,
        "sigma_intr_true_kms": float(sigma_intr),
        "ratio_raw_over_intr": sigma_raw / sigma_intr,
        "u_recovered_kms": float(out["U_kms"]),
        "v_recovered_kms": float(out["V_kms"]),
        "w_recovered_kms": float(out["W_kms"]),
        "vc_mag_recovered_kms": vc_mag,
        "n_stars": int(len(df)),
    }


def run_test_0a_3(rng: np.random.Generator) -> dict:
    """500 cluster + 50 000 uniform-field; 5-D Poisson density selection."""
    vc_uvw = (-42.24, -19.00, -1.48)
    ra_cp, dec_cp, vc_mag = uvw_to_cp(*vc_uvw)
    vc_dict = {
        "U_kms": vc_uvw[0],
        "V_kms": vc_uvw[1],
        "W_kms": vc_uvw[2],
        "vc_mag_kms": vc_mag,
        "ra_cp_deg": ra_cp,
        "dec_cp_deg": dec_cp,
    }
    df, is_cluster = _make_synthetic_cluster_field(
        rng,
        n_cluster=500,
        n_field=50_000,
        centre_xyz_pc=(100.0, 0.0, 0.0),
        sigma_spatial_pc=2.0,
        field_half_size_pc=100.0,
        vc_uvw_kms=vc_uvw,
        sigma_intr_kms=0.3,
        field_v_half_size_kms=14.0,
    )

    mask, p_cont = poisson_5d_neighbourhood(df, vc_dict=vc_dict)
    cluster_recovered = float(mask[is_cluster].sum()) / float(is_cluster.sum())
    field_rejected = float((~mask[~is_cluster]).sum()) / float((~is_cluster).sum())

    recovery_ok = cluster_recovered >= 0.95
    rejection_ok = field_rejected >= 0.98
    passed = bool(recovery_ok and rejection_ok)

    return {
        "test_id": "0a.3",
        "name": "poisson_5d_signal_recovery",
        "passed": passed,
        "cluster_recovered_pct": 100.0 * cluster_recovered,
        "field_rejected_pct": 100.0 * field_rejected,
        "n_cluster_truth": int(is_cluster.sum()),
        "n_field_truth": int((~is_cluster).sum()),
        "n_selected": int(mask.sum()),
        "cluster_p_cont_median": float(np.median(p_cont[is_cluster])),
        "field_p_cont_median": float(np.median(p_cont[~is_cluster])),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_perspective_figure(extras: dict, out_path: Path) -> None:
    """Overlay unchanged residual arrays with explicit component-basis labels.

    Red: RA/Dec PM residuals converted with individual parallaxes.
    Blue: CP parallel/perpendicular residual velocities. Their common display
    scale compares widths; the two colours do not share a coordinate basis.
    """
    configure_matplotlib()

    res_ra = extras["res_ra_kms"]
    res_dec = extras["res_dec_kms"]
    v_par_res = extras["v_par_residual_kms"]
    v_perp = extras["v_perp_obs_kms"]
    sigma_raw = extras["sigma_raw"]
    sigma_cp = extras["sigma_cp"]
    sigma_intr = extras["sigma_intr_true"]

    # Symmetric limit covering both clouds.
    lim = float(
        np.max(
            np.abs(
                np.concatenate([res_ra, res_dec, v_par_res, v_perp])
            )
        )
    )
    lim = max(lim * 1.05, 1.0)

    # Preserve both clouds and velocity limits; identify their different bases.
    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH_INCH, SINGLE_COLUMN_WIDTH_INCH)
    )

    ax.scatter(
        res_ra, res_dec, s=4.0, alpha=0.30, color="tab:red",
        edgecolors="none", rasterized=True, zorder=2,
        label=rf"PM (RA/Dec): $\sigma_{{\rm obs}}={sigma_raw:.2f}$ $\mathrm{{km}}\,\mathrm{{s}}^{{-1}}$",
    )
    ax.scatter(
        v_par_res, v_perp, s=6.0, alpha=0.8, color="tab:blue",
        edgecolors="none", rasterized=True, zorder=3,
        label=rf"CP: $\sigma_{{\rm CP}}={sigma_cp:.2f}$ $\mathrm{{km}}\,\mathrm{{s}}^{{-1}}$",
    )
    ax.axhline(0.0, color="black", lw=0.4, ls=":")
    ax.axvline(0.0, color="black", lw=0.4, ls=":")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta V_{\alpha *}$ (red), $\Delta V_\parallel$ (blue) [$\mathrm{km}\,\mathrm{s}^{-1}$]", fontsize=8)
    ax.set_ylabel(r"$\Delta V_\delta$ (red), $V_\perp$ (blue) [$\mathrm{km}\,\mathrm{s}^{-1}$]", fontsize=8)
    ax.text(0.03, 0.03, rf"Injected $\sigma_{{\rm intr}}={sigma_intr:.2f}$ $\mathrm{{km}}\,\mathrm{{s}}^{{-1}}$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    ax.legend(loc="upper right", fontsize=7, markerscale=1.6,
              handletextpad=0.3, borderaxespad=0.4,
              frameon=True, framealpha=0.85, edgecolor="none")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def figure_inputs():
    """Replay the original RNG sequence; return both results and plotted arrays."""
    rng = np.random.default_rng(SEED)
    results = [run_test_0a_1(rng)]
    extras = {}
    results.append(run_test_0a_2(rng, extras=extras))
    return results, extras, rng
