"""Tests for `crosscat.cp_kinematics` — σ_∥, σ_⊥, σ_RV combiner.

Plan §5 (Röser+2019 + Van Leeuwen 2009 + Katz+2023) error-deconvolved
velocity dispersions:

    σ²_∥  = ⟨(V∥_obs − V∥_pred)²⟩ − ⟨σ²_V∥,err⟩
    σ²_⊥  = ⟨V⊥_obs²⟩            − ⟨σ²_V⊥,err⟩
    σ²_RV = ⟨(v_r' − ⟨v_r'⟩)²⟩    − ⟨σ²_v_r,err⟩

All clamp to 0. σ_1D is dispatched between tangential-only (σ²_∥ + σ²_⊥)/2
and tangential+RV (σ²_∥ + σ²_⊥ + σ²_RV)/3 based on the RV-quality count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crosscat.cp_kinematics import (
    sigma_1d_combined,
    sigma_rv,
    sigma_tangential,
)
from crosscat.cp_method import fit_cp_lsq, vc_from_prior_members

from .conftest import make_mock_cluster


# ---------------------------------------------------------------------------
# sigma_tangential
# ---------------------------------------------------------------------------


def test_sigma_tangential_recovers_input(rng, capsys):
    """Hyades-like mock (σ_intr=0.3, σ_pm=0.05): σ_par, σ_perp ∈ (0.20, 0.45) km/s."""
    centre = (-44.77, 0.40, -16.24)  # Hyades-like, Galactic Cartesian pc
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
    # Use the parallax-inverse distance and the corresponding parallax_error.
    # Realistic DR3 Hyades σ_ϖ/ϖ ~ 0.1% — at 2% the distance-error term
    # (V∥_obs · σ_d/d)² ~ (23·0.02)² = 0.21 km/s² would swamp σ²_intr.
    df = df.copy()
    df["parallax_corrected"] = df["parallax"].to_numpy()
    df["parallax_error"] = 0.001 * df["parallax"].to_numpy()
    df["d_pc"] = 1000.0 / df["parallax"].to_numpy()

    vc_dict = fit_cp_lsq(df)
    out = sigma_tangential(df, vc_dict=vc_dict)

    with capsys.disabled():
        print(
            f"\n  test_sigma_tangential_recovers_input:\n"
            f"    sigma_par  = {out['sigma_par_kms']:.3f} km/s "
            f"(clamped={out['sigma_par_clamped']})\n"
            f"    sigma_perp = {out['sigma_perp_kms']:.3f} km/s "
            f"(clamped={out['sigma_perp_clamped']})\n"
            f"    N = {out['N']}"
        )

    assert 0.20 < out["sigma_par_kms"] < 0.45, out
    assert 0.20 < out["sigma_perp_kms"] < 0.45, out
    assert out["sigma_par_clamped"] is False, out
    assert out["sigma_perp_clamped"] is False, out
    assert out["N"] == 500


def test_sigma_tangential_clamps_when_measurement_limited(rng, capsys):
    """σ_intr=0, σ_pm=0.5 mas/yr (huge) → σ²_∥ or σ²_⊥ < 0 before clamp."""
    centre = (-44.77, 0.40, -16.24)
    vc_uvw = (-42.24, -19.00, -1.48)

    df = make_mock_cluster(
        rng,
        n_stars=500,
        centre_xyz_pc=centre,
        radius_pc=10.0,
        vc_uvw_kms=vc_uvw,
        sigma_intr_kms=0.0,        # perfectly co-moving
        pm_err_masyr=0.5,           # large PM error → dominates raw variance
    )
    df = df.copy()
    df["parallax_corrected"] = df["parallax"].to_numpy()
    df["parallax_error"] = 0.001 * df["parallax"].to_numpy()
    df["d_pc"] = 1000.0 / df["parallax"].to_numpy()

    vc_dict = fit_cp_lsq(df)
    out = sigma_tangential(df, vc_dict=vc_dict)

    with capsys.disabled():
        print(
            f"\n  test_sigma_tangential_clamps_when_measurement_limited:\n"
            f"    sigma_par  = {out['sigma_par_kms']:.3f} km/s "
            f"(clamped={out['sigma_par_clamped']})\n"
            f"    sigma_perp = {out['sigma_perp_kms']:.3f} km/s "
            f"(clamped={out['sigma_perp_clamped']})"
        )

    # At least one of the two clamps should fire when σ_intr=0 and PM err is huge.
    assert out["sigma_par_clamped"] or out["sigma_perp_clamped"], out
    # Both σ outputs are non-negative.
    assert out["sigma_par_kms"] >= 0.0
    assert out["sigma_perp_kms"] >= 0.0


def test_sigma_tangential_sigma_clip_drops_outliers(rng, capsys):
    """3σ-clip removes injected ΔV_par outliers → σ_∥ recovers σ_intr.

    Stage 0b (Hyades) diagnostic showed the (-5, +3) × (-0.8, +1.1) CP
    rectangle admits ~1.5 % kinematic outliers (tail-leakage + unresolved
    binaries) whose ΔV_par scatter inflates σ_∥ from 0.33 → 0.52 km/s.
    Iterative σ-clipping on ΔV_par before deconvolution recovers the
    Röser+2019 σ_1D = 0.30 km/s target. The clipping iterates on the
    sample mean / sample std until either no further stars are dropped
    or ``clip_max_iters`` is hit.
    """
    centre = (-44.77, 0.40, -16.24)
    vc_uvw = (-42.24, -19.00, -1.48)
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
    df = df.copy()
    df["parallax_corrected"] = df["parallax"].to_numpy()
    df["parallax_error"] = 0.001 * df["parallax"].to_numpy()
    df["d_pc"] = 1000.0 / df["parallax"].to_numpy()

    # Inject 10 ΔV_par outliers (2 %) by adding a large signed PM
    # perturbation. At d ≈ 47 pc, κd/1000 = 0.22 km·yr/(s·mas), so
    # ±15 mas/yr maps to ±3.3 km/s ΔV∥ — clearly outside any plausible
    # cluster member, mimicking the −4.8 / +2.8 km/s tail-leakage stars
    # we observed in real Hyades data.
    n = len(df)
    out_idx = rng.choice(n, size=10, replace=False)
    signs = rng.choice([-1.0, 1.0], size=10)
    df.loc[out_idx, "pmra"] = df.loc[out_idx, "pmra"].to_numpy() + 15.0 * signs

    vc_dict = fit_cp_lsq(df)

    out_raw = sigma_tangential(df, vc_dict=vc_dict)
    out_clip = sigma_tangential(df, vc_dict=vc_dict, clip_sigma=3.0)

    with capsys.disabled():
        print(
            f"\n  test_sigma_tangential_sigma_clip_drops_outliers:\n"
            f"    raw   : sigma_par = {out_raw['sigma_par_kms']:.3f}, "
            f"sigma_perp = {out_raw['sigma_perp_kms']:.3f}, N = {out_raw['N']}\n"
            f"    clip3 : sigma_par = {out_clip['sigma_par_kms']:.3f}, "
            f"sigma_perp = {out_clip['sigma_perp_kms']:.3f}, N = {out_clip['N']}, "
            f"n_clipped = {out_clip['n_clipped']}, n_iter = {out_clip['n_clip_iters']}"
        )

    # Raw σ_∥ inflated well above the intrinsic 0.3 km/s by the 10 outliers.
    assert out_raw["sigma_par_kms"] > 1.5 * sigma_intr, out_raw
    # Clipped σ_∥ is back in [0.20, 0.45] like the no-outlier test.
    assert 0.20 < out_clip["sigma_par_kms"] < 0.45, out_clip
    # At least ~10 stars were clipped (the injected ones).
    assert out_clip["n_clipped"] >= 8, out_clip
    # Clip iteration is bounded and ran at least once.
    assert 1 <= out_clip["n_clip_iters"] <= 5, out_clip
    # N reported is the kept count after clipping.
    assert out_clip["N"] == 500 - out_clip["n_clipped"]


def test_sigma_tangential_clip_disabled_matches_raw(rng):
    """``clip_sigma=None`` preserves the existing plan §5 raw-variance behaviour."""
    centre = (-44.77, 0.40, -16.24)
    vc_uvw = (-42.24, -19.00, -1.48)
    df = make_mock_cluster(
        rng,
        n_stars=500,
        centre_xyz_pc=centre,
        radius_pc=10.0,
        vc_uvw_kms=vc_uvw,
        sigma_intr_kms=0.3,
        pm_err_masyr=0.05,
    )
    df = df.copy()
    df["parallax_corrected"] = df["parallax"].to_numpy()
    df["parallax_error"] = 0.001 * df["parallax"].to_numpy()
    df["d_pc"] = 1000.0 / df["parallax"].to_numpy()

    vc_dict = fit_cp_lsq(df)
    out_a = sigma_tangential(df, vc_dict=vc_dict)
    out_b = sigma_tangential(df, vc_dict=vc_dict, clip_sigma=None)
    assert out_a["sigma_par_kms"] == out_b["sigma_par_kms"]
    assert out_a["sigma_perp_kms"] == out_b["sigma_perp_kms"]
    assert out_a["N"] == out_b["N"]
    # New keys present and indicate no clipping happened.
    assert out_b["n_clipped"] == 0
    assert out_b["n_clip_iters"] == 0


def test_sigma_tangential_uses_bj_distance_err(rng):
    """``d_err_col`` overrides parallax SNR for σ_d/d propagation.

    With a *huge* per-star distance error (50%), V∥/V⊥ error budget is
    inflated → measurement-limited regime where σ²_∥ < ⟨σ²_V∥,err⟩ and
    the clamp fires. Same cluster with d_err = 0 (parallax-inverse SNR
    of 2%) does *not* clamp.
    """
    centre = (-44.77, 0.40, -16.24)
    vc_uvw = (-42.24, -19.00, -1.48)

    df = make_mock_cluster(
        rng,
        n_stars=500,
        centre_xyz_pc=centre,
        radius_pc=10.0,
        vc_uvw_kms=vc_uvw,
        sigma_intr_kms=0.3,
        pm_err_masyr=0.05,
    )
    df = df.copy()
    df["parallax_corrected"] = df["parallax"].to_numpy()
    df["parallax_error"] = 0.001 * df["parallax"].to_numpy()  # realistic DR3
    df["d_pc"] = 1000.0 / df["parallax"].to_numpy()

    # Bailer-Jones-style brackets: huge per-star distance error.
    d = df["d_pc"].to_numpy()
    df["d_lo_pc"] = d * 0.5
    df["d_hi_pc"] = d * 1.5
    df["d_err"] = 0.5 * (df["d_hi_pc"].to_numpy() - df["d_lo_pc"].to_numpy())  # = 0.5·d

    vc_dict = fit_cp_lsq(df)

    out_bj = sigma_tangential(df, vc_dict=vc_dict, d_err_col="d_err")
    out_pi = sigma_tangential(df, vc_dict=vc_dict, d_err_col=None)

    # The BJ path's error budget is dominated by σ_d/d=0.5, so the
    # deconvolved σ² collapses to (near) zero and the clamp fires for
    # at least one of (par, perp). The parallax-inverse path keeps a
    # finite signal because σ_d/d = 0.02.
    assert out_bj["sigma_par_clamped"] or out_bj["sigma_perp_clamped"], out_bj
    # Sanity: with d_err=None we recover the input signal (not clamped).
    assert (out_pi["sigma_par_clamped"] is False) or (
        out_pi["sigma_perp_clamped"] is False
    ), out_pi


# ---------------------------------------------------------------------------
# sigma_rv
# ---------------------------------------------------------------------------


def test_sigma_rv_basic(rng, capsys):
    """30 stars: intrinsic σ=0.3 + per-star Gaussian noise σ_err=0.5.

    Observed v_r has variance ≈ σ_intr² + σ_err² = 0.09 + 0.25 = 0.34;
    error deconvolution against ⟨σ²_err⟩ = 0.25 recovers σ_rv ≈ √0.09 = 0.3
    within sampling variance at N=30.
    """
    n = 30
    rv_err_kms = 0.5
    intrinsic = rng.normal(0.0, 0.3, size=n)
    noise = rng.normal(0.0, rv_err_kms, size=n)
    v_r = 40.0 + intrinsic + noise
    df = pd.DataFrame(
        {
            "radial_velocity_corrected": v_r,
            "radial_velocity_error": np.full(n, rv_err_kms),
        }
    )
    out = sigma_rv(df)
    with capsys.disabled():
        print(
            f"\n  test_sigma_rv_basic: sigma_rv = {out['sigma_rv_kms']:.3f} km/s "
            f"(clamped={out['sigma_rv_clamped']}) N={out['N_rv']}"
        )
    # σ_rv ≈ 0.3 ± 0.1 km/s (allowing for sample variance at N=30)
    assert 0.15 < out["sigma_rv_kms"] < 0.5, out
    assert out["N_rv"] == 30
    assert out["sigma_rv_clamped"] is False, out


def test_sigma_rv_below_min_N():
    """5 rows < min_N=10 → NaN σ_rv, N_rv = 5."""
    df = pd.DataFrame(
        {
            "radial_velocity_corrected": [40.0, 40.5, 39.8, 40.1, 40.2],
            "radial_velocity_error": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    out = sigma_rv(df)
    assert np.isnan(out["sigma_rv_kms"])
    assert np.isnan(out["sigma_rv_err_kms"])
    assert out["N_rv"] == 5


# ---------------------------------------------------------------------------
# sigma_1d_combined
# ---------------------------------------------------------------------------


def test_sigma_1d_combined_tangential_only():
    """N_rv_qual < 10 → tangential-only, method = 'CP_tangential'."""
    out = sigma_1d_combined(
        sigma_par_kms=0.3, sigma_perp_kms=0.3, sigma_rv_kms=0.5, N_rv_qual=5
    )
    assert out["sigma_method"] == "CP_tangential"
    # σ_1D,tang = √((0.3² + 0.3²) / 2) = 0.3
    assert abs(out["sigma_1d_tangential_kms"] - 0.3) < 1e-9
    assert abs(out["sigma_1d_combined_kms"] - 0.3) < 1e-9


def test_sigma_1d_combined_tang_plus_rv():
    """N_rv_qual ≥ 10 → tang+rv, method = 'CP_tang+rv'."""
    out = sigma_1d_combined(
        sigma_par_kms=0.3, sigma_perp_kms=0.3, sigma_rv_kms=0.3, N_rv_qual=15
    )
    assert out["sigma_method"] == "CP_tang+rv"
    # σ_1D = √((0.3² + 0.3² + 0.3²) / 3) = 0.3
    assert abs(out["sigma_1d_tangential_kms"] - 0.3) < 1e-9
    assert abs(out["sigma_1d_combined_kms"] - 0.3) < 1e-9


def test_sigma_1d_combined_nan_propagates():
    """NaN σ_par → both outputs NaN, method = 'CP_failed'."""
    out = sigma_1d_combined(
        sigma_par_kms=float("nan"),
        sigma_perp_kms=0.3,
        sigma_rv_kms=0.3,
        N_rv_qual=15,
    )
    assert out["sigma_method"] == "CP_failed"
    assert np.isnan(out["sigma_1d_tangential_kms"])
    assert np.isnan(out["sigma_1d_combined_kms"])
