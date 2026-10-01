"""Reproduce the main Hyades CP result and two relaxed-selection checks offline."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from crosscat import cp_method, cp_membership, cp_kinematics, gaia_quality, parallax_zp, distance, rv_quality
DATA_DIR = Path(__file__).resolve().parent / "data"
SEED_INNER_CONE_DEG = 1.0
SEED_PLX_RANGE_MAS = (18.0, 25.0)
DV_PAR_BOX = (-5.0, 3.0)
DV_PERP_BOX = (-0.8, 1.1)
CORE_RADIUS_PC = 18.0
CLIP_SIGMA = 3.0
CLIP_MAX_ITERS = 5

def seed_vc_from_inner_cone(stars_df: pd.DataFrame, anchor: dict) -> dict:
    """Seed V_c: fit_cp_lsq on the inner 1° cone + Hyades parallax slice."""
    ra = stars_df["ra"].to_numpy(dtype=float)
    dec = stars_df["dec"].to_numpy(dtype=float)
    plx = stars_df["parallax_corrected"].to_numpy(dtype=float)

    cos_d = np.sin(np.deg2rad(dec)) * np.sin(np.deg2rad(anchor["dec_deg"])) + np.cos(
        np.deg2rad(dec)
    ) * np.cos(np.deg2rad(anchor["dec_deg"])) * np.cos(
        np.deg2rad(ra - anchor["ra_deg"])
    )
    sep_deg = np.rad2deg(np.arccos(np.clip(cos_d, -1.0, 1.0)))

    inner = (
        (sep_deg <= SEED_INNER_CONE_DEG)
        & (plx >= SEED_PLX_RANGE_MAS[0])
        & (plx <= SEED_PLX_RANGE_MAS[1])
        & np.isfinite(stars_df["pmra"].to_numpy(dtype=float))
        & np.isfinite(stars_df["pmdec"].to_numpy(dtype=float))
    )
    n_inner = int(inner.sum())
    if n_inner < 5:
        raise RuntimeError(
            f"seed_vc_from_inner_cone: only {n_inner} seed stars; cannot "
            "bootstrap the pure-PM CP fit."
        )
    seed = cp_method.fit_cp_lsq(stars_df.loc[inner].reset_index(drop=True))
    print(
        f"[cp-hyades] seed V_c (U,V,W) = ({seed['U_kms']:+.2f}, "
        f"{seed['V_kms']:+.2f}, {seed['W_kms']:+.2f}) km/s "
        f"(inner {SEED_INNER_CONE_DEG:.1f}° cone, N = {n_inner})"
    )
    return seed


def fit_cp_iteratively(stars_df: pd.DataFrame, seed_vc: dict) -> dict:
    """cp_iterate with the Röser+2019 Hyades rectangle as membership."""

    def membership_fn(df: pd.DataFrame, ra_cp: float, dec_cp: float, vc_mag: float):
        return cp_membership.rectangle_select(
            df,
            dV_par_box=DV_PAR_BOX,
            dV_perp_box=DV_PERP_BOX,
            vc_dict={
                "ra_cp_deg": ra_cp,
                "dec_cp_deg": dec_cp,
                "vc_mag_kms": vc_mag,
            },
            parallax_col="parallax_corrected",
        )

    return cp_method.cp_iterate(
        stars_df,
        vc_seed_dict=seed_vc,
        membership_fn=membership_fn,
        max_iter=10,
        tol_deg=0.01,
        tol_vc_kms=0.1,
    )


def select_core_18pc(
    stars_df: pd.DataFrame, anchor: dict, member_mask: np.ndarray
) -> pd.DataFrame:
    """Restrict CP-selected members to the 18 pc heliocentric core sphere."""
    cp_members = stars_df.loc[member_mask].reset_index(drop=True)

    def _xyz(ra_deg, dec_deg, d_pc):
        ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
        dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
        return (
            d_pc * np.cos(dec) * np.cos(ra),
            d_pc * np.cos(dec) * np.sin(ra),
            d_pc * np.sin(dec),
        )

    cx, cy, cz = _xyz(anchor["ra_deg"], anchor["dec_deg"], anchor["dist_pc"])
    sx, sy, sz = _xyz(
        cp_members["ra"].to_numpy(),
        cp_members["dec"].to_numpy(),
        cp_members["d_pc"].to_numpy(),
    )
    sep_pc = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2)
    return cp_members.loc[sep_pc <= CORE_RADIUS_PC].reset_index(drop=True)


def measure_sigma(core_members: pd.DataFrame, vc_dict: dict) -> dict:
    """σ_∥, σ_⊥ (deconvolved, clipped) → σ_1D,tang; σ_RV as diagnostic."""
    tang = cp_kinematics.sigma_tangential(
        core_members,
        vc_dict=vc_dict,
        parallax_col="parallax_corrected",
        d_pc_col="d_pc",
        d_err_col=None,
        clip_sigma=CLIP_SIGMA,
        clip_max_iters=CLIP_MAX_ITERS,
    )
    sigma_1d = float(
        np.sqrt(
            0.5 * (float(tang["sigma_par_kms"]) ** 2 + float(tang["sigma_perp_kms"]) ** 2)
        )
    )
    sigma_1d_err = float(
        0.5
        * np.sqrt(
            (tang["sigma_par_kms"] * tang["sigma_par_err_kms"]) ** 2
            + (tang["sigma_perp_kms"] * tang["sigma_perp_err_kms"]) ** 2
        )
        / max(sigma_1d, 1e-12)
    )

    try:
        rv_clean, n_rv_qual = rv_quality.apply_rv_quality(core_members)
        sig_rv = cp_kinematics.sigma_rv(rv_clean)
        sigma_rv = float(sig_rv["sigma_rv_kms"])
    except KeyError as exc:
        raise ValueError("The supplied Hyades table lacks a required RV diagnostic column") from exc

    return {
        "N_core": int(tang["N"]),
        "sigma_par_kms": float(tang["sigma_par_kms"]),
        "sigma_par_err_kms": float(tang["sigma_par_err_kms"]),
        "sigma_perp_kms": float(tang["sigma_perp_kms"]),
        "sigma_perp_err_kms": float(tang["sigma_perp_err_kms"]),
        "sigma_1d_tang_kms": sigma_1d,
        "sigma_1d_tang_err_kms": sigma_1d_err,
        "sigma_rv_kms": sigma_rv,
        "N_rv_qual": int(n_rv_qual),
    }



def clean_gaia(df_raw: pd.DataFrame, *, accept_6p=False, apply_chromaticity=True):
    """Apply the original full quality, ZP and nearby-distance steps to a table."""
    apply_binary_indicators = True
    n_raw = len(df_raw)

    # --- Quality stack (with toggles for matching Röser+2019-era cuts) ---
    quality_masks = [
        gaia_quality.lindegren2018_ruwe_mask(df_raw["ruwe"]),
        gaia_quality.excess_noise_mask(df_raw["astrometric_excess_noise"]),
        gaia_quality.riello2021_flux_excess_mask(
            df_raw["bp_rp"], df_raw["phot_bp_rp_excess_factor"]
        ),
        gaia_quality.parallax_snr_mask(
            df_raw["parallax"], df_raw["parallax_error"], snr_min=10.0
        ),
        gaia_quality.solution_type_mask(
            df_raw["astrometric_params_solved"], accept_6p=accept_6p
        ),
    ]
    if apply_binary_indicators:
        quality_masks.insert(
            1,
            gaia_quality.fabricius2021_binary_mask(
                df_raw["ipd_frac_multi_peak"],
                df_raw["ipd_gof_harmonic_amplitude"],
            ),
        )
    if apply_chromaticity:
        quality_masks.append(
            gaia_quality.chromaticity_mask(
                df_raw["nu_eff_used_in_astrometry"],
                df_raw["astrometric_params_solved"],
            )
        )
    mask = gaia_quality.combine_and(*quality_masks)
    df = df_raw.loc[mask].reset_index(drop=True)
    n_quality = len(df)
    cut_summary = (
        f"5p={'+6p' if accept_6p else 'only'}, "
        f"chrom={'on' if apply_chromaticity else 'off'}, "
        f"bin_ind={'on' if apply_binary_indicators else 'off'}"
    )
    print(f"[hyades] Gaia cone: {n_raw} stars  →  {n_quality} after quality cuts ({cut_summary})")

    # --- Parallax zero-point (Lindegren+2021b Z5 for 5p, Z6 for 6p) ---
    sin_b = np.sin(np.deg2rad(df["ecl_lat"].to_numpy(dtype=float)))
    df["sin_ecl_lat"] = sin_b
    # apply_zp expects ONE colour column; the value should be nu_eff for 5p
    # rows and pseudocolour for 6p rows (mixed when accept_6p=True).
    params_solved = df["astrometric_params_solved"].to_numpy()
    nu_eff = df["nu_eff_used_in_astrometry"].to_numpy(dtype=float)
    pseudocol = df["pseudocolour"].to_numpy(dtype=float)
    colour = np.where(params_solved == 95, pseudocol, nu_eff)
    df["parallax_corrected"] = parallax_zp.apply_zp(
        df["parallax"].to_numpy(dtype=float),
        df["phot_g_mean_mag"].to_numpy(dtype=float),
        colour,
        sin_b,
        params_solved,
    )
    # Out-of-range Z5 returns NaN; rows with NaN corrected parallax can
    # not be used downstream so drop them now.
    df = df.loc[np.isfinite(df["parallax_corrected"])].reset_index(drop=True)
    print(f"[hyades] {len(df)} stars after Lindegren+2021b ZP correction")

    # --- Distance (1/ϖ' for ϖ' ≥ 1 mas; BJ for the rest — at d < 200 pc
    # essentially every star is in the naive branch) ---
    if (df["parallax_corrected"] < 1.0).any():
        raise ValueError("Unexpected distant stars: this offline Hyades path requires parallax >= 1 mas.")
    df = distance.choose_distance(df, threshold_mas=1.0)
    return df
