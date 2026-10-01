"""DR3 quality boolean masks.

Stateless cuts following Lindegren+2018 Eq C.1 (RUWE), Fabricius+2021
(binary indicators + excess-noise), Riello+2021 Eq C.2 (BP/RP flux
excess), and Lindegren+2021a §2.3 (chromaticity window for 5p sources).

All functions take ndarray-coercible inputs and return boolean numpy
arrays. NaN in any required column is treated as failing the cut.
"""

from __future__ import annotations

import numpy as np


def lindegren2018_ruwe_mask(ruwe, threshold: float = 1.4) -> np.ndarray:
    """RUWE strictly below ``threshold`` (Lindegren+2018 Eq C.1)."""
    r = np.asarray(ruwe, dtype=float)
    return np.isfinite(r) & (r < threshold)


def fabricius2021_binary_mask(
    ipd_frac_multi_peak, ipd_gof_harmonic_amplitude
) -> np.ndarray:
    """ipd_frac_multi_peak ≤ 2 AND ipd_gof_harmonic_amplitude < 0.1 (Fabricius+2021)."""
    f = np.asarray(ipd_frac_multi_peak, dtype=float)
    g = np.asarray(ipd_gof_harmonic_amplitude, dtype=float)
    return np.isfinite(f) & np.isfinite(g) & (f <= 2.0) & (g < 0.1)


def excess_noise_mask(astrometric_excess_noise, threshold: float = 1.0) -> np.ndarray:
    """astrometric_excess_noise < ``threshold`` (Fabricius+2021 §3.2)."""
    e = np.asarray(astrometric_excess_noise, dtype=float)
    return np.isfinite(e) & (e < threshold)


def riello2021_flux_excess_mask(bp_rp, phot_bp_rp_excess_factor) -> np.ndarray:
    """Babusiaux+2018 §2.1 / Gaia HRD 2018 Eq B.2 BP/RP flux-excess window.

    Window: ``1 + 0.015·C² ≤ phot_bp_rp_excess_factor ≤ 1.3 + 0.06·C²`` with
    ``C = bp_rp``. The function is named ``riello2021_*`` because Riello+2021 §9.4
    is the DR3-era reference for the same cleaning step (they additionally
    introduce a corrected ``C* = C − f(BP-RP)`` with Table 2 polynomial f, which
    we do not adopt here — we keep the Babusiaux+2018 window for direct
    comparability with Röser+2019 Hyades §2.1)."""
    c = np.asarray(bp_rp, dtype=float)
    excess = np.asarray(phot_bp_rp_excess_factor, dtype=float)
    c2 = c * c
    lo = 1.0 + 0.015 * c2
    hi = 1.3 + 0.06 * c2
    return np.isfinite(c) & np.isfinite(excess) & (excess >= lo) & (excess <= hi)


def parallax_snr_mask(parallax, parallax_error, snr_min: float = 10.0) -> np.ndarray:
    """Positive parallax AND parallax / parallax_error ≥ ``snr_min``."""
    p = np.asarray(parallax, dtype=float)
    s = np.asarray(parallax_error, dtype=float)
    finite = np.isfinite(p) & np.isfinite(s) & (s > 0.0) & (p > 0.0)
    safe_s = np.where(finite, s, 1.0)
    return finite & ((p / safe_s) >= snr_min)


def solution_type_mask(astrometric_params_solved, accept_6p: bool = False) -> np.ndarray:
    """5p solutions (==31); also 6p (==95) when ``accept_6p=True``."""
    p = np.asarray(astrometric_params_solved)
    is_5p = p == 31
    if accept_6p:
        return is_5p | (p == 95)
    return is_5p


def chromaticity_mask(
    nu_eff_used_in_astrometry,
    astrometric_params_solved,
    lo: float = 1.24,
    hi: float = 1.72,
) -> np.ndarray:
    """ν_eff ∈ [lo, hi] required only for 5p sources (Lindegren+2021a §2.3)."""
    nu = np.asarray(nu_eff_used_in_astrometry, dtype=float)
    p = np.asarray(astrometric_params_solved)
    is_5p = p == 31
    in_window = np.isfinite(nu) & (nu >= lo) & (nu <= hi)
    # Non-5p sources: cut does not apply — pseudocolour-fit (6p) has no
    # clamping, and any other solution code likewise passes through.
    return np.where(is_5p, in_window, True)


def combine_and(*masks) -> np.ndarray:
    """Element-wise AND across N boolean arrays."""
    if len(masks) == 0:
        raise ValueError("combine_and: at least one mask required")
    arrs = [np.asarray(m, dtype=bool) for m in masks]
    shape = arrs[0].shape
    for i, a in enumerate(arrs[1:], start=1):
        if a.shape != shape:
            raise ValueError(
                f"combine_and: mask {i} shape {a.shape} != mask 0 shape {shape}"
            )
    out = arrs[0].copy()
    for a in arrs[1:]:
        out &= a
    return out
