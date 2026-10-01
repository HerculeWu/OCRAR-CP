"""Lindegren+2021b Z5/Z6 parallax zero-point correction.

For Gaia EDR3/DR3 5-parameter (5p) astrometric solutions, the published
parallax has a systematic bias depending on G magnitude, effective
wavenumber ``ν_eff`` (colour), and ecliptic latitude β. Lindegren+2021b
(arXiv:2012.01742) defines Z5(G, ν_eff, sin β) for 5p sources and the
analogous Z6(G, pseudocolour, sin β) for 6p sources where the colour is
the astrometric estimate ``pseudocolour``.

Corrected parallax (Lindegren+2021b Eq. for ϖ_corrected)::

    ϖ' = ϖ − Z(G, colour, sin β) / 1000

with Z in μas and ϖ in mas.

This module wraps the official ``gaiadr3-zeropoint`` package (Ramos &
Brown, Lindegren's Matlab code → Python; distributed with the paper).
The wrapper:

- Returns Z5 / Z6 in μas (the official ``zpt.get_zpt`` returns mas).
- Accepts ``sin_ecl_lat`` directly (the official API takes ecliptic
  latitude in degrees); we convert internally via ``arcsin``.
- Dispatches by ``astrometric_params_solved`` (31 → 5p via Z5,
  95 → 6p via Z6; any other value — including 3 for 2-parameter
  solutions — passes the input parallax through unchanged).
- Returns NaN for inputs outside the Lindegren+2021b §8.1 calibrated
  range (6 ≤ G ≤ 21; 1.1 ≤ ν_eff ≤ 1.9 μm⁻¹ for 5p; 1.24 ≤ pc ≤ 1.72
  μm⁻¹ for 6p) — no extrapolation.
"""

from __future__ import annotations

import numpy as np

from zero_point import zpt


# ---------------------------------------------------------------------------
# Lazy table-load — initialise the global coefficient tables on first use.
# ---------------------------------------------------------------------------

_TABLES_LOADED = False


def _ensure_tables() -> None:
    """Load Z5/Z6 coefficient tables once; cache the loaded state."""
    global _TABLES_LOADED
    if not _TABLES_LOADED:
        zpt.load_tables()
        _TABLES_LOADED = True


# ---------------------------------------------------------------------------
# Internal: invoke zpt.get_zpt with array-shaped inputs (numpy-2 compatible).
# ---------------------------------------------------------------------------


def _zpt_array(
    g: np.ndarray,
    nu: np.ndarray,
    pc: np.ndarray,
    sin_b: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    """Call ``zpt.get_zpt`` with array inputs and return Z in **μas**.

    The official ``zpt.get_zpt`` scalar path uses ``np.can_cast`` on
    Python scalars, which numpy 2.x disallows. We always pass arrays.

    Conversion: ``zpt.get_zpt`` returns mas; multiply by 1000 → μas.
    Ecliptic latitude is provided to the official API in degrees, so
    we convert ``sin_ecl_lat`` → ``arcsin`` → degrees here.
    """
    _ensure_tables()
    ecl_lat_deg = np.degrees(np.arcsin(np.clip(sin_b, -1.0, 1.0)))
    # _warnings=False makes the official package return NaN for out-of-range
    # inputs (G ∉ [6, 21]; ν_eff ∉ [1.1, 1.9] for 5p; pc ∉ [1.24, 1.72] for 6p)
    # rather than warning and silently extrapolating.
    z_mas = zpt.get_zpt(g, nu, pc, ecl_lat_deg, params, _warnings=False)
    return np.asarray(z_mas, dtype=float) * 1000.0


# ---------------------------------------------------------------------------
# Public API: z5, z6
# ---------------------------------------------------------------------------


def z5(phot_g_mean_mag, nu_eff_used_in_astrometry, sin_ecl_lat) -> np.ndarray:
    """Z5 zero-point bias in μas for 5-parameter astrometric solutions.

    Inputs are scalar or array-like (broadcast together).

    Returns: bias in microarcseconds (μas), as float64 ndarray shaped to
    the broadcast inputs (i.e. ``np.atleast_1d`` shape). Callers wanting a
    Python scalar can use ``float(z5(...).item())`` or ``z5(...)[0]``.

    Out-of-range inputs return NaN per Lindegren+2021b §8.1 (no
    extrapolation outside 6 ≤ G ≤ 21 and 1.1 ≤ ν_eff ≤ 1.9 μm⁻¹).
    """
    g = np.atleast_1d(np.asarray(phot_g_mean_mag, dtype=float))
    nu = np.atleast_1d(np.asarray(nu_eff_used_in_astrometry, dtype=float))
    sin_b = np.atleast_1d(np.asarray(sin_ecl_lat, dtype=float))
    g, nu, sin_b = np.broadcast_arrays(g, nu, sin_b)
    pc = np.full_like(g, np.nan)
    params = np.full_like(g, 31, dtype=int)

    return _zpt_array(g, nu, pc, sin_b, params)


def z6(phot_g_mean_mag, pseudocolour, sin_ecl_lat) -> np.ndarray:
    """Z6 zero-point bias in μas for 6-parameter astrometric solutions.

    Same semantics as :func:`z5` but uses ``pseudocolour`` (the
    astrometric estimate of ν_eff). Returns a float64 ndarray shaped to
    the broadcast inputs (no scalar collapse).

    Out-of-range inputs return NaN per Lindegren+2021b §8.1 (no
    extrapolation outside 6 ≤ G ≤ 21 and 1.24 ≤ pc ≤ 1.72 μm⁻¹).
    """
    g = np.atleast_1d(np.asarray(phot_g_mean_mag, dtype=float))
    pc = np.atleast_1d(np.asarray(pseudocolour, dtype=float))
    sin_b = np.atleast_1d(np.asarray(sin_ecl_lat, dtype=float))
    g, pc, sin_b = np.broadcast_arrays(g, pc, sin_b)
    nu = np.full_like(g, np.nan)
    params = np.full_like(g, 95, dtype=int)

    return _zpt_array(g, nu, pc, sin_b, params)


# ---------------------------------------------------------------------------
# Public API: apply_zp — dispatch by astrometric_params_solved
# ---------------------------------------------------------------------------


def apply_zp(
    parallax_mas,
    phot_g_mean_mag,
    colour,
    sin_ecl_lat,
    astrometric_params_solved,
) -> np.ndarray:
    """Apply Z5 or Z6 correction to parallax based on solution type.

    Inputs are scalar or array-like (broadcast together).

    Parameters
    ----------
    parallax_mas : array-like
        Catalogue parallax in mas (Gaia ``parallax`` column).
    phot_g_mean_mag : array-like
        Gaia G magnitude.
    colour : array-like
        For 5p sources, ``nu_eff_used_in_astrometry`` (μm⁻¹).
        For 6p sources, ``pseudocolour`` (μm⁻¹).
        The caller is responsible for passing the right value per row.
    sin_ecl_lat : array-like
        sin of ecliptic latitude β.
    astrometric_params_solved : array-like
        Dispatch code per row:

        - ``31`` → 5-parameter solution: apply Z5 correction, with
          ``colour`` interpreted as ``nu_eff_used_in_astrometry``.
        - ``95`` → 6-parameter solution: apply Z6 correction, with
          ``colour`` interpreted as ``pseudocolour``.
        - **Any other value** (including ``3`` for 2-parameter solutions,
          legacy ``5`` / ``7`` codes, pipeline bugs, or ``NaN``) → input
          parallax passes through unchanged.

    Returns
    -------
    ndarray (float64)
        Corrected parallax in mas: ``ϖ' = ϖ − Z(G, colour, sin β) / 1000``
        for rows with ``astrometric_params_solved ∈ {31, 95}``. Rows with
        any other dispatch code keep their input parallax unchanged
        (NaN inputs stay NaN). Rows where Z evaluates to NaN
        (out-of-range G or colour for the relevant 5p/6p calibration)
        return NaN.
    """
    plx = np.atleast_1d(np.asarray(parallax_mas, dtype=float))
    g = np.atleast_1d(np.asarray(phot_g_mean_mag, dtype=float))
    col = np.atleast_1d(np.asarray(colour, dtype=float))
    sin_b = np.atleast_1d(np.asarray(sin_ecl_lat, dtype=float))
    params = np.atleast_1d(np.asarray(astrometric_params_solved))
    plx, g, col, sin_b, params = np.broadcast_arrays(plx, g, col, sin_b, params)
    # Make a writable, contiguous output (broadcast can return read-only views).
    out = np.array(plx, dtype=np.float64, copy=True)

    is_5p = params == 31
    is_6p = params == 95

    # Build masked inputs for zpt.get_zpt: it requires `astrometric_params_solved
    # ∈ {31, 95}` everywhere and uses the colour column matching the type. We
    # therefore compute Z separately for the 5p and 6p subsets.

    if np.any(is_5p):
        idx = np.where(is_5p)[0]
        z_uas = _zpt_array(
            g[idx],
            col[idx],
            np.full(idx.size, np.nan),
            sin_b[idx],
            np.full(idx.size, 31, dtype=int),
        )
        out[idx] = plx[idx] - z_uas / 1000.0

    if np.any(is_6p):
        idx = np.where(is_6p)[0]
        z_uas = _zpt_array(
            g[idx],
            np.full(idx.size, np.nan),
            col[idx],
            sin_b[idx],
            np.full(idx.size, 95, dtype=int),
        )
        out[idx] = plx[idx] - z_uas / 1000.0

    # Rows with params ∉ {31, 95} pass parallax through unchanged — this
    # covers the documented 2p case (params == 3) but also any other value
    # (legacy 5 / 7 codes, NaN, pipeline bugs). Already handled by the
    # np.array(plx, ...) initialiser above; we keep the safe default rather
    # than raising on unknown codes so that mixed-DR / partial catalogues
    # still flow through this function. Callers that need strict dispatch
    # should validate `astrometric_params_solved` upstream.
    return out
