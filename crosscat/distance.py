"""Distance dispatch with explicit, offline Bailer-Jones catalogue input.

The shipped Hyades examples all use inverse corrected parallax (>=1 mas).
For distant user data pass a parquet cache explicitly; no network or fallback.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

class BailerJonesUnavailable(RuntimeError):
    """Required geometric distances are not supplied."""

def bailer_jones_lookup(source_ids, *, cache_path=None):
    ids = np.asarray(list(source_ids), dtype=np.int64)
    if cache_path is None or not Path(cache_path).is_file():
        raise BailerJonesUnavailable("Provide bj_cache_path for parallaxes below 1 mas; no network query is made.")
    cache = pd.read_parquet(cache_path).set_index("source_id")
    if not cache.index.is_unique or not np.isin(ids, cache.index).all():
        raise BailerJonesUnavailable("Distance cache must have unique IDs and cover every requested source.")
    return cache.reindex(ids)[["r_med_geo_pc", "r_lo_geo_pc", "r_hi_geo_pc"]]

def naive_distance(parallax_corrected_mas) -> np.ndarray:
    """Heliocentric distance in pc from corrected parallax: 1000/ϖ'.

    Vectorised. Returns NaN for ϖ' ≤ 0 or non-finite inputs.

    Parameters
    ----------
    parallax_corrected_mas : array-like
        Lindegren+2021b zero-point-corrected parallax in mas
        (the output of :func:`crosscat.parallax_zp.apply_zp`).

    Returns
    -------
    ndarray (float64)
        d_pc = 1000 / ϖ' where ϖ' > 0; NaN otherwise.
    """
    plx = np.asarray(parallax_corrected_mas, dtype=float)
    out = np.full(plx.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(plx) & (plx > 0.0)
    out[positive] = 1000.0 / plx[positive]
    return out


def choose_distance(
    stars_df: pd.DataFrame,
    *,
    threshold_mas: float = 1.0,
    parallax_col: str = "parallax_corrected",
    parallax_err_col: str = "parallax_error",
    source_id_col: str = "source_id",
    bj_cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Per-star distance: naive 1/ϖ' for ϖ' ≥ threshold; Bailer-Jones otherwise.

    Adds these columns to the input frame (without dropping existing ones):

    - ``d_pc``           heliocentric distance in pc
    - ``d_lo_pc``        lower bracket
    - ``d_hi_pc``        upper bracket
    - ``distance_method`` ∈ ``{"parallax_inverse", "bailer_jones_med", "invalid"}``

    Naive branch (ϖ' ≥ ``threshold_mas``)::

        d_pc    = 1000 / ϖ'
        d_lo_pc = d_pc · (1 - σ_ϖ / ϖ')
        d_hi_pc = d_pc · (1 + σ_ϖ / ϖ')

    Bailer-Jones branch (0 < ϖ' < ``threshold_mas``): the per-star
    posterior median ``r_med_geo`` plus its 14th- and 86th-percentile
    bracket (``r_lo_geo``, ``r_hi_geo``), looked up from an explicitly supplied offline catalogue via
    :func:`bailer_jones_lookup`. If a star is not in the Bailer-Jones+2021
    catalogue, NaN is recorded for all three distance columns and
    ``distance_method`` is set to ``"bailer_jones_med"`` anyway (so
    the dispatcher's intent is recorded; the NaN signals data lacking).

    Invalid branch (ϖ' ≤ 0 or non-finite): NaN distance, NaN bracket,
    ``distance_method = "invalid"``.

    Parameters
    ----------
    stars_df : pandas.DataFrame
        Per-star table. Must contain columns named by ``parallax_col``,
        ``parallax_err_col``, ``source_id_col``.
    threshold_mas : float
        Boundary between naive and Bailer-Jones paths (default 1.0).
    bj_cache_path : str, pathlib.Path, or None
        Local Bailer-Jones parquet cache. Required only for the distant-star branch; no default external path.

    Returns
    -------
    pandas.DataFrame
        Copy of ``stars_df`` with the four new columns appended.

    Raises
    ------
    BailerJonesUnavailable
        Propagated from :func:`bailer_jones_lookup` when stars in the
        Bailer-Jones branch are not covered by the supplied catalogue.
    """
    out = stars_df.copy()
    n = len(out)
    plx = np.asarray(out[parallax_col].to_numpy(), dtype=float)
    sigma = np.asarray(out[parallax_err_col].to_numpy(), dtype=float)
    src = np.asarray(out[source_id_col].to_numpy(), dtype=np.int64)

    d_pc = np.full(n, np.nan, dtype=np.float64)
    d_lo = np.full(n, np.nan, dtype=np.float64)
    d_hi = np.full(n, np.nan, dtype=np.float64)
    method = np.full(n, "invalid", dtype=object)

    finite_pos = np.isfinite(plx) & (plx > 0.0)
    naive_mask = finite_pos & (plx >= threshold_mas)
    bj_mask = finite_pos & (plx < threshold_mas)

    # Naive branch
    if np.any(naive_mask):
        idx = np.where(naive_mask)[0]
        d = 1000.0 / plx[idx]
        ratio = np.where(plx[idx] > 0.0, sigma[idx] / plx[idx], np.nan)
        d_pc[idx] = d
        d_lo[idx] = d * (1.0 - ratio)
        d_hi[idx] = d * (1.0 + ratio)
        method[idx] = "parallax_inverse"

    # Bailer-Jones branch
    if np.any(bj_mask):
        idx = np.where(bj_mask)[0]
        bj_ids = src[idx]
        bj = bailer_jones_lookup(bj_ids, cache_path=bj_cache_path)
        # bj is indexed by source_id, in input order. Match positions by index.
        bj_arr = bj.to_numpy()  # cols: r_med, r_lo, r_hi
        d_pc[idx] = bj_arr[:, 0]
        d_lo[idx] = bj_arr[:, 1]
        d_hi[idx] = bj_arr[:, 2]
        method[idx] = "bailer_jones_med"

    out["d_pc"] = d_pc
    out["d_lo_pc"] = d_lo
    out["d_hi_pc"] = d_hi
    out["distance_method"] = method
    return out
