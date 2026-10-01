"""Shared pytest fixtures and helpers for the crosscat test suite."""

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


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy RNG so tests are deterministic."""
    return np.random.default_rng(20260512)


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

    Stars are uniformly distributed inside a sphere of radius ``radius_pc``
    centred at ``centre_xyz_pc`` in Galactic Cartesian coordinates. Each
    star's velocity is V_c (Galactic Cartesian) plus an isotropic 3D
    Gaussian intrinsic dispersion of ``sigma_intr_kms`` per axis. The
    function returns observable quantities (ra, dec, parallax, pmra,
    pmdec, radial_velocity) plus the per-star errors. Gaussian
    measurement noise of ``pm_err_masyr`` is added to pmra and pmdec.
    """
    n = int(n_stars)
    # Uniform-in-sphere positions: pick radii ∝ u**(1/3).
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


@pytest.fixture(autouse=True)
def disable_network(monkeypatch):
    """All tests are offline, including failure paths."""
    import socket
    def blocked(*args, **kwargs):
        raise RuntimeError("Network is disabled during release tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
