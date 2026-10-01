# Implementation details and fidelity

This document describes the **executable code**, including historical choices
that should not be mistaken for universally recommended Gaia cuts. Numerical
behaviour is preserved, not revised to match abbreviated manuscript prose.

## Data flow and units

`reproduce.py` verifies the bundled SHA256 hashes before reading any data.
`hyades.clean_gaia` applies the masks below, subtracts the parallax zero-point,
rejects nonfinite corrected parallaxes and chooses distances. The pure-PM modes
fit a bulk velocity, select a CP rectangle and then a core sphere. The prior
mode derives the velocity directly from RV-bearing prior members, then applies
the same rectangle and sphere. All three measure clipped tangential dispersions.

Inputs use Gaia DR3 column names: `ra`, `dec`, `ecl_lat` in degrees;
`parallax`, `parallax_error`, `astrometric_excess_noise` in mas;
`pmra`, `pmdec` and their errors in mas yr⁻¹; RVs/errors in km s⁻¹;
`rv_template_teff` in K; `nu_eff_used_in_astrometry`, `pseudocolour` in μm⁻¹;
`pmra_pmdec_corr` is dimensionless. **pmra already contains cos(dec)**.
`ipd_frac_multi_peak` is a percentage, not a fraction in [0,1].

`U` points to the Galactic centre, `V` in the direction of Galactic rotation,
and `W` to the north Galactic pole. The tangential conversion constant is
`κ = 4.74047 km s⁻¹ / (mas yr⁻¹ kpc)`; equivalently `V = κ μ / parallax_mas`.

## Astrometric and photometric quality

| Columns | Literal criterion | Provenance / status |
|---|---|---|
| `ruwe` | `< 1.4` | Adopted RUWE screening; Lindegren et al. (2018), Fabricius et al. (2021); not a literal RUWE equation in the DR2 appendix |
| `ipd_frac_multi_peak`, `ipd_gof_harmonic_amplitude` | `<= 2`, `< 0.1` | Adopted DR3 double-source diagnostics; Fabricius et al. (2021) |
| `astrometric_excess_noise` | `< 1 mas` | Adopted excess-noise threshold; Fabricius et al. (2021) |
| `bp_rp = C`, `phot_bp_rp_excess_factor` | `1 + 0.015 C² <= excess <= 1.3 + 0.06 C²` | Gaia Collaboration/Babusiaux et al. (2018) window; **not** the corrected C* prescription of Riello et al. (2021), despite the historical function name |
| `parallax`, `parallax_error` | positive, finite; `parallax/parallax_error >= 10` | Adopted precision cut; equality is accepted |
| `astrometric_params_solved` | strict: `31`; validation: `31 or 95` | 5p versus 5p+6p selection |
| `nu_eff_used_in_astrometry` | strict: `[1.24,1.72]` for 5p; validation: disabled | Adopted heuristic motivated by chromaticity calibration, not a universally prescribed Lindegren et al. (2021a) cut |

Nonfinite inputs fail the relevant masks; the chromaticity mask only applies to
5p sources. The two IPD bounds are a single combined mask. Both real-data
validation modes retain the IPD cuts, RUWE, excess-noise, flux-excess and parallax
precision cuts.

### Parallax and distance

`parallax_zp.apply_zp` uses the separately installed published
`gaiadr3-zeropoint` package. Z5 uses G, effective wavenumber and ecliptic latitude;
Z6 uses G, pseudocolour and ecliptic latitude. It returns
`parallax_corrected = parallax - Z_mas`, with out-of-calibration inputs rejected
through NaNs. The wrapper's `z5`/`z6` functions expose Z in μas; `apply_zp` converts
back to mas. No calibration tables are manually copied or approximated.

The shipped Hyades runs all have corrected parallax >= 1 mas and use
`d_pc = 1000 / parallax_corrected`. Parallax inversion is an adopted approximation,
not exactly unbiased for noisy parallaxes. No Bailer-Jones lookup is needed for
these examples. The reusable `choose_distance` function supports lower positive
parallaxes only with an explicitly supplied offline table containing
`source_id, r_med_geo_pc, r_lo_geo_pc, r_hi_geo_pc`. The latter bounds are the
14th/86th percentiles. Missing coverage raises an error; there is no automatic
query or silent substitution. The Hyades cleaning driver deliberately rejects
this distant branch rather than pretending to be a general catalogue pipeline.

## Bulk motion and membership

- **Prior-member check:** 495 directly matched Gaia HRD 2018 IDs, 322 with finite
  DR3 RVs. `vc_from_prior_members` averages the individual Galactic velocities.
  It uses raw catalogue parallaxes and no new quality/RV gate on the prior
  velocity sample, matching the historical calculation. The source-ID
  continuity assumption is **not an official DR2→DR3 neighbourhood match**.
- **Pure-PM modes:** first fit stars within **1 degree** of the HR2024 position,
  corrected parallax in `[18,25] mas`, and finite proper motions. This gives 11
  seed stars (strict) or 19 (relaxed). It is not a median motion of an inner
  half-number-radius selection. The low-level initial CP direction uses a
  mean-PM pointing. It minimizes the squared perpendicular-PM residual divided
  by its propagated formal variance (Nelder–Mead then BFGS), then estimates the
  speed from parallel motion. This is an implemented adaptation of CP geometry,
  not a claim of a literal replication of a published full inference pipeline.
- Membership/fit iterations stop at a CP-direction change below `0.01 deg` and a
  speed change below `0.1 km/s`, with at most 10 iterations. Failure raises
  `CPNotConverged`; these are numerical stopping thresholds, not uncertainties.
- **Real-Hyades membership:** the inclusive rectangle
  `-5 <= ΔV_parallel <= 3`, `-0.8 <= V_perp <= 1.1 km/s`, followed by a Euclidean
  ICRS Cartesian distance **<=18 pc** from the HR2024 centre. HR2024 supplies only
  position/distance fields, **not its member list or proper-motion dispersion**.
- The separate `poisson_5d_neighbourhood` selector is tested on the synthetic
  cluster/field population. Defaults are `a=1.2`, `b=0.5 km/s`, spatial scale
  `r_lim=15 pc` and at least 3 neighbours (self excluded). A scaled 5D Euclidean
  **unit ball** is stricter than the intersection of a 3D sphere and 2D velocity
  ellipse; it is an adaptation of Röser & Schilbach (2019). It is not used to
  generate the three real-Hyades results. Its approximate Poisson-background
  ratios are not calibrated real-data membership probabilities.

**Historical parallax convention:** `fit_cp_lsq` reads `parallax` unchanged,
whereas the seed selection, CP membership and dispersion calculation use
`parallax_corrected`. The prior-member velocity also uses `parallax`. This mixed
convention is retained to reproduce the published numbers; the statement that
all velocity computations use corrected parallaxes is not literally true of
this implementation. Changing this would require a separate scientific rerun.

## Dispersions and errors

Rotate each star's proper motion and its 2×2 within-star covariance into the CP
frame. Subtract the bulk projection `V_parallel,pred = |Vc| sin(lambda)`;
`V_perp,pred = 0`. The per-star velocity error includes rotated PM covariance
and the approximate distance term `(V_obs sigma_d/d)²`. For the nearby examples
`sigma_d/d = parallax_error/parallax_corrected`. Cross-star errors, and
parallax–PM cross terms, are not included.

The real-data code subtracts mean formal variance from the **centred sample
variance with ddof=1**, not an uncentred mean square. Non-positive results are
clamped to zero and flagged; they represent unresolved dispersion. Before
variance calculation it clips parallel residuals about their mean at 3 sample
standard deviations, at most five times; both components use the same survivors.
The exact code retains `abs(residual - mean) < 3*std` (strict inequality).

`σ_tang = sqrt[(σ_parallel² + σ_perp²)/2]`.
Each component error is approximated as `σ / sqrt[2(N-1)]`; the tangential error
propagates these assuming independent component errors. These conditional errors
omit uncertainties from bulk fitting, membership, clipping, correlated Gaia
systematics and binaries; they are **not** a complete calibrated CP likelihood.

The synthetic perspective test instead reports residual sample standard
deviations **without measurement-error subtraction**, retaining its injected
0.05 mas yr⁻¹ noise. Tests 0a.1, 0a.2 and 0a.3 consume the same RNG sequentially
with seed `20260512`; running 0a.2 with a fresh seed gives a different mock.

## Radial-velocity diagnostic

The optional routines implement the following conjunction:

- `3900 <= rv_template_teff <= 8000 K`;
- finite `radial_velocity_error < 2 km/s`;
- `rv_nb_transits >= 10`, `rv_renormalised_gof <= 4`, `rv_chisq_pvalue > 0.01`.

The last two inequalities together are **stricter than** the logical complement
of Katz et al.'s variable-star criterion. The 2 km/s cap and minimum sample
size of ten are implementation choices, not universal literature prescriptions.
For `g = grvs_mag`, subtract `0.02755*g² - 0.55863*g + 2.81129 km/s` at `g>=11`;
subtract zero below 11. Finite RVs/errors are checked in the dispersion function.

**No separate SB1/SB2 classifier is used.** The diagnostic may retain binaries.
All reported Hyades results in this release use the two tangential components,
not the diagnostic RV dispersion. `sigma_1d_combined` is included as a reusable
API and tested; it adds an RV term only for a sufficiently large usable RV sample.

## Literature and data references

- van Leeuwen (2009), A&A 497, 209, DOI
  [10.1051/0004-6361/200811382](https://doi.org/10.1051/0004-6361/200811382):
  CP/projection geometry. Historical source comments contain section-number
  shorthand; the executable adaptation is specified above.
- Röser, Schilbach & Goldman (2019), A&A 621, L2,
  [arXiv:1811.03845](https://arxiv.org/abs/1811.03845): Hyades CP box and spatial selection.
- Röser & Schilbach (2019), A&A 627, A4, DOI
  [10.1051/0004-6361/201935502](https://doi.org/10.1051/0004-6361/201935502): Praesepe neighbourhood method.
- Gaia Collaboration, Babusiaux et al. (2018), A&A 616, A10,
  [VizieR J/A+A/616/A10](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/616/A10): prior members and historical flux-excess window.
- Hunt & Reffert (2024), A&A 686, A42,
  [VizieR J/A+A/686/A42](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/686/A42): cluster centre/distance anchor.
- Lindegren et al. (2018), A&A 616, A2; Lindegren et al. (2021a), A&A 649, A2;
  Fabricius et al. (2021), A&A 649, A5: astrometric quality/calibration context.
- Lindegren et al. (2021b), A&A 649, A4,
  [arXiv:2012.01742](https://arxiv.org/abs/2012.01742): parallax zero-point.
- Riello et al. (2021), A&A 649, A3: DR3-era photometric validation; the code
  retains the older flux-excess window, not its corrected C*.
- Bailer-Jones et al. (2021), AJ 161, 147,
  [arXiv:2012.05220](https://arxiv.org/abs/2012.05220): optional geometric distances.
- Katz et al. (2023), A&A 674, A5,
  [arXiv:2206.05902](https://arxiv.org/abs/2206.05902): RV quality and zero-point.

## Scope of reproduction

This release establishes executable reproducibility of the included CP
calculations and their numerical regression anchors. It does not validate the
physical interpretation of the gravitational anomaly, remeasure the 18-cluster
Risbud sample, refit HR2024 masses, or recompute PIT/CvM significances. Some
manuscript descriptions of initial members, selection, parallax treatment,
variance convention, and binary screening are more general than (or differ from)
the executed Hyades path. They require author review, not silent algorithm
changes in a packaging task.
