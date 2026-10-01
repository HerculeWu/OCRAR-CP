# Bundled data and provenance

Only public astronomical catalogue quantities needed by these examples are
included. No private manuscript, correspondence, credentials, general Gaia cache,
full HR2024 member table, or parent Git history is distributed. Input checksums
are in `data/manifest.json`; the runner refuses missing/modified inputs.

## Inventory

| File | Contents and provenance |
|---|---|
| `hyades_anchor.json` | One HR2024 `Melotte_25` cluster row, restricted to `Name`, `RAdeg`, `DEdeg`, `Plx`, `dist50` and renamed to the API keys. Source: Hunt & Reffert 2024, VizieR `J/A+A/686/A42` |
| `hyades_cone_strict.parquet` | 4,466 Gaia DR3 rows from the original cache `d3bd03867b228278.parquet`, copied byte-for-byte |
| `hyades_cone_validation.parquet` | 4,532 Gaia DR3 rows from the earlier cache `d55e06b4448e74d6.parquet`, copied byte-for-byte |
| `hyades_cone_*.json` | Original query-centre/radius/parallax-range/count sidecars |
| `hyades_prior_members.parquet` | 495 directly matched DR3 sources from the original 515 Gaia HRD 2018 Hyades identifiers, augmented with radial velocities; 322 finite RVs |
| `prior_rv_supplement.csv` | Gaia DR3 RV query for the two prior IDs absent from the cone, obtained on 2026-10-01 |

The cone centre is `(66.70864584, 16.08079780) deg`, radius **25 deg**, with
catalogue parallax **15–30 mas inclusive**. The anchor's distance is
47.19065976 pc and parallax 21.23204937 mas. The smaller cone snapshot uses the
later query's `parallax_over_error >= 10` prefilter; the historical larger
snapshot predates this query version. Both are preserved separately so the
main result and historical validations use their actual original inputs.
The Python cleaning layer repeats the precision cut.

The Gaia table is `gaiadr3.gaia_source`. The relevant ADQL selection is:

```sql
SELECT source_id, ra, dec, parallax, parallax_error,
       pmra, pmra_error, pmdec, pmdec_error, pmra_pmdec_corr,
       radial_velocity, radial_velocity_error, rv_template_teff,
       rv_renormalised_gof, rv_chisq_pvalue, rv_nb_transits, grvs_mag,
       phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp, ruwe,
       phot_bp_rp_excess_factor, ipd_frac_multi_peak,
       ipd_gof_harmonic_amplitude, astrometric_excess_noise,
       astrometric_params_solved, nu_eff_used_in_astrometry,
       pseudocolour, ecl_lat
FROM gaiadr3.gaia_source
WHERE CONTAINS(POINT('ICRS', ra, dec),
              CIRCLE('ICRS', 66.70864584, 16.0807978, 25.0)) = 1
  AND parallax BETWEEN 15.0 AND 30.0
  AND parallax_over_error >= 10
```

For the older snapshot omit the last clause. This query documents acquisition;
it is **not run** by the reproduction program. Exact reretrieval can depend on
archive representation/version, so the shipped snapshots and their hashes are
the reproducibility inputs. Original retrieval timestamps are not established by
the cache sidecars; local filesystem modification dates are not acquisition dates.

## Prior-member reconstruction

The historical driver requested `Cluster='Hyades'` from
`J/A+A/616/A10/tablea1a` (Gaia Collaboration/Babusiaux et al. 2018). It used those
515 DR2 identifiers directly in a DR3 source-ID query, yielding 495 rows cached
as `dr3_1efc5fbdda67780f.parquet`. This assumes ID continuity, rather than using
an official DR2–DR3 cross-match; 20 unmatched identifiers are not recovered.

The cached astrometry was retained. RVs for 493 of these IDs come from the
bundled validation cone; a fresh public query supplied the remaining two:

```sql
SELECT source_id, radial_velocity
FROM gaiadr3.gaia_source
WHERE source_id IN (3410466133406341504, 51902691204229888)
```

The first RV is null; the second is 34.40454 km/s. The response is preserved in
`prior_rv_supplement.csv`. It was requested from
`https://gea.esac.esa.int/tap-server/tap/sync` using public ADQL, no credentials.
The merge is one-to-one on integer `source_id`, yielding 322 finite RVs. Computing
mean Galactic velocities from this table agrees with the original stored prior
bulk vector to better than 1e-8 km/s per component. The release therefore
**recomputes** that mean rather than hard-coding the saved bulk velocity.

## Columns and units

Both cone parquets contain exactly the columns in the query above. Gaia names
and original floating-point precision are preserved. `source_id` is a signed
64-bit integer, never converted to floating point. See `../METHODS.md` for the
units and precise selection inequalities. Diagnostic columns such as
`parallax_corrected`, `sin_ecl_lat`, `d_pc` and `distance_method` are computed at
runtime without modifying the input files.

For user-supplied distant sources, `crosscat.distance.choose_distance` accepts an
explicit Bailer-Jones table. That optional API is not needed for the bundled
Hyades examples, which all use inverse corrected parallax; no such large distant
catalogue is shipped.

## Attribution and scope

Gaia: ESA Gaia mission and DPAC; DR3 archive at
<https://gea.esac.esa.int/archive/>. Catalogue access/provenance: CDS/VizieR,
<https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/686/A42> and
<https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/616/A10>.

Please retain the Gaia/DPAC acknowledgement and cite these data papers in work
using the inputs. Original provider terms continue to apply. The data are
reproduction subsets, not a replacement for the public catalogues, a new member
catalogue, or an assertion of complete/pure Hyades membership.
