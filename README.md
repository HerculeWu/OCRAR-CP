# OCRAR convergent-point code

Self-contained reproduction of the **Hyades convergent-point (CP) implementation
and validation checks** accompanying *A significant detection of a gravitational
anomaly in open clusters* (Wu et al.).

This is a focused release of the code referenced in the manuscript's
implementation appendix, **not the full catalogue/RAR/PIT analysis**. It contains
all Python modules and the small public-catalogue input subsets needed for the
commands below. After installing the dependencies, reproduction is **offline**:
no Gaia login, downloads, external disk, parent project, LaTeX installation or
unpublished local files are needed.

## Quick start

Python **3.11+** is required; the pinned environment was tested with Python 3.13.

```bash
git clone https://github.com/HerculeWu/OCRAR-CP.git
cd OCRAR-CP
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt
python reproduce.py --output outputs
python -m pytest -q
```

`requirements-lock.txt` pins the tested dependency versions. For other supported
Python versions, use `python -m pip install -r requirements.txt` to let pip resolve
compatible versions. The latter is a compatibility range, not a claim that every
combination has been tested.

The default runs **all three** synthetic checks, including the 50,000-field-star
membership experiment, and all three real-Hyades modes. The full run takes seconds
to tens of seconds on a typical laptop. Use `--skip-large-test` for a shorter
smoke test. Run from any working directory using an absolute path to
`reproduce.py`; bundled input paths are resolved relative to the code, not the
working directory. The output directory must be new: existing results are never
overwritten.

Outputs:

- `results.json`: fitted velocities, sample sizes, dispersions, synthetic recovery
  metrics and software versions;
- `hyades_summary.csv`: the three real-data selections in tabular form;
- `synthetic_perspective.pdf`: the synthetic before/after CP comparison, rendered
  with Matplotlib MathText (same numerical arrays as the manuscript, portable
  fonts rather than a pixel-identical LaTeX figure).

## Expected results

All dispersions below are in **km s⁻¹**. Counts distinguish the pre-clipping core
from the stars actually entering the final two tangential variances.

| Mode | Core before → after clipping | σ₁D,tang | Historical conditional error |
|---|---:|---:|---:|
| `main_strict` | 232 → 227 | 0.325346 | 0.010822 |
| `validation_prior` | 394 → 387 | 0.331061 | 0.008426 |
| `validation_purepm` | 393 → 385 | 0.332896 | 0.008497 |

The two validation modes admit 5- and 6-parameter astrometry and omit the
chromaticity cut; the main mode admits 5-parameter solutions only and applies it.
They are **not identical member samples**. The prior-member velocity is
recomputed from 322 RV-bearing matched sources, giving approximately
`(U,V,W) = (-42.168383, -19.245474, -1.281816) km/s`.

The seeded synthetic tests recover a CP direction within about 0.445° in the
narrow-angle test, reduce the perspective-broadened dispersion from 4.225861 to
0.291264 km/s for a 0.30 km/s injection, and select all 500 injected members while
rejecting 99.99% of 50,000 simulated field stars. These last rates apply only to
the specified synthetic population, not to real-catalogue completeness/purity.

Small optimizer/library/platform roundoff differences are expected. Regression
tests use numerical tolerances, not PDF byte equality. The historically requested
501 ± 30 real-Hyades core members were **not** recovered by these cuts; this
release does not silently turn that old sample-size failure into a successful
literal replication of Röser et al. (2019).

## Code summary

| File | Role |
|---|---|
| `reproduce.py` | Checks input hashes; runs the three Hyades modes and seeded synthetic sequence; writes fresh outputs |
| `hyades.py` | Gaia quality stack, parallax correction, inner-cone seed, iterative CP fit, 18 pc core, clipped dispersions |
| `synthetic.py` | Mock observables, bulk-motion/perspective recovery, 5D membership experiment, diagnostic figure |
| `crosscat/cp_method.py` | ICRS/Galactic transformations, parallel/perpendicular PM rotation, prior-member velocity, pure-PM least-squares fitting and iteration |
| `crosscat/cp_membership.py` | Hyades CP velocity rectangle and separate scaled-5D neighbourhood selector |
| `crosscat/cp_kinematics.py` | PM covariance and distance-error propagation, variance subtraction, clipping, tangential and optional RV combinations |
| `crosscat/gaia_quality.py` | Explicit astrometric/photometric masks, including finite-value handling |
| `crosscat/parallax_zp.py` | Wrapper around the published `gaiadr3-zeropoint` Z5/Z6 calibration |
| `crosscat/distance.py` | Corrected-parallax inversion; explicit offline Bailer-Jones table support for user-supplied distant sources |
| `crosscat/rv_quality.py` | Temperature/error/variability gates and magnitude-dependent RV correction |
| `crosscat/plotting.py`, `constants.py` | Portable plotting style and numerical unit constants |
| `data/` | Two Gaia cone snapshots, matched prior members and the HR2024 Hyades position/distance anchor |
| `tests/` | Geometry, selection, zero-point, uncertainty, seeded reproduction and offline/portability tests |

## Methods, provenance and limitations

Read **[METHODS.md](METHODS.md)** before reusing the calculations. It documents the
exact thresholds, input columns, units, references and differences between the
implemented procedures and some manuscript descriptions. In particular, the
real-Hyades driver uses a **velocity rectangle plus an 18 pc sphere**, not the
5D selector; the fit uses the original `parallax` column while membership and
dispersion calculations use `parallax_corrected`; and no separate SB1/SB2
classification is applied. These behaviours are retained for reproducibility,
not concealed or silently changed to force agreement with the prose.

The conditional error estimates do not propagate CP fitting, clipping,
membership reselection, binary motion or between-star astrometric correlations.
The large Gaia RV dispersion is a diagnostic, **not** a cleaned independent
Hyades dispersion; it is not combined into the quoted tangential result.

- [Data provenance and field units](docs/DATA.md)
- [Original source manifest](docs/source_manifest.json)
- [Input checksums](data/manifest.json)

The scientific routines were copied/extracted from the analysis sources;
packaging replaces local cache discovery with bundled inputs, removes network
fallbacks, and uses portable plotting fonts. Two misleading historical
docstrings (5D geometry and RV-gate logic) were corrected without changing their
numerical algorithms. The release adds regression and offline tests rather than
rerunning the manuscript's gravitational-model significance analysis.

## Attribution

Please cite the accompanying manuscript and the relevant method/data papers
listed in METHODS.md. Gaia data are from ESA's Gaia mission and DPAC; the
Hunt & Reffert and Gaia HRD catalogues are distributed through CDS/VizieR.
Dependency licences and the original data-provider terms remain applicable.
No additional software licence has been assigned to this release.
