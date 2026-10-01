# Release validation — 2026-10-01

- Created a new isolated Python 3.13.13 virtual environment and installed only
  `requirements.txt`; pinned the resulting environment in `requirements-lock.txt`.
- All **46 tests** passed, including the 42 retained component tests and four
  release/integration tests. Python network connections are blocked during tests.
- End-to-end integration copies only this release into a temporary directory,
  runs it from a different working directory with an empty executable PATH
  (no TeX), disables network access in the child process, and confirms that no
  bundled input hash changes. Missing/corrupt inputs and existing output paths
  are explicit failures.
- Recomputed the strict Hyades result and both relaxed-selection validations;
  counts and dispersions agree with the original analysis. Reconstructed the
  prior bulk vector from 322 RV-bearing sources to <1e-8 km/s per component.
- Replayed the entire original synthetic RNG sequence, including the
  500-cluster/50,000-field-star membership test. Regression values are saved in
  `validated_results.json`; that file is not read by the calculations.
- Checked that the numerical ASTs of all seven copied scientific modules and
  the synthetic numerical functions are unchanged (documentation excluded).
- Inspected the portable synthetic figure. Only the text-rendering backend and
  fonts differ from the original manuscript figure; no active manuscript figure
  is overwritten by this release.

This is reproducibility validation, not a new assessment of dynamical equilibrium,
binary contamination, catalogue completeness, the CP uncertainty calibration or
the gravitational-model significances. Known prose/implementation differences
are explicit in METHODS.md. The GitHub workflow provides an additional Linux
check; local tests alone do not establish that remote CI has completed.
