#!/usr/bin/env python3
"""Standalone offline reproduction of the Hyades and synthetic CP checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def verify_data():
    """Fail before calculation if a bundled input is missing or altered."""
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = ROOT / "data" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled input: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Bundled input checksum mismatch: {path}")


def run_hyades():
    import numpy as np
    import pandas as pd
    import hyades as h
    from crosscat import cp_method, cp_membership, cp_kinematics

    anchor = json.loads((h.DATA_DIR / "hyades_anchor.json").read_text())
    prior = pd.read_parquet(h.DATA_DIR / "hyades_prior_members.parquet")
    prior_vc = cp_method.vc_from_prior_members(prior)
    rows = []
    for mode, filename, relaxed in [
        ("main_strict", "hyades_cone_strict.parquet", False),
        ("validation_prior", "hyades_cone_validation.parquet", True),
        ("validation_purepm", "hyades_cone_validation.parquet", True),
    ]:
        raw = pd.read_parquet(h.DATA_DIR / filename)
        stars = h.clean_gaia(raw, accept_6p=relaxed, apply_chromaticity=not relaxed)
        if mode == "validation_prior":
            fit = prior_vc
            mask = cp_membership.rectangle_select(stars, vc_dict=fit)
        else:
            fit = h.fit_cp_iteratively(stars, h.seed_vc_from_inner_cone(stars, anchor))
            mask = fit["selected_mask"]
        core = h.select_core_18pc(stars, anchor, mask)
        sigma = h.measure_sigma(core, fit)
        tang = cp_kinematics.sigma_tangential(
            core, vc_dict=fit, clip_sigma=3.0, clip_max_iters=5
        )
        row = {
            "mode": mode,
            "N_input": len(raw),
            "N_clean": len(stars),
            "N_cp": int(mask.sum()),
            "N_core_preclip": len(core),
            "N_core_postclip": sigma.pop("N_core"),
            "n_clipped": tang["n_clipped"],
            "n_clip_iters": tang["n_clip_iters"],
            "sigma_par_clamped": tang["sigma_par_clamped"],
            "sigma_perp_clamped": tang["sigma_perp_clamped"],
            **{key: fit[key] for key in (
                "U_kms", "V_kms", "W_kms", "ra_cp_deg", "dec_cp_deg", "vc_mag_kms"
            )},
            "fit_iterations": fit.get("n_iter"),
            **sigma,
        }
        # This exactly preserves the historical error calculation. It is not a
        # calibrated uncertainty in the fitted/clipped CP estimator (see METHODS).
        rows.append(row)
        print(f"{mode}: N={len(core)} -> {row['N_core_postclip']}, "
              f"sigma_tang={row['sigma_1d_tang_kms']:.6f} km/s")
    return {
        "prior_N_matched": len(prior),
        "prior_N_rv": int(np.isfinite(prior.radial_velocity).sum()),
        "hyades": rows,
    }


def _jsonable(value):
    """Strict JSON: replace missing numerical values with null."""
    import math
    import numpy as np
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"),
                        help="New output directory (default: ./outputs); existing paths are refused")
    parser.add_argument("--skip-large-test", action="store_true",
                        help="Skip the 50,000-field-star synthetic membership test")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must not already exist; inputs and previous results are never overwritten")
    verify_data()
    # Agg and MathText avoid GUI and external TeX dependencies.
    import matplotlib
    matplotlib.use("Agg")
    import synthetic
    summary = run_hyades()
    results, extras, rng = synthetic.figure_inputs()
    if not args.skip_large_test:
        results.append(synthetic.run_test_0a_3(rng))
    if not all(row["passed"] for row in results):
        raise RuntimeError(f"Synthetic checks failed: {results}")
    summary["synthetic"] = results
    from importlib.metadata import version
    summary["environment"] = {
        "python": sys.version.split()[0],
        **{name: version(name) for name in (
            "numpy", "scipy", "pandas", "astropy", "matplotlib", "pyarrow", "gaiadr3-zeropoint"
        )},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    synthetic.make_perspective_figure(extras, args.output / "synthetic_perspective.pdf")
    (args.output / "results.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=False) + "\n"
    )
    import pandas as pd
    pd.DataFrame(summary["hyades"]).to_csv(args.output / "hyades_summary.csv", index=False)
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
