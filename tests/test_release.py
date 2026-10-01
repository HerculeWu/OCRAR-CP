"""End-to-end regression, no-network execution and self-contained-copy checks."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import reproduce
from crosscat import distance

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_checksums():
    reproduce.verify_data()


def test_hyades_regression():
    result = reproduce.run_hyades()
    assert result['prior_N_matched'] == 495
    assert result['prior_N_rv'] == 322
    expected = [
        (232, 227, 0.32534578848229245, 0.010822398114642874),
        (394, 387, 0.3310613016776101, 0.008425707899634979),
        (393, 385, 0.3328956409302448, 0.00849658635507887),
    ]
    for row, (before, after, sig, err) in zip(result['hyades'], expected):
        assert row['N_core_preclip'] == before
        assert row['N_core_postclip'] == after
        assert row['sigma_1d_tang_kms'] == pytest.approx(sig, abs=2e-6)
        assert row['sigma_1d_tang_err_kms'] == pytest.approx(err, abs=2e-7)
    prior = result['hyades'][1]
    np.testing.assert_allclose(
        [prior[k] for k in ['U_kms', 'V_kms', 'W_kms']],
        [-42.168383273267466, -19.245474007367186, -1.2818160483318115],
        rtol=0, atol=1e-8,
    )
    pure = result['hyades'][2]
    np.testing.assert_allclose(
        [pure[k] for k in ['U_kms', 'V_kms', 'W_kms']],
        [-42.700049538864086, -19.0972314143266, -1.465893588548132],
        rtol=0, atol=2e-5,
    )


def test_distance_offline_dispatch(tmp_path):
    df = pd.DataFrame({'source_id': [1, 2], 'parallax_corrected': [20., 0.5],
                       'parallax_error': [0.1, 0.05]})
    with pytest.raises(distance.BailerJonesUnavailable):
        distance.choose_distance(df)
    cache = tmp_path / 'distances.parquet'
    pd.DataFrame({'source_id': [2], 'r_med_geo_pc': [1800.],
                  'r_lo_geo_pc': [1600.], 'r_hi_geo_pc': [2000.]}).to_parquet(cache)
    out = distance.choose_distance(df, bj_cache_path=cache)
    np.testing.assert_allclose(out.d_pc, [50., 1800.])
    assert out.distance_method.tolist() == ['parallax_inverse', 'bailer_jones_med']
    with pytest.raises(distance.BailerJonesUnavailable):
        distance.bailer_jones_lookup([3], cache_path=cache)


def test_standalone_copy_without_network_or_tex(tmp_path):
    isolated = tmp_path / 'standalone'
    shutil.copytree(ROOT, isolated, ignore=shutil.ignore_patterns(
        '.git', '.venv', '__pycache__', '.pytest_cache', 'outputs'
    ))
    # These files are the entire code/data tree visible to the child imports.
    blocker = tmp_path / 'blocker'
    blocker.mkdir()
    (blocker / 'sitecustomize.py').write_text(
        'import socket\n'
        'def blocked(*a, **k):\n'
        '    raise RuntimeError("Network is disabled during release validation")\n'
        'socket.socket.connect = blocked\n'
        'socket.socket.connect_ex = blocked\n'
        'socket.create_connection = blocked\n'
    )
    env = dict(os.environ, PYTHONPATH=str(blocker), MPLBACKEND='Agg',
               MPLCONFIGDIR=str(tmp_path / 'mpl'), PATH='',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (isolated / 'data').iterdir()}
    output = tmp_path / 'results'
    cmd = [sys.executable, str(isolated / 'reproduce.py'), '--output', str(output)]
    run = subprocess.run(cmd, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads((output / 'results.json').read_text())
    syn = result['synthetic']
    assert len(syn) == 3 and all(row['passed'] for row in syn)
    assert syn[0]['cp_separation_deg'] == pytest.approx(0.44515977, abs=2e-5)
    assert syn[1]['sigma_raw_kms'] == pytest.approx(4.225860552096609, abs=1e-7)
    assert syn[1]['sigma_cp_kms'] == pytest.approx(0.2912636316168946, abs=2e-6)
    assert syn[2]['cluster_recovered_pct'] == 100
    assert syn[2]['field_rejected_pct'] == 99.99
    assert syn[2]['n_selected'] == 505
    assert (output / 'synthetic_perspective.pdf').stat().st_size > 1000
    assert len(pd.read_csv(output / 'hyades_summary.csv')) == 3
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (isolated / 'data').iterdir()}
    assert before == after
    rerun = subprocess.run(cmd, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert rerun.returncode != 0 and 'must not already exist' in rerun.stderr
    # Missing/corrupt inputs are errors, never downloads or invented fallbacks.
    datafile = isolated / 'data/hyades_anchor.json'
    datafile.write_text('{}\n')
    corrupt = subprocess.run([*cmd[:-1], str(tmp_path / 'corrupt')], cwd=tmp_path,
                             env=env, text=True, capture_output=True)
    assert corrupt.returncode != 0 and 'checksum mismatch' in corrupt.stderr
    datafile.unlink()
    missing = subprocess.run([*cmd[:-1], str(tmp_path / 'missing')], cwd=tmp_path,
                             env=env, text=True, capture_output=True)
    assert missing.returncode != 0 and 'Missing bundled input' in missing.stderr
