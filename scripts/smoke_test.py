"""Smoke test: import every module of qising, load the village data, build one village,
fit the EMVS model on a short trajectory and run plotting/make_all.py into a scratch directory.

Run from the repository root:  python scripts/smoke_test.py
Nothing is written inside the repository (figures go to a temporary directory).
"""
import importlib
import json
import os
import pkgutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# 1. import every module of the package
import qising

n_mod = 0
for m in pkgutil.walk_packages(qising.__path__, prefix="qising."):
    importlib.import_module(m.name)
    n_mod += 1
print(f"[1] imported {n_mod} qising modules")

ADJ_FILE = ROOT / "data" / "villages" / "adjacency_all.json"
HAVE_DATA = ADJ_FILE.exists()
if not HAVE_DATA:
    print("[2-4] skipped: data/villages/adjacency_all.json is not present (see README, Data)")
else:
    # 2. load the village adjacency matrices
    with open(ADJ_FILE) as f:
        adj_all = json.load(f)
    print(f"[2] loaded {len(adj_all)} village adjacency matrices")

    # 3. build one village (same rates as the experiments) and fit EMVS on a short trajectory
    from qising.environments.edge_bet_env import EdgeBetCommunitySISEnv
    from qising.ising.group_ising import fit_group_ising_emvs
    from qising.utils import build_village_envdata

    BETA_LIST = [0.01, 0.5, 0.05, 0.07, 0.06, 0.02, 0.01, 0.4, 0.1, 0.3]
    DELTA_LIST = [0.5, 0.9, 0.9, 0.6, 0.5, 0.50, 0.7, 0.6, 0.5, 0.8]
    vid = "50"
    A = np.array(adj_all[vid], dtype=float)
    np.fill_diagonal(A, 0)
    vd = build_village_envdata(A, BETA_LIST, DELTA_LIST, comm_method="edge_betweenness", min_comm_size=10)
    print(f"[3] built village {vid}: N={vd['N']}, communities={vd['n_bins']}")

    T_SHORT = 100  # short trajectory
    env = EdgeBetCommunitySISEnv(vd["adj"], vd["beta_vec"], vd["delta_vec"], vd["block_labels"],
                                 T=T_SHORT, seed=42, init_rate=0.0)
    traj = env.collect_behavioral_episode(skip_prob=0, verbose=False)
    X = traj["adopted"][:-1].astype(float)[np.newaxis]
    Y = traj["adopted"][1:].astype(float)[np.newaxis]
    TR = traj["seeded_nodes"][:-1][np.newaxis]
    res = fit_group_ising_emvs(X, Y, TR, vd["adj"], vd["block_labels"],
                               v0=0.01, v1=10.0, c=1.0, sigma_sq=10.0, verbose=False)
    assert "group_params" in res and len(res["group_params"]) == vd["n_bins"]
    print(f"[4] EMVS fitted on a trajectory of length {T_SHORT}: {len(res['group_params'])} community parameter sets")


# 5. run all figure scripts into a scratch directory (the repository's figures/ is untouched)
with tempfile.TemporaryDirectory() as tmp:
    env_vars = dict(os.environ, QISING_FIG_DIR=tmp, MPLBACKEND="Agg")
    subprocess.run([sys.executable, str(ROOT / "plotting" / "make_all.py")], check=True, cwd=ROOT, env=env_vars)
    pdfs = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*.pdf"))
print(f"[5] plotting/make_all.py wrote {len(pdfs)} PDFs to a scratch directory")
assert len(pdfs) >= (10 if HAVE_DATA else 9), pdfs
print("smoke test passed")
