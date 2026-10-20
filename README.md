# Q-Ising: Dynamic Treatment on Networks

Code for "Dynamic Treatment on Networks" by Bengusu Nar, Jiguang Li, Veronika Rockova and Panos Toulis ([arXiv:2605.06564](https://arxiv.org/abs/2605.06564)). Q-Ising combines a dynamic Ising model of network dynamics, fitted by EMVS or MCMC, with offline conservative Q-learning to allocate treatment sequentially on a network. An ensemble over posterior draws gives uncertainty estimates for the chosen decisions.

## Installation

```
python -m venv .venv && source .venv/bin/activate        # Python 3.11 or 3.12
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` explains the CPU build of torch and a macOS note for PyMC. Check the installation with `python scripts/smoke_test.py`.

## Example

Q-Ising on a toy network with three communities:

```python
import numpy as np, networkx as nx
from qising.environments.edge_bet_env import EdgeBetCommunitySISEnv
from qising.baselines import QIsingPolicy

# A small network with 3 communities of 10 nodes; each community has its own spread and churn rates
G = nx.stochastic_block_model([10, 10, 10], [[0.5, 0.05, 0.05], [0.05, 0.5, 0.05], [0.05, 0.05, 0.5]], seed=0)
adj = nx.to_numpy_array(G)
blocks = np.repeat([0, 1, 2], 10)
beta = np.repeat([0.05, 0.3, 0.1], 10)      # spread rate per node
delta = np.repeat([0.5, 0.6, 0.5], 10)      # churn rate per node

# 1. Collect an offline panel under a random behaviour policy
env = EdgeBetCommunitySISEnv(adj, beta, delta, blocks, T=100, seed=0, init_rate=0.0)
panel = env.collect_behavioral_episode(skip_prob=0)

# 2. Fit the dynamic Ising model (EMVS) and train the conservative Q-learning policy on its state
policy = QIsingPolicy(ising_augmentation=True, discount=0.8, alpha_cql=0.1, verbose=False).fit([panel], adj, blocks)

# 3. Deploy: at each period the policy picks a community, one of its nodes is treated
test = EdgeBetCommunitySISEnv(adj, beta, delta, blocks, T=25, seed=1, init_rate=0.0)
obs, done, total = test.reset(), False, 0.0
while not done:
    obs, reward, done, info = test.step(policy(obs, test))
    total += reward
print("mean adoption rate:", total / (25 * len(adj)))
```

## Reproduce the figures from the saved results

```
python plotting/make_all.py
```

## Run an experiment

| Experiment | Command |
|---|---|
| Covariate | `python experiments/covariate/run_cluster_sweep.py` |
| Misspecification | `python experiments/misspecification/run_qising_sweep.py` |
| Small network UQ | `python experiments/uq_small_network/confirm_instability_sis.py`, then `uq_calibration_sis.py` |
| Villages, all policies | notebook `experiments/villages/policy_eval.ipynb` |
| Villages, EMVS ROC | `python experiments/villages/emvs_roc/run_emvs_roc.py` |
| Village 50, MCMC | `python experiments/villages/ensemble_v50/run_v50.py mcmc` |
| Village 50, ensemble | `python experiments/villages/ensemble_v50/run_v50.py ensemble` |
| Seeding example | `python experiments/villages/seeding_example/run_fig3_once.py` |
| Villages, UQ calibration | `python experiments/villages/uq_calibration_villages.py` |

Each experiment folder has a README with its environment variables and output files.

## Repository layout

```
qising/                  package: ising/ (group Ising model, EMVS, MCMC), baselines/ (policies, ensemble),
                         environments/, utils/
data/villages/           place adjacency_all.json here (not included, see Data)
experiments/
  villages/              notebooks, UQ calibration, and re-runs: emvs_roc/, ensemble_v50/, seeding_example/
  covariate/             covariate experiment
  misspecification/      misspecification experiment
  uq_small_network/      uncertainty calibration in a small network
plotting/                figure scripts (make_all.py runs all)
figures/                 the figure PDFs used in the paper
scripts/smoke_test.py    quick check of the installation
```

Numbers quoted in the paper (tables, correlations) are recomputed from the saved results with `python scripts/paper_numbers.py`.

## Data

The village networks are not redistributed here. They come from the microfinance data collected in Karnataka, India by Banerjee, Chandrasekhar, Duflo and Jackson (2013), "The Diffusion of Microfinance", *Science* 341(6144):1236498, and can be obtained from the paper's archive in the [J-PAL Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl%3A1902.1%2F21538). The village experiments and the village figures read `data/villages/adjacency_all.json`, a JSON object that maps the village number to its symmetric 0/1 adjacency matrix (list of lists); create this file from the archive. Everything else (the covariate and misspecification experiments, the toy example, the saved results) runs without it.

## Citation

```
@misc{nar2026dynamic,
  title         = {Dynamic Treatment on Networks},
  author        = {Nar, Bengusu and Li, Jiguang and Ro{\v c}kov{\'a}, Veronika and Toulis, Panos},
  year          = {2026},
  eprint        = {2605.06564},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.06564}
}
```

Repository: https://github.com/narbengusu/dynamic_treatment_on_networks. Released under the MIT License (see `LICENSE`).
