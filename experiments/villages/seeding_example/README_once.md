# Fig. 3, train-once protocol

`python experiments/villages/seeding_example/run_fig3_once.py` (from the repository root) trains one Q-Ising agent each for villages 59 and 71 (seed 0), saves it to `out/models/` (`v{V}_cql.d3`, `v{V}_state.pkl`), and runs test episodes with seeds 0..99. Output: `out/fig3_once_summary.json`. The displayed episode seeds the most distinct communities (ties to the lowest seed).

Reload: `from run_fig3_once import reload_policy; pol = reload_policy("59")` (needs `pip install -e .`). Figure: `python plotting/make_seeding_sequence.py`.

The network layouts in `out/*.npz` come from `run_fig3.py` (community-aware spring layout, seed 42).
