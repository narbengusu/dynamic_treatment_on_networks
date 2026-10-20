"""Run every figure script (from a clean shell: python plotting/make_all.py)."""
import subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
scripts = ["make_e1_sweep", "make_improvement_hist", "make_e1_village_panel",
           "make_emvs_roc_auc", "make_posterior_v50", "make_seeding_sequence", "make_village_details",
           "make_ensemble_tree"]
ADJ = here.parent / "data" / "villages" / "adjacency_all.json"
for s in scripts:
    if s == "make_village_details" and not ADJ.exists():
        print(f"--- {s}: skipped (needs data/villages/adjacency_all.json, see README)")
        continue
    print(f"--- {s}")
    subprocess.run([sys.executable, str(here / f"{s}.py")], check=True, cwd=here)
