"""Shared style for the paper figures. Vector PDF only."""
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent  # repository root
EXP = ROOT / "experiments"
# Output directory for the PDFs: the repository's figures/ unless QISING_FIG_DIR is set.
FIG = Path(os.environ.get("QISING_FIG_DIR", ROOT / "figures"))

FULL_WIDTH = 5.5
HALF_WIDTH = 2.65
TITLE_SIZE = 12
TICK_SIZE = 10

COLORS = {
    "q_ising": "#0072B2", "ensemble": "#009E73", "dqn": "#D55E00", "dqn_all": "#E69F00",
    "degree_bin": "#CC79A7", "lir": "#56B4E9", "random": "#7F7F7F", "oracle": "#000000",
}
MARKERS = {"q_ising": "o", "ensemble": "s", "dqn": "^", "dqn_all": "D",
           "degree_bin": None, "lir": None, "random": None, "oracle": None}
LINESTYLES = {"q_ising": "-", "ensemble": "-", "dqn": "-", "dqn_all": "-",
              "degree_bin": "-.", "lir": "-.", "random": ":", "oracle": "--"}
LABELS = {"q_ising": "Q-Ising", "ensemble": "Ensemble Q-Ising", "dqn": "DQN",
          "dqn_all": "DQN-all", "degree_bin": "Degree-bin", "lir": "LIR", "random": "Random",
          "oracle": "Oracle"}
BLUE, VERMILLION, BAND = "#0072B2", "#D55E00", "#BFD9EA"
# Okabe-Ito sequence for communities, then lighter tints
OKABE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
COMM_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">"]

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#FFFFFF", "#0072B2"])
DIV_CMAP = LinearSegmentedColormap.from_list("div_bvr", ["#0072B2", "#FFFFFF", "#D55E00"])


def community_color(i):
    """Okabe-Ito first, then lighter tints of the same sequence."""
    base = mpl.colors.to_rgb(OKABE[i % len(OKABE)])
    tint = (i // len(OKABE)) * 0.4
    return tuple(c + (1 - c) * min(tint, 0.8) for c in base)


def apply_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE, "axes.labelsize": TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE, "legend.title_fontsize": TICK_SIZE,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.5, "lines.markersize": 5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.format": "pdf",
    })


def grid(ax):
    ax.grid(True, linestyle="--", color="0.8", linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name, subdir="main"):
    """Write figures/<subdir>/<name>.pdf (PDF only, no tight bbox)."""
    out = FIG / subdir if subdir else FIG
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    w, h = fig.get_size_inches()
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"wrote {shown}  {w:.2f} x {h:.2f} in")
    return path
