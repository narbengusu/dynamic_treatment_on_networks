import os


def make_savefig(fig_dir):
    """Return a savefig(fig, name) function bound to fig_dir (PDF only)."""
    def savefig(fig, name):
        fig.savefig(os.path.join(fig_dir, name + ".pdf"), bbox_inches="tight")
    return savefig


def get_color(val):
    """Map an improvement percentage to a display colour."""
    if val < 0:
        return "#C2510B"
    elif val <= 5:
        return "#808080"
    else:
        return "#0577BE"
