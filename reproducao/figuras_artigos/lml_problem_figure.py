"""Problem-setup figure of the LML article: meshes, one permeability realization and the
reference pressure statistics of the three boundary-value problems.

Usage (from reproducao/artigos_MOS_LML_COST, after `python run_base.py 1.1 1.3 1.4`):
    python ../figuras_artigos/lml_problem_figure.py OUT_DIR
"""
import os, sys, pickle
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs_lml"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.grid": False, "axes.titleweight": "bold",
                     "savefig.dpi": 300, "figure.dpi": 150})
RED, BLUE, GREY = "#c62828", "#1f6fe0", "#8a8a8a"  # palette of the original article figures
L = 15.0
PROB = [("Problem 1", "1.1"), ("Problem 2", "1.3"), ("Problem 3", "1.4")]
R = {k: pickle.load(open(f"_cache_P{k.replace('.', '_')}_NR1000_v14.pkl", "rb")) for _, k in PROB}


def edges(ax, bc):
    seg = {"esq": ([0, 0], [0, L]), "dir": ([L, L], [0, L]), "baixo": ([0, L], [0, 0]), "cima": ([0, L], [L, L])}
    for face, (kind, val) in bc.items():
        xs, ys = seg[face]
        if kind == "D":
            ax.plot(xs, ys, color=RED if val == 1.0 else BLUE, lw=4, solid_capstyle="butt", clip_on=False, zorder=5)
        else:
            ax.plot(xs, ys, color=GREY, lw=2.2, ls=(0, (2, 1.5)), clip_on=False, zorder=5)


fig = plt.figure(figsize=(7.4, 3.9))
gs = fig.add_gridspec(2, 6, width_ratios=[1, 0.05, 0.28, 1, 1, 1], wspace=0.1, hspace=0.25)
r0 = R["1.4"]
lk = np.log(r0["K_micro_first_realiz"]); lo, hi = lk.min(), lk.max()
a = fig.add_subplot(gs[0, 0]); im = a.imshow(lk, origin="lower", extent=[0, L, 0, L], cmap="viridis", vmin=lo, vmax=hi)
for g in np.linspace(0, L, 16):
    a.axvline(g, color="white", lw=0.25, alpha=0.6); a.axhline(g, color="white", lw=0.25, alpha=0.6)
for g in np.linspace(0, L, 6):
    a.axvline(g, color="white", lw=1.2); a.axhline(g, color="white", lw=1.2)
a.set_title("(a) ln κ, fine"); a.set_xticks([0, 5, 10, 15]); a.set_yticks([0, 5, 10, 15]); a.tick_params(labelsize=6.5)
a.set_ylabel("y (m)", fontsize=7)
b = fig.add_subplot(gs[1, 0]); b.imshow(np.log(r0["Kxx_all"][0]), origin="lower", extent=[0, L, 0, L], cmap="viridis", vmin=lo, vmax=hi)
b.set_title("(b) ln K$_{xx}$, coarse"); b.set_xticks([0, 5, 10, 15]); b.set_yticks([0, 5, 10, 15]); b.tick_params(labelsize=6.5)
b.set_xlabel("x (m)", fontsize=7); b.set_ylabel("y (m)", fontsize=7)
cax = fig.add_subplot(gs[:, 1]); fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6.5)
smax = max(R[k]["Pmic_std"].max() for _, k in PROB)
for j, (name, k) in enumerate(PROB):
    r = R[k]
    t = fig.add_subplot(gs[0, 3 + j]); m = t.imshow(r["Pmic_mean"], origin="lower", extent=[0, L, 0, L], cmap="viridis", vmin=0, vmax=1)
    u = fig.add_subplot(gs[1, 3 + j]); s = u.imshow(r["Pmic_std"], origin="lower", extent=[0, L, 0, L], cmap="viridis", vmin=0, vmax=smax)
    for x in (t, u):
        edges(x, r["bc"]); x.set_xticks([0, 5, 10, 15]); x.set_yticks([0, 5, 10, 15]); x.tick_params(labelsize=6.5); x.set_yticklabels([])
    t.set_xticklabels([]); t.set_title(f"({'cde'[j]}) {name}"); u.set_title(f"({'fgh'[j]}) {name}"); u.set_xlabel("x (m)", fontsize=7)
pos = fig.add_axes([0.915, 0.53, 0.012, 0.35]); cb = fig.colorbar(m, cax=pos); cb.ax.tick_params(labelsize=6.5); cb.set_label("mean pressure", fontsize=7)
pos = fig.add_axes([0.915, 0.11, 0.012, 0.35]); cb = fig.colorbar(s, cax=pos); cb.ax.tick_params(labelsize=6.5); cb.set_label("standard deviation", fontsize=7)
h = [plt.Line2D([], [], color=RED, lw=4), plt.Line2D([], [], color=BLUE, lw=4), plt.Line2D([], [], color=GREY, lw=2.2, ls=(0, (2, 1.5)))]
fig.legend(h, ["u = 1 (Dirichlet)", "u = 0 (Dirichlet)", "no flow"], loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.6, -0.06))
fig.subplots_adjust(left=0.07, right=0.9, top=0.93, bottom=0.1)
fig.savefig(os.path.join(OUT, "fig_problem.png"), bbox_inches="tight"); plt.close(fig)
print("ok")
