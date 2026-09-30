"""Figures 1-4 of the LML verification article.

Usage (from reproducao/artigos_MOS_LML_COST, after `python lml_paper.py 1.1 1.3 1.4`,
which writes master_lml_paper.json and lml_paper_arrays.pkl):
    python ../figuras_artigos/lml_article_figures.py OUT_DIR
"""
import os, sys
import json, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# palette of the original article figures
BLUE, ORANGE, AQUA, YEL, INK, MUTED = "#1f6fe0", "#e07b1b", "#128a3e", "#c62828", "#1b2a4a", "#8a8a8a"
GREEN, RED, NAVY, GREY = AQUA, YEL, INK, MUTED
plt.rcParams.update({"patch.force_edgecolor": True, "patch.edgecolor": "black", "patch.linewidth": 0.6, "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.35, "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8", "savefig.dpi": 300, "figure.dpi": 150})
OUT = (sys.argv[1] if len(sys.argv) > 1 else "figs_lml") + "/"
os.makedirs(OUT, exist_ok=True)
from matplotlib.ticker import FuncFormatter
PCT = FuncFormatter(lambda v, _: f"{v:g}")
J = json.load(open("master_lml_paper.json")); A = pickle.load(open("lml_paper_arrays.pkl", "rb"))
PROB = {"1.1": "Problem 1", "1.3": "Problem 2", "1.4": "Problem 3"}

def hm(ax, f, title, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(f, origin="lower", cmap=cmap, extent=[0, 15, 0, 15], vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.grid(False); ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 5, 10, 15])
    return im

# Figure 1: operator stages, Problem 3
a = A["1.4"]
fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.2))
panels = [(a["Pmac_mean"], "(a) coarse mean", 0, 1, "viridis"), (a["m_coarse"], "(b) projection", 0, 1, "viridis"),
          (a["sig"], "(c) residual scale", 0, None, "viridis"), (a["m_down"], "(d) reconstructed mean", 0, 1, "viridis")]
for x, (f, t, lo, hi, cm) in zip(ax, panels):
    n = f.shape[0]; im = x.imshow(f, origin="lower", cmap=cm, extent=[0, n, 0, n], vmin=lo, vmax=hi)
    x.set_title(t); x.grid(False); x.set_xticks(range(0, n+1, 1 if n <= 5 else 5)); x.set_yticks(range(0, n+1, 1 if n <= 5 else 5))
    x.set_xlabel("macro index i" if n <= 5 else "micro index i", fontsize=7); x.set_ylabel("macro index j" if n <= 5 else "micro index j", fontsize=7); x.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=x, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
fig.tight_layout(w_pad=0.8); fig.savefig(OUT+"fig1.png", bbox_inches="tight"); plt.close(fig)

# Figure 2: standard-deviation fields, three problems x four columns
fig, ax = plt.subplots(3, 4, figsize=(7.4, 5.6))
for i, p in enumerate(PROB):
    a = A[p]; v = a["s_ref"].max()
    cols = [(a["s_ref"], "reference (test half)"), (a["s_coarse"], "coarse only"), (a["s_raw"], "raw operator"), (a["s_down"], "rescaled operator")]
    for j, (f, t) in enumerate(cols):
        im = hm(ax[i, j], f, t if i == 0 else "", 0, v, "viridis")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[i, 0].set_ylabel(PROB[p])
    fig.colorbar(im, ax=ax[i, :].tolist(), fraction=0.02, pad=0.02)
fig.savefig(OUT+"fig2.png", bbox_inches="tight"); plt.close(fig)

# Figure 3: the tautology
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
x = np.arange(3); w = 0.2
for k, (key, lab, col) in enumerate([("coarse_only", "coarse only (no fine statistics)", GREY), ("raw_no_rescaling", "raw operator", RED),
                                      ("holdout", "calibrated operator, hold-out", GREEN), ("split_gap", "reference split gap", BLUE)]):
    for m, a_ in zip(("mean", "sd"), ax):
        vals = [100*J[p][key][m] for p in PROB]
        a_.bar(x+(k-1.5)*w, vals, w*0.92, color=col, label=lab)
for m, a_, t in zip(("mean", "sd"), ax, ("(a) mean field", "(b) standard-deviation field")):
    a_.set_xticks(x); a_.set_xticklabels(PROB.values()); a_.set_ylabel("relative L$^2$ error (%)"); a_.set_title(t); a_.set_yscale("log")
    a_.yaxis.set_major_formatter(PCT); a_.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(round(v, 6))[0] in "25" else ""))
h, l = ax[1].get_legend_handles_labels(); fig.legend(h, l, fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.07, 1, 1]); fig.savefig(OUT+"fig3.png", bbox_inches="tight"); plt.close(fig)

# Figure 4: calibration-size experiment
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for p, col, mk in zip(PROB, (BLUE, ORANGE, GREEN), ("o", "s", "^")):
    cs = J[p]["calibration_size"]; n = np.array(sorted(int(k) for k in cs))
    op = np.array([cs[str(k)]["op_sd"] for k in n]); nu = np.array([cs[str(k)]["null_sd"] for k in n])
    ax[0].plot(n, 100*op, color=col, marker=mk, ms=4, lw=1.3, label=f"{PROB[p]}: operator")
    ax[0].plot(n, 100*nu, color=col, ls=":", lw=1.2, label=f"{PROB[p]}: n fine solves alone")
    opm = np.array([cs[str(k)]["op_mean"] for k in n]); num = np.array([cs[str(k)]["null_mean"] for k in n])
    ax[1].plot(n, 100*opm, color=col, marker=mk, ms=4, lw=1.3); ax[1].plot(n, 100*num, color=col, ls=":", lw=1.2)
nn = np.logspace(1, np.log10(500), 50)
ax[0].plot(nn, 100/np.sqrt(2*nn), color=INK, lw=0.8, ls="--", label="1/√(2n)")
for a_, t in zip(ax, ("(a) standard-deviation field", "(b) mean field")):
    a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("fine realizations used for calibration, n"); a_.set_ylabel("hold-out error (%)"); a_.set_title(t)
    a_.xaxis.set_major_formatter(PCT); a_.yaxis.set_major_formatter(PCT)
    a_.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(round(v, 6))[0] in "25" else ""))
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=[0, 0.13, 1, 1]); fig.savefig(OUT+"fig4.png", bbox_inches="tight"); plt.close(fig)
print("ok")
