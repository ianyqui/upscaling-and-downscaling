import json, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YEL, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "axes.edgecolor": "#6b6a66", "axes.linewidth": 0.6, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.5, "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": INK,
                     "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150})
OUT = "../build/lml/"
J = json.load(open("master_lml_paper.json")); A = pickle.load(open("lml_paper_arrays.pkl", "rb"))
PROB = {"1.1": "Problem 1", "1.3": "Problem 2", "1.4": "Problem 3"}

def hm(ax, f, title, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(f, origin="lower", cmap=cmap, extent=[0, 15, 0, 15], vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.grid(False); ax.set_xticks([0, 5, 10, 15]); ax.set_yticks([0, 5, 10, 15])
    return im

# Figure 1: operator stages, Problem 3
a = A["1.4"]
fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.0))
panels = [(a["Pmac_mean"], "(a) coarse mean", 0, 1, "viridis"), (a["m_coarse"], "(b) projection", 0, 1, "viridis"),
          (a["sig"], "(c) residual scale", 0, None, "Blues"), (a["m_down"], "(d) reconstructed mean", 0, 1, "viridis")]
for x, (f, t, lo, hi, cm) in zip(ax, panels):
    im = x.imshow(f, origin="lower", cmap=cm, extent=[0, 15, 0, 15], vmin=lo, vmax=hi)
    x.set_title(t); x.grid(False); x.set_xticks([0, 15]); x.set_yticks([0, 15])
    cb = fig.colorbar(im, ax=x, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
fig.tight_layout(w_pad=0.8); fig.savefig(OUT+"fig1.png", bbox_inches="tight"); plt.close(fig)

# Figure 2: standard-deviation fields, three problems x four columns
fig, ax = plt.subplots(3, 4, figsize=(7.4, 5.6))
for i, p in enumerate(PROB):
    a = A[p]; v = a["s_ref"].max()
    cols = [(a["s_ref"], "reference (test half)"), (a["s_coarse"], "coarse only"), (a["s_raw"], "raw operator"), (a["s_down"], "rescaled operator")]
    for j, (f, t) in enumerate(cols):
        im = hm(ax[i, j], f, t if i == 0 else "", 0, v, "Blues")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[i, 0].set_ylabel(PROB[p])
    fig.colorbar(im, ax=ax[i, :].tolist(), fraction=0.02, pad=0.02)
fig.savefig(OUT+"fig2.png", bbox_inches="tight"); plt.close(fig)

# Figure 3: the tautology
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
x = np.arange(3); w = 0.2
for k, (key, lab, col) in enumerate([("coarse_only", "coarse only (no fine statistics)", MUTED), ("raw_no_rescaling", "raw operator", BLUE),
                                      ("holdout", "calibrated operator, hold-out", ORANGE), ("split_gap", "reference split gap", AQUA)]):
    for m, a_ in zip(("mean", "sd"), ax):
        vals = [100*J[p][key][m] for p in PROB]
        a_.bar(x+(k-1.5)*w, vals, w*0.92, color=col, label=lab)
for m, a_, t in zip(("mean", "sd"), ax, ("(a) mean field", "(b) standard-deviation field")):
    a_.set_xticks(x); a_.set_xticklabels(PROB.values()); a_.set_ylabel("relative L$^2$ error (%)"); a_.set_title(t); a_.set_yscale("log")
ax[1].legend(fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT+"fig3.png"); plt.close(fig)

# Figure 4: calibration-size experiment
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for p, col, mk in zip(PROB, (BLUE, ORANGE, AQUA), ("o", "s", "^")):
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
ax[0].legend(fontsize=5.8, ncol=1, loc="lower left")
fig.tight_layout(); fig.savefig(OUT+"fig4.png"); plt.close(fig)
print("ok")
