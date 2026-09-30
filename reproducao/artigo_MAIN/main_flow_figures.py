"""Figures 7-9 of the MAIN article: fine (15x15) versus coarse (5x5) pressure fields.
Style of the original article: viridis maps, micro panels in micro-mesh indices (0-15), macro panels in macro-mesh indices (0-5)."""
import json, pickle, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]      # matplotlib default (tab10)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold",
                     "axes.grid": False, "savefig.dpi": 300, "figure.dpi": 150})
OUT = os.environ.get("OUT", "figuras/"); os.makedirs(OUT, exist_ok=True)
F = {tuple(k.split("|")): {kk: np.array(vv) for kk, vv in v.items()} for k, v in pickle.load(open("main_flow_fields.pkl", "rb")).items()}
d = json.load(open("main_flow.json"))
PN = {"1.1": "Problem 1", "1.3": "Problem 2", "1.4": "Problem 3"}

def hm(fig, ax, arr, title, vmin=None, vmax=None, cbl=None):
    n = arr.shape[0]; macro = n <= 5
    im = ax.imshow(arr, origin="lower", cmap="viridis", extent=[0, n, 0, n], vmin=vmin, vmax=vmax, aspect="equal")
    t = np.arange(0, n+1, 1 if macro else 3)
    ax.set_xticks(t); ax.set_yticks(t); ax.tick_params(labelsize=6)
    ax.set_xlabel("macro index i" if macro else "micro index i", fontsize=7)
    ax.set_ylabel("macro index j" if macro else "micro index j", fontsize=7)
    ax.set_title(title, fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6)
    if cbl: cb.set_label(cbl, fontsize=6.5)

# Figure 7: mean pressure
fig, ax = plt.subplots(3, 4, figsize=(9.6, 7.2))
for r, p in enumerate(("1.1", "1.3", "1.4")):
    fine = F[(p, "REF")]["fmean"]; ref = F[(p, "REF")]["mean"]; est = F[(p, "DAC")]["mean"]
    ae = np.abs(est-ref); re_ = 100*ae/np.abs(ref); L = "abcdefghijkl"[4*r:4*r+4]
    hm(fig, ax[r, 0], fine, f"({L[0]}) {PN[p]} — micromesh ({fine.shape[0]}×{fine.shape[0]})", 0, 1)
    hm(fig, ax[r, 1], est, f"({L[1]}) {PN[p]} — macromesh ({est.shape[0]}×{est.shape[0]}, DA)", 0, 1)
    hm(fig, ax[r, 2], ae, f"({L[2]}) absolute error |P$_{{mac}}$ − P$_{{mic}}$|")
    hm(fig, ax[r, 3], re_, f"({L[3]}) relative error (%)")
fig.tight_layout(); fig.savefig(OUT+"fig7.png", bbox_inches="tight"); plt.close(fig)

# Figure 8: standard deviation of pressure
fig, ax = plt.subplots(3, 5, figsize=(12.0, 7.2))
for r, p in enumerate(("1.1", "1.3", "1.4")):
    fine = F[(p, "REF")]["fsd"]; ref = F[(p, "REF")]["sd"]; vm = max(fine.max(), ref.max())
    ri = 100*np.abs(F[(p, "DAI")]["sd"]-ref)/ref; rc = 100*np.abs(F[(p, "DAC")]["sd"]-ref)/ref; em = max(ri.max(), rc.max())
    L = "abcdefghijklmno"[5*r:5*r+5]
    hm(fig, ax[r, 0], fine, f"({L[0]}) {PN[p]} — micromesh sd", 0, vm)
    hm(fig, ax[r, 1], F[(p, "DAI")]["sd"], f"({L[1]}) macromesh sd, independent", 0, vm)
    hm(fig, ax[r, 2], F[(p, "DAC")]["sd"], f"({L[2]}) macromesh sd, copula", 0, vm)
    hm(fig, ax[r, 3], ri, f"({L[3]}) relative error (%), independent", 0, em)
    hm(fig, ax[r, 4], rc, f"({L[4]}) relative error (%), copula", 0, em)
fig.tight_layout(); fig.savefig(OUT+"fig8.png", bbox_inches="tight"); plt.close(fig)

# Figure 9: summary
V = [("noise_floor", "micro vs micro (500/500)", C[7]), ("UPS", "upscaled, no compression", C[0]), ("DA25", "DA, 25 independent", C[1]),
     ("DAI", "DA, 1000 independent", C[2]), ("DAC", "DA, 1000 with copula", C[3])]
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6)); x = np.arange(3); w = 0.16
for k, (key, lab, col) in enumerate(V):
    for j, (q, t) in enumerate((("mean", "(a) Mean pressure, rel. L$^2$ error (%)"), ("sd", "(b) Sd of pressure, rel. L$^2$ error (%)"))):
        ax[j].bar(x+(k-2)*w, [d[p][key][q]["rel_L2"] for p in ("1.1", "1.3", "1.4")], w, color=col, label=lab); ax[j].set_title(t)
    ax[2].bar(x+(k-2)*w, [d[p][key]["ks_mean"] for p in ("1.1", "1.3", "1.4")], w, color=col); ax[2].set_title("(c) KS distance per cell (mean)")
for a in ax: a.set_xticks(x); a.set_xticklabels(["Problem 1", "Problem 2", "Problem 3"], fontsize=7); a.set_axisbelow(True)
ax[0].legend(fontsize=5.8, loc="upper left", framealpha=0.9)
fig.tight_layout(); fig.savefig(OUT+"fig9.png", bbox_inches="tight"); plt.close(fig)
print("ok")
