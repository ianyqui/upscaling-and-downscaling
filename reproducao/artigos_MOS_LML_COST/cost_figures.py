import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, YEL, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "axes.edgecolor": "#6b6a66", "axes.linewidth": 0.6, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.5, "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": INK,
                     "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150})
OUT = "../build/cost/"
sc = json.load(open("master_cost_scaling.json")); r = json.load(open("master_cost.json")); an = json.load(open("master_cost_analysis.json"))
NAMES = {"1.1": "S1", "1.3": "S2", "1.4": "S3", "1.2": "S4"}
VAR = [("upscaled_no_compression", "upscaled tensors, no compression", MUTED, "o"), ("DA_independent", "DA, independent cells", BLUE, "s"), ("DA_copula", "DA, Gaussian copula", ORANGE, "^")]

# Fig 1 solver scaling
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
c = np.array([s["cells"] for s in sc["scaling"]]); tf = np.array([s["t_fine"] for s in sc["scaling"]]); tc = np.array([s["t_coarse_a3"] for s in sc["scaling"]])
ax[0].loglog(c, 1000*tf, "o-", color=BLUE, ms=4, lw=1.3, label="fine solve")
ax[0].loglog(c, 1000*tc, "s-", color=MUTED, ms=4, lw=1.3, label="coarse solve, α = 3")
p = sc["scaling_exponent_last3"]; xx = np.array([c[-3], c[-1]]); ax[0].loglog(xx, 1000*tf[-1]*(xx/c[-1])**p, "--", color=INK, lw=0.8, label=f"∝ n$^{{{p:.2f}}}$")
ax[0].set_xlabel("fine-mesh cells n"); ax[0].set_ylabel("time per solve (ms)"); ax[0].set_title("(a) Solver scaling"); ax[0].legend(fontsize=7)
ra = sc["ratio_vs_alpha_NF480"]; a = [x["alpha"] for x in ra]; rr = [x["ratio"] for x in ra]
ax[1].loglog(a, rr, "o-", color=ORANGE, ms=4, lw=1.3); ax[1].axvline(12, color=MUTED, ls=":", lw=1)
ax[1].set_xlabel("coarsening factor α"); ax[1].set_ylabel("fine / coarse solve time"); ax[1].set_title("(b) Ratio at 480 × 480 cells")
fig.tight_layout(); fig.savefig(OUT+"fig1.png"); plt.close(fig)

# Fig 2 cost anatomy
To = r["T_offline"]; S = r["scenarios"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.4]))
lab = ["upscaling\n(500 fine solves)", "DA compression\n(200 anneals)"]; val = [To["upscaling"], To["DA_compression"]]
b = ax[0].bar(lab, val, color=[BLUE, ORANGE], width=0.6)
for x, v in zip(b, val): ax[0].text(x.get_x()+x.get_width()/2, v*1.02, f"{v:.0f} s", ha="center", fontsize=7.5)
ax[0].set_ylabel("offline wall-clock (s)"); ax[0].set_title("(a) Offline, paid once")
sn = list(NAMES.values()); x = np.arange(4); w = 0.2
bf = [S[k]["t_bruteforce_250"] for k in NAMES]
ax[1].bar(x-1.5*w, bf, w, color=INK, label="fine Monte Carlo, 250 solves")
for j, (v, l, col, _) in enumerate(VAR):
    ax[1].bar(x+(j-0.5)*w, [S[k]["variants"][v]["t_online"] for k in NAMES], w, color=col, label=l)
ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(sn); ax[1].set_ylabel("per-scenario wall-clock (s)")
ax[1].set_title("(b) Online, per scenario (250 realizations)"); ax[1].set_ylim(0.03, 60); ax[1].legend(fontsize=6.2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(OUT+"fig2.png"); plt.close(fig)

# Fig 3 accuracy vs Monte Carlo
MET = [("err_mean_p", "mean pressure (rel. L$^2$)"), ("err_sd_p", "sd of pressure (rel. L$^2$)"), ("ks_Q", "KS distance of the flux")]
fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.6))
for i, key in enumerate(("1.1", "1.4")):
    mc = S[key]["mc_with_n"]; n = np.array(sorted(int(k) for k in mc))
    for j, (m, t) in enumerate(MET):
        a_ = ax[i, j]; a_.loglog(n, [mc[str(k)][m] for k in n], "-", color=INK, marker="o", ms=3, lw=1.2, label="fine Monte Carlo, n solves")
        for v, l, col, mk in VAR:
            a_.axhline(S[key]["variants"][v][m], color=col, lw=1.3, ls="--" if v != "DA_copula" else "-", label=l)
        a_.set_title(f"{NAMES[key]}: {t}", fontsize=8); a_.set_xlabel("n")
ax[0, 0].legend(fontsize=6, loc="lower left")
fig.tight_layout(); fig.savefig(OUT+"fig3.png"); plt.close(fig)

# Fig 4 break-even
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
for j, (m, t) in enumerate(MET):
    a_ = ax[j]; x = np.arange(4); w = 0.26
    for k, (v, l, col, _) in enumerate(VAR):
        vals = [an["scen"][s][v][m]["S_star"] for s in sn]
        vals = [1e5 if y is None else y for y in vals]
        a_.bar(x+(k-1)*w, vals, w, color=col, label=l)
    a_.set_yscale("log"); a_.set_ylim(1, 2e4); a_.set_xticks(x); a_.set_xticklabels(sn); a_.set_title(t, fontsize=8)
ax[0].set_ylabel("break-even scenarios S*"); ax[0].legend(fontsize=6, loc="upper left")
fig.tight_layout(); fig.savefig(OUT+"fig4.png"); plt.close(fig)
print("ok")
