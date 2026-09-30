import json, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "axes.edgecolor": "#6b6a66", "axes.linewidth": 0.6, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.5, "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": INK,
                     "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 150})
OUT = "../build/mos/"
agg = json.load(open("master_mos_agg.json")); base = json.load(open("master_base.json"))
res = pickle.load(open("_cache_P1_4_NR1000_v14.pkl", "rb"))
P = "1.4"; A = agg[P]; rows = A["rows"]; S = A["summary"]; I, J = S["cell"]
N = res["N_R"]; NC = res["N_C"]

# ---------------- Figure 1 ----------------
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw=dict(width_ratios=[1, 1.25, 1.25]))
var = res["Kxx_all"].var(axis=0)
im = ax[0].imshow(var, origin="lower", cmap="Blues", extent=[0, 15, 0, 15])
ax[0].add_patch(plt.Rectangle((J*3, I*3), 3, 3, fill=False, ec=ORANGE, lw=1.6))
ax[0].set_xticks([0, 5, 10, 15]); ax[0].set_yticks([0, 5, 10, 15]); ax[0].grid(False)
ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].set_title("(a) Var($K_{xx}$) per coarse cell")
cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03); cb.outline.set_linewidth(0.4)
e = next(d for d in res["da_all_xx"] if tuple(d["IJ"]) == (I, J))
s, yk, qk = e["samples"], e["yk"], e["qk"]/e["qk"].sum()
bins = np.linspace(s.min(), np.quantile(s, 0.995), 40)
ax[1].hist(s, bins=bins, density=True, color=BLUE, alpha=0.35, lw=0)
o = np.argsort(yk); ys = yk[o]; qs = qk[o]
edges = np.concatenate([[ys[0]-(ys[1]-ys[0])/2], (ys[1:]+ys[:-1])/2, [ys[-1]+(ys[-1]-ys[-2])/2]])
dens = qs/np.diff(edges)
ax[1].vlines(ys, 0, dens, color=ORANGE, lw=1.0); ax[1].plot(ys, dens, "o", color=ORANGE, ms=2.5)
ax[1].set_xlim(bins[0], bins[-1]); ax[1].set_xlabel("$K_{xx}$ in the highlighted cell"); ax[1].set_ylabel("density")
ax[1].set_title(f"(b) Samples and K = {len(yk)} values")
ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.35), Line2D([], [], color=ORANGE, marker="o", ms=3, lw=1)],
             labels=[f"samples (N = {N})", "representative values"], fontsize=6.5, loc="upper right", bbox_to_anchor=(1.0, 1.0))
ax[1].set_ylim(0, dens.max()*1.35)
h = e["h"]
sc = ax[2].scatter(h[:, 1], h[:, 2], c=h[:, 3], cmap="Blues", s=6, vmin=0, edgecolors="none")
ax[2].set_xscale("log"); ax[2].set_xlabel("distortion D"); ax[2].set_ylabel("rate I(X;Q) (nats)")
ax[2].set_title("(c) Rate–distortion curve")
cb = fig.colorbar(sc, ax=ax[2], fraction=0.046, pad=0.03); cb.set_label("K", fontsize=7); cb.outline.set_linewidth(0.4)
fig.tight_layout(w_pad=1.2); fig.savefig(OUT+"fig1.png"); plt.close(fig)

# ---------------- Figure 2 ----------------
K = np.array([r["K"][0] for r in rows]); Ksd = np.array([r["K"][1] for r in rows])
def m(key): return np.array([r[key][0] for r in rows]), np.array([r[key][1] for r in rows])
mis, mis_s = m("misfit"); pen, _ = m("penalty"); mdl, mdl_s = m("MDL"); ks, ks_s = m("KS"); kst, kst_s = m("KS_test")
sel = K >= 5  # the K = 2-3 points have misfits one order of magnitude larger and are omitted for legibility
Kopt = S["MDL_opt"]["K"]; Kpl = S["KS_plateau"]["K"]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
ax[0].errorbar(K[sel], mis[sel]-mis[sel].min(), yerr=mis_s[sel], color=BLUE, marker="o", ms=3, lw=1.2, capsize=1.5, label="misfit − min")
ax[0].plot(K[sel], pen[sel], color=ORANGE, marker="s", ms=3, lw=1.2, label="penalty (K/2) ln N")
ax[0].set_xlabel("attained cardinality K"); ax[0].set_ylabel("nats"); ax[0].set_title("(a) The two parts of the code")
ax[0].legend(fontsize=7); ax[0].set_ylim(0, 150)
ax[1].errorbar(K[sel], mdl[sel], yerr=mdl_s[sel], color=INK, marker="o", ms=3, lw=1.2, capsize=1.5)
ax[1].set_ylim(1270, 1450); ax[1].axvline(Kopt, color=ORANGE, ls="--", lw=1); ax[1].text(Kopt+1, 1440, f"minimum\nK ≈ {Kopt:.0f}", fontsize=7, va="top", color="#52514e")
ax[1].set_xlabel("attained cardinality K"); ax[1].set_ylabel("MDL (nats)"); ax[1].set_title("(b) Description length")
ax[2].errorbar(K, ks, yerr=ks_s, color=BLUE, marker="o", ms=3, lw=1.2, capsize=1.5, label="in sample (N = 1000)")
ax[2].errorbar(np.array([r["K_half"][0] for r in rows]), kst, yerr=kst_s, color=AQUA, marker="^", ms=3, lw=1.2, capsize=1.5, label="held out (500/500)")
ax[2].axvline(Kopt, color=ORANGE, ls="--", lw=1); ax[2].axvline(Kpl, color=MUTED, ls=":", lw=1)
ax[2].set_ylim(0, 0.45); ax[2].set_xlabel("attained cardinality K"); ax[2].set_ylabel("Kolmogorov–Smirnov distance")
ax[2].set_title("(c) Distributional error"); ax[2].legend(fontsize=7)
fig.tight_layout(w_pad=1.0); fig.savefig(OUT+"fig2.png"); plt.close(fig)

# ---------------- Figure 3 ----------------
full = json.load(open("master_mos_multiseed.json"))[P]["full"]
Ka = np.array([r["K"] for r in full]); Ksa = np.array([r["Ks"] for r in full]); KSa = np.array([r["KS"] for r in full])
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
ax[0].plot([0, 60], [0, 60], color=MUTED, ls="--", lw=0.8)
ax[0].scatter(Ka, Ksa, s=8, color=BLUE, alpha=0.5, edgecolors="none"); ax[0].plot(K, m("Ks")[0], color=BLUE, lw=1.4)
ax[0].set_xlim(0, 50); ax[0].set_ylim(0, 50); ax[0].set_xlabel("attained cardinality K"); ax[0].set_ylabel("supported values $K_s$")
ax[0].set_title("(a) Sample-support test")
ax[1].scatter(Ksa, KSa, s=8, color=BLUE, alpha=0.5, edgecolors="none"); ax[1].set_ylim(0, 0.6)
ax[1].set_xlabel("supported values $K_s$"); ax[1].set_ylabel("KS distance"); ax[1].set_title("(b) Error against support")
Kprod = np.concatenate([[len(d["yk"]) for d in res[f"da_all_{c}"]] for c in ("xx", "yy", "xy")])
Ksprod = np.concatenate([[int(((d["qk"]/d["qk"].sum())*N >= 5).sum()) for d in res[f"da_all_{c}"]] for c in ("xx", "yy", "xy")])
ax[2].hist(Kprod, bins=np.arange(24.5, 51.5, 1), color=BLUE, alpha=0.8, lw=0, label="attained K")
ax[2].hist(Ksprod, bins=np.arange(14.5, 41.5, 1), color=AQUA, alpha=0.6, lw=0, label="supported $K_s$")
ax[2].axvline(Kopt, color=ORANGE, ls="--", lw=1, label="MDL optimum"); ax[2].axvline(Kpl, color=MUTED, ls=":", lw=1.2, label="KS plateau")
ax[2].set_xlabel("cardinality"); ax[2].set_ylabel("(component, cell) pairs"); ax[2].set_title("(c) Tuned annealing, 75 pairs")
ax[2].legend(fontsize=6.5, loc="upper right")
fig.tight_layout(w_pad=1.0); fig.savefig(OUT+"fig3.png"); plt.close(fig)

# ---------------- Figure 4 ----------------
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw=dict(width_ratios=[1, 1.2, 1.2]))
km = np.array([len(d["yk"]) for d in res["da_all_xx"]]).reshape(NC, NC)
im = ax[0].imshow(km, origin="lower", cmap="Blues", extent=[0, 5, 0, 5], vmin=20)
for a in range(NC):
    for b in range(NC):
        ax[0].text(b+0.5, a+0.5, str(km[a, b]), ha="center", va="center", fontsize=7, color="white" if km[a, b] > 36 else INK)
ax[0].set_xticks(range(6)); ax[0].set_yticks(range(6)); ax[0].grid(False)
ax[0].set_xlabel("coarse index i"); ax[0].set_ylabel("coarse index j"); ax[0].set_title("(a) Attained K, $K_{xx}$")
vv = []; kk = []; pp = []; lab = []
for c, col in zip(("xx", "yy", "xy"), (BLUE, ORANGE, AQUA)):
    for d in res[f"da_all_{c}"]:
        q = d["qk"]/d["qk"].sum(); vv.append(np.var(d["samples"])); kk.append(len(d["yk"])); pp.append(np.exp(-(q[q>0]*np.log(q[q>0])).sum())); lab.append(c)
vv, kk, pp, lab = map(np.array, (vv, kk, pp, lab))
d_ = lab != "xy"; r = np.corrcoef(np.log(vv[d_]), kk[d_])[0, 1]
for c, col, mk in zip(("xx", "yy"), (BLUE, ORANGE), ("o", "s")):
    s_ = lab == c; ax[1].scatter(vv[s_], kk[s_], s=10, color=col, marker=mk, edgecolors="none", alpha=0.8, label=f"$K_{{{c}}}$")
ax[1].set_xlabel("total variance of the cell"); ax[1].set_ylabel("attained K")
ax[1].set_title(f"(b) K against variance (r = {r:.2f})"); ax[1].legend(fontsize=7)
ax[2].plot([20, 52], [20, 52], color=MUTED, ls="--", lw=0.8, label="equal masses")
for c, col, mk in zip(("xx", "yy", "xy"), (BLUE, ORANGE, AQUA), ("o", "s", "^")):
    s_ = lab == c; ax[2].scatter(kk[s_], pp[s_], s=10, color=col, marker=mk, edgecolors="none", alpha=0.8, label=f"$K_{{{c}}}$")
ax[2].set_xlim(22, 52); ax[2].set_ylim(10, 52); ax[2].set_xlabel("attained K"); ax[2].set_ylabel("perplexity exp H(Q)")
ax[2].set_title(f"(c) Effective size ≈ {np.mean(pp)/np.mean(kk):.2f} K"); ax[2].legend(fontsize=6.5, loc="upper left")
fig.tight_layout(w_pad=1.0); fig.savefig(OUT+"fig4.png"); plt.close(fig)
json.dump(dict(r_logvar_K_diag=float(r), pp_over_K=float(np.mean(pp)/np.mean(kk)), K_mean=float(kk.mean()), Ks_mean_prod=float(Ksprod.mean()),
               frac_below_floor=float(1-Ksprod.sum()/Kprod.sum())), open("mos_fig_stats.json", "w"), indent=1)
print(open("mos_fig_stats.json").read())
