"""Figures 1–4 and Tables 1–2 of the MOS article (model-order selection), in the
article's visual style, from the reproducible results.

Usage (from reproducao/artigos_MOS_LML_COST, after run_base.py / mos_multiseed.py /
mos_aggregate.py, or with the result files shipped there):
    python ../figuras_artigos/mos_article_figures.py OUT_DIR
The cache _cache_P1_4_NR1000_v14.pkl is read if present; otherwise it is recomputed
with rdupscale (identical result, about 1.5 min)."""
import json, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs_mos"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.alpha": 0.35, "savefig.dpi": 200})
BLUE, GREEN, RED, NAVY, GREY, ORANGE = "#1f6fe0", "#128a3e", "#c62828", "#1b2a4a", "#8a8a8a", "#e07b1b"
P = "1.4"
agg = json.load(open("master_mos_agg.json"))
full = json.load(open("master_mos_multiseed.json"))[P]["full"]
cache = f"_cache_P{P.replace('.', '_')}_NR1000_v14.pkl"
if os.path.exists(cache):
    res = pickle.load(open(cache, "rb"))
else:
    import rdupscale as rd
    K = rd.sample_permeability(1000, seed=2026104)
    res = rd.run_problem(P, rd.PROBLEMS[P], K, da_params=rd.DAParams.legacy(), n_R=25, seed_macro=7)
N, NC = res["N_R"], res["N_C"]
A = agg[P]; rows = A["rows"]; S = A["summary"]; I, J = S["cell"]
Kopt, Kpl = S["MDL_opt"]["K"], S["KS_plateau"]["K"]
m = lambda key: (np.array([r[key][0] for r in rows]), np.array([r[key][1] for r in rows]))
Kat = m("K")[0]


def legend_below(fig, handles, labels, ncol):
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.02))


def save(fig, name):
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.6)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig)


# ---------------- Figure 1: the scalar source and its compression ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1, 1.45, 1.45]))
var = res["Kxx_all"].var(axis=0)
im = ax[0].imshow(var, origin="lower", cmap="viridis", extent=[0, 15, 0, 15])
ax[0].add_patch(plt.Rectangle((J * 3, I * 3), 3, 3, fill=False, ec=RED, lw=2.2))
ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].grid(False)
ax[0].set_xticks([0, 5, 10, 15]); ax[0].set_yticks([0, 5, 10, 15])
ax[0].set_title("(a) One scalar source per macro-cell")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03, label="Var($K_{xx}$)")
e = next(d for d in res["da_all_xx"] if tuple(d["IJ"]) == (I, J))
s, yk, qk = e["samples"], e["yk"], e["qk"] / e["qk"].sum()
bins = np.linspace(0, np.quantile(s, 0.995), 36)
ax[1].hist(s, bins=bins, density=True, color=BLUE, alpha=0.45, lw=0)
o = np.argsort(yk); ys, qs = yk[o], qk[o]
edges = np.concatenate([[ys[0] - (ys[1] - ys[0]) / 2], (ys[1:] + ys[:-1]) / 2, [ys[-1] + (ys[-1] - ys[-2]) / 2]])
dens = qs / np.diff(edges)
ax[1].vlines(ys, 0, dens, color=ORANGE, lw=1.4)
ax[1].plot(ys, dens, "o", color=ORANGE, mec="k", mew=0.5, ms=4.5)
ax[1].set_xlim(0, bins[-1]); ax[1].set_ylim(0, None)
ax[1].set_xlabel("$K_{xx}$  in the highlighted cell"); ax[1].set_ylabel("probability density")
ax[1].set_title("(b) The source and its compression")
h = e["h"]
ax[2].plot(h[:, 1], h[:, 2], color=GREY, lw=0.8, zorder=1)
sc = ax[2].scatter(h[:, 1], h[:, 2], c=h[:, 3], cmap="viridis", s=14, zorder=2)
ax[2].set_xscale("log"); ax[2].set_xlabel("distortion  $D$"); ax[2].set_ylabel("rate  $I(X;Q)$  [nats]")
ax[2].set_title("(c) The rate-distortion curve")
fig.colorbar(sc, ax=ax[2], fraction=0.046, pad=0.03, label="$K$ in use")
legend_below(fig, [Patch(color=BLUE, alpha=0.45), Line2D([], [], color=ORANGE, marker="o", mec="k", mew=0.5, lw=1.4),
                   Line2D([], [], color=RED, lw=2)],
             [f"micromesh samples  ($N_R$ = {N})", f"representative realizations  ($K$ = {len(yk)})", "highlighted macro-cell"], 3)
save(fig, "fig1.png")

# ---------------- Figure 2: two criteria that disagree ----------------
mis, mis_s = m("misfit"); pen, _ = m("penalty"); mdl, mdl_s = m("MDL"); ks, ks_s = m("KS"); kst, kst_s = m("KS_test")
Kh = np.array([r["K_half"][0] for r in rows])
ok = Kat >= 5
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].errorbar(Kat[ok], mis[ok], yerr=mis_s[ok], color=BLUE, marker="o", ms=4.5, lw=2, capsize=2)
ax[0].plot(Kat[ok], pen[ok], color=GREEN, marker="s", ms=4.5, lw=2)
ax[0].set_yscale("log"); ax[0].set_xlabel("cardinality  $K$"); ax[0].set_ylabel("code length  [nats]")
ax[0].set_title("(a) The two parts of the description length")
ax[1].errorbar(Kat[ok], mdl[ok], yerr=mdl_s[ok], color=RED, marker="o", ms=4.5, lw=2, capsize=2)
io = int(np.argmin(np.where(ok, mdl, np.inf)))
ax[1].plot(Kat[io], mdl[io], "*", color=RED, mec="k", ms=14, zorder=5)
ax[1].annotate(f"MDL optimum\nK ≈ {Kopt:.0f}", (Kat[io], mdl[io]), xytext=(Kat[io] + 5, mdl[io] + 25), color=RED,
               fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec=RED))
ax[1].set_ylim(mdl[io] - 30, mdl[ok].max() + 30)
ax[1].set_xlabel("cardinality  $K$"); ax[1].set_ylabel("MDL (sum)  [nats]")
ax[1].set_title(f"(b) An interior minimum near K ≈ {Kopt:.0f}")
ax[2].errorbar(Kat, ks, yerr=ks_s, color=GREEN, marker="o", ms=4.5, lw=2, capsize=2)
ax[2].errorbar(Kh, kst, yerr=kst_s, color=NAVY, marker="^", ms=4.5, lw=1.4, ls="--", capsize=2)
ax[2].axvline(Kopt, color=RED, ls="--", lw=2); ax[2].axvline(Kpl, color=GREEN, ls="--", lw=1.6)
ks_opt, ks_pl = S["MDL_opt"]["KS"], S["KS_plateau"]["KS"]
ax[2].axhline(ks_opt, color=GREY, ls=":", lw=1.2); ax[2].axhline(ks_pl, color=GREY, ls=":", lw=1.2)
xa = Kpl + 6
ax[2].annotate("", (xa, ks_pl), (xa, ks_opt), arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.4))
ax[2].text(xa - 1.5, (ks_opt + ks_pl) / 2, f"{ks_opt / ks_pl:.1f}×", ha="right", va="center", fontweight="bold", color=NAVY,
           bbox=dict(boxstyle="round", fc="white", ec=NAVY))
ax[2].set_ylim(0, 0.45); ax[2].set_xlabel("cardinality  $K$"); ax[2].set_ylabel("Kolmogorov–Smirnov statistic")
ax[2].set_title(f"(c) Fidelity keeps improving to K ≈ {Kpl:.0f}")
legend_below(fig, [Line2D([], [], color=BLUE, marker="o", lw=2), Line2D([], [], color=GREEN, marker="s", lw=2),
                   Line2D([], [], color=RED, marker="o", lw=2), Line2D([], [], color=GREEN, marker="o", lw=2),
                   Line2D([], [], color=NAVY, marker="^", ls="--"), Line2D([], [], color=RED, ls="--", lw=2),
                   Line2D([], [], color=GREEN, ls="--")],
             ["misfit  $-\\log L$", "parsimony penalty  $(K/2)\\log N_R$", "MDL (their sum)", "K-S in sample",
              "K-S held out (500/500)", "MDL optimum", "K-S plateau"], 4)
save(fig, "fig2.png")

# ---------------- Figure 3: the sample-support ceiling ----------------
Ka = np.array([r["K"] for r in full]); Ksa = np.array([r["Ks"] for r in full]); KSa = np.array([r["KS"] for r in full])
Ksm = m("Ks")[0]
Kprod = np.concatenate([[len(d["yk"]) for d in res[f"da_all_{c}"]] for c in ("xx", "yy", "xy")])
Ksprod = np.concatenate([[int(((d["qk"] / d["qk"].sum()) * N >= 5).sum()) for d in res[f"da_all_{c}"]] for c in ("xx", "yy", "xy")])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].plot([0, 40], [0, 40], color=GREY, ls="--", lw=2)
ax[0].fill_between(Kat, Ksm, Kat, color=RED, alpha=0.15)
ax[0].scatter(Ka, Ksa, s=12, color=BLUE, alpha=0.35)
ax[0].plot(Kat, Ksm, color=BLUE, marker="o", ms=4.5, lw=2.4)
ax[0].set_xlabel("attained cardinality  $K$"); ax[0].set_ylabel("resolvable  $K_{\\rm valid}$")
ax[0].set_title("(a) The five-sample ceiling")
ax[1].scatter(Ksa, KSa, s=14, color=GREEN, alpha=0.45)
order = np.argsort(Ksm); ax[1].plot(Ksm[order], ks[order], color=GREEN, lw=2.4, marker="o", ms=4.5)
ax[1].set_ylim(0, 0.6); ax[1].set_xlabel("resolvable values  $K_{\\rm valid}$"); ax[1].set_ylabel("Kolmogorov–Smirnov statistic")
ax[1].set_title("(b) Gains flatten where resolvability ends")
ax[2].hist(Kprod, bins=np.arange(14.5, 51.5, 1), color=BLUE, alpha=0.6, lw=0)
ax[2].hist(Ksprod, bins=np.arange(14.5, 51.5, 1), color=GREEN, alpha=0.45, lw=0)
ax[2].axvline(Kopt, color=RED, ls="--", lw=2); ax[2].axvline(Kpl, color=GREEN, ls="--", lw=2)
ax[2].axvline(Kprod.mean(), color=NAVY, lw=2.4)
ax[2].set_xlabel("cardinality of the tuned annealing"); ax[2].set_ylabel("(component, macro-cell) pairs")
ax[2].set_title(f"(c) Where the annealing lands (mean = {Kprod.mean():.1f})")
legend_below(fig, [Line2D([], [], color=BLUE, marker="o", lw=2.4), Patch(color=RED, alpha=0.15), Line2D([], [], color=GREY, ls="--", lw=2),
                   Line2D([], [], color=GREEN, marker="o", lw=2.4), Patch(color=BLUE, alpha=0.6), Patch(color=GREEN, alpha=0.45),
                   Line2D([], [], color=RED, ls="--", lw=2), Line2D([], [], color=GREEN, ls="--", lw=2), Line2D([], [], color=NAVY, lw=2.4)],
             ["resolvable values (seed mean)", "below the five-sample floor", "all K", "K-S statistic", "attained K", "resolvable $K_{\\rm valid}$",
              "MDL optimum", "K-S plateau", "attained mean"], 5)
save(fig, "fig3.png")

# ---------------- Figure 4: allocation and effective size ----------------
km = np.array([len(d["yk"]) for d in res["da_all_xx"]]).reshape(NC, NC)
vv, kk, pp, lab = [], [], [], []
for c in ("xx", "yy", "xy"):
    for d in res[f"da_all_{c}"]:
        q = d["qk"] / d["qk"].sum(); q = q[q > 0]
        vv.append(np.var(d["samples"])); kk.append(len(d["yk"])); pp.append(np.exp(-(q * np.log(q)).sum())); lab.append(c)
vv, kk, pp, lab = map(np.array, (vv, kk, pp, lab))
dg = lab != "xy"; r = np.corrcoef(np.log(vv[dg]), kk[dg])[0, 1]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3]))
im = ax[0].imshow(km, origin="lower", cmap="viridis", extent=[0, 15, 0, 15])
for a in range(NC):
    for b in range(NC):
        ax[0].text(b * 3 + 1.5, a * 3 + 1.5, str(km[a, b]), ha="center", va="center", color="white", fontweight="bold")
ax[0].grid(False); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
ax[0].set_xticks([0, 5, 10, 15]); ax[0].set_yticks([0, 5, 10, 15])
ax[0].set_title("(a) Cardinality allocated per macro-cell")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03, label="attained $K$")
ax[1].scatter(vv[dg], kk[dg], c=kk[dg], cmap="viridis", s=45, edgecolors="k", lw=0.5)
xs = np.linspace(vv[dg].min(), vv[dg].max(), 50); cf = np.polyfit(np.log(vv[dg]), kk[dg], 1)
ax[1].plot(xs, np.polyval(cf, np.log(xs)), color=RED, ls="--", lw=2)
from matplotlib.ticker import ScalarFormatter, NullFormatter
ax[1].set_xscale("log"); ax[1].xaxis.set_major_formatter(ScalarFormatter()); ax[1].xaxis.set_minor_formatter(NullFormatter())
ax[1].set_xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.4]); ax[1].set_xlabel("local source variance  Var($K_{xx}$), Var($K_{yy}$)"); ax[1].set_ylabel("attained  $K$")
ax[1].set_title(f"(b) Allocation against local variance (r = {r:.2f})")
ax[2].plot([22, 52], [22, 52], color=GREY, ls="--", lw=2)
ax[2].scatter(kk, pp, s=40, color=BLUE, edgecolors="k", lw=0.5, alpha=0.85)
ax[2].axhline(pp.mean(), color=ORANGE, lw=2.4)
ax[2].set_xlabel("nominal cardinality  $K$"); ax[2].set_ylabel("perplexity  exp$H(Q)$")
ax[2].set_title(f"(c) Effective size is {pp.mean() / kk.mean():.2f} of nominal")
legend_below(fig, [Line2D([], [], marker="o", color="w", mfc=BLUE, mec="k", ms=8), Line2D([], [], color=RED, ls="--", lw=2),
                   Line2D([], [], color=GREY, ls="--", lw=2), Line2D([], [], color=ORANGE, lw=2.4)],
             ["macro-cell (three tensor components)", "log-linear fit", "uniform masses (perplexity = K)", "mean perplexity"], 4)
save(fig, "fig4.png")

# ---------------- Tables and summary numbers ----------------
tab1 = [dict(K_max=r["K_max"], K=r["K"], misfit=r["misfit"][0], penalty=r["penalty"][0], MDL=r["MDL"], KS=r["KS"],
             KS_test=r["KS_test"], Ks=r["Ks"]) for r in rows]
tab2 = {}
for c in ("xx", "yy", "xy"):
    s_ = lab == c; Kc = kk[s_]
    Hc = [-(q[q > 0] * np.log(q[q > 0])).sum() for q in (d["qk"] / d["qk"].sum() for d in res[f"da_all_{c}"])]
    tab2[c] = dict(K_mean=float(Kc.mean()), K_min=int(Kc.min()), K_max=int(Kc.max()), H=float(np.mean(Hc)),
                   PP=float(pp[s_].mean()), compression=float(N / Kc.mean()))
stats = dict(r_logvar_K_diag=float(r), pp_over_K=float(pp.mean() / kk.mean()), K_mean=float(kk.mean()),
             K_range=[int(kk.min()), int(kk.max())], Ks_mean_prod=float(Ksprod.mean()),
             frac_below_floor=float(1 - Ksprod.sum() / Kprod.sum()), cell=[I, J], K_fig1=int(len(yk)))
json.dump(dict(table1=tab1, table2=tab2, stats=stats, summary={p: agg[p]["summary"] for p in agg}),
          open(os.path.join(OUT, "mos_article_numbers.json"), "w"), indent=1)
print(json.dumps(stats, indent=1)); print(json.dumps(tab2, indent=1))
