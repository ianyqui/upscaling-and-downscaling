import json, sys, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# palette of the original article figures
BLUE, ORANGE, AQUA, YEL, INK, MUTED = "#1f6fe0", "#e07b1b", "#128a3e", "#c62828", "#1b2a4a", "#8a8a8a"
GREEN, RED = AQUA, YEL
plt.rcParams.update({"patch.force_edgecolor": True, "patch.edgecolor": "black", "patch.linewidth": 0.6, "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.35, "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8", "savefig.dpi": 300, "figure.dpi": 150})
import os
OUT = "figuras/"; os.makedirs(OUT + "supp", exist_ok=True)
PN = {"1.1": "Problem 1", "1.3": "Problem 2", "1.4": "Problem 3"}; PC = {"1.1": BLUE, "1.3": ORANGE, "1.4": AQUA}
CN = {"xx": "K$_{xx}$", "yy": "K$_{yy}$", "xy": "K$_{xy}$"}
which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]

if "1" in which:
    # Figure 1: boundary conditions (style of the original article) and one permeability realization per problem
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(2, 3, figsize=(7.4, 5.0))
    BC = {"1.1": dict(left=1, right=0), "1.3": dict(bottom=1, left=0), "1.4": dict(left=1, right=0, top=0, bottom=0)}
    ARR = {"1.1": ((2.5, 7.5), (12.5, 7.5)), "1.3": ((11.0, 3.0), (3.0, 11.5)), "1.4": ((2.5, 7.5), (12.5, 7.5))}
    seg = dict(left=([0, 0], [0, 15]), right=([15, 15], [0, 15]), bottom=([0, 15], [0, 0]), top=([0, 15], [15, 15]))
    lab = dict(left=(-1.4, 7.5, 90), right=(16.4, 7.5, 90), bottom=(7.5, -1.4, 0), top=(7.5, 16.4, 0))
    for j, p in enumerate(PN):
        a = ax[0, j]; bc = BC[p]
        a.set_xlim(-2.4, 17.4); a.set_ylim(-2.4, 17.4); a.set_aspect("equal"); a.axis("off")
        for i in range(16):
            a.plot([i, i], [0, 15], color="#d9d9d9", lw=0.5); a.plot([0, 15], [i, i], color="#d9d9d9", lw=0.5)
        for i in range(0, 16, 3):
            a.plot([i, i], [0, 15], color="0.15", lw=1.0); a.plot([0, 15], [i, i], color="0.15", lw=1.0)
        for f, (xs, ys) in seg.items():
            v = bc.get(f)
            a.plot(xs, ys, color=GREEN if v is None else BLUE, lw=4.5, solid_capstyle="butt")
            x, y, r = lab[f]
            a.text(x, y, "∂p/∂n = 0" if v is None else f"p = {v:.1f}", ha="center", va="center", rotation=r, fontsize=7,
                   color=GREEN if v is None else BLUE, fontweight="bold")
        (x0, y0), (x1, y1) = ARR[p]
        a.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2, mutation_scale=14))
        a.text((x0 + x1) / 2 + 0.6, (y0 + y1) / 2 + 0.9, "q", color=RED, fontsize=10, fontweight="bold")
        a.set_title(f"({'abc'[j]}) {PN[p]}")
        r = pickle.load(open(f"_cache_P{p.replace('.', '_')}_NR1000_v14.pkl", "rb"))
        b = ax[1, j]; lk = np.log(r["K_micro_first_realiz"])
        im = b.imshow(lk, origin="lower", cmap="viridis", extent=[0, 15, 0, 15], vmin=-2.5, vmax=2.5)
        for g in range(0, 16, 3):
            b.axvline(g, color="white", lw=1.0); b.axhline(g, color="white", lw=1.0)
        b.grid(False); b.set_xticks([0, 5, 10, 15]); b.set_yticks([0, 5, 10, 15]); b.tick_params(labelsize=7)
        b.set_xlabel("x (m)", fontsize=7.5); b.set_title(f"({'def'[j]}) ln κ, one realization")
        if j == 0: b.set_ylabel("y (m)", fontsize=7.5)
    
    h = [Line2D([], [], color=BLUE, lw=4.5), Line2D([], [], color=GREEN, lw=4.5), Line2D([], [], color="#d9d9d9", lw=1.2),
         Line2D([], [], color="0.15", lw=1.2), Line2D([], [], color=RED, lw=2.2, marker=">", ms=6)]
    fig.legend(h, ["Dirichlet: imposed pressure", "Neumann: no flow", "fine grid 15 × 15", "coarse grid 5 × 5 (α = 3)", "Darcy flux q"],
               loc="lower center", ncol=5, fontsize=6.5, frameon=True, bbox_to_anchor=(0.46, -0.01))
    fig.subplots_adjust(left=0.06, right=0.9, top=0.96, bottom=0.12, hspace=0.12, wspace=0.18)
    pos = ax[1, 2].get_position(); cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.013, pos.height])
    cb = fig.colorbar(im, cax=cax); cb.set_label("ln κ", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    fig.savefig(OUT+"fig1.png", bbox_inches="tight"); plt.close(fig)

if "2" in which:
    d = json.load(open("main_tc_cells.json")); alpha = 0.953
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for c, col in (("xx", BLUE), ("xy", ORANGE)):
        for e in d["1.4"][c]:
            T = np.array(e["T"])/e["Tc_theory"]; K = np.array(e["K"])
            ax[0].step(T, K, where="post", color=col, lw=0.6, alpha=0.55)
        ax[0].plot([], [], color=col, label=CN[c])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].axvline(1, color=INK, ls="--", lw=0.8)
    ax[0].set_xlabel("T / 2Var(X)"); ax[0].set_ylabel("number of values K"); ax[0].set_title("(a) Cascade, Problem 3"); ax[0].legend(fontsize=7, loc="lower left")
    allr = []
    for p in d:
        for c, mk in (("xx", "o"), ("yy", "s"), ("xy", "^")):
            x = [e["Tc_theory"] for e in d[p][c]]; y = [e["T_split"] for e in d[p][c]]; allr += list(np.array(y)/np.array(x))
            ax[1].scatter(x, y, s=7, marker=mk, color=PC[p], alpha=0.7, lw=0)
    lim = [0.1, 10]; ax[1].plot(lim, lim, color=INK, lw=0.8); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("2Var(X)"); ax[1].set_ylabel("measured split temperature"); ax[1].set_title("(b) Measured vs predicted")
    for p in PN: ax[1].scatter([], [], s=12, color=PC[p], label=PN[p])
    ax[1].legend(fontsize=6.5, loc="upper left")
    allr = np.array(allr); grid = 2*alpha**np.arange(10, 20)
    u, cnt = np.unique(np.round(allr, 3), return_counts=True)
    ax[2].bar(u, cnt, width=0.012, color=BLUE)
    for g in grid:
        if 0.8 < g < 1.2: ax[2].axvline(g, color=MUTED, ls=":", lw=0.8)
    ax[2].axvline(1, color=INK, ls="--", lw=0.8); ax[2].set_xlim(0.8, 1.16)
    ax[2].set_xlabel("measured / 2Var(X)"); ax[2].set_ylabel("runs"); ax[2].set_title("(c) Ratio, 225 runs")
    fig.tight_layout(); fig.savefig(OUT+"fig2.png", bbox_inches="tight"); plt.close(fig)

if "3" in which or "4" in which:
    rd = json.load(open("main_rd.json"))
if "3" in which:
    fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.3))
    for a, c in zip(ax[:3], ("xx", "yy", "xy")):
        e = max(rd["1.4"][c], key=lambda e: e["var"]); D = np.array(e["D"])/e["var"]; I = np.array(e["I"])
        dd = np.logspace(-3.3, 0, 200); g = np.maximum(0, 0.5*np.log(1/dd)); s = np.maximum(0, e["h"]-0.5*np.log(2*np.pi*np.e*dd*e["var"]))
        a.fill_between(dd, s, g, color="#eef3fb"); a.plot(dd, g, "--", color=INK, lw=0.9, label="Gaussian bound"); a.plot(dd, s, ":", color=INK, lw=1.1, label="Shannon lower bound")
        a.plot(D, I, "-", color=BLUE if c != "xy" else ORANGE, lw=1.4, label="annealing"); a.plot(D[-1], I[-1], "*", color=INK, ms=7)
        a.set_xscale("log"); a.set_xlabel("D / Var(X)"); a.set_title(f"({'abc'['xx yy xy'.split().index(c)]}) {CN[c]}"); a.set_xlim(4e-4, 1.2); a.set_ylim(0, 4)
    ax[0].set_ylabel("rate (nats)"); ax[0].legend(fontsize=5.8, loc="upper right")
    for p in rd:
        for c, mk in (("xx", "o"), ("yy", "s"), ("xy", "^")):
            x = [e["negentropy"] for e in rd[p][c]]
            y = [0.5*np.log(e["var"]/e["D"][-1])-e["I"][-1] for e in rd[p][c]]
            ax[3].scatter(x, y, s=7, marker=mk, color=PC[p], alpha=0.7, lw=0)
    ax[3].plot([0.1, 0.8], [0.1, 0.8], color=INK, lw=0.8); ax[3].set_xlim(0.15, 0.7); ax[3].set_ylim(0, 0.7)
    ax[3].set_xlabel("negentropy (nats)"); ax[3].set_ylabel("Gaussian bound − rate"); ax[3].set_title("(d) Gap vs negentropy")
    fig.tight_layout(); fig.savefig(OUT+"fig3.png", bbox_inches="tight"); plt.close(fig)

if "4" in which:
    fig = plt.figure(figsize=(7.4, 2.5)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    ax = [fig.add_subplot(gs[i]) for i in range(4)]
    grid = np.linspace(0, 25, 300)
    for a, c in zip(ax[:3], ("xx", "yy", "xy")):
        for p in rd:
            Is = []
            for e in rd[p][c]:
                S = 10*np.log10(e["var"]/np.array(e["D"])); I = np.array(e["I"]); o = np.argsort(S)
                Is.append(np.interp(grid, S[o], I[o], right=np.nan))
            a.plot(grid, np.nanmean(Is, axis=0), color=PC[p], lw=1.2, label=PN[p])
            a.plot(np.mean([10*np.log10(e["var"]/e["D"][-1]) for e in rd[p][c]]), np.mean([e["I"][-1] for e in rd[p][c]]), "*", color=PC[p], ms=6)
        a.plot(grid, grid*np.log(10)/20, ":", color=MUTED, lw=0.9)
        a.set_xlabel("10 log$_{10}$(Var/D) (dB)"); a.set_title(CN[c]); a.set_ylim(0, 3); a.set_xlim(0, 25)
    ax[0].set_ylabel("rate I(X;Q) (nats)"); ax[0].legend(fontsize=6, loc="upper left")
    for key, col, lab in (("D", BLUE, "D / Var"), ("TI", ORANGE, "T·I / Var"), ("F", INK, "F / Var")):
        curves = []
        for e in rd["1.4"]["xx"]:
            T = np.array(e["T"]); D = np.array(e["D"]); I = np.array(e["I"]); v = e["var"]
            y = {"D": D/v, "TI": T*I/v, "F": (D+T*I)/v}[key]; curves.append(y)
        n = min(len(y) for y in curves); m = np.mean([y[:n] for y in curves], axis=0)
        e = rd["1.4"]["xx"][0]; Tn = np.array(e["T"])[:n]/(2*e["var"])
        ax[3].plot(Tn, m, color=col, lw=1.3 if key != "F" else 1.0, ls="--" if key == "F" else "-", label=lab)
    ax[3].axvline(1, color=MUTED, ls=":", lw=1)
    ax[3].set_xscale("log"); ax[3].set_xlabel("T / 2Var(X)"); ax[3].set_title("(b) Free energy, K$_{xx}$"); ax[3].legend(fontsize=6.5, loc="upper left")
    fig.text(0.01, 0.95, "(a)", fontweight="bold", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT+"fig4.png", bbox_inches="tight"); plt.close(fig)

if "5" in which:
    sm = json.load(open("main_summary.json")); res = pickle.load(open("_cache_P1_4_NR1000_v14.pkl", "rb"))
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for c, col in (("xx", BLUE), ("xy", ORANGE)):
        s = np.sort(np.concatenate([e["samples"] for e in res[f"da_all_{c}"]])); ax[0].plot(s, np.arange(1, s.size+1)/s.size, color=col, lw=1.4, label=f"{CN[c]} samples")
        yy = np.concatenate([e["yk"] for e in res[f"da_all_{c}"]]); qq = np.concatenate([e["qk"] for e in res[f"da_all_{c}"]])/25; o = np.argsort(yy)
        ax[0].step(yy[o], np.cumsum(qq[o]), where="post", color=INK, lw=0.7, ls="--")
    ax[0].plot([], [], color=INK, ls="--", lw=0.7, label="representative values"); ax[0].set_xlim(-1.5, 5); ax[0].set_xlabel("value"); ax[0].set_ylabel("distribution function")
    ax[0].legend(fontsize=6, loc="lower right"); ax[0].set_title("(a) Pooled, Problem 3")
    labs = []; x0 = 0
    for p in ("1.1", "1.3", "1.4"):
        for c in ("xx", "yy", "xy"):
            h = sm[p][c]["argmax_quartile_hist"]; b = 0
            for q, col in zip(range(4), (BLUE, AQUA, ORANGE, YEL)):
                ax[1].bar(x0, h[q], bottom=b, color=col, width=0.75, label=f"Q{q+1}" if x0 == 0 else None); b += h[q]
            labs.append(f"{'123'[['1.1','1.3','1.4'].index(p)]}{c}"); x0 += 1
    ax[1].set_xticks(range(9)); ax[1].set_xticklabels(labs, rotation=90, fontsize=6.5); ax[1].set_ylabel("cells"); ax[1].set_title("(b) Quartile of largest error")
    ax[1].legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.2); ax[1].set_ylim(0, 36); ax[1].set_yticks([0, 5, 10, 15, 20, 25])
    w = 0.26
    for k, (c, col) in enumerate((("xx", BLUE), ("yy", AQUA), ("xy", ORANGE))):
        ax[2].bar(np.arange(4)+(k-1)*w, sm["1.4"][c]["worst_ks_quart"], w, color=col, label=CN[c])
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["Q1", "Q2", "Q3", "Q4"]); ax[2].set_ylabel("largest deviation"); ax[2].set_title("(c) Worst cell, Problem 3"); ax[2].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(OUT+"fig5.png", bbox_inches="tight"); plt.close(fig)

if "6" in which:
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5)); ref = json.load(open("main_search_ref.json")); rng = np.random.default_rng(0)
    for p in ("1.1", "1.3", "1.4"):
        S = json.load(open(f"main_search_P{p.replace('.','_')}.json")); R = S["rows"]
        t = np.array([r["t_call"] for r in R]); J = np.array([r["J"] for r in R]); ni = np.array([r["cfg"]["n_inner"] for r in R])
        H = np.array([np.mean([r["per"][c]["H"] for c in r["per"]]) for r in R])
        ax[0].scatter(t, J, s=10, color=PC[p], label=PN[p], alpha=0.8, lw=0)
        o = np.argsort(t); best = np.inf; px = []; py = []
        for i in o:
            if J[i] < best: best = J[i]; px.append(t[i]); py.append(J[i])
        px.append(t.max()); py.append(py[-1]); ax[0].step(px, py, where="post", color=PC[p], lw=0.9)
        rr = ref[p]; ax[0].plot(np.mean([v["t"] for v in rr.values()]), np.mean([v["J"] for v in rr.values()]), "*", color=PC[p], ms=9, mec=INK, mew=0.5)
        ax[1].scatter(ni + rng.uniform(-6, 6, ni.size), J, s=10, color=PC[p], alpha=0.8, lw=0)
        ax[2].scatter(H, J, s=10, color=PC[p], alpha=0.8, lw=0)
    ax[0].set_xscale("log"); ax[0].set_xlabel("time per annealing run (s)"); ax[0].set_ylabel("objective J"); ax[0].set_title("(a) Accuracy–cost"); ax[0].set_ylim(0.07, 0.132); ax[0].legend(fontsize=5.8, loc="upper center", ncol=3, handletextpad=0.1, columnspacing=0.6)
    ax[1].set_xticks([40, 75, 100, 150]); ax[1].set_xlabel("inner iterations n$_{inner}$"); ax[1].set_title("(b) Objective vs n$_{inner}$")
    ax[2].set_xlabel("entropy H(Q) (nats)"); ax[2].set_title("(c) Objective vs entropy")
    fig.tight_layout(); fig.savefig(OUT+"fig6.png", bbox_inches="tight"); plt.close(fig)
print("ok")
