"""Figuras do pipeline (painéis da tese), portadas sem mudança visual.

As dimensões da malha passam a ser lidas do próprio resultado (``res``) em vez
de constantes globais."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = {
    "micro": "#1f6FE0", "macro": "#E07B1B", "down": "#0E8B43",
    "phase": "#188F4E", "exact": "#3F3F3F", "accent": "#C12626",
}

def _hm(fig, ax, arr, title, cmap="viridis", vmin=None, vmax=None,
        macro=False, cbl=None):
    """Heatmap with title, axis labels and colorbar.
    macro=True uses macro-mesh coordinates (0..NC); macro=False uses
    micro-mesh coordinates (0..NF). Ticks derived from arr.shape[0]."""
    n = arr.shape[0]                              # usa dim REAL do array
    ext = [0, n, 0, n]
    if macro:
        xlab, ylab = "macro index i", "macro index j"
        step = max(1, n // 5)
    else:
        xlab, ylab = "micro index i", "micro index j"
        step = max(1, n // 5)
    ticks = np.arange(0, n + 1, step)
    im = ax.imshow(arr, origin="lower", cmap=cmap, extent=ext,
                   aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbl)


def plot_campos(res, fname):
    NF, NC, HF, HC = res["N_F"], res["N_C"], res["HF"], res["HC"]
    """Panel 1 -- 10 figures: pressure fields (mean and std) on the three
    representations and the absolute & relative error maps. Macro panels
    have axes 0..5 (macro index); downscaling has 0..15 (micro index)."""
    n = res["name"]
    eps = 1e-9
    abs_m = np.abs(res["Pmic_mean"] - res["Pdown_mean"])
    rel_m = 100.0 * abs_m / (np.abs(res["Pmic_mean"]) + eps)
    abs_s = np.abs(res["Pmic_std"] - res["Pdown_std"])
    rel_s = 100.0 * abs_s / (np.abs(res["Pmic_std"]) + eps)
    fig, ax = plt.subplots(2, 5, figsize=(21, 8.6))
    vmn = min(res["Pmic_mean"].min(), res["Pdown_mean"].min())
    vmx = max(res["Pmic_mean"].max(), res["Pdown_mean"].max())
    _hm(fig, ax[0, 0], res["Pmic_mean"], f"1.a  Mean pressure -- micro-mesh ({NF}x{NF})", "viridis", vmn, vmx)
    _hm(fig, ax[0, 1], res["Pmac_mean"], f"1.b  Mean pressure -- macro-mesh ({NC}x{NC}, DA)", "viridis", vmn, vmx, macro=True)
    _hm(fig, ax[0, 2], res["Pdown_mean"], f"1.c  Mean pressure -- LML downscaling ({NF}x{NF})", "viridis", vmn, vmx)
    _hm(fig, ax[0, 3], abs_m, "1.d  Absolute error -- mean", "viridis")
    _hm(fig, ax[0, 4], rel_m, "1.e  Relative error -- mean (%)", "viridis")
    smx = max(res["Pmic_std"].max(), res["Pdown_std"].max())
    _hm(fig, ax[1, 0], res["Pmic_std"], f"2.a  Std deviation -- micro-mesh ({NF}x{NF})", "viridis", 0.0, smx)
    _hm(fig, ax[1, 1], res["Pmac_std"], f"2.b  Std deviation -- macro-mesh ({NC}x{NC}, DA)", "viridis", 0.0, smx, macro=True)
    _hm(fig, ax[1, 2], res["Pdown_std"], f"2.c  Std deviation -- LML downscaling ({NF}x{NF})", "viridis", 0.0, smx)
    _hm(fig, ax[1, 3], abs_s, "2.d  Absolute error -- std", "viridis")
    _hm(fig, ax[1, 4], rel_s, "2.e  Relative error -- std (%)", "viridis")
    fig.suptitle(f"Panel 1 -- Problem {n}: pressure fields "
                 f"(second-order statistics) and error maps",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fname}")


def plot_da(res, fname, dkey="xx"):
    NF, NC, HF, HC = res["N_F"], res["N_C"], res["HF"], res["HC"]
    """Panel 3 -- DA results: figures 3.a-3.e are MEANS over the 25
    macroelements (with the 25 individual curves shown transparently);
    3.f is the codebook-size map. Direction in dkey={'xx','yy','xy'}.
    Enriched legends: K-S distance, theoretical T_c1 = 2*lambda_max(Cov),
    sub-sampling reference levels, sample sizes, cooling arrow."""
    n = res["name"]
    da = res[f"da_all_{dkey}"]
    label_K = {"xx": r"$K_{xx}$", "yy": r"$K_{yy}$", "xy": r"$K_{xy}$"}[dkey]
    H = np.stack([e["h"] for e in da])
    Hm = H.mean(axis=0)
    Tc_avg = float(np.mean([e["Tc"] for e in da]))
    Kfin = np.array([e["yk"].size for e in da])
    M_per_mac = np.array([e["samples"].size for e in da])
    M_mic_tot = int(M_per_mac.sum())

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.2))

    # ----- 3.a rate-distortion -----
    a = ax[0, 0]
    for e in da:
        a.plot(e["h"][:, 1], e["h"][:, 2], "-", color=PAL["micro"],
               lw=0.6, alpha=0.16)
    Hs = H.std(axis=0)
    # mean +- std bands (3.a): plot in D-I space using paired (D,I) curves
    a.plot(Hm[:, 1], Hm[:, 2] + Hs[:, 2], "--", color=PAL["micro"], lw=1.0,
           alpha=0.75, label="mean +/- std (25 macroelements)")
    a.plot(Hm[:, 1], np.maximum(Hm[:, 2] - Hs[:, 2], 0.0), "--",
           color=PAL["micro"], lw=1.0, alpha=0.75)
    a.fill_between(Hm[:, 1], np.maximum(Hm[:, 2] - Hs[:, 2], 0.0),
                   Hm[:, 2] + Hs[:, 2], color=PAL["micro"], alpha=0.15)
    a.plot(Hm[:, 1], Hm[:, 2], "-", color=PAL["micro"], lw=2.2,
           label="mean over 25 macroelements")
    a.set_xscale("log")
    a.set_xlabel(r"mean distortion $D = \mathbb{E}\,[(r(X) - r(Y))^2]$")
    a.set_ylabel(r"mutual information $I(Y;Q)$ [nats]")
    a.set_title(f"3.a  Rate-distortion curve ({label_K})", fontsize=9)
    a.legend(fontsize=8, framealpha=0.9, loc="lower left")

    # ----- 3.b phase transitions -----
    a = ax[0, 1]
    for e in da:
        a.plot(e["h"][:, 0], e["h"][:, 3], "-", color=PAL["phase"],
               lw=0.6, alpha=0.16)
    a.fill_between(Hm[:, 0], np.maximum(Hm[:, 3] - Hs[:, 3], 0.0),
                   Hm[:, 3] + Hs[:, 3], color=PAL["phase"], alpha=0.15,
                   label="mean +/- std (25 macroelements)")
    a.plot(Hm[:, 0], Hm[:, 3] + Hs[:, 3], "--", color=PAL["phase"],
           lw=1.0, alpha=0.75)
    a.plot(Hm[:, 0], np.maximum(Hm[:, 3] - Hs[:, 3], 0.0), "--",
           color=PAL["phase"], lw=1.0, alpha=0.75)
    a.plot(Hm[:, 0], Hm[:, 3], "-", color=PAL["phase"], lw=2.2,
           label=fr"mean K(T)  (final $\overline{{K}}$ = {Kfin.mean():.1f})")
    a.axvline(Tc_avg, ls="--", lw=1.3, color=PAL["accent"],
              label=fr"$T_{{c,1}} = 2\,\lambda_{{\max}}(\mathrm{{Cov}})$"
                    f" = {Tc_avg:.2e}")
    a.set_xscale("log"); a.invert_xaxis()
    a.set_xlabel(r"temperature $T$  (cooling left-to-right)")
    a.set_ylabel(r"realizations (codebook) size  $K(T)$")
    a.set_title(f"3.b  Phase transitions ({label_K})", fontsize=9)
    a.legend(fontsize=8, framealpha=0.9, loc="upper left")

    # ----- 3.c stochastic sub-sampling -----
    a = ax[1, 1] if False else ax[0, 2]
    for e in da:
        a.plot(e["h"][:, 0], e["h"][:, 4], "-", color=PAL["exact"],
               lw=0.6, alpha=0.16)
    a.plot(Hm[:, 0], Hm[:, 4], "-", color=PAL["exact"], lw=2.2,
           label="mean over 25 macroelements")
    m_lo = float(np.nanmin(Hm[:, 4])); m_hi = float(np.nanmax(Hm[:, 4]))
    a.axhline(m_lo, ls=":", lw=1.2, color=PAL["micro"],
              label=fr"$f_{{\mathrm{{lo}}}}\,M$ floor  ({m_lo:.0f})")
    a.axhline(m_hi, ls=":", lw=1.2, color=PAL["macro"],
              label=fr"$f_{{\mathrm{{hi}}}}\,M$ ceil   ({m_hi:.0f})")
    a.set_xscale("log"); a.invert_xaxis()
    a.set_xlabel(r"temperature $T$  (cooling left-to-right)")
    a.set_ylabel(r"realizations per step  $m(T)$")
    a.set_title(f"3.c  Stochastic sub-sampling schedule ({label_K})",
                fontsize=9)
    a.legend(fontsize=8, framealpha=0.9, loc="lower right")

    # ----- 3.d  CDF MEDIA: r(X)_micro vs r(Y)_DA (SEM amostragem) -----
    # range FISICO completo (apenas para definir eixo X)
    all_samples_for_range = np.concatenate([e["samples"] for e in da])
    lo, hi = float(all_samples_for_range.min()), float(all_samples_for_range.max())
    if hi <= lo: hi = lo + 1e-12
    xg = np.linspace(lo, hi, 400)
    # Para cada macroel, calcula r(X) (CDF empirica das amostras micro)
    # e r(Y) (CDF do DA via cumsum dos q_k). Depois faz a MEDIA ponto a ponto.
    cdfs_X = np.empty((len(da), xg.size))
    cdfs_Y = np.empty((len(da), xg.size))
    for k, e in enumerate(da):
        sX = np.sort(e["samples"])
        cdfs_X[k] = np.searchsorted(sX, xg, side="right") / len(sX)
        o = np.argsort(e["yk"]); sY = e["yk"][o]
        cs = np.cumsum(e["qk"][o]) / e["qk"].sum()
        idx = np.searchsorted(sY, xg, side="right")
        cdfs_Y[k] = np.where(idx > 0,
                             cs[np.clip(idx - 1, 0, len(cs) - 1)], 0.0)
    # MEDIA e STD ponto-a-ponto (NAO pooled, NAO amostragem)
    mic_m, mic_s = cdfs_X.mean(0), cdfs_X.std(0)
    mac_m, mac_s = cdfs_Y.mean(0), cdfs_Y.std(0)
    # K-S = sup |<r(X)> - <r(Y)>| das curvas medias
    ks_d = float(np.max(np.abs(mic_m - mac_m)))

    a = ax[1, 0]
    # bandas mean +- std (CDF fica em [0,1])
    a.fill_between(xg, np.clip(mic_m - mic_s, 0, 1),
                   np.clip(mic_m + mic_s, 0, 1),
                   color=PAL["micro"], alpha=0.22,
                   label=r"$\langle r(X)\rangle \pm \sigma$  (25 macroel.)")
    a.fill_between(xg, np.clip(mac_m - mac_s, 0, 1),
                   np.clip(mac_m + mac_s, 0, 1),
                   color=PAL["macro"], alpha=0.22,
                   label=r"$\langle r(Y)\rangle \pm \sigma$  (25 macroel.)")
    a.plot(xg, mic_m, "-",  color=PAL["micro"], lw=2.4,
           label=r"$\langle r(X)\rangle$  (mean micro)")
    a.plot(xg, mac_m, "--", color=PAL["macro"], lw=2.2,
           label=r"$\langle r(Y)\rangle$  (mean DA)")
    a.fill_between(xg, mic_m, mac_m, color=PAL["accent"], alpha=0.10,
                   label=fr"|$\Delta$| envelope  --  K-S = {ks_d:.3f}")
    a.set_ylim(-0.02, 1.04)
    a.set_xlabel(f"upscaled {label_K} (permeability)")
    a.set_ylabel(r"CDF  $r(\cdot)$")
    a.set_title(fr"3.d  Mean CDFs $\langle r(\cdot)\rangle \pm \sigma$ "
                fr"({label_K})  --  K-S = {ks_d:.3f}", fontsize=9)
    a.legend(fontsize=7.5, framealpha=0.9, loc="lower right")

    # ----- 3.e  PDFs medias +- std (derivada das 25 CDFs por macroel.) -----
    dx = xg[1] - xg[0]
    def _smooth(y, w=9):
        if w < 2 or w >= len(y): return y
        k = np.ones(w) / w
        return np.convolve(y, k, mode="same")
    # Deriva CADA UMA das 25 CDFs individualmente, depois media e std
    pdfs_X = np.empty_like(cdfs_X)
    pdfs_Y = np.empty_like(cdfs_Y)
    for k in range(len(da)):
        pdfs_X[k] = _smooth(np.gradient(cdfs_X[k], dx), 9)
        pdfs_Y[k] = _smooth(np.gradient(cdfs_Y[k], dx), 9)
    pdf_X_m, pdf_X_s = pdfs_X.mean(0), pdfs_X.std(0)
    pdf_Y_m, pdf_Y_s = pdfs_Y.mean(0), pdfs_Y.std(0)

    a = ax[1, 1]
    # bandas mean +- std (clipa em 0 pra PDF nao ir negativa)
    a.fill_between(xg, np.maximum(pdf_X_m - pdf_X_s, 0.0),
                   pdf_X_m + pdf_X_s, color=PAL["micro"], alpha=0.22,
                   label=r"$\langle p(X)\rangle \pm \sigma$  (25 macroel.)")
    a.fill_between(xg, np.maximum(pdf_Y_m - pdf_Y_s, 0.0),
                   pdf_Y_m + pdf_Y_s, color=PAL["macro"], alpha=0.22,
                   label=r"$\langle p(Y)\rangle \pm \sigma$  (25 macroel.)")
    a.plot(xg, pdf_X_m, "-",  color=PAL["micro"], lw=2.4,
           label=r"$\langle p(X)\rangle$  (mean micro PDF)")
    a.plot(xg, pdf_Y_m, "--", color=PAL["macro"], lw=2.2,
           label=r"$\langle p(Y)\rangle$  (mean DA PDF)")
    a.set_xlabel(f"upscaled {label_K} (permeability)")
    a.set_ylabel(r"PDF  $p(\cdot) = \mathrm{d}r/\mathrm{d}y$")
    a.set_title(fr"3.e  Mean PDFs $\langle p(\cdot)\rangle \pm \sigma$  "
                fr"({label_K})", fontsize=9)
    a.legend(fontsize=7.5, framealpha=0.9, loc="upper right")
    ks_cdf = ks_d  # reusado pelo subtitle

    # ----- 3.f codebook-size map -----
    a = ax[1, 2]
    nm = res[f"nmac_map_{dkey}"]
    Nc_cache = nm.shape[0]                            # usa dim do cache, nao global
    im = a.imshow(nm, origin="lower", cmap="Blues",
                  extent=[0, Nc_cache, 0, Nc_cache], aspect="equal")
    thr = 0.5 * (nm.min() + nm.max())
    for I in range(Nc_cache):
        for J in range(Nc_cache):
            a.text(J + 0.5, I + 0.5, str(nm[I, J]),
                   ha="center", va="center", fontsize=9, fontweight="bold",
                   color="white" if nm[I, J] > thr else "#1A1A1A")
    a.set_xticks(np.arange(Nc_cache + 1)); a.set_yticks(np.arange(Nc_cache + 1))
    a.set_title(fr"3.f  $n_R$ map ({label_K})  --  "
                fr"min = {int(nm.min())}, mean = {nm.mean():.1f}, "
                fr"max = {int(nm.max())}", fontsize=9)
    a.set_xlabel("macro index i"); a.set_ylabel("macro index j")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04,
                 label="no. of macro realiz.")

    # Sub-titulo dinamico com N_R, n_R, alpha (lidos do cache, nao hardcoded)
    NR_eff = res.get("N_R", "?")
    nR_eff = res.get("n_R", "?")
    fig.suptitle(f"Panel 3 -- Problem {n}: Deterministic Annealing "
                 f"(means over {NC*NC} macroel., dir {label_K})  --  "
                 f"$N_R$={NR_eff}, $n_R$={nR_eff}, "
                 fr"$\alpha$={_fmt(res.get('da_params', {}).get('alpha'))}, $K_{{\max}}$={res.get('da_params', {}).get('K_max', '?')}",
                 fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fname}")


def plot_lml(res, fname):
    NF, NC, HF, HC = res["N_F"], res["N_C"], res["HF"], res["HC"]
    """Panel 4 -- LML downscaling: micro reference (NFxNF) + macro input
    (NCxNC) + LML output (NFxNF) + scatter quality. 8 sub-figures, 2x4."""
    n = res["name"]
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    # ---- linha 1: MEAN ----
    _hm(fig, ax[0, 0], res["Pmic_mean"], f"1.a  Reference micro-mesh -- "
        f"mean ({NF}x{NF})")
    _hm(fig, ax[0, 1], res["Pmac_mean"], f"1.b  Input macro-mesh -- "
        f"mean ({NC}x{NC})", macro=True)
    _hm(fig, ax[0, 2], res["Pdown_mean"], f"1.c  LML output -- "
        f"mean ({NF}x{NF})")
    a = ax[0, 3]
    xm, ym = res["Pmic_mean"].ravel(), res["Pdown_mean"].ravel()
    a.scatter(xm, ym, s=12, color=PAL["down"], alpha=0.6, label="cells")
    lim = [min(xm.min(), ym.min()), max(xm.max(), ym.max())]
    a.plot(lim, lim, "k--", lw=1, label="perfect agreement")
    r2 = 1.0 - np.sum((xm - ym) ** 2) / np.sum((xm - xm.mean()) ** 2)
    a.set_title(f"1.d  LML vs micro -- mean  (R2 = {r2:.4f})", fontsize=9)
    a.set_xlabel("mean -- micro-mesh"); a.set_ylabel("mean -- LML")
    a.legend(fontsize=8, framealpha=0.9); a.set_aspect("equal", "box")
    # ---- linha 2: STD ----
    _hm(fig, ax[1, 0], res["Pmic_std"], f"2.a  Reference micro-mesh -- "
        f"std ({NF}x{NF})", "viridis")
    _hm(fig, ax[1, 1], res["Pmac_std"], f"2.b  Input macro-mesh -- "
        f"std ({NC}x{NC})", "viridis", macro=True)
    _hm(fig, ax[1, 2], res["Pdown_std"], f"2.c  LML output -- "
        f"std ({NF}x{NF})", "viridis")
    a = ax[1, 3]
    xs, ys = res["Pmic_std"].ravel(), res["Pdown_std"].ravel()
    a.scatter(xs, ys, s=12, color=PAL["phase"], alpha=0.6, label="cells")
    lim = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
    a.plot(lim, lim, "k--", lw=1, label="perfect agreement")
    r2s = 1.0 - np.sum((xs - ys) ** 2) / np.sum((xs - xs.mean()) ** 2)
    a.set_title(f"2.d  LML vs micro -- std  (R2 = {r2s:.4f})", fontsize=9)
    a.set_xlabel("std -- micro-mesh"); a.set_ylabel("std -- LML")
    a.legend(fontsize=8, framealpha=0.9); a.set_aspect("equal", "box")
    fig.suptitle(f"Panel 4 -- Problem {n}: LML downscaling  "
                 f"(mean sigma^2 = {res['sig2_mean']:.1e}, "
                 f"std sigma^2 = {res['sig2_std']:.1e})",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fname}")



def plot_erros(res, fname):
    NF, NC, HF, HC = res["N_F"], res["N_C"], res["HF"], res["HC"]
    n = res["name"]
    abs_m = np.abs(res["Pmic_mean"] - res["Pdown_mean"])
    abs_s = np.abs(res["Pmic_std"]  - res["Pdown_std"])
    rms_m = np.sqrt(np.mean((res["Pmic_mean"] - res["Pdown_mean"])**2))
    rms_s = np.sqrt(np.mean((res["Pmic_std"]  - res["Pdown_std"])**2))
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6))
    XF = (np.arange(NF) + 0.5) * HF
    XC = (np.arange(NC) + 0.5) * HC
    def cut(a, mic, mac, dow, vert, ttl, ylab):
        if vert:
            a.plot(mic[:, NF // 2], XF, "-o", ms=3, color=PAL["micro"], label="micro-mesh")
            a.plot(mac[:, NC // 2], XC, "s--", ms=7, color=PAL["macro"], label="macro-mesh (DA)")
            a.plot(dow[:, NF // 2], XF, "-", lw=1.7, color=PAL["down"], label="LML downscaling")
            a.set_xlabel(ylab); a.set_ylabel("y (m)")
        else:
            a.plot(XF, mic[NF // 2], "-o", ms=3, color=PAL["micro"], label="micro-mesh")
            a.plot(XC, mac[NC // 2], "s--", ms=7, color=PAL["macro"], label="macro-mesh (DA)")
            a.plot(XF, dow[NF // 2], "-", lw=1.7, color=PAL["down"], label="LML downscaling")
            a.set_xlabel("x (m)"); a.set_ylabel(ylab)
        a.set_title(ttl, fontsize=9); a.legend(fontsize=8, framealpha=0.9)
    cut(ax[0,0], res["Pmic_mean"], res["Pmac_mean"], res["Pdown_mean"], False,
        "1.a  Mean pressure -- horizontal cut (y = L/2)", "mean pressure")
    cut(ax[0,1], res["Pmic_mean"], res["Pmac_mean"], res["Pdown_mean"], True,
        "1.b  Mean pressure -- vertical cut (x = L/2)", "mean pressure")
    a = ax[0,2]
    a.hist(abs_m.ravel(), bins=30, color=PAL["micro"], alpha=0.85)
    a.axvline(rms_m, ls="--", color=PAL["accent"], lw=1.4, label=f"SRSS = {rms_m:.4f}")
    a.set_title("1.c  Histogram -- error of the mean", fontsize=9)
    a.set_xlabel("absolute error |micro - LML|"); a.set_ylabel("no. of cells")
    a.legend(fontsize=8, framealpha=0.9)
    cut(ax[1,0], res["Pmic_std"], res["Pmac_std"], res["Pdown_std"], False,
        "2.a  Std deviation -- horizontal cut (y = L/2)", "std deviation")
    cut(ax[1,1], res["Pmic_std"], res["Pmac_std"], res["Pdown_std"], True,
        "2.b  Std deviation -- vertical cut (x = L/2)", "std deviation")
    a = ax[1,2]
    a.hist(abs_s.ravel(), bins=30, color=PAL["phase"], alpha=0.85)
    a.axvline(rms_s, ls="--", color=PAL["accent"], lw=1.4, label=f"SRSS = {rms_s:.4f}")
    a.set_title("2.c  Histogram -- error of the std", fontsize=9)
    a.set_xlabel("absolute error |micro - LML|"); a.set_ylabel("no. of cells")
    a.legend(fontsize=8, framealpha=0.9)
    fig.suptitle(f"Panel 2 -- Problem {n}: field cuts and distribution of LML downscaling errors",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {fname}")
def plot_geostat(K_micro, fname, lx=15.0):
    NF = K_micro.shape[1]; LX = LY = lx
    """Geostatistics: adopted autocorrelation function and sample fields."""
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    lag = np.linspace(0, 8, 200)
    for ell in (2.0, 3.0, 4.5):
        ax[0].plot(lag, np.exp(-(lag / ell) ** 2), lw=1.8,
                   label=fr"$\ell$ = {ell}")
    ax[0].set_title("Adopted autocorrelation function\n"
                    r"$\rho(\Delta)=\exp[-(\Delta/\ell)^2]$ (2-D Gaussian)")
    ax[0].set_xlabel(r"lag $\Delta$ (cells)")
    ax[0].set_ylabel(r"autocorrelation $\rho$")
    ax[0].legend(framealpha=0.9, title="correlation length")
    for idx, a in zip([0, 1, 2], ax[1:]):
        im = a.imshow(K_micro[idx], origin="lower", cmap="turbo",
                      extent=[0, LX, 0, LY], aspect="equal")
        a.set_title(f"Absolute permeability -- realization {idx + 1}")
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label="k")
    fig.suptitle(f"Application 1 -- geostatistical absolute permeability "
                 f"field (micro-mesh {NF}x{NF})", fontsize=12,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fname}")


# ---------------------------------------------------------------------------
# 5 panels per macroelement -- NC x NC subplots, one per macroelement.
# These are parametrized by the direction dkey in {"xx","yy","xy"} so the
# same code produces panels for K_xx, K_yy and K_xy.
# ---------------------------------------------------------------------------

def _grid25(da_all, fname, draw, suptitle):
    NC = int(round(np.sqrt(len(da_all))))
    """Build a NC x NC figure: one subplot per macroelement (I=0 at bottom)."""
    cell_w, cell_h = 3.1, 2.8
    fig, axes = plt.subplots(NC, NC, figsize=(cell_w * NC, cell_h * NC))
    fig.subplots_adjust(left=0.052, right=0.985, bottom=0.05, top=0.925,
                        hspace=0.62, wspace=0.42)
    for e in da_all:
        I, J = e["IJ"]
        ax = axes[NC - 1 - I, J]
        draw(ax, e)
        ax.set_title(f"macroelement ({I},{J})", fontsize=7.8)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"  saved -> {fname}")


def _ecdf(v):
    v = np.sort(v)
    return v, np.arange(1, len(v) + 1) / len(v)


def plot_rd_25(da_all, label_K, fname, name):
    """5xx/5yy/5xy -- rate-distortion curve per macroelement."""
    def draw(ax, e):
        h = e["h"]
        ax.plot(h[:, 1], h[:, 2], "-", color=PAL["micro"], lw=1.2)
        ax.scatter(h[:, 1], h[:, 2], c=PAL["micro"], s=5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("distortion D"); ax.set_ylabel("I [nats]")
    _grid25(da_all, fname, draw,
            f"Problem {name}: rate-distortion curve per macroelement "
            f"(direction {label_K})")


def plot_codebook_25(da_all, label_K, fname, name):
    """6xx/6yy/6xy -- DA realizations vs micro samples per macroelement."""
    def draw(ax, e):
        s, yk, qk = e["samples"], e["yk"], e["qk"]
        cnt, edg = np.histogram(s, bins=18, density=True)
        ax.bar(edg[:-1], cnt, width=np.diff(edg), align="edge",
               alpha=0.35, color=PAL["micro"])
        peak = cnt.max() if cnt.max() > 0 else 1.0
        scale = peak / (qk.max() + 1e-12)
        ax.vlines(yk, 0, qk * scale, color=PAL["macro"], lw=1.1)
        ax.plot(yk, qk * scale, "o", color=PAL["macro"], ms=2.6)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(f"upscaled {label_K}"); ax.set_ylabel("density")
    _grid25(da_all, fname, draw,
            f"Problem {name}: DA realizations vs micro samples "
            f"per macroelement (direction {label_K})")


def plot_phase_25(da_all, label_K, fname, name):
    """7xx/7yy/7xy -- phase transitions during annealing per macroelement."""
    def draw(ax, e):
        h = e["h"]
        ax.plot(h[:, 0], h[:, 3], "-", color=PAL["phase"], lw=1.2)
        ax.scatter(h[:, 0], h[:, 3], c=PAL["phase"], s=5, zorder=3)
        ax.axvline(e["Tc"], ls="--", lw=1.0, color=PAL["accent"])
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("T (cooling ->)"); ax.set_ylabel("macro realizations K")
    _grid25(da_all, fname, draw,
            f"Problem {name}: phase transitions during annealing per "
            f"macroelement (direction {label_K}; dashed = T_c,1)")


def plot_cdf_25(da_all, label_K, fname, name):
    """8xx/8yy/8xy -- exact r(X) vs upscaled r(Y) CDF per macroelement."""
    def draw(ax, e):
        s, yk, qk = e["samples"], e["yk"], e["qk"]
        xe, ye = _ecdf(s)
        ax.step(xe, ye, where="post", color=PAL["micro"], lw=1.7,
                label="exact r(X)")
        o = np.argsort(yk)
        ys = yk[o]; cs = np.cumsum(qk[o]) / qk.sum()
        ax.step(np.r_[ys[0], ys], np.r_[0.0, cs], where="post",
                color=PAL["macro"], lw=1.5, ls="--", label="upscaled r(Y)")
        ax.set_xlabel(f"{label_K} (permeability / response)")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=5.5, framealpha=0.9, loc="lower right")
    _grid25(da_all, fname, draw,
            f"Problem {name}: exact r(X) vs upscaled r(Y) CDF per "
            f"macroelement (direction {label_K})")


def plot_subsampling_25(da_all, label_K, fname, name):
    """9xx/9yy/9xy -- stochastic sub-sampling schedule per macroelement."""
    def draw(ax, e):
        h = e["h"]
        ax.plot(h[:, 0], h[:, 4], "-", color=PAL["exact"], lw=1.2)
        ax.scatter(h[:, 0], h[:, 4], c=PAL["exact"], s=5, zorder=3)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("T (cooling ->)"); ax.set_ylabel("batch m(T)")
    _grid25(da_all, fname, draw,
            f"Problem {name}: stochastic sub-sampling schedule per "
            f"macroelement (direction {label_K})")


def _fmt(v):
    return "?" if v is None else f"{v:.4f}"
