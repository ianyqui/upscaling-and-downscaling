"""Problem-setup figures of the cost article: boundary conditions with the reference pressure
statistics of the four scenarios, and the fine and coarse meshes with one permeability realization.

Usage (from reproducao/artigos_MOS_LML_COST):
    python ../figuras_artigos/cost_problem_figures.py OUT_DIR
Regenerates the 500 fields with the seed of cost_study.py and solves the 250 held-out
realizations of each scenario on the 120 x 120 mesh (about one minute).
"""
import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "legacy"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tese_eliptico as te

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs_cost"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "savefig.dpi": 300, "figure.dpi": 150})
RED, BLUE, GREY = "#d62728", "#1f77b4", "#7f7f7f"

NF, NC = 120, 10
te.NF, te.NC = NF, NC; te.ALPHA = NF // NC; te.HF = te.LX / NF; te.HC = te.LX / NC
L = te.LX


def spectral_fields(N, n, ell, sigma, seed):
    """Identical to cost_study.py (circulant embedding, one global variance normalization)."""
    rng = np.random.default_rng(seed); M = 2 * n; x = np.arange(M); d = np.minimum(x, M - x)
    DX, DY = np.meshgrid(d, d, indexing="ij"); S = np.clip(np.fft.fft2(np.exp(-(DX / ell) ** 2 - (DY / ell) ** 2)).real, 0, None)
    sq = np.sqrt(S); Z = np.empty((N, n, n))
    for i in range(N):
        w = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
        Z[i] = np.fft.ifft2(sq * w).real[:n, :n] * M / np.sqrt(M * M)
    Z /= Z.std()
    return np.exp(sigma * Z)


Kf = spectral_fields(500, NF, 3.0, 0.9, 2026120)
SCEN = [("S1", "1.1", "v"), ("S2", "1.3", "v"), ("S3", "1.4", "v"), ("S4", "1.2", "h")]
REF = {}
_cache = os.path.join(OUT, "cost_reference_fields.npz")
if os.path.exists(_cache):
    _z = np.load(_cache); REF = {n: (_z[f"{n}_mean"], _z[f"{n}_sd"]) for n, _, _ in SCEN}
for name, key, _ in ([] if REF else SCEN):
    bc = te.PROBLEMAS[key]
    P = np.array([te.tpfa(Kf[r], Kf[r], bc, te.HF) for r in range(250, 500)])
    REF[name] = (P.mean(0), P.std(0))
    print(name, "done", flush=True)
np.savez_compressed(_cache, **{f"{k}_mean": v[0] for k, v in REF.items()},
                    **{f"{k}_sd": v[1] for k, v in REF.items()})


def edges(ax, bc, mid):
    seg = {"esq": ([0, 0], [0, L]), "dir": ([L, L], [0, L]), "baixo": ([0, L], [0, 0]), "cima": ([0, L], [L, L])}
    for face, (kind, val) in bc.items():
        xs, ys = seg[face]
        if kind == "D":
            ax.plot(xs, ys, color=RED if val == 1.0 else BLUE, lw=4, solid_capstyle="butt", clip_on=False, zorder=5)
        else:
            ax.plot(xs, ys, color=GREY, lw=2.2, ls=(0, (2, 1.5)), clip_on=False, zorder=5)
    if mid == "v":
        ax.plot([L / 2, L / 2], [0, L], color="white", lw=1.2, ls=":", zorder=4)
    else:
        ax.plot([0, L], [L / 2, L / 2], color="white", lw=1.2, ls=":", zorder=4)


# Figure: boundary conditions and reference statistics
fig, ax = plt.subplots(2, 4, figsize=(7.4, 4.3))
for j, (name, key, mid) in enumerate(SCEN):
    m, s = REF[name]
    im0 = ax[0, j].imshow(m, origin="lower", extent=[0, L, 0, L], cmap="viridis", vmin=0, vmax=1)
    im1 = ax[1, j].imshow(s, origin="lower", extent=[0, L, 0, L], cmap="magma", vmin=0, vmax=max(REF[n][1].max() for n in REF))
    for i in range(2):
        edges(ax[i, j], te.PROBLEMAS[key], mid)
        ax[i, j].set_xticks([0, 5, 10, 15]); ax[i, j].set_yticks([0, 5, 10, 15]); ax[i, j].tick_params(labelsize=6.5)
        if j:
            ax[i, j].set_yticklabels([])
    ax[0, j].set_title(name); ax[0, j].set_xticklabels([])
    ax[1, j].set_xlabel("x (m)", fontsize=7)
ax[0, 0].set_ylabel("mean pressure\ny (m)", fontsize=7.5); ax[1, 0].set_ylabel("standard deviation\ny (m)", fontsize=7.5)
fig.subplots_adjust(left=0.08, right=0.9, top=0.93, bottom=0.2, wspace=0.08, hspace=0.18)
c0 = fig.add_axes([0.915, 0.58, 0.015, 0.33]); fig.colorbar(im0, cax=c0).ax.tick_params(labelsize=6.5)
c1 = fig.add_axes([0.915, 0.2, 0.015, 0.33]); fig.colorbar(im1, cax=c1).ax.tick_params(labelsize=6.5)
h = [plt.Line2D([], [], color=RED, lw=4), plt.Line2D([], [], color=BLUE, lw=4), plt.Line2D([], [], color=GREY, lw=2.2, ls=(0, (2, 1.5))),
     plt.Line2D([], [], color="k", lw=1.2, ls=":")]
fig.legend(h, ["u = 1 (Dirichlet)", "u = 0 (Dirichlet)", "no flow", "section of the flux"], loc="lower center", ncol=4,
           fontsize=7, frameon=True, bbox_to_anchor=(0.5, 0.02))
fig.savefig(os.path.join(OUT, "fig_problem.png"), bbox_inches="tight"); plt.close(fig)

# Figure: meshes and one realization
k0 = Kf[0]; Kxx, Kyy, Kxy = te.upscale_realisation(k0, te.HF)
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
lk = np.log(k0); lo, hi = lk.min(), lk.max()
im = ax[0].imshow(lk, origin="lower", extent=[0, L, 0, L], cmap="cividis", vmin=lo, vmax=hi)
for g in np.linspace(0, L, NC + 1):
    ax[0].axvline(g, color="white", lw=0.5); ax[0].axhline(g, color="white", lw=0.5)
I, J = 6, 3; hc = L / NC
ax[0].add_patch(Rectangle((J * hc, I * hc), hc, hc, fill=False, ec=RED, lw=1.5))
ax[0].set_title("(a) ln κ, fine mesh"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03).ax.tick_params(labelsize=6.5)
a = te.ALPHA; sub = lk[I * a:(I + 1) * a, J * a:(J + 1) * a]
im = ax[1].imshow(sub, origin="lower", extent=[J * hc, (J + 1) * hc, I * hc, (I + 1) * hc], cmap="cividis", vmin=lo, vmax=hi)
for g in np.linspace(0, hc, a + 1):
    ax[1].axvline(J * hc + g, color="white", lw=0.4); ax[1].axhline(I * hc + g, color="white", lw=0.4)
for sp in ax[1].spines.values():
    sp.set_color(RED); sp.set_linewidth(1.5)
ax[1].set_title("(b) one coarse cell"); ax[1].set_xlabel("x (m)")
ax[1].set_xticks([J * hc, (J + 1) * hc]); ax[1].set_yticks([I * hc, (I + 1) * hc])
im = ax[2].imshow(np.log(Kxx), origin="lower", extent=[0, L, 0, L], cmap="cividis", vmin=lo, vmax=hi)
ax[2].add_patch(Rectangle((J * hc, I * hc), hc, hc, fill=False, ec=RED, lw=1.5))
ax[2].set_title("(c) ln K$_{xx}$, coarse mesh"); ax[2].set_xlabel("x (m)")
fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03).ax.tick_params(labelsize=6.5)
for x in ax:
    x.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mesh.png"), bbox_inches="tight"); plt.close(fig)
print("ok")
