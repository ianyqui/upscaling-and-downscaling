"""Rate-distortion bounds, information plane and free-energy decomposition from the cached DA histories."""
import json, pickle, numpy as np
from scipy.special import digamma
from scipy.stats import skew, kurtosis
def kl_entropy(x, k=4):
    x = np.sort(np.asarray(x, float)); N = x.size
    # distance to k-th nearest neighbour in 1-D
    eps = np.empty(N)
    for i in range(N):
        lo, hi = max(0, i-k), min(N, i+k+1)
        d = np.sort(np.abs(x[lo:hi]-x[i]))[k]
        eps[i] = max(d, 1e-12)
    return float(digamma(N)-digamma(k)+np.log(2.0)+np.mean(np.log(eps)))
def main():
  out = {}
  for name in ("1.1", "1.3", "1.4"):
      res = pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb")); o = {}
      for d in ("xx", "yy", "xy"):
          cells = []
          for e in res[f"da_all_{d}"]:
              s = e["samples"]; v = float(np.var(s)); h = kl_entropy(s)
              negent = 0.5*np.log(2*np.pi*np.e*v)-h
              H = e["h"]; T, D, I, K, m = H.T
              slb = np.maximum(0, h-0.5*np.log(2*np.pi*np.e*np.maximum(D, 1e-300)))
              gub = np.maximum(0, 0.5*np.log(v/np.maximum(D, 1e-300)))
              tol = 0.02
              inside = (I >= slb-tol) & (I <= gub+tol)
              below = I < slb-tol; above = I > gub+tol
              p = e["qk"][e["qk"] > 0]
              cells.append(dict(IJ=list(e["IJ"]), var=v, h=h, negentropy=float(negent), skew=float(skew(s)), exkurt=float(kurtosis(s)),
                                frac_inside=float(inside.mean()), frac_below_slb=float(below.mean()), frac_above_g=float(above.mean()),
                                D_cross=float(D[below].max()/v) if below.any() else None,
                                D_final_rel=float(D[-1]/v), I_final=float(I[-1]), K_final_hist=int(K[-1]), lnm_final=float(np.log(m[-1])),
                                K=int(len(e["yk"])), Hq=float(-(p*np.log(p)).sum()),
                                T=T.tolist(), D=D.tolist(), I=I.tolist(), Kh=K.astype(int).tolist()))
          o[d] = cells
          a = lambda k: np.array([c[k] for c in cells], float)
          print(name, d, f"negent {a('negentropy').mean():.3f}  skew {a('skew').mean():.2f}  inside {a('frac_inside').mean():.3f} below {a('frac_below_slb').mean():.3f} above {a('frac_above_g').mean():.3f}  Dcross/var {np.nanmean([c['D_cross'] or np.nan for c in cells]):.2e}  Dfin/var {a('D_final_rel').mean():.2e} Ifin {a('I_final').mean():.2f} lnm {a('lnm_final').mean():.2f} K {a('K').mean():.1f} Hq {a('Hq').mean():.2f}")
      out[name] = o
  json.dump(out, open("main_rd.json", "w"))

if __name__ == "__main__":
    main()
