import pickle, json, numpy as np
import tese_eliptico as te
def sat_frac(s, n_inner, seed, alpha=0.953, K_max=50, perturb=4.6e-3, merge_atol=3.8e-4, f_lo=0.76):
    rng=np.random.default_rng(seed); sn=np.asarray(s,float); M=sn.size; sd=sn.std()+1e-12
    Tc=2*np.var(sn,ddof=1); T=2*Tc; Tmin=3e-3*Tc; N=int(np.ceil(np.log(Tmin/T)/np.log(alpha)))
    y=np.array([sn.mean()]); q=np.array([1.0]); step=0; sat=0; tot=0
    while T>Tmin:
        frac=f_lo+(1-f_lo)*min(step/N,1); m=max(int(round(frac*M)),40); bt=sn[rng.choice(M,m,replace=False)] if m<M else sn
        conv=False
        for _ in range(n_inner):
            d=(bt[:,None]-y[None,:])**2; lw=np.log(q+1e-300)[None,:]-d/T; lw-=lw.max(1,keepdims=True)
            P=np.exp(lw); P/=P.sum(1,keepdims=True); qn=P.mean(0); yn=(P*bt[:,None]).sum(0)/(qn*m+1e-300)
            if np.max(np.abs(yn-y))+np.max(np.abs(qn-q))<1e-7: y,q=yn,qn; conv=True; break
            y,q=yn,qn
        sat+= (not conv); tot+=1
        keep=q>1e-6; y,q=y[keep],q[keep]; o=np.argsort(y); y,q=y[o],q[o]
        ky,kq=[y[0]],[q[0]]
        for i in range(1,len(y)):
            if y[i]-ky[-1]<merge_atol: kq[-1]+=q[i]
            else: ky.append(y[i]); kq.append(q[i])
        y=np.array(ky); q=np.array(kq); q/=q.sum(); T*=alpha; step+=1
        if 2*len(y)<=K_max: y=np.concatenate([y,y+perturb*sd*rng.standard_normal(len(y))]); q=np.concatenate([q,q])/2
    return sat/tot, tot
res=pickle.load(open("_cache_P1_4_NR1000_v14.pkl","rb")); out={}
for d in ("xx","xy"):
    e=max(res[f"da_all_{d}"],key=lambda e:e["samples"].var())
    for ni in (40,150):
        f,t=sat_frac(e["samples"],ni,1); out[f"{d}_{ni}"]=dict(frac=f,steps=t); print(d,ni,round(f,3),t,flush=True)
json.dump(out,open("picard_check.json","w"),indent=1)
