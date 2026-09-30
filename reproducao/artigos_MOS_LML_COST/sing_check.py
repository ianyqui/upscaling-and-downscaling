import os; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, json, tese_eliptico as te
def inflow(p, kx, ky, bc):
    Q = 0.0
    for face, (kind, val) in bc.items():
        if kind == "D" and val == 1.0:
            if face == "esq": Q += np.sum(2*kx[:, 0]*(1-p[:, 0]))
            elif face == "dir": Q += np.sum(2*kx[:, -1]*(1-p[:, -1]))
            elif face == "baixo": Q += np.sum(2*ky[0, :]*(1-p[0, :]))
            else: Q += np.sum(2*ky[-1, :]*(1-p[-1, :]))
    return Q
def midflux(p, kx, ky, vl):
    h = p.shape[0]//2
    if vl:
        T = 2*kx[:, h-1]*kx[:, h]/(kx[:, h-1]+kx[:, h]); return float(abs(np.sum(T*(p[:, h-1]-p[:, h]))))
    T = 2*ky[h-1, :]*ky[h, :]/(ky[h-1, :]+ky[h, :]); return float(abs(np.sum(T*(p[h-1, :]-p[h, :]))))
out={}
for s,vl in [("1.1",True),("1.3",True),("1.4",True),("1.2",False)]:
    bc=te.PROBLEMAS[s]; out[s]={}
    for n in (10,20,40,60,120,240,480):
        k=np.ones((n,n)); p=te.tpfa(k,k,bc,te.LX/n)
        out[s][n]=(round(inflow(p,k,k,bc),4),round(midflux(p,k,k,vl),4))
    print(s,out[s])
json.dump(out,open("master_singularity.json","w"),indent=1)
