from fit import *
import numpy as np
cache={}; imgs=root.findall("image")
def ev(cfg):
    s=[run(im,cfg,cache) for im in imgs]; return 0.6*np.mean([x[0] for x in s])+0.4*np.mean([x[1] for x in s]), s
base=dict(idx="neg_a",thr=4,blur=2,close=0,open=0,half=0.30,minA=0.19)
res=[]
for thr in [3,3.5,4,4.5,5]:
    for blur in [1,1.5,2,2.5,3]:
        c=dict(base,thr=thr,blur=blur); sc,s=ev(c); res.append((sc,thr,blur,s))
res.sort(key=lambda x:-x[0])
for r in res[:6]: print("thr %.1f blur %.1f score %.3f"%(r[1],r[2],r[0]), [(round(a,3),round(b,3),n) for a,b,n in r[3]])
best=dict(base,thr=res[0][1],blur=res[0][2])
for h in [0.25,0.28,0.30,0.32,0.35]:
    print("half",h, round(ev(dict(best,half=h))[0],3))
for mA in [0.1,0.15,0.19,0.25]:
    print("minA",mA, round(ev(dict(best,minA=mA))[0],3))
