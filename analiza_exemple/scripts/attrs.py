import numpy as np, cv2, tifffile
from axes import *
def band(a_line, b_line, off=0.30):
    # quad between line a shifted toward b and line b shifted toward a; both lines as 2-pt segments
    da=a_line[1]-a_line[0]; na=np.array([-da[1],da[0]])/np.linalg.norm(da)
    if (b_line.mean(0)-a_line.mean(0))@na<0: na=-na
    db=b_line[1]-b_line[0]
    if db@da<0: b_line=b_line[::-1]
    nb=na
    return np.array([a_line[0]+na*off/G, a_line[1]+na*off/G, b_line[1]-nb*off/G, b_line[0]-nb*off/G])
def order(rows):
    d=rows[0][1]-rows[0][0]; n=np.array([-d[1],d[0]])/np.linalg.norm(d)
    return sorted(rows,key=lambda r:r.mean(0)@n)
def ir_mask(rows):
    m=np.zeros((H,H),np.uint8); rows=order(rows)
    for a,b in zip(rows[:-1],rows[1:]):
        cv2.fillPoly(m,[np.round(band(a,b)*16).astype(np.int32)],1,shift=4)
    return m
def row_gap(lab_mask, r):
    # max gap (m) along axis r between canopy px within corridor, incl. ends to axis ends
    d=r[1]-r[0]; L=np.linalg.norm(d); u=d/L; nn=np.array([-u[1],u[0]])
    ys,xs=np.nonzero(lab_mask); P=np.stack([xs,ys],1)-r[0]
    t=P@u; o=np.abs(P@nn); sel=(o<0.3/G)&(t>=0)&(t<=L)
    if sel.sum()==0: return L*G
    occ=np.zeros(int(L)+1,bool); occ[t[sel].astype(int)]=True
    best=cur=0
    for v in occ:
        cur=0 if v else cur+1; best=max(best,cur)
    return best*G
for img in root.findall("image"):
    n=img.get("name"); im=tifffile.imread(B+"images/"+n)
    ref_rows=[pts(e.get("points")) for e in img.findall("polyline")]
    irs=[(pts(e.get("points")),{a.get("name"):a.text for a in e.findall("attribute")}) for e in img.findall("polygon") if e.get("label")=="interrow_area"]
    cans=[pts(e.get("points")) for e in img.findall("polygon") if e.get("label")=="vineyard"]
    refc=np.zeros((H,H),np.uint8)
    for p in cans: cv2.fillPoly(refc,[np.round(p).astype(np.int32)],1)
    refir=np.zeros((H,H),np.uint8)
    for p,_ in irs: cv2.fillPoly(refir,[np.round(p).astype(np.int32)],1)
    m=veg_mask(im); rows,_=detect_rows(m)
    for nm,rr in [("refaxes",ref_rows),("predaxes",rows)]:
        pm=ir_mask(rr); iou=(pm&refir).sum()/(pm|refir).sum()
        print(f"{n} {nm}: interrow IoU={iou:.3f} area pred={pm.sum()*G*G:.0f} ref={refir.sum()*G*G:.0f} m2")
    # interrow cover: veg fraction (a* mask) in each ref interrow
    print(" cover label vs veg fraction:")
    for p,a in irs:
        mm=np.zeros((H,H),np.uint8); cv2.fillPoly(mm,[np.round(p).astype(np.int32)],1)
        for thr in [4]:
            pass
        print("   %-9s veg=%.2f"%(a["interrow_cover"], m[mm>0].mean()))
    # row structure: gap using ref canopies and predicted canopies
    predlab=(canopies(m,ref_rows)>0).astype(np.uint8)
    print(" row_structure: label | maxgap(ref canopies) | maxgap(pred canopies, ref axes)")
    for e,r in zip(img.findall("polyline"),ref_rows):
        lab={a.get("name"):a.text for a in e.findall("attribute")}["row_structure"]
        print("   %-9s %5.1f %5.1f"%(lab,row_gap(refc,r),row_gap(predlab,r)))
