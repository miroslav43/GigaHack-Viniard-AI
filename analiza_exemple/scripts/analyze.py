import xml.etree.ElementTree as ET, numpy as np, cv2, tifffile, collections, json, sys
BASE = "/Users/maleticimiroslav/Vin Gigahack/data & info/05_examples/siret3_examples_cvat/"
GSD = 0.025
root = ET.parse(BASE + "annotations.xml").getroot()

def pts(s):
    return np.array([[float(v) for v in p.split(",")] for p in s.split(";")])

def area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def attrs(el):
    return {a.get("name"): a.text for a in el.findall("attribute")}

def q(a, qs=(0, 5, 25, 50, 75, 95, 100)):
    return " ".join(f"p{k}={np.percentile(a, k):.3g}" for k in qs) + f" mean={np.mean(a):.3g} n={len(a)}"

out = {}
for img in root.findall("image"):
    name = img.get("name")
    print("\n==========", name)
    rows = [(pts(e.get("points")), attrs(e), e) for e in img.findall("polyline")]
    cans = [(pts(e.get("points")), attrs(e), e) for e in img.findall("polygon") if e.get("label") == "vineyard"]
    irs = [(pts(e.get("points")), attrs(e), e) for e in img.findall("polygon") if e.get("label") == "interrow_area"]
    boxes = img.findall("box")
    print("rows", len(rows), "canopies", len(cans), "interrows", len(irs), "boxes", len(boxes))
    # misc element attributes
    print("keys seen:", collections.Counter(k for e in img for k in e.keys()))
    print("sources:", collections.Counter(e.get("source") for e in img), "z:", collections.Counter(e.get("z_order") for e in img), "occl:", collections.Counter(e.get("occluded") for e in img))
    print("vineyard_ids:", collections.Counter(a.get("vineyard_id") for _, a, _ in rows + cans + irs))
    # rows
    print("--- ROWS")
    print("n points per row:", collections.Counter(len(p) for p, _, _ in rows))
    print("row_structure:", collections.Counter(a.get("row_structure") for _, a, _ in rows))
    L = [np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) * GSD for p, _, _ in rows]
    print("row length m:", q(L))
    ang = [np.degrees(np.arctan2(p[-1, 1] - p[0, 1], p[-1, 0] - p[0, 0])) % 180 for p, _, _ in rows]
    print("row angle deg:", q(ang))
    ends_on_edge = []
    for p, a, _ in rows:
        e = [bool(min(v[0], v[1], 2048 - v[0], 2048 - v[1]) < 1.0) for v in (p[0], p[-1])]
        ends_on_edge.append(sum(e))
    print("row endpoints on tile edge (0/1/2):", collections.Counter(ends_on_edge))
    print("row ids:", [a.get("row_id") for _, a, _ in rows])
    # spacing between neighbouring rows: perpendicular distance at tile centre-ish
    th = np.radians(np.median(ang))
    d = np.array([np.cos(th), np.sin(th)]); nvec = np.array([-d[1], d[0]])
    offs = sorted([(np.mean(p @ nvec), a.get("row_id")) for p, a, _ in rows])
    sp = np.diff([o for o, _ in offs]) * GSD
    print("row spacing m (perp, sorted):", q(sp))
    print("row order by offset:", [r for _, r in offs])
    # straightness: max deviation of multi-point rows from chord
    for p, a, _ in rows:
        if len(p) > 2:
            c = p[-1] - p[0]; c = c / np.linalg.norm(c); nn = np.array([-c[1], c[0]])
            dev = np.abs((p - p[0]) @ nn).max() * GSD
            print("  multi-pt row", a.get("row_id"), len(p), "pts, max dev from chord m=%.3f" % dev)
    # canopies
    print("--- CANOPIES")
    A = np.array([area(p) for p, _, _ in cans]) * GSD * GSD
    V = np.array([len(p) for p, _, _ in cans])
    print("area m2:", q(A))
    print("vertices:", q(V))
    # extent along row / across row
    along = []; across = []; perim = []
    for p, _, _ in cans:
        al = p @ d; ac = p @ nvec
        along.append((al.max() - al.min()) * GSD); across.append((ac.max() - ac.min()) * GSD)
        perim.append(np.sum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)) * GSD)
    print("along-row extent m:", q(along)); print("across-row extent m:", q(across))
    seg = []
    for p, _, _ in cans:
        s = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1) * GSD
        seg += list(s)
    print("edge length m:", q(seg))
    # assign canopies to nearest row
    def dist_to_poly(pt, pl):
        best = 1e9
        for a_, b_ in zip(pl[:-1], pl[1:]):
            ab = b_ - a_; t = np.clip(np.dot(pt - a_, ab) / np.dot(ab, ab), 0, 1)
            best = min(best, np.linalg.norm(pt - (a_ + t * ab)))
        return best
    assign = collections.defaultdict(list); dists = []
    for i, (p, _, _) in enumerate(cans):
        c = p.mean(0)
        ds = [dist_to_poly(c, rp) for rp, _, _ in rows]
        j = int(np.argmin(ds)); dists.append(ds[j] * GSD)
        assign[j].append((c @ d, i))
    print("canopy centroid dist to nearest axis m:", q(dists))
    print("canopies per row:", sorted(len(v) for v in assign.values()))
    gaps = []; pitch = []; bigg = []
    for j, lst in assign.items():
        lst.sort()
        cs = [x for x, _ in lst]
        pitch += list(np.diff(cs) * GSD)
        # edge-to-edge gaps along row
        ext = sorted([((cans[i][0] @ d).min(), (cans[i][0] @ d).max()) for _, i in lst])
        for (a0, a1), (b0, b1) in zip(ext[:-1], ext[1:]):
            g = (b0 - a1) * GSD; gaps.append(g)
            if g >= 5: bigg.append((rows[j][1].get("row_id"), round(g, 1), rows[j][1].get("row_structure")))
    print("centroid pitch along row m:", q(pitch))
    print("edge gap along row m (neg = touching/overlap):", q(gaps))
    print("share of gaps <=0.05m (touching):", np.mean(np.array(gaps) <= 0.05))
    print("gaps >= 5 m:", bigg)
    for j, (p, a, _) in enumerate(rows):
        lst = assign.get(j, [])
        mg = 0
        ext = sorted([((cans[i][0] @ d).min(), (cans[i][0] @ d).max()) for _, i in lst])
        for (a0, a1), (b0, b1) in zip(ext[:-1], ext[1:]): mg = max(mg, (b0 - a1) * GSD)
        # also gap from axis endpoints to first/last canopy
        al = p @ d
        s0 = (ext[0][0] - al.min()) * GSD if ext else None
        s1 = (al.max() - ext[-1][1]) * GSD if ext else None
        print(f"  {a.get('row_id')}: {a.get('row_structure'):10s} n={len(lst):3d} len={L[j]:5.1f}m maxgap={mg:4.1f}m axis-overhang start={s0 if s0 is None else round(s0,2)} end={s1 if s1 is None else round(s1,2)}")
    # canopies touching tile edge
    edge = sum(1 for p, _, _ in cans if (p.min() <= 0.5) or (p.max() >= 2047.5))
    print("canopies touching tile edge:", edge)
    # interrows
    print("--- INTERROWS")
    print("cover:", collections.Counter(a.get("interrow_cover") for _, a, _ in irs))
    IA = np.array([area(p) for p, _, _ in irs]) * GSD * GSD
    print("area m2:", q(IA)); print("vertices:", q([len(p) for p, _, _ in irs]))
    W = []
    for p, _, _ in irs:
        ac = p @ nvec; al = p @ d
        W.append(area(p) * GSD * GSD / ((al.max() - al.min()) * GSD))
    print("mean width (area/length) m:", q(W))
    # overlap canopy vs interrow and coverage, via raster
    H = 2048
    mc = np.zeros((H, H), np.uint8); mi = np.zeros((H, H), np.uint8); cnt = np.zeros((H, H), np.uint8)
    for p, _, _ in cans:
        m = np.zeros((H, H), np.uint8); cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1); cnt += m
    mc = (cnt > 0).astype(np.uint8)
    for p, _, _ in irs: cv2.fillPoly(mi, [np.round(p).astype(np.int32)], 1)
    print("canopy px overlap between canopies (cnt>1) m2:", (cnt > 1).sum() * GSD**2)
    print("canopy union area m2:", mc.sum() * GSD**2, " sum of canopy areas m2:", A.sum())
    print("interrow union m2:", mi.sum() * GSD**2, "; canopy∩interrow m2:", (mc & mi).sum() * GSD**2)
    # gap between interrow and canopy: dilate canopy by k px and measure how much of interrow border touches
    kernel = np.ones((3, 3), np.uint8)
    border = mi - cv2.erode(mi, kernel)
    dt = cv2.distanceTransform((1 - mc).astype(np.uint8), cv2.DIST_L2, 5)
    bd = dt[border > 0] * GSD
    print("interrow border distance to nearest canopy m:", q(bd, (5, 25, 50, 75, 90, 95)))
    # image stats
    im = tifffile.imread(BASE + "images/" + name).astype(np.float32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    s = R + G + B + 1e-6
    exg = 2 * G / s - R / s - B / s
    nodata = (im.sum(-1) == 0)
    print("nodata px share:", nodata.mean())
    for nm, m in [("canopy", mc > 0), ("interrow", mi > 0), ("other", (mc == 0) & (mi == 0) & ~nodata)]:
        print(f"ExG {nm}:", q(exg[m][::50], (5, 25, 50, 75, 95)))
    # best Otsu-like threshold separation canopy vs interrow
    xs = np.linspace(-0.1, 0.4, 51)
    best = max(((np.mean(exg[mc > 0] > t) + np.mean(exg[mi > 0] <= t)) / 2, t) for t in xs)
    print("best ExG thr canopy vs interrow: balanced acc=%.3f thr=%.3f" % best)
    # iou of simple ExG mask vs canopy mask within vineyard zone
    zone = (mc > 0) | (mi > 0)
    for t in [0.02, 0.05, 0.08, 0.1, 0.12, 0.15]:
        pm = (exg > t) & zone
        inter = (pm & (mc > 0)).sum(); uni = (pm | (mc > 0)).sum()
        print(f"  ExG>{t}: IoU in zone={inter/uni:.3f}; whole tile IoU={((exg>t)&(mc>0)).sum()/((exg>t)|(mc>0)).sum():.3f}")
    np.save(f"/private/tmp/claude-501/-Users-maleticimiroslav-Vin-Gigahack-data---info/62a47fe7-0155-4b5f-a470-d6f1a44f9238/scratchpad/mask_{name}.npy", np.stack([mc, mi]))
