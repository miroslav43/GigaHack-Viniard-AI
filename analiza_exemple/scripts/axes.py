"""Prototype row-axis detector + end-to-end score on the two example tiles."""
import numpy as np, cv2, tifffile
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from fit import root, pts, B, G, H, indices, corridor, score

def veg_mask(im, thr=4.0, blur=2.5):
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)
    na = -(lab[..., 1] - 128)
    na = cv2.GaussianBlur(na, (0, 0), blur)
    return (na > thr).astype(np.uint8)

def dominant_angle(m):
    # angle whose perpendicular projection profile has max variance (rows = periodic peaks)
    best = None
    ys, xs = np.nonzero(m)
    sel = np.random.default_rng(0).choice(len(xs), min(len(xs), 200000), replace=False)
    xs, ys = xs[sel].astype(np.float32), ys[sel].astype(np.float32)
    for a in np.arange(0, 180, 0.5):
        t = np.radians(a); n = np.array([-np.sin(t), np.cos(t)])
        off = xs * n[0] + ys * n[1]
        h, _ = np.histogram(off, bins=np.arange(off.min(), off.max() + 4, 4))  # 0.1 m bins
        v = h.var()
        if best is None or v > best[0]: best = (v, a)
    return best[1]

def detect_rows(m, spacing_min_m=1.8):
    a = dominant_angle(m)
    t = np.radians(a); d = np.array([np.cos(t), np.sin(t)]); n = np.array([-d[1], d[0]])
    ys, xs = np.nonzero(m); P = np.stack([xs, ys], 1).astype(np.float32)
    off = P @ n; al = P @ d
    lo = off.min(); bins = np.arange(lo, off.max() + 2, 2)  # 5 cm
    h, _ = np.histogram(off, bins=bins); h = gaussian_filter1d(h.astype(float), 3)
    pk, _ = find_peaks(h, distance=int(spacing_min_m / G / 2), prominence=h.max() * 0.05)
    rows = []
    for p in pk:
        c = bins[p] + 1
        band = np.abs(off - c) < 0.45 / G
        if band.sum() < 0.2 / G / G: continue
        Q = P[band]
        for it in range(3):
            vx, vy, x0, y0 = cv2.fitLine(Q, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            u = np.array([vx, vy]); nu = np.array([-vy, vx])
            r = np.abs((P - [x0, y0]) @ nu)
            Q = P[(r < 0.35 / G) & (np.abs(off - c) < 0.8 / G)]
        if abs(((np.degrees(np.arctan2(vy, vx)) - a) + 90) % 180 - 90) > 3: continue
        tq = (Q - [x0, y0]) @ u
        ends = [np.array([x0, y0]) + u * np.percentile(tq, q) for q in (0.5, 99.5)]
        e = clip_to_tile(np.array(ends))
        if e is not None: rows.append(e)
    return rows, a

def clip_to_tile(seg, snap_px=120):
    a, b = seg
    # snap endpoints that lie within snap_px of tile border onto the border by extending the line
    dv = b - a
    out = []
    for p, sgn in ((a, -1), (b, 1)):
        if min(p[0], p[1], H - p[0], H - p[1]) < snap_px:
            # extend in direction sgn*dv until hitting border
            ts = []
            for i in range(2):
                step = sgn * dv[i]
                if abs(step) > 1e-6:
                    lim = H if step > 0 else 0
                    ts.append(max((lim - p[i]) / step, 0))
            tt = min(ts) if ts else 0
            p = p + sgn * dv * tt
        out.append(np.clip(p, 0, H))
    return np.array(out)

def cover(a, b, tol_px):
    # share of polyline a (sampled) within tol of polyline b (segment)
    s = np.linspace(0, 1, 200)[:, None]; A = a[0] + s * (a[1] - a[0])
    ab = b[1] - b[0]; t = np.clip(((A - b[0]) @ ab) / (ab @ ab), 0, 1)
    dist = np.linalg.norm(A - (b[0] + t[:, None] * ab), axis=1)
    return np.mean(dist <= tol_px)

def row_f1(pred, ref, tol_m=0.4, need=0.8):
    used = set(); tp = 0
    for r in ref:
        for j, p in enumerate(pred):
            if j in used: continue
            if cover(p, r, tol_m / G) >= need and cover(r, p, tol_m / G) >= need:
                used.add(j); tp += 1; break
    return 2 * tp / (len(pred) + len(ref)), tp

def canopies(m, rows, minA=0.19):
    mm = m & corridor(rows, 0.30)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mm, 8)
    keep = np.zeros(nl, np.int32); j = 0
    for i in range(1, nl):
        if st[i, cv2.CC_STAT_AREA] * G * G >= minA: j += 1; keep[i] = j
    return keep[lab]

for img in root.findall("image"):
    n = img.get("name"); im = tifffile.imread(B + "images/" + n)
    ref_rows = [pts(e.get("points")) for e in img.findall("polyline")]
    cans = [pts(e.get("points")) for e in img.findall("polygon") if e.get("label") == "vineyard"]
    ref = np.zeros((H, H), np.int32)
    for i, p in enumerate(cans): cv2.fillPoly(ref, [np.round(p).astype(np.int32)], i + 1)
    m = veg_mask(im)
    rows, ang = detect_rows(m)
    f1, tp = row_f1(rows, ref_rows)
    lab = canopies(m, rows)
    iou, cf1, npred = score(lab, ref, len(cans))
    print(f"{n}: angle={ang} rows pred={len(rows)} ref={len(ref_rows)} rowF1={f1:.3f} (tp={tp}) | canopy e2e IoU={iou:.3f} F1={cf1:.3f} npred={npred} ref={len(cans)} score={0.6*iou+0.4*cf1:.3f}")
    # per-row coverage diagnostics
    for r in ref_rows:
        best = max((min(cover(p, r, 16), cover(r, p, 16)) for p in rows), default=0)
        if best < 0.8: print("   missed ref row, best mutual cover=%.2f len=%.1fm" % (best, np.linalg.norm(r[1] - r[0]) * G))
    vis = im[..., ::-1].copy()
    for r in ref_rows: cv2.line(vis, tuple(np.int32(r[0])), tuple(np.int32(r[-1])), (0, 0, 255), 3)
    for r in rows: cv2.line(vis, tuple(np.int32(r[0])), tuple(np.int32(r[1])), (255, 255, 0), 1)
    cv2.imwrite(f"axes_{n[:-4]}.jpg", cv2.resize(vis, (1024, 1024), interpolation=cv2.INTER_AREA))
