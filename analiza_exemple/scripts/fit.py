"""Reproduce reference canopies from reference axes: veg mask ∩ corridor -> components -> score."""
import xml.etree.ElementTree as ET, numpy as np, cv2, tifffile, itertools, sys
B = "/Users/maleticimiroslav/Vin Gigahack/data & info/05_examples/siret3_examples_cvat/"
G = 0.025; H = 2048
root = ET.parse(B + "annotations.xml").getroot()
def pts(s): return np.array([[float(v) for v in p.split(",")] for p in s.split(";")])

def corridor(rows, half_m):
    m = np.zeros((H, H), np.uint8)
    for r in rows:
        a, b = r[0], r[-1]; d = (b - a) / np.linalg.norm(b - a); n = np.array([-d[1], d[0]]) * half_m / G
        a2, b2 = a, b  # rows end on tile edges here
        poly = np.array([a2 + n, b2 + n, b2 - n, a2 - n])
        cv2.fillPoly(m, [np.round(poly * 16).astype(np.int32)], 1, shift=4)
    return m

def indices(im):
    im = im.astype(np.float32); R, Gc, Bc = im[..., 0], im[..., 1], im[..., 2]; s = R + Gc + Bc + 1e-6
    r, g, b = R / s, Gc / s, Bc / s
    exg = 2 * g - r - b
    exgr = exg - (1.4 * r - g)
    hsv = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    return {"exg": exg, "exgr": exgr, "neg_a": -(lab[..., 1] - 128), "vari": (Gc - R) / (Gc + R - Bc + 1e-6)}

def score(pred_lab, ref_lab, nref):
    pm, rm = pred_lab > 0, ref_lab > 0
    iou = (pm & rm).sum() / max((pm | rm).sum(), 1)
    npred = pred_lab.max()
    # pairwise intersections
    both = (pm & rm)
    pairs = np.stack([pred_lab[both], ref_lab[both]], 1)
    ua, cnt = np.unique(pairs, axis=0, return_counts=True)
    pa = np.bincount(pred_lab.ravel(), minlength=npred + 1); ra = np.bincount(ref_lab.ravel(), minlength=nref + 1)
    tp = 0
    for (p, r), c in zip(ua, cnt):
        if c / (pa[p] + ra[r] - c) >= 0.5: tp += 1
    f1 = 2 * tp / max(npred + nref, 1)
    return iou, f1, npred

def run(img, cfg, cache):
    n = img.get("name")
    if n not in cache:
        im = tifffile.imread(B + "images/" + n)
        rows = [pts(e.get("points")) for e in img.findall("polyline")]
        cans = [pts(e.get("points")) for e in img.findall("polygon") if e.get("label") == "vineyard"]
        ref = np.zeros((H, H), np.int32)
        for i, p in enumerate(cans): cv2.fillPoly(ref, [np.round(p).astype(np.int32)], i + 1)
        cache[n] = (indices(im), rows, ref, len(cans))
    idx, rows, ref, nref = cache[n]
    corr = corridor(rows, cfg["half"])
    v = idx[cfg["idx"]]
    if cfg["blur"]: v = cv2.GaussianBlur(v, (0, 0), cfg["blur"])
    m = (v > cfg["thr"]).astype(np.uint8)
    if cfg["close"]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["close"], cfg["close"])); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if cfg["open"]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["open"], cfg["open"])); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m &= corr
    nl, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(nl, np.int32); j = 0
    for i in range(1, nl):
        if st[i, cv2.CC_STAT_AREA] * G * G >= cfg["minA"]: j += 1; keep[i] = j
    lab = keep[lab]
    return score(lab, ref, nref)

if __name__ == "__main__":
    cache = {}
    imgs = root.findall("image")
    grid = []
    for idx_name, thrs in [("exg", [0.04, 0.06, 0.08, 0.1]), ("neg_a", [2, 4, 6, 8, 10]), ("exgr", [-0.2, -0.15, -0.1, -0.05])]:
        for thr, blur, close, minA in itertools.product(thrs, [0, 2, 4], [0, 5, 9, 13], [0.19]):
            grid.append(dict(idx=idx_name, thr=thr, blur=blur, close=close, open=0, half=0.30, minA=minA))
    res = []
    for cfg in grid:
        s = [run(im, cfg, cache) for im in imgs]
        res.append((np.mean([x[0] for x in s]) * 0.6 + np.mean([x[1] for x in s]) * 0.4, cfg, s))
    res.sort(key=lambda x: -x[0])
    for sc, cfg, s in res[:15]:
        print(f"{sc:.3f}", cfg, [(round(a, 3), round(b, 3), c) for a, b, c in s])
