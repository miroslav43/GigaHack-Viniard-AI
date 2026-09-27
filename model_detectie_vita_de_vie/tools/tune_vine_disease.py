"""Chooses the "suspect vine" rule on EscaYard B7 and checks it on B9 (never used to choose).

Today a vine is suspect when the diseased-leaf model finds at least 3 leaves at conf >= 0.5 on the whole photo:
B7 28/38 sick vines found, but B9 only 3/12 (on B9 the model sees ~1.9 leaves per sick vine). Two levers, no training:
  - tiles: the leaf model also runs on 3 x 3 pieces of the photo (leaves are small in whole-vine photos)
  - the rule: "N leaves at conf >= c" or "sum of the leaf confidences >= S", chosen on B7 only

The detections are run once at a low threshold and cached, so trying rules is instant.
Run (from the project folder):  python -m tools.tune_vine_disease
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from tools.evaluate_escayard import find_photos, load_datasheet, share, shrink
from vineyard.config import MODELS

LOW = 0.2
COUNT_RULES = [("count", c, n) for c in (0.3, 0.4, 0.5, 0.6) for n in (1, 2, 3, 4, 5, 6, 8, 10)]
SUM_RULES = [("sum", LOW, s) for s in (0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10)]
MAX_HEALTHY_FLAGGED = 0.10  # choose the rule that finds the most sick B7 vines while flagging at most 10% healthy ones


def detections(tiles, cache_dir):
    """{photo: [conf of each diseased leaf]} for all EscaYard photos, cached."""
    cache = Path(cache_dir) / f"escayard_leaves_tiles{tiles}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import cv2
    from vineyard.detector import VineyardDetector

    leaf = next(m for m in MODELS if "diseased leaf" in m.classes.values())
    det = VineyardDetector(models=[replace(leaf, conf=LOW, tile=True)], tiles=tiles)
    photos = find_photos("data/escayard/photos")
    out = {}
    for i, (name, path) in enumerate(sorted(photos.items()), 1):
        out[name] = [d.conf for d in det.detect([shrink(cv2.imread(str(path)))])[0] if d.label == "diseased leaf"]
        if i % 25 == 0 or i == len(photos):
            print(f"tiles={tiles}: {i}/{len(photos)} ({i / len(photos):.0%})", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def suspect(confs, rule):
    kind, c, x = rule
    return sum(v for v in confs if v >= c) >= x if kind == "sum" else sum(v >= c for v in confs) >= x


def rates(vines, dets, rule):
    """(sick found, sick, healthy flagged, healthy) for vines = [(photo, esca)]"""
    sick = [p for p, e in vines if e]
    healthy = [p for p, e in vines if not e]
    return (sum(suspect(dets[p], rule) for p in sick), len(sick),
            sum(suspect(dets[p], rule) for p in healthy), len(healthy))


def choose(vines, dets, rules, max_fp=MAX_HEALTHY_FLAGGED):
    """Most sick vines found with at most max_fp of the healthy ones flagged (ties: fewer healthy flagged)."""
    ok = []
    for rule in rules:
        tp, n_sick, fp, n_healthy = rates(vines, dets, rule)
        if fp <= max_fp * n_healthy:
            ok.append((tp, -fp, rule))
    return max(ok, key=lambda r: (r[0], r[1]))[2] if ok else None


def describe(rule):
    kind, c, x = rule
    return f"sum of confidences >= {x}" if kind == "sum" else f"at least {x} leaves at conf >= {c}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="results/vine_disease")
    args = p.parse_args()
    sheet = load_datasheet("data/escayard/datasheet.csv")
    blocks = {b: [(n, r["esca"]) for n, r in sheet.items() if r["vineyard"] == b and r["esca"] is not None]
              for b in ("B7", "B9")}
    lines = ["Rule chosen on B7 (most sick vines found, at most 10% healthy flagged), then checked on B9.", ""]
    for tiles in (1, 3):
        dets = detections(tiles, args.cache)
        missing = [n for b in blocks.values() for n, _ in b if n not in dets]
        blocks_ok = {b: [(n, e) for n, e in v if n in dets] for b, v in blocks.items()}
        if missing:
            print(f"WARNING: {len(missing)} photos not found")
        for label, rules in (("today's rule", [("count", 0.5, 3)]), ("count rules", COUNT_RULES),
                             ("sum rules", SUM_RULES), ("all rules", COUNT_RULES + SUM_RULES)):
            rule = rules[0] if len(rules) == 1 else choose(blocks_ok["B7"], dets, rules)
            if rule is None:
                lines.append(f"tiles={tiles} {label}: no rule keeps B7 false alarms under 10%")
                continue
            b7 = rates(blocks_ok["B7"], dets, rule)
            b9 = rates(blocks_ok["B9"], dets, rule)
            lines.append(f"tiles={tiles} | {label:13s} | {describe(rule)}")
            lines.append(f"   B7: sick found {share(b7[0], b7[1])}, healthy flagged {share(b7[2], b7[3])}")
            lines.append(f"   B9: sick found {share(b9[0], b9[1])}, healthy flagged {share(b9[2], b9[3])}")
        lines.append("")
    report = "\n".join(lines)
    Path(args.cache, "vine_rule_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
