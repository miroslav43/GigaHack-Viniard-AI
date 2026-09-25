from __future__ import annotations

import csv
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vineyard.perception.waste.confirm import read_confirmations
from vineyard.perception.waste.review import (
    CSV_COLUMNS,
    crop_name,
    render_review_html,
    review_items,
    write_review_crops,
    write_review_csv,
    write_review_html,
)
from vineyard.perception.waste.verify import RuleOnlyVerifier, load_verifier, waste_version_string

T = "siret3_r021_c012"


def frame() -> pd.DataFrame:
    base = {"tile_id": T, "area_m2": 0.1, "colour_class": "vivid", "vineyard_id": "V01", "dist_block_m": 0.0}
    rows = [
        {
            **base,
            "waste_id": f"{T}:W0001",
            "cand_key": f"{T}@0100_0100",
            "px_xtl": 90.0,
            "px_ytl": 90.0,
            "px_xbr": 110.0,
            "px_ybr": 110.0,
            "rank_score": 0.4,
            "review": True,
        },
        {
            **base,
            "waste_id": f"{T}:W0002",
            "cand_key": f"{T}@0200_0200",
            "px_xtl": 190.0,
            "px_ytl": 190.0,
            "px_xbr": 215.0,
            "px_ybr": 210.0,
            "rank_score": 0.9,
            "review": True,
        },
        {
            **base,
            "waste_id": f"{T}:W0003",
            "cand_key": f"{T}@0240_0020",
            "px_xtl": 230.0,
            "px_ytl": 10.0,
            "px_xbr": 250.0,
            "px_ybr": 30.0,
            "rank_score": 0.95,
            "review": False,
        },
    ]
    return pd.DataFrame(rows)


def test_items_ordered_by_rank_and_filtered() -> None:
    items = review_items(frame())
    assert [it.rank for it in items] == [1, 2]
    assert [it.cand_key for it in items] == [f"{T}@0200_0200", f"{T}@0100_0100"]
    assert items[0].crop == f"waste_crops/{T}@0200_0200.jpg"
    assert items[0].line.startswith(f"{T},190.0,190.0,215.0,210.0,accept,unknown,,")
    assert review_items(frame().iloc[0:0]) == ()


def test_csv_lines_are_pasteable(tmp_path: Path) -> None:
    items = review_items(frame())
    path = write_review_csv(items, tmp_path / "waste_candidates.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert tuple(rows[0]) == CSV_COLUMNS
    confirmed = tmp_path / "waste_confirmed.csv"
    confirmed.write_text(
        "tile_id,xtl,ytl,xbr,ybr,decision,category,reviewer,note\n"
        + "\n".join(r["confirm_line"] for r in rows)
        + "\n",
        encoding="utf-8",
    )
    confs = read_confirmations(confirmed, frozenset({T}))
    assert [c.box.as_tuple() for c in confs] == [it.box.as_tuple() for it in items]


def test_crops_written_with_box(tmp_path: Path) -> None:
    img = np.full((256, 256, 3), 120, np.uint8)
    img[195:205, 195:210] = (20, 60, 230)
    paths = write_review_crops(
        review_items(frame()), lambda t: img, tmp_path, context=0.5, min_px=64, out_px=256
    )
    assert [p.name for p in paths] == [f"{T}@0200_0200.jpg", f"{T}@0100_0100.jpg"]
    crop = cv2.imread(str(paths[0]))
    assert crop.shape == (256, 256, 3)
    assert crop[:, :, 2].max() > 200 and crop[:, :, 0].max() > 200


class _Collect(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.imgs: list[str] = []
        self.inputs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "img":
            self.imgs.append(a["src"] or "")
        if tag == "input":
            self.inputs.append(a["value"] or "")


def test_html_renders_in_rank_order_with_lines(tmp_path: Path) -> None:
    items = review_items(frame())
    path = write_review_html(items, tmp_path / "waste_review.html", {"level": "L3", "tiles": 2})
    parser = _Collect()
    parser.feed(path.read_text(encoding="utf-8"))
    assert parser.imgs == [it.crop for it in items]
    assert parser.inputs == [it.line for it in items]
    assert "level: L3" in path.read_text(encoding="utf-8")


def test_html_empty_and_escaping() -> None:
    text = render_review_html((), {"<b>": "&"})
    assert "Niciun candidat" in text and "&lt;b&gt;" in text


def test_crop_name_is_sanitised() -> None:
    assert crop_name("a/b c@1") == "waste_crops/a_b_c@1.jpg"


def test_rule_verifier_and_version(cfg) -> None:
    verifier, status = load_verifier(cfg.waste, cfg.grid.gsd_m)
    assert isinstance(verifier, RuleOnlyVerifier)
    assert (status.level, status.probe, status.sam) == ("L3", False, False)
    assert waste_version_string(status) == "rule@1"
    from dataclasses import replace

    assert waste_version_string(replace(status, probe=True, sam=True)) == "probe+sam3@1"
    from tests.waste.synth_waste import make_candidate

    (s,) = verifier.score([make_candidate()])
    assert s.probe_p is None and s.rule_score == pytest.approx(s.rule_score) and 0.0 < s.rule_score <= 1.0
