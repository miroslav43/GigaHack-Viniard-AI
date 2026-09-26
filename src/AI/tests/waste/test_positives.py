"""UAVVaste positives (design 03 W4): per-image scale to 2.5 cm/px, crop policy, cap, crop store."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from vineyard.perception.waste.crop_store import read_crop_store
from vineyard.perception.waste.positives import (
    SOURCE_PASTE,
    SOURCE_UAVVASTE,
    PositiveError,
    PositiveParams,
    build_positive_store,
    crop_geometry,
    gsd_estimate,
    image_scales,
    jpeg_roundtrip,
    object_alpha,
    object_scale,
    paste_crop,
    positive_crop,
    select_objects,
)
from vineyard.perception.waste.uavvaste import CocoImage, CocoObject, UavvasteDataset

RED = (255, 0, 0)
GREY = 128


def params(**over: object) -> PositiveParams:
    base = PositiveParams(
        target_gsd_m=0.025,
        assumed_object_m=0.25,
        long_side_px=(8.0, 40.0),
        min_objects_for_scale=3,
        max_per_image=2,
        max_n=100,
        context=0.5,
        min_px=64,
        out_px=224,
        seed=0,
    )
    return replace(base, **over)


def obj(ann_id: int, image_id: int, bbox: tuple[float, float, float, float]) -> CocoObject:
    x, y, w, h = bbox
    return CocoObject(ann_id, image_id, bbox, w * h, ((x, y, x + w, y, x + w, y + h, x, y + h),))


def dataset(root: Path, objects: tuple[CocoObject, ...], images: dict[int, CocoImage]) -> UavvasteDataset:
    return UavvasteDataset(root=root, images=images, objects=objects)


# ------------------------------------------------------------------ scale


def test_gsd_estimate_uses_median_long_side() -> None:
    assert gsd_estimate([80.0, 100.0, 120.0], 0.25) == pytest.approx(0.0025)
    with pytest.raises(PositiveError, match="object"):
        gsd_estimate([], 0.25)


def test_image_scales_fall_back_to_batch_then_dataset(tmp_path: Path) -> None:
    images = {
        0: CocoImage(0, "BATCH_d07_img_1.jpg", 1000, 800),  # 3 objects -> own estimate
        1: CocoImage(1, "BATCH_d07_img_2.jpg", 1000, 800),  # 1 object -> batch d07
        2: CocoImage(2, "GOPR0001.JPG", 1000, 800),  # 1 object, batch misc with 1 object -> dataset
    }
    objects = (
        obj(0, 0, (0, 0, 100, 50)),
        obj(1, 0, (0, 0, 100, 50)),
        obj(2, 0, (0, 0, 100, 50)),
        obj(3, 1, (0, 0, 200, 50)),
        obj(4, 2, (0, 0, 50, 50)),
    )
    scales = image_scales(dataset(tmp_path, objects, images), params())
    assert scales[0].source == "image" and scales[0].gsd_m == pytest.approx(0.0025)
    assert scales[1].source == "batch" and scales[1].gsd_m == pytest.approx(0.25 / 100.0)
    assert scales[2].source == "dataset" and scales[2].gsd_m == pytest.approx(0.25 / 100.0)
    assert scales[0].scale == pytest.approx(0.1)  # source px -> target px (0.0025 / 0.025)


def test_object_scale_clamps_long_side() -> None:
    p = params()
    assert object_scale(100.0, 0.1, p) == pytest.approx(0.1)  # 10 px: inside [8, 40]
    assert object_scale(500.0, 0.1, p) == pytest.approx(40.0 / 500.0)  # 50 px -> 40
    assert object_scale(30.0, 0.1, p) == pytest.approx(8.0 / 30.0)  # 3 px -> 8


# ------------------------------------------------------------------ crop


def _scene(h: int = 800, w: int = 1000, bbox: tuple[int, int, int, int] = (400, 300, 100, 100)) -> np.ndarray:
    img = np.full((h, w, 3), GREY, np.uint8)
    x, y, bw, bh = bbox
    img[y : y + bh, x : x + bw] = RED
    return img


def test_positive_crop_centres_object_at_target_scale() -> None:
    crop, info = positive_crop(_scene(), (400.0, 300.0, 100.0, 100.0), 0.1, params())
    assert crop.shape == (224, 224, 3) and crop.dtype == np.uint8
    assert info["side_t"] == 64 and info["side_src"] == 640 and info["long_t"] == pytest.approx(10.0)
    red = (crop[..., 0] > 200) & (crop[..., 1] < 60)
    ys, xs = np.nonzero(red)
    assert abs(xs.mean() - 112) < 3 and abs(ys.mean() - 112) < 3
    assert 25 <= xs.max() - xs.min() <= 40  # 10 of 64 px -> ~35 of 224


def test_positive_crop_window_shifted_inside_image() -> None:
    crop, info = positive_crop(_scene(bbox=(0, 0, 50, 50)), (0.0, 0.0, 50.0, 50.0), 0.1, params())
    assert info["x0"] == 0 and info["y0"] == 0
    assert crop[0, 0, 0] > 200  # object in the top-left corner of the crop


def test_positive_crop_window_larger_than_image_keeps_object_scale() -> None:
    img = _scene(h=300, w=400, bbox=(150, 100, 100, 100))
    crop, info = positive_crop(img, (150.0, 100.0, 100.0, 100.0), 0.1, params())
    assert info["side_src"] == 300 and info["side_t"] == 30
    assert crop.shape == (224, 224, 3)


def test_positive_crop_rejects_bad_bbox() -> None:
    with pytest.raises(PositiveError, match="bbox"):
        positive_crop(_scene(), (990.0, 0.0, 100.0, 10.0), 0.1, params())


# ------------------------------------------------------------------ selection and store


def _objects_many() -> tuple[CocoObject, ...]:
    return tuple(obj(i, i // 5, (10 + i, 10, 80, 60)) for i in range(15))  # 3 images x 5 objects


def test_select_objects_caps_per_image_and_total() -> None:
    picked = select_objects(_objects_many(), params(max_per_image=2))
    per = {}
    for o in picked:
        per[o.image_id] = per.get(o.image_id, 0) + 1
    assert per == {0: 2, 1: 2, 2: 2}
    assert select_objects(_objects_many(), params(max_per_image=2)) == picked  # deterministic
    assert len(select_objects(_objects_many(), params(max_per_image=5, max_n=4))) == 4


def _write_images(root: Path, specs: dict[str, tuple[int, int]]) -> None:
    (root / "images").mkdir(parents=True)
    for name, (h, w) in specs.items():
        cv2.imwrite(str(root / "images" / name), _scene(h=h, w=w, bbox=(100, 100, 60, 40))[..., ::-1])


def test_build_positive_store(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _write_images(
        root, {"BATCH_d07_img_1.jpg": (400, 500), "BATCH_d07_img_2.jpg": (400, 500), "bad.jpg": (500, 400)}
    )
    images = {
        0: CocoImage(0, "BATCH_d07_img_1.jpg", 500, 400),
        1: CocoImage(1, "BATCH_d07_img_2.jpg", 500, 400),
        2: CocoImage(2, "bad.jpg", 500, 400),  # file is 400 x 500: orientation mismatch -> skipped
    }
    objects = (
        obj(0, 0, (100, 100, 60, 40)),
        obj(1, 0, (300, 200, 60, 40)),
        obj(2, 1, (100, 100, 60, 40)),
        obj(3, 2, (100, 100, 60, 40)),
    )
    out = build_positive_store(dataset(root, objects, images), params(), tmp_path / "pos")
    store = read_crop_store(out)
    assert store.crops.shape == (3, 224, 224, 3)
    assert [r.key for r in store.records] == ["uav:0", "uav:1", "uav:2"]
    assert all(
        r.label == 1 and r.source == SOURCE_UAVVASTE and r.group == "uav:batch_d07" for r in store.records
    )
    assert store.meta["licence"] == "CC-BY-4.0" and "Kraft" in store.meta["attribution"]
    assert store.meta["skipped_images"] == [{"file": "bad.jpg", "reason": "shape (500, 400) != (400, 500)"}]
    assert store.records[0].meta["scale_source"] == "batch"


# ------------------------------------------------------------------ paste mode

BLUE = (0, 0, 255)


def _blue_bg(side: int) -> np.ndarray:
    bg = np.empty((side, side, 3), np.uint8)
    bg[...] = BLUE
    return bg


def test_params_validate_mode_and_range() -> None:
    with pytest.raises(PositiveError, match="mode"):
        params(mode="nope")
    with pytest.raises(PositiveError, match="long_side_px"):
        params(long_side_px=(40.0, 8.0))


def test_object_alpha_matches_object_area_at_target_scale() -> None:
    o = obj(0, 0, (400, 300, 100, 100))
    g = crop_geometry((800, 1000), o.bbox, 0.1, params())
    alpha = object_alpha(o, g, 0.0)
    assert alpha.shape == (64, 64) and alpha.dtype == np.float32
    assert alpha.sum() == pytest.approx(100.0, rel=0.1)  # 10 x 10 px at 2.5 cm/px
    ys, xs = np.nonzero(alpha > 0.5)
    assert abs(xs.mean() - 32) <= 1 and abs(ys.mean() - 32) <= 1
    blurred = object_alpha(o, g, 0.5)
    assert blurred.sum() == pytest.approx(alpha.sum(), rel=0.05) and blurred.max() <= 1.0


def test_paste_crop_puts_object_on_background() -> None:
    o = obj(0, 0, (400, 300, 100, 100))
    crop, info = paste_crop(_scene(), o, 0.1, _blue_bg, params(jpeg_quality=95))
    assert crop.shape == (224, 224, 3) and info["side_t"] == 64
    centre, corner = crop[112, 112].astype(int), crop[5, 5].astype(int)
    assert centre[0] > 200 and centre[2] < 60  # the red object
    assert corner[2] > 200 and corner[0] < 60  # blue background, no grey source context
    red = (crop[..., 0] > 150) & (crop[..., 2] < 100)
    ys, xs = np.nonzero(red)
    assert 28 <= xs.max() - xs.min() <= 42  # long side 10 of 64 px -> ~35 of 224


def test_paste_crop_rejects_wrong_background() -> None:
    o = obj(0, 0, (400, 300, 100, 100))
    with pytest.raises(PositiveError, match="background"):
        paste_crop(_scene(), o, 0.1, lambda side: _blue_bg(side + 1), params())


def test_jpeg_roundtrip_keeps_shape() -> None:
    img = _scene(h=40, w=50)
    out = jpeg_roundtrip(img, 90)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 10


def test_build_positive_store_both_modes(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    _write_images(root, {"BATCH_d07_img_1.jpg": (400, 500)})
    images = {0: CocoImage(0, "BATCH_d07_img_1.jpg", 500, 400)}
    objects = (obj(0, 0, (100, 100, 60, 40)), obj(1, 0, (300, 200, 60, 40)), obj(2, 0, (200, 50, 60, 40)))
    ds = dataset(root, objects, images)
    with pytest.raises(PositiveError, match="background"):
        build_positive_store(ds, params(mode="both"), tmp_path / "pos")
    out = build_positive_store(ds, params(mode="both"), tmp_path / "pos", background=_blue_bg)
    store = read_crop_store(out)
    keys = [r.key for r in store.records]  # max_per_image = 2: a plain + pasted pair per chosen object
    assert len(keys) == 4 and [k.split(":")[1] for k in keys[::2]] == [k.split(":")[1] for k in keys[1::2]]
    assert all(k.startswith("uav:") for k in keys[::2]) and all(k.startswith("uavp:") for k in keys[1::2])
    assert [r.source for r in store.records[:2]] == [SOURCE_UAVVASTE, SOURCE_PASTE]
    assert {r.group for r in store.records} == {"uav:batch_d07"} and store.meta["mode"] == "both"
    only_paste = read_crop_store(
        build_positive_store(ds, params(mode="paste"), tmp_path / "p2", background=_blue_bg)
    )
    assert [r.source for r in only_paste.records] == [SOURCE_PASTE, SOURCE_PASTE]


def test_target_blur_softens_the_crop() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(800, 1000, 3), dtype=np.uint8)
    sharp, _ = positive_crop(img, (400.0, 300.0, 100.0, 100.0), 0.1, params())
    soft, _ = positive_crop(img, (400.0, 300.0, 100.0, 100.0), 0.1, params(target_blur_sigma_px=0.8))

    def energy(x: np.ndarray) -> float:
        return float(
            np.abs(cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.CV_32F)).mean()
        )

    assert energy(soft) < 0.6 * energy(sharp)


def test_params_from_config_harmonise_sharpness() -> None:
    from vineyard.config import load_config

    p = PositiveParams.from_config(load_config())
    assert p.target_gsd_m == 0.025 and p.mode == "both" and p.target_blur_sigma_px > 0
    assert p.long_side_px == (8.0, 40.0) and p.out_px == 224
