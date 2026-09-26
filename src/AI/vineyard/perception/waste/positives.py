"""Probe positives from UAVVaste (A§4.9 step 3, design 03 W4).

Scale assumption (UAVVaste publishes no GSD or altitude): the median long side of the rubbish objects of
an image is taken to be `assumed_object_m` (0.25 m: bottles, cans, bags, paper). That gives a per-image
GSD estimate; images with fewer than `min_objects_for_scale` objects use the dataset median. Each object
is then rescaled to the Sireț3 GSD (2.5 cm/px) and clamped so its long side lies in `long_side_px`.
The crop is made like the pipeline's probe crop (crops.crop_window policy at the target GSD: box + 50 %
context, >= 64 px, square), first INTER_AREA down to the target GSD, then resized to 224 like
`crops.extract_crop`, so positives carry the same up-sampling blur as the tile candidates. The patch at
the target GSD is softened (target_blur_sigma_px) and JPEG round-tripped to match the Sireț3 tiles.

Modes (waste.probe.positives.mode): "plain" = the UAVVaste crop as is; "paste" = the object cut out with its
COCO polygon (alpha blurred by paste_blur_sigma_px) and pasted at the same place onto a Sireț3 background
window of the same size (BackgroundFn, non-example tiles), then a JPEG round trip, so the probe cannot
separate the classes by background or sensor; "both" = one of each per object. CV group = UAVVaste flight
batch for both, so an object never sits on both sides of a fold.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import cv2
import numpy as np

from vineyard.errors import VineyardError
from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.waste.crop_store import CropRecord, write_crop_store
from vineyard.perception.waste.crops import CropWindow, extract_crop

if TYPE_CHECKING:
    from vineyard.config import AppConfig
    from vineyard.perception.waste.uavvaste import CocoImage, CocoObject, UavvasteDataset

SOURCE_UAVVASTE: Final = "uavvaste"
SOURCE_PASTE: Final = "uavvaste_paste"
POSITIVE: Final = 1
UAV_GROUP_PREFIX: Final = "uav:"
PASTE_PREFIX: Final = "uavp:"
MODES: Final = ("plain", "paste", "both")
_PLAIN_MODES: Final = frozenset({"plain", "both"})
_PASTE_MODES: Final = frozenset({"paste", "both"})
_BLUR_RADII: Final = 3  # Gaussian kernel half-width in sigmas
BackgroundFn = Callable[[int], np.ndarray]  # side -> uint8 (side, side, 3) Sireț3 window at the target GSD
# waste.probe.positives.target_blur_sigma_px (0.8): at 2.5 cm/px UAVVaste patches are ~4x sharper than Sireț3
# windows (median |Laplacian| / std 1.94 vs 0.51); a Gaussian of 0.8 px matches (0.53), so sharpness cannot
# tell the classes apart.

_log = get_logger("perception.waste.positives")


class PositiveError(VineyardError):
    """UAVVaste positive-crop failure."""


# ------------------------------------------------------------------ UAVVaste positives


@dataclass(frozen=True)
class PositiveParams:
    target_gsd_m: float
    assumed_object_m: float
    long_side_px: tuple[float, float]
    min_objects_for_scale: int
    max_per_image: int
    max_n: int
    context: float
    min_px: int
    out_px: int
    seed: int
    mode: str = "plain"
    blur_sigma_px: float = 0.0  # alpha edge of pasted objects
    jpeg_quality: int = 90
    target_blur_sigma_px: float = 0.0  # UAVVaste patch at the target GSD, to match Sireț3 sharpness

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise PositiveError("unknown positives mode", mode=self.mode, allowed=MODES)
        lo, hi = self.long_side_px
        if not 0 < lo <= hi:
            raise PositiveError("long_side_px must satisfy 0 < lo <= hi", long_side_px=self.long_side_px)

    @classmethod
    def from_config(cls, cfg: AppConfig) -> PositiveParams:
        probe = cfg.waste.probe
        lo, hi = probe.positives.long_side_px
        return cls(
            target_gsd_m=cfg.grid.gsd_m,
            assumed_object_m=probe.positives.assumed_object_m,
            long_side_px=(float(lo), float(hi)),
            min_objects_for_scale=probe.positives.min_objects_for_scale,
            max_per_image=probe.positives.max_per_image,
            max_n=probe.positives.max_n,
            context=probe.crop_context,
            min_px=probe.crop_min_px,
            out_px=probe.crop_resize_px,
            seed=cfg.runtime.seed,
            mode=probe.positives.mode,
            blur_sigma_px=probe.positives.paste_blur_sigma_px,
            jpeg_quality=probe.positives.jpeg_quality,
            target_blur_sigma_px=probe.positives.target_blur_sigma_px,
        )


@dataclass(frozen=True)
class ImageScale:
    gsd_m: float  # estimated source GSD (m/px)
    scale: float  # source px -> target px (gsd_m / target_gsd_m)
    source: str  # "image" | "batch" | "dataset"


def gsd_estimate(long_sides: Sequence[float], assumed_object_m: float) -> float:
    """Source GSD if the median object long side measures `assumed_object_m`."""
    if not long_sides or min(long_sides) <= 0:
        raise PositiveError("need positive object sizes to estimate a GSD", n=len(long_sides))
    return assumed_object_m / float(np.median(long_sides))


def image_scales(ds: UavvasteDataset, p: PositiveParams) -> dict[int, ImageScale]:
    """Per-image estimate; images with too few objects use their flight batch, then the dataset."""
    from vineyard.perception.waste.uavvaste import MISC_BATCH

    by_image = ds.objects_by_image()
    by_batch: dict[str, list[float]] = {}
    for image_id, objs in by_image.items():
        by_batch.setdefault(ds.images[image_id].batch, []).extend(o.long_side for o in objs)
    everything = [o.long_side for o in ds.objects]
    out: dict[int, ImageScale] = {}
    for image_id, objs in by_image.items():
        batch = ds.images[image_id].batch
        if len(objs) >= p.min_objects_for_scale:
            sides, source = [o.long_side for o in objs], "image"
        elif batch != MISC_BATCH and len(by_batch[batch]) >= p.min_objects_for_scale:
            sides, source = by_batch[batch], "batch"
        else:
            sides, source = everything, "dataset"
        gsd = gsd_estimate(sides, p.assumed_object_m)
        out[image_id] = ImageScale(gsd_m=gsd, scale=gsd / p.target_gsd_m, source=source)
    return out


def object_scale(long_side_src: float, scale: float, p: PositiveParams) -> float:
    """Image scale, adjusted so the object's long side lands inside `long_side_px` at the target GSD."""
    lo, hi = p.long_side_px
    return float(np.clip(long_side_src * scale, lo, hi)) / long_side_src


def select_objects(objects: Sequence[CocoObject], p: PositiveParams) -> tuple[CocoObject, ...]:
    """At most max_per_image random objects per image, then at most max_n overall (seeded; by ann id)."""
    rng = np.random.default_rng(p.seed)
    by_image: dict[int, list[CocoObject]] = {}
    for o in sorted(objects, key=lambda o: o.ann_id):
        by_image.setdefault(o.image_id, []).append(o)
    picked = [
        objs[i]
        for _, objs in sorted(by_image.items())
        for i in sorted(rng.permutation(len(objs))[: p.max_per_image])
    ]
    if len(picked) > p.max_n:
        picked = [picked[i] for i in sorted(rng.permutation(len(picked))[: p.max_n])]
    return tuple(sorted(picked, key=lambda o: o.ann_id))


def _placed(centre: float, side: int, extent: int) -> int:
    return int(min(max(round(centre - side / 2.0), 0), extent - side))


@dataclass(frozen=True)
class CropGeometry:
    """Square source window [x0, x0 + side_src)^2 that becomes side_t px at the target GSD."""

    x0: int
    y0: int
    side_src: int
    side_t: int
    long_t: float

    def info(self) -> dict[str, Any]:
        return {
            "side_t": self.side_t,
            "side_src": self.side_src,
            "long_t": self.long_t,
            "x0": self.x0,
            "y0": self.y0,
        }


def crop_geometry(
    shape_hw: tuple[int, int], bbox: tuple[float, float, float, float], scale: float, p: PositiveParams
) -> CropGeometry:
    """Probe crop policy at the target GSD (box + context, >= min_px), mapped back to source px."""
    h, w = shape_hw
    x, y, bw, bh = bbox
    if x < 0 or y < 0 or x + bw > w or y + bh > h or bw <= 0 or bh <= 0:
        raise PositiveError("bbox outside the image", bbox=bbox, image=(w, h))
    long_t = max(bw, bh) * scale
    side_t = max(p.min_px, math.ceil((1.0 + p.context) * long_t))
    side_src = round(side_t / scale)
    if side_src > min(h, w):  # keep the object's scale, lose some context
        side_src = min(h, w)
        side_t = max(1, round(side_src * scale))
    x0, y0 = _placed(x + bw / 2.0, side_src, w), _placed(y + bh / 2.0, side_src, h)
    return CropGeometry(x0, y0, side_src, side_t, long_t)


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    k = 2 * math.ceil(_BLUR_RADII * sigma) + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def _target_patch(img: np.ndarray, g: CropGeometry, p: PositiveParams) -> np.ndarray:
    """Source window at the target GSD (INTER_AREA), softened to the Sireț3 sharpness."""
    patch = img[g.y0 : g.y0 + g.side_src, g.x0 : g.x0 + g.side_src]
    return _gauss(
        cv2.resize(patch, (g.side_t, g.side_t), interpolation=cv2.INTER_AREA), p.target_blur_sigma_px
    )


def positive_crop(
    img: np.ndarray, bbox: tuple[float, float, float, float], scale: float, p: PositiveParams
) -> tuple[np.ndarray, dict[str, Any]]:
    """Probe-policy crop around a COCO bbox, resampled to the target GSD, then resized to out_px."""
    g = crop_geometry(img.shape[:2], bbox, scale, p)
    patch = jpeg_roundtrip(_target_patch(img, g, p), p.jpeg_quality)
    return extract_crop(patch, CropWindow(0, 0, g.side_t), p.out_px), g.info()


def object_alpha(obj: CocoObject, g: CropGeometry, blur_sigma_px: float) -> np.ndarray:
    """float32 (side_t, side_t) alpha of the object (its polygons, else its bbox) in the target window."""
    mask = np.zeros((g.side_src, g.side_src), np.uint8)
    x, y, bw, bh = obj.bbox
    polys = obj.polygons or ((x, y, x + bw, y, x + bw, y + bh, x, y + bh),)
    for poly in polys:
        pts = np.asarray(poly, np.float64).reshape(-1, 2) - (g.x0, g.y0)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    alpha = cv2.resize(mask.astype(np.float32), (g.side_t, g.side_t), interpolation=cv2.INTER_AREA)
    return np.clip(_gauss(alpha, blur_sigma_px), 0.0, 1.0)


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise PositiveError("JPEG encoding failed", shape=rgb.shape, quality=quality)
    return np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1])


def paste_crop(
    img: np.ndarray, obj: CocoObject, scale: float, background: BackgroundFn, p: PositiveParams
) -> tuple[np.ndarray, dict[str, Any]]:
    """The object alone (alpha-blended) on a Sireț3 background window, JPEG round trip, resized to out_px."""
    g = crop_geometry(img.shape[:2], obj.bbox, scale, p)
    bg = background(g.side_t)
    if bg.shape != (g.side_t, g.side_t, 3) or bg.dtype != np.uint8:
        raise PositiveError("background window has the wrong shape", want=g.side_t, shape=bg.shape)
    alpha = object_alpha(obj, g, p.blur_sigma_px)[..., None]
    blend = np.rint(bg * (1.0 - alpha) + _target_patch(img, g, p) * alpha).astype(np.uint8)
    out = jpeg_roundtrip(blend, p.jpeg_quality)
    return extract_crop(out, CropWindow(0, 0, g.side_t), p.out_px), g.info()


def read_rgb(path: Path) -> np.ndarray:
    """RGB uint8, raw pixel orientation (EXIF ignored, like the COCO annotations)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise PositiveError("unreadable image", path=str(path))
    return np.ascontiguousarray(bgr[..., ::-1])


def _image_crops(
    img: np.ndarray,
    image: CocoImage,
    objs: Sequence[CocoObject],
    s: ImageScale,
    p: PositiveParams,
    background: BackgroundFn | None,
) -> Iterator[tuple[CropRecord, np.ndarray]]:
    group = f"{UAV_GROUP_PREFIX}{image.batch}"
    for o in objs:
        scale = object_scale(o.long_side, s.scale, p)
        base = {
            "image": image.file_name,
            "ann_id": o.ann_id,
            "scale": scale,
            "gsd_src_m": s.gsd_m,
            "scale_source": s.source,
        }
        if p.mode in _PLAIN_MODES:
            crop, info = positive_crop(img, o.bbox, scale, p)
            yield (
                CropRecord(
                    f"{UAV_GROUP_PREFIX}{o.ann_id}", SOURCE_UAVVASTE, group, POSITIVE, {**base, **info}
                ),
                crop,
            )
        if p.mode in _PASTE_MODES and background is not None:
            crop, info = paste_crop(img, o, scale, background, p)
            yield (
                CropRecord(f"{PASTE_PREFIX}{o.ann_id}", SOURCE_PASTE, group, POSITIVE, {**base, **info}),
                crop,
            )


def _positive_meta(p: PositiveParams, skipped: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    from vineyard.perception.waste.uavvaste import ATTRIBUTION, DOI, LICENCE

    return {
        "kind": "positives",
        "dataset": "UAVVaste",
        "doi": DOI,
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "skipped_images": list(skipped),
        "scale_assumption": f"median object long side = {p.assumed_object_m} m",
        "mode": p.mode,
        "params": {
            "target_gsd_m": p.target_gsd_m,
            "long_side_px": list(p.long_side_px),
            "max_per_image": p.max_per_image,
            "max_n": p.max_n,
            "context": p.context,
            "min_px": p.min_px,
        },
    }


def build_positive_store(
    ds: UavvasteDataset,
    p: PositiveParams,
    out_dir: Path,
    read_image: Callable[[Path], np.ndarray] = read_rgb,
    background: BackgroundFn | None = None,
) -> Path:
    """Crop the selected UAVVaste objects into a store (images whose size differs from COCO are skipped)."""
    if p.mode in _PASTE_MODES and background is None:
        raise PositiveError("paste mode needs a Sireț3 background sampler", mode=p.mode)
    scales = image_scales(ds, p)
    chosen: dict[int, list[CocoObject]] = {}
    for o in select_objects(ds.objects, p):
        chosen.setdefault(o.image_id, []).append(o)
    pairs: list[tuple[CropRecord, np.ndarray]] = []
    skipped: list[dict[str, str]] = []
    for image_id, objs in sorted(chosen.items()):
        image = ds.images[image_id]
        img = read_image(ds.image_path(image))
        if img.shape[:2] != (image.height, image.width):
            reason = f"shape {img.shape[:2]} != {(image.height, image.width)}"
            log_event(
                _log,
                "waste.positives.image_skipped",
                level=logging.WARNING,
                file=image.file_name,
                reason=reason,
            )
            skipped.append({"file": image.file_name, "reason": reason})
            continue
        pairs.extend(_image_crops(img, image, objs, scales[image_id], p, background))
    crops = (c for _, c in pairs)
    return write_crop_store(out_dir, crops, [r for r, _ in pairs], p.out_px, _positive_meta(p, skipped))
