"""Build the pitch images from the RC10f QA run (overview, row preview, canopy zoom, waste crops).

Run from the repo root:  python3 prezentare/scripts/make_figures.py
Needs Python 3.10+ and Pillow. Writes prezentare/img/*.jpg. The farm route map is make_farm_map.py.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
QA = ROOT / "src/AI/work/runs/rc10f/qa"
OUT_IMG = ROOT / "prezentare/img"

NIGHT_RGB = (0x12, 0x10, 0x3A)
BLACK_CUTOFF = 10
JPEG_QUALITY = 88

WASTE_CROPS = {
    "waste_ok1.jpg": "siret3_r023_c016@0600_0591.jpg",
    "waste_ok2.jpg": "siret3_r008_c000@0661_1644.jpg",
    "waste_rej.jpg": "siret3_r009_c005@0347_1358.jpg",
}
ROWS_PREVIEW = "siret3_r006_c004.jpg"
PREVIEW_HEADER_PX = 26
CANOPY_BOX = (200, 260, 480, 540)


def _load(name: str) -> list[dict]:
    path = BUNDLE / name
    if not path.exists():
        sys.exit(f"missing {path} (run the web bundle first)")
    return json.loads(path.read_text())["features"]


def _night_background(img: Image.Image) -> Image.Image:
    """Paint the black nodata frame in the slide's night colour."""
    rgb = img.convert("RGB")
    mask = rgb.convert("L").point(lambda v: 255 if v < BLACK_CUTOFF else 0)
    out = rgb.copy()
    out.paste(NIGHT_RGB, mask=mask)
    return out


def build_images() -> None:
    if not QA.exists():
        print(f"skip images: {QA} not found (git-ignored run folder)")
        return
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    overview = Image.open(QA / "overview.jpg")
    overview.thumbnail((1600, 1600))
    _night_background(overview).save(OUT_IMG / "overview.jpg", quality=JPEG_QUALITY)

    preview = Image.open(QA / "previews" / ROWS_PREVIEW).convert("RGB")
    w, h = preview.size
    preview.crop((0, PREVIEW_HEADER_PX, w, h)).save(OUT_IMG / "rows_preview.jpg", quality=JPEG_QUALITY)
    preview.crop(CANOPY_BOX).resize((600, 600), Image.LANCZOS).save(OUT_IMG / "canopy_zoom.jpg", quality=JPEG_QUALITY)

    for out_name, crop in WASTE_CROPS.items():
        Image.open(QA / "waste_crops" / crop).convert("RGB").save(OUT_IMG / out_name, quality=JPEG_QUALITY)


if __name__ == "__main__":
    build_images()
    print(f"wrote {OUT_IMG}")
