# Pitch deck — Solemtrix · GigaHack 2026

Beamer (16:9) deck for the 5 min pitch and the 5 min Q&A. The layout is the classic Madrid structure:
a full-width title bar, a 3-part footline (author | title | date and page), ball items, rounded blocks and a
section outline at the start of every section. The colours come from the web theme
(`src/Web/frontend/src/theme/tokens.ts`: indigo `#4F46E5` and night `#12103A`).

| File | Content |
|---|---|
| `gigahack/gigahack.tex` | **the deck to present and submit**: the organisers' template (`GigaHack_2026_Pitch_Template.pptx`) rebuilt in Beamer, with the template's backgrounds, logo, boxes, colours and Arial at the template's own coordinates. 9 main slides plus 4 backup. Build: `cd gigahack && xelatex -output-directory=build gigahack.tex` twice, then `cp build/gigahack.pdf .` |
| `gigahack/make_pptx.mjs` | `node make_pptx.mjs <template.pptx>` writes `gigahack/gigahack.pptx` (the platform asks for a .pptx): every PDF page as a full-slide picture in the template's own package |
| `pitch.tex` | main deck (10 slides), in the organisers' order: title (team), problem (roles, user stories), business case, architecture, data pipeline, route result, demo, impact (economics, deployment time), scaling, summary |
| `tech.tex` | technical detail moved to backup (slides 11–17): challenge, CV first, rows, canopies, waste, measures, engineering |
| `CONTINUT_TEMPLATE.md` | slide-by-slide text to paste into the organisers' mandatory template |
| `export/slide-NN.png` | the 10 main slides at 200 dpi, for pasting diagrams into the template |
| `robot.tex` | the 3 robot slides, now backup (18–20) |
| `backup.tex` | Q&A backup slides (21–30), numbered "backup N" in the footline |
| `beamerthemesolemtrix.sty` | theme: `rounded` inner + `infolines` outer, colours, fonts, blocks, stats, steps, score chips |
| `scripts/make_figures.py` | builds `img/*.jpg` (overview, row preview, canopy zoom, waste crops) from the RC10f QA run |
| `scripts/make_farm_map.py` | builds `img/farm_ortho.jpg` and `fig/map_farm.tex`: farm F09's orthophoto with its route (needs rasterio, run it in `src/AI`) |
| `data/traseu_F09.gpx` | per-farm route of F09, exported from the web app's farm route tool |
| `SCRIPT.md` | what to say on each slide, with timings |
| `pitch.pdf` | the compiled deck |

## Build

Needs XeLaTeX (BasicTeX is enough: beamer, tikz, fontspec, booktabs, etoolbox). Fonts: Avenir Next and
Menlo on macOS, with a fallback to Helvetica Neue or Latin Modern elsewhere.

```bash
cd prezentare
python3 scripts/make_figures.py        # only when the data changes (needs Pillow)
(cd ../src/AI && uv run --no-sync python ../../prezentare/scripts/make_farm_map.py)   # farm route map
xelatex -output-directory=build pitch.tex && xelatex -output-directory=build pitch.tex
cp build/pitch.pdf pitch.pdf
```

The second run places the scoring chips (TikZ `remember picture`) and the "N / 10" page total, which
`\sxmainend` (just before `\appendix`) writes to the aux file.

## Adding the robot photos and the web screenshot

- `robot/robot.jpg` replaces the drawing on slide 15, and `robot/detection.jpg` the one on slide 16.
- `img/web_map.png` appears above the web-app card on slide 12.

Rebuild after adding them. The robot slides describe the concept only. Add the real model, the disease
classes and the measured numbers in `robot.tex` once the team confirms them.

## Sources of the numbers

- Run `rc10f` and its post run `20260927T0452-post-rc10f`: `summary.json`, `metrics/route_baseline.json`.
- `README.md` §6 (timings) and §10 (results).
- `src/AI/reports/nn_ablation.md` (the CV vs U-Net ablation).
- The official route visits 147/147 reachable must-visit targets; 99 more are unreachable (76 disconnected from START, 23 too far
  from any allowed path), per `targets.geojson`.
- Example-tile metrics: `src/AI/work/runs/rc10f/metrics/eval_examples.md` (the organisers' 2 example tiles); every slide that
  shows them says so.
- Farm F09 route: 9.71 km vs a 27.46 km serpentine (−65 %), 455/455 targets, recomputed with the web app's planner
  (`src/Web/frontend/src/lib/farmRoute`) from the GPX start point.
- The SAM 3 figure of about 18 h is an extrapolation: 311 tiles × 16 crops of 512 px × 13.1 s per crop.
  The slides label it as such.
