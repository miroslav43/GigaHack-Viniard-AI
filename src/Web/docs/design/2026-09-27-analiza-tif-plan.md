# Analiză TIF: plan de implementare (27.09.2026)

Spec: [`2026-09-27-analiza-tif-spec.md`](2026-09-27-analiza-tif-spec.md). Două fluxuri paralele, legate doar prin contractul
dosarului de job din spec.

## A. Python (`src/AI/vineyard/demo/`)

1. `tile_input.py`: `check_tile(path) -> TileInput(tile_id, path)`, erori `DemoInputError(code)`; teste în
   `tests/demo/test_tile_input.py` (tile bun, CRS, benzi, dimensiune, GSD, în afara grilei).
2. `job.py`: pregătește `data/01_tiles/demo.zip` (tile-ul redenumit `<tile_id>.tif`), `data/02_route` (symlink),
   `overrides.yaml` și `waste_confirmed.csv` goale, `--set`-urile de izolare; rulează `ingest … assemble` prin `run_pipeline`;
   un `logging.Handler` scrie `status.json` la fiecare `stage.start`.
3. `export.py`: AnnSet + `layers/waste_candidates.parquet` → GeoJSON EPSG:4326, `preview.jpg`, `veg_mask.png`, `result.json`.
4. `cli.py`: `vineyard demo tile <tif> --out <dir>`; montat în `vineyard/cli.py` cu `mount_subapp`.
5. Verificare: rulare pe `siret3_r021_c012` și pe un tile fără vie; ruff + pytest pe `tests/demo`.

## B. Web (`src/Web/frontend/`)

1. `src/lib/analiza/`: tipuri pentru `status.json` / `result.json`, gestiunea joburilor (dosar, spawn fără shell,
   un singur job activ), lista albă de fișiere; teste unitare.
2. API: `app/api/analiza/route.ts` (POST upload), `app/api/analiza/[job]/route.ts` (GET status + rezultat),
   `app/api/analiza/[job]/[file]/route.ts` (GET fișier din lista albă). Runtime Node.
3. Pagina `app/[locale]/(app)/analiza/page.tsx` + componente în `src/components/analiza/`: upload, progres, hartă cu
   straturi și comutatoare, panou de rezultate. Culori din `src/theme`. Texte în `messages/*`.
4. Link în navigație; notă în `src/Web/CLAUDE.md` (pagini + env `VINEYARD_AI_DIR`, `ANALIZA_JOBS_DIR`).
5. Verificare: `pnpm lint`, `pnpm typecheck`, `pnpm test`; test manual în browser cu tile-ul probei.

## C. Integrare

Rulare cap-coadă din browser pe 2 tile-uri; timpul afișat corespunde celui măsurat; commit separat AI / Web.
