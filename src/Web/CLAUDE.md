# src/Web/ — Solemtrix Vineyard, aplicația web (GigaHack 2026, challenge Marcaj)

Documentul-sursă al părții web. Planul complet: `docs/PLAN.md`. Deciziile: `docs/DECISIONS.md`. Scenariul de pitch: `docs/DEMO.md`.

**Stare (25.09, seara): site-ul există în `frontend/`**, pe date statice (`siret3-mock`), cu login Supabase. API-ul FastAPI și datele în PostGIS nu sunt încă implementate; restul documentului rămâne planul.

Implementat: panou general, hartă (ortofoto 311 tile-uri, straturi, atribute, căutare, deep link-uri, rută cu săgeți de direcție, unealtă de măsurare în UTM 35N, vedere oblică 3D cu relief sintetic din coroane), blocuri și rânduri (DataGrid, filtre, export CSV), rută (GPX, GeoJSON EPSG:32635), **RO / EN / RU** (next-intl, RO implicit fără prefix, fără detectare din browser), e2e Playwright (desktop + mobil), CI `.github/workflows/web.yml`.

**Login (Supabase Auth, proiect `htdnuwmnztenevnfahie`, eu-west-1):**
- email + parolă, `@supabase/ssr`; `src/proxy.ts` = next-intl + reîmprospătarea sesiunii (`getClaims`) + protecția paginilor; „Demo fără cont” = cookie `solemtrix_demo` (date publice, doar vizualizare);
- primăria și rolul vin din `app_metadata` (`uat` = `sireti` | `cojusna`, `uat_role`), niciodată din `user_metadata`; o primărie fără zbor în limita ei primește o stare goală și nu i se încarcă datele altui UAT (`src/lib/uats.ts`, `src/lib/viewer.ts`);
- conturi demo (seed: `supabase/seed/demo_users.sql`, parola NU e în git — se înlocuiește `__DEMO_PASSWORD__` la rulare): `admin@solemtrix.demo` (platform_admin), `primar@sireti.demo` (uat_admin), `inspector@sireti.demo` (inspector), `primar@cojusna.demo` (uat_admin Cojușna);
- „Ați uitat parola?” pe `/login` (`src/components/auth/ForgotPasswordForm.tsx`): `resetPasswordForEmail` cu un client în flux implicit (`createRecoveryClient`, linkul merge și pe alt dispozitiv) → `/parola-noua?flow=reset`; răspuns identic pentru adrese cu și fără cont; emailurile reale cer SMTP propriu în Supabase;
- `frontend/.env.local` (ignorat de git, model în `.env.example`): URL-ul proiectului + cheia publishable. Fără ele (CI) aplicația rulează în mod demo deschis;
**Super-admin (`/super-admin`): accesibil doar tastând URL-ul (nu apare în meniu) și doar pentru `uat_role = platform_admin`; oricine altcineva (anonim, demo, primar, inspector) primește HTTP 403 „Acces interzis” din `src/proxy.ts`, fără redirect la login; pagina și fiecare server action verifică din nou rolul:**
- **UAT-uri:** înrolare din OpenStreetMap (căutare Nominatim, doar relații `boundary=administrative` din MD/RO; geometria se descarcă din nou pe server la salvare), activare / dezactivare, ștergere, hartă cu toate primăriile;
- **Utilizatori:** invitație pe email (utilizatorul își alege singur parola pe `/parola-noua`; nimeni nu setează parola altcuiva), primărie + rol (`app_metadata`), link de resetare a parolei pe email, dezactivare, ștergere — prin Admin API, necesită `SUPABASE_SECRET_KEY` în `frontend/.env.local` (doar server, `src/lib/supabase/admin.ts`, `server-only`); fără cheie tab-ul arată pașii de configurare;
- **Survey-uri:** registrul survey-urilor și primăriile pe care le acoperă (intersecție spațială); înregistrarea pachetelor locale din `public/data/<id>/`;
- **Sistem:** verificări (Supabase, Auth, DB/RLS, sesiune, date, ortofoto), versiuni, link-uri, comenzi; **Jurnal:** `public.admin_audit_log` (append-only).
- **Baza de date** (migrație `supabase/migrations/20260926000100_platform_admin.sql`, aplicată pe proiect): `public.uat`, `public.survey`, `public.admin_audit_log`, view-uri `uat_public`, `survey_public`, `uat_survey` (security_invoker), RPC `enroll_uat`, `upsert_survey`; RLS: un utilizator citește doar primăria lui și survey-urile care îi intersectează limita, scrie doar `platform_admin`. Seed: `supabase/seed/uats_surveys.sql` (generat de `frontend/scripts/gen-seed-sql.mjs`). Aplicația citește primăria utilizatorului din DB (`src/lib/viewer.ts`); demo-ul folosește fixture-ul Sireți.
**Echipă și sarcini de teren (roluri pe primărie):**
- **Super-admin** înrolează primăria și pe **administratorul UAT** (buton „Adaugă administrator UAT” pe rândul primăriei → dialog precompletat, rol `uat_admin`).
- **Administratorul UAT** (`uat_admin`) își gestionează echipa în **`/echipa`** (doar el; oricine altcineva → 403): invită `inspector` / `viewer` numai în primăria lui (email Supabase `inviteUserByEmail` → membrul își alege parola pe **`/parola-noua`**, pagină deschisă fără sesiune care citește tokenii din fragmentul URL; „invitație trimisă” + „Retrimite invitația” până la prima setare a parolei), schimbă rolul, trimite link de resetare a parolei pe email, dezactivează, șterge; vede încărcarea fiecăruia (sarcini deschise / în lucru / rezolvate). Server actions în `src/app/[locale]/(app)/echipa/actions.ts` + `src/lib/team.ts` (Admin API, rol și primărie re-verificate din Auth; nu poate atinge alți administratori sau alte primării).
- **Sarcini** (**`/sarcini`**, orice cont de primărie): administratorul transformă țintele analizei AI (`targets.geojson`: goluri în rânduri, plante lipsă, deșeuri) în sarcini, le atribuie (responsabil, termen, prioritate), le șterge; inspectorul își actualizează sarcinile (în lucru / rezolvată + notă de teren); vizualizatorul doar citește. „Vezi pe hartă” → `/harta?tinta=T003`. Sarcinile, țintele și încărcarea echipei urmează doar survey-ul servit (`NEXT_PUBLIC_SURVEY_ID`; `activeSurveys` / `ACTIVE_SURVEY_FILTER` în `src/lib/tasks.ts`): o primărie cu mai multe zboruri (ex. `siret3-mock` și `siret3`) nu le amestecă; sarcinile fără survey rămân vizibile.
- **Notificări în aplicație** (ADR-026): clopoțelul din meniu (desktop: lângă logo; mobil: bara de sus) arată notificările contului, cu contor de necitite și toast live (Supabase Realtime). Acum: „V-a fost alocată o sarcină” (trigger pe `public.task` → `public.notification`, RLS: doar propriile notificări, clienții nu pot insera); click → `/sarcini?sarcina=<id>` (sarcina evidențiată). Componenta: `src/components/notifications/NotificationBell.tsx`; migrația `20260926000400_notifications.sql`; testat în SQL pe roluri și în `e2e/roles.spec.ts`.
- **Baza de date** (migrațiile `20260926000200_field_tasks.sql`, `…0300_task_guard_fix.sql`): `public.task` + view `task_public`; RLS: citire doar în propria primărie; creare/ștergere/atribuire doar `uat_admin` al primăriei; inspectorul actualizează doar sarcinile atribuite lui, iar trigger-ul `tg_task_guard` îi permite doar starea și nota (nu anulare, nu alte câmpuri); locația trebuie să fie în limita primăriei. Testat în SQL pe toate rolurile și în browser (`e2e/roles.spec.ts`, local cu `E2E_PASSWORD=…`).
- **Asistent** (ADR-027): butonul „Asistent” din meniu deschide un panou non-modal care explică pașii din aplicație și dă linkuri directe (rămâne deschis la navigare). `POST /api/assistant` → Gemini (`GEMINI_API_KEY`, `GEMINI_MODEL` în `frontend/.env.local`, doar server); promptul e în `src/lib/assistant/prompt.ts` și citează etichetele din `messages/*.json` (`pnpm test:scripts` le verifică). O funcție nouă în UI se descrie și acolo.
- limită cunoscută: fișierele din `/data` sunt publice (survey CC BY 4.0); izolarea reală pe UAT vine cu RLS în PostGIS (planul, §5).

Ce rulează acum (din `src/Web/frontend/`):

| Comandă | Ce face |
|---|---|
| `pnpm install` | dependențele (versiuni exacte, `pnpm-lock.yaml`) |
| `pnpm data` | `scripts/build-data.mjs`: ortofoto din cele 311 GeoTIFF (`public/ortho/`, ~33 s) + survey-ul `siret3-mock` din exemplele CVAT (`public/data/`), măsurat în EPSG:32635. Fără GeoTIFF-uri (CI) folosește `data/tiles_index.json` și sare peste ortofoto; cifrele rămân identice |
| `pnpm data:fast` | la fel, fără regenerarea ortofoto |
| `pnpm data:survey --survey siret3` | `scripts/build-survey.mjs`: bundle-ul AI real din `src/Web/data/surveys/siret3/pipeline/` (EPSG:32635) → `public/data/siret3/` (EPSG:4326, același format ca mock-ul); validează contractul §6 și iese cu 1 la erori. `--check` doar validează; `--emit-seed` scrie `supabase/seed/survey_siret3.sql`. Apoi `NEXT_PUBLIC_SURVEY_ID=siret3 pnpm build` (variabila se fixează la build) |
| `pnpm data:terrain --survey siret3` | `scripts/build-terrain.mjs`: relieful sintetic al vederii 3D din `public/data/<id>/canopies.geojson` → `public/data/<id>/terrain/{z}/{x}/{y}.png` (raster-dem terrarium, z14–z19) + `terrain.json`; opțiuni `--height 1.1 --blur 0.45`. `pnpm data`/`data:fast` îl rulează pentru mock; după `data:survey` se rulează din nou (ADR-025). Fără el, butonul 3D e dezactivat |
| `pnpm test:scripts` | teste `node:test` pentru convertor (fixture mini-bundle, cu `tiles.geojson` + 2 măști; `node scripts/survey/fixtures/make-mini-bundle.mjs` îl regenerează) și pentru relieful 3D |
| `pnpm dev` | `http://localhost:3000` (copiază întâi worker-ul MapLibre în `public/maplibre/`) |
| `pnpm build` · `pnpm start` | build de producție și server |
| `pnpm lint` · `pnpm typecheck` | ESLint, TypeScript |
| `pnpm e2e` | Playwright pe build-ul de producție (port 3100, sau `E2E_PORT`): KPI, deep link, rută, CSV, tabel, schimbarea limbii, straturile opționale tile-uri / mască (`e2e/overlays.spec.ts`), ferme și drumuri (`e2e/farms-roads.spec.ts`) |

Abateri față de plan, deliberate, pentru viteză:
- fișierele statice stau în `frontend/public/` (generate, ignorate de git), nu în `src/Web/data/` servite de o rută `/data`;
- ortofoto nu e PMTiles, ci un mozaic de ansamblu WebP (~0,4 m/px) plus JPEG-uri de 1024 px pe tile, încărcate doar la zoom ≥ 17;
- MapLibre 6 + Turbopack: worker-ul se servește din `public/maplibre/` prin `setWorkerUrl` (`scripts/copy-maplibre-worker.mjs`, rulat de `predev`/`prebuild`).

Pagini: `/` (panou general), `/harta`, `/blocuri`, `/ruta`. Deep link-uri: `/harta?rand=V02-R16`, `/harta?bloc=V01`.

---

## 1. Regula de aur

Tot codul, configul, asset-urile, testele, documentația și dependențele aplicației web trăiesc **exclusiv în `src/Web/`**. Nu crea și nu modifica fișiere în afara `src/Web/` pentru partea web.

- Datele produse de pipeline-ul AI se citesc din locațiile din §6 „Data contracts”. **Nu se copiază manual** în alte locuri. Fișierele derivate (static, PMTiles) le generează doar scripturile din `src/Web/scripts/`.
- Excepții în afara `src/Web/` (aprobate):
  - secțiunea „Web” din `CLAUDE.md`-ul din rădăcină;
  - `.github/workflows/web.yml`: CI-ul care rulează `make ci` în `src/Web/` (lint, teste unitare, e2e static);
  - linkul și instrucțiunile de rulare ale interfeței în `README.md`-ul din rădăcină (cerut de regulament).
- `src/Web/readme.md`, existent în repo, rămâne; la implementare devine indexul scurt spre acest fișier.
- Ignorările de git pentru web stau în `src/Web/.gitignore`, nu în `.gitignore`-ul din rădăcină.

## 2. Scop

1. **Juriu (obligatoriu, prioritate absolută):** o hartă cu:
   - ortofoto Sireț3;
   - coroane, axe de rând, inter-rânduri, deșeuri și ținte de inspecție;
   - ruta de mers cu `length_m` și START/FINISH;
   - `vineyard_id` / `row_id` la click;
   - aria coroanelor și aria inter-rândurilor (m² și ha), numărul de blocuri și de rânduri, lungimea fiecărui rând și lungimea totală.

   Totul trebuie să meargă **offline, în mod static**.
2. **Produs — Solemtrix:** o platformă SaaS pentru UAT-uri (primării), cu geofence, izolare reală a datelor pe UAT (RLS), dashboard, inspecții și rute, mod teren și audit pentru subvenții. UAT-ul demo este **comuna Sireți, raionul Strășeni, Republica Moldova**, cu limita administrativă reală din OSM (§6.6).

**Brand:** Solemtrix, cu logo și nume proprii. De la Marcaj preluăm doar paleta și structura vizuală, nu logo-ul și nici textele. În subsol apare mențiunea „adnotări corectate în Marcaj”.

**Infrastructură:** doar **Supabase** (proiectul și baza de date există deja). Frontend-ul și API-ul rulează pe laptop, cu Docker, iar demo-ul se face de pe laptop, cum permite regulamentul. Nu folosim Vercel, Railway sau Cloudflare.

## 3. Stack (versiuni exacte, verificate pe npm/PyPI la 25.09.2026)

| Strat | Tehnologie | Versiune |
|---|---|---|
| Runtime JS | Node.js · pnpm | 22.x LTS (local 22.23.2) · 10.33.0 |
| Frontend | Next.js (App Router, Turbopack) · React | 16.3.6 · 19.3.0 |
| Limbaj | TypeScript | **5.9.3**. Nu 7.x: `openapi-typescript` cere `^5`, `typescript-eslint` cere `<6.1` |
| UI | @mui/material · @mui/icons-material · @mui/material-nextjs · @emotion/react · @emotion/styled · @emotion/cache | 9.4.0 · 9.4.0 · 9.4.0 · 11.14.0 · 11.14.1 · 11.14.0 |
| Tabele / grafice | @mui/x-data-grid · @mui/x-charts (MIT) | 9.14.0 · 9.14.0 |
| Font | @fontsource-variable/inter | 5.3.0 |
| Hartă | maplibre-gl · @vis.gl/react-maplibre · pmtiles · proj4 | 6.11.2 · 8.1.3 · 4.5.0 · 2.22.0 |
| i18n | next-intl | 4.14.7 |
| Auth / DB client | @supabase/supabase-js · @supabase/ssr | 2.117.2 · 0.12.7 |
| Date / validare | @tanstack/react-query · zod | 5.103.2 · 4.6.5 |
| Client API | openapi-typescript · openapi-fetch | 7.13.0 · 0.17.0 |
| Calitate FE | eslint · eslint-config-next · eslint-config-prettier · prettier · vitest · @testing-library/react · @playwright/test | **9.39.5** (10 nu e suportat de eslint-plugin-react) · 16.3.6 · 10.1.8 · 3.9.9 · 5.0.2 · 16.3.3 · 1.63.0 |
| Runtime Python | Python · uv | 3.12 · 0.12.19 |
| API | fastapi · uvicorn · pydantic · pydantic-settings · python-multipart · orjson | 0.141.1 · 0.54.0 · 2.13.5 · 2.15.0 · 0.0.32 · 3.12.0 |
| Geo (Python) | shapely · pyproj · geopandas · pyogrio · numpy | 2.1.2 · 3.8.0 · 1.1.4 · 0.13.0 · 2.5.3 |
| DB (Python) | psycopg[binary] · psycopg-pool | 3.3.6 · 3.3.3 |
| Auth (Python) | pyjwt[crypto] | 2.15.0 |
| Exporturi | gpxpy · simplekml · weasyprint · jinja2 | 1.6.2 · 1.3.6 · 70.0 · 3.1.6 |
| Calitate BE | ruff · mypy · pytest · pytest-asyncio · httpx | 0.16.9 · 2.3.1 · 9.1.1 · 1.4.0 · 0.28.1 |
| Supabase | Supabase CLI · Postgres + PostGIS | 2.118.0 (local 2.116.0, de actualizat) · proiectul cloud existent (`supabase link`) |
| Tile-uri | GDAL (Docker) · go-pmtiles · tippecanoe | 3.13.3 · 1.31.2 · 2.79.0 |

Versiunile se fixează exact: `package.json` fără `^`/`~` plus `pnpm-lock.yaml`; `pyproject.toml` cu `==` plus `uv.lock`. O versiune se schimbă doar cu o notă în `docs/DECISIONS.md`.

## 4. Structura de foldere

```
src/Web/
├── CLAUDE.md                 acest fișier
├── readme.md                 existent; index scurt spre CLAUDE.md
├── Makefile                  punctul unic de intrare (vezi §5)
├── docker-compose.yml        frontend + api (conectate la Supabase cloud); profilul `static` = doar frontend
├── .env.example              toate variabilele, fără valori secrete
├── .gitignore                data/**, .env*, node_modules, .next, .venv, *.pmtiles …
├── docs/                     PLAN.md · DECISIONS.md · DEMO.md · PERF.md (la implementare) · design/
├── frontend/                 Next.js
│   ├── messages/             ro.json · en.json · ru.json
│   ├── public/               logo propriu, iconițe, manifest.webmanifest, sw.js
│   └── src/
│       ├── app/[locale]/     (auth)/login · (app)/{dashboard,map,blocks,routes,field,audit,admin}
│       ├── app/data/[...path]/route.ts   servește src/Web/data/…/static cu suport Range (mod static local)
│       ├── proxy.ts          next-intl + refresh sesiune Supabase (Next 16: proxy.ts, nu middleware.ts)
│       ├── components/       shell/ · map/ · kpi/ · tables/ · common/
│       ├── features/         map · measurements · route · targets · audit · admin
│       ├── lib/api/          schema.d.ts (GENERAT) · client.ts (openapi-fetch + bearer)
│       ├── lib/data/         DataSource (interfață) · static.ts · api.ts · index.ts (alege după DATA_MODE)
│       ├── lib/supabase/     server.ts · browser.ts
│       ├── lib/format/       units.ts (m, m², ha, km, min; Intl per locale)
│       └── theme/            tokens.ts · theme.ts · mapPalette.ts  ← SINGURUL loc cu culori
├── api/                      FastAPI
│   ├── pyproject.toml · uv.lock · Dockerfile
│   ├── app/
│   │   ├── main.py · config.py · auth.py (JWKS/HS256) · db.py (pool + rls_tx)
│   │   ├── routers/          me · uats · surveys · features · tiles · route · targets · measure · audits · admin
│   │   ├── geo/              crs.py · measure.py (SURSA ADEVĂRULUI) · bundle.py (validarea contractului) · mvt.py
│   │   ├── exports/          csv.py · gpx.py · kml.py · pdf.py
│   │   └── templates/        audit.html
│   └── tests/                fixtures/ · test_measure_examples.py · test_crs_control.py · test_contract.py · test_rls_api.py
├── supabase/                 config.toml · migrations/ · seed.sql · tests/ (pgTAP, RLS)
├── scripts/                  Python rulat cu `uv run --project api` (+ shell pentru GDAL/tippecanoe în Docker)
└── data/                     ignorat de git, cu excepția data/README.md
    ├── surveys/<survey_id>/pipeline/   INTRARE de la AI (§6)
    ├── surveys/<survey_id>/static/     IEȘIRE generată pentru modul static
    └── ortho/<survey_id>.pmtiles       ortofoto generat
```

## 5. Comenzi (toate PLANIFICATE, se rulează din `src/Web/`)

| Comandă | Ce face |
|---|---|
| `make up` | `docker compose up`: frontend :3000 + api :8000, conectate la proiectul Supabase cloud existent (din `.env`) |
| `make demo` | modul static, offline: `docker compose --profile static up` (doar frontend, fără API, fără Supabase) |
| `make dev` | frontend `pnpm dev` + api `uv run uvicorn --reload`, fără Docker |
| `make build` | `pnpm build` + `docker build` api |
| `make test` | vitest + pytest + `supabase test db --linked` (pgTAP, fiecare test într-o tranzacție cu ROLLBACK) |
| `make e2e` | Playwright pe fluxul de demo, în mod static |
| `make lint` | eslint + prettier --check + tsc --noEmit + ruff + mypy |
| `make check` | lint + test + e2e (obligatoriu înainte de push) |
| `make ci` | ce rulează CI-ul (`.github/workflows/web.yml`): lint + vitest + pytest fără DB + e2e static pe `siret3-mock` |
| `make db-push` | `supabase db push` (migrațiile din `supabase/migrations/` în proiectul legat) |
| `make seed` | `scripts/seed_demo.py`: UAT-urile Sireți și Cojușna (limite OSM), utilizatorii demo, vizite și audituri exemplu |
| `make gen-api` | exportă `/openapi.json` și regenerează `frontend/src/lib/api/schema.d.ts` |
| `make mock` | `scripts/cvat_to_bundle.py` → bundle `siret3-mock` din cele 2 tile-uri exemplu |
| `make data SURVEY=siret3` | validează bundle-ul → `build_static` → PMTiles coroane → (opțional) import în DB |
| `make ortho SURVEY=siret3` | 311 GeoTIFF → `data/ortho/siret3.pmtiles` (GDAL + pmtiles, în Docker) |
| `make bench` | `scripts/bench.py`: timpi de import, build static, latențe API → `docs/PERF.md` |

## 6. Data contracts

> **Stare:** propunere. Se confirmă cu echipa AI când vin primele date. Până atunci, UI-ul se construiește pe mock (`make mock`), care respectă exact acest contract.

### 6.1 Locații

| Ce | Unde | Cine scrie |
|---|---|---|
| Bundle AI (intrare), EPSG:32635 | `src/Web/data/surveys/<survey_id>/pipeline/` | pipeline-ul AI (`--out` direct acolo) sau `make mock` |
| Date organizatori | `data & info/02_route/*.geojson` (citite pe loc, nu copiate) | organizatorii |
| Tile-uri GeoTIFF | `data & info/01_tiles/siret3_challenge_tiles_part*of5/*.tif` | organizatorii |
| Exemple CVAT | `data & info/05_examples/siret3_examples_cvat/annotations.xml` | organizatorii |
| Static (ieșire), EPSG:4326 | `src/Web/data/surveys/<survey_id>/static/` | `scripts/build_static.py` |
| Ortofoto | `src/Web/data/ortho/<survey_id>.pmtiles` | `scripts/build_ortho.sh` |
| Livrabilele din rădăcină | `/route.geojson`, `/measurements.csv` | pipeline-ul AI (NU web-ul) |

`survey_id`: `siret3` (real) și `siret3-mock` (din exemple). Căile se configurează prin `WEB_DATA_DIR` (implicit `src/Web/data`) și `ORGANIZER_DATA_DIR` (implicit `data & info`).

### 6.2 Reguli comune pentru bundle-ul de intrare

- GeoJSON `FeatureCollection` în **EPSG:32635** (metri), cu `"crs": {"type":"name","properties":{"name":"urn:ogc:def:crs:EPSG::32635"}}`. Coordonate cu ≥ 3 zecimale (mm).
- Toate ID-urile sunt string-uri, **globale pe survey**: aceeași valoare înseamnă același obiect fizic.
- Enum-urile sunt exact cele din regulament, cu litere mici:
  - `row_structure` ∈ {`regular`, `disrupted`, `unassessable`};
  - `interrow_cover` ∈ {`bare_soil`, `vegetation`, `mixed`, `unassessable`}.
- Câmpurile marcate „opțional” pot lipsi. Web-ul **recalculează** toate ariile și lungimile (`api/app/geo/measure.py`), iar valorile pipeline-ului se folosesc doar la verificare: diferența de peste 0,5% dă o avertizare la import.
- `canopies` poate fi și GeoJSONSeq (`canopies.geojsonl`, un Feature pe linie) pentru volum mare.

### 6.3 Fișierele bundle-ului

| Fișier | Geometrie | Proprietăți (obligatorii **îngroșat**) |
|---|---|---|
| `manifest.json` | — | **`survey_id`**, **`name`**, **`captured_at`** (`2025-05-20`), **`gsd_m`** (0.025), **`crs`** (`EPSG:32635`), **`source`** (`3DATA COLLECT / OpenAerialMap`), **`license`** (`CC BY 4.0`), **`stage`** (`model` \| `marcaj_corrected`), **`generated_at`** (ISO 8601), `pipeline_version` (git sha), `tiles` (311) |
| `blocks.geojson` | Polygon \| MultiPolygon | **`vineyard_id`**, `area_m2` |
| `rows.geojson` | LineString \| MultiLineString, **un Feature per rând fizic** (piesele din tile-uri unite) | **`row_id`**, **`vineyard_id`**, **`row_structure`** (agregat: `disrupted` dacă vreo piesă e disrupted; `unassessable` dacă toate sunt; altfel `regular`), `length_m`, `plant_count`, `max_gap_m`, `tile_structures` (`{"siret3_r021_c012": "regular", …}`) |
| `canopies.geojson(l)` | Polygon | **`canopy_id`** (`<tile>#<n>`), **`vineyard_id`**, **`tile`**, `row_id` |
| `interrows.geojson` | Polygon | **`interrow_id`**, **`vineyard_id`**, **`interrow_cover`**, **`tile`**, `row_ids` (`["V01-R03","V01-R04"]`), `area_m2` |
| `waste.geojson` | Polygon (casetă aliniată la axe) | **`waste_id`**, **`vineyard_id`** (sau `null` dacă e la peste 10 m de un bloc), **`tile`**, `confidence` (0–1) |
| `targets.geojson` | Point | **`target_id`** (`T001`…), **`type`** (`gap` \| `missing` \| `waste`), **`vineyard_id`**, **`row_id`** (sau `null`), `waste_id`, **`route_order`** (int sau `null` dacă e neatins), **`reachable`** (bool), `gap_length_m`, `note` |
| `route.geojson` | un Feature LineString (bare sau într-un FeatureCollection) | **`length_m`**, `duration_min` (la 4 km/h), `baseline_length_m` (ruta de referință pentru economii), `outside_share` (fracția din lungime în afara zonei permise) |
| `measurements.csv` | — | vezi §6.4 |
| `tiles.geojson` *(opțional, web bundle v3)* | Polygon, **exact un Feature per tile furnizat** (311): pătratul de 51,2 m al grilei §6.6, CCW, 3 zecimale | **`tile`** (`siret3_rNNN_cNNN`), **`status`** (`vineyard` dacă tile-ul are vreo piesă de rând sau coroană în AnnSet, altfel `no_vineyard`), **`n_rows`**, **`n_canopies`**, **`n_interrows`**, **`n_waste`** (int), **`veg_frac`**, **`nodata_frac`** (0–1, 3 zecimale, sau `null`), **`review_priority`** (int sau `null`, din coada de QA), **`review_note`**, **`review_status`** (`null` sau `missed` \| `partial` \| `verify`, din `src/AI/configs/tile_review.csv`: tile-ul trebuie completat în Marcaj), **`has_mask`** (bool). Chei nullable prezente mereu |
| `masks/<tile>.png` *(opțional)* | raster | masca de vegetație a pipeline-ului (`tile_prep`, Lab a*), 1024×1024 px, un canal, ≠ 0 = vegetație, pixel (0,0) = colțul NV al tile-ului (aceeași amprentă ca în `tiles.geojson`). Există exact pentru tile-urile cu `has_mask: true`; `manifest.json` are atunci `counts.tiles` și `"masks": {"dir": "masks", "px": 1024, "n": <nr>, "source": "tile_prep vegetation mask (Lab a*)"}` |
| `farms.geojson` *(opțional)*, `"name": "farms"` | Polygon \| MultiPolygon: o fermă = blocuri de vie la ≤ 10 m unul de altul, fără drum public între ele | **`farm_id`** (`F01`…), **`vineyard_ids`** (listă de `vineyard_id`, **sursa adevărului pentru apartenență**: fiecare bloc în cel mult o fermă), `n_blocks` (int), `area_m2` (aria conturului); opțional (instantaneu cadastral): `n_parcels` (int \| `null`), `cadastral_codes` (string[] \| `null`), `landuse_counts` (`{folosință: nr}` \| `null`) |
| `roads.geojson` *(opțional)*, `"name": "roads"` | LineString \| MultiLineString | **`road_id`** (`D0001`), **`road_class`** (`public` \| `field` \| `internal`), `highway` (valoarea OSM: `residential`, `tertiary`, `track`, `path`, `service`…, sau `cross_path` = trecere prin rânduri detectată de AI), `name`, `surface`, `farm_id` (setat pe drumurile interne), `length_m` (float), `source` (`osm` \| `detected` \| `cadastre` = drum public găsit doar în cadastru), `cadastral` (bool \| `null`: o parcelă oficială „Cale de comunicaţie” acoperă drumul). Cheile lipsă se citesc `null`; `length_m` lipsă se măsoară în UTM |

`blocks.geojson` primește `farm_id` (string \| `null`) și, opțional, aceleași 3 câmpuri cadastrale ca fermele; `manifest.json` `counts` poate avea `farms` și `roads` (verificate ca celelalte contoare).

Un bundle fără `tiles.geojson` (mock-ul, bundle-urile mai vechi) e valid: convertorul nu scrie nimic pentru ele, iar comutatoarele „Tile-uri” și „Mască vegetație” din hartă apar dezactivate, cu o explicație scurtă. Verificări (`scripts/survey/tiles.mjs`, `masks.mjs`): ID-uri, enum-uri, contoare, fracții, amprenta = pătratul din grilă (±2 mm), PNG lizibil de 1024 px pentru fiecare `has_mask`; număr de tile-uri ≠ `manifest.tiles`, `status` în dezacord cu contoarele, `masks.n` greșit sau PNG-uri în plus sunt doar avertizări.

La fel pentru `farms.geojson` / `roads.geojson` (fiecare opțional; `scripts/survey/farms.mjs`): erori = CRS, geometrie, `farm_id` / `road_id` unice, `vineyard_ids` gol sau cu blocuri necunoscute, un bloc în două ferme, `road_class` / `source` în afara enum-ului, `length_m` negativ, tipuri greșite ale câmpurilor cadastrale; avertizări = blocuri fără fermă (fermele nu mai dau totalul), `n_blocks` ≠ `vineyard_ids`, `farm_id` al blocului în dezacord cu fermele (fermele câștigă), `farm_id` al unui drum necunoscut.

### 6.4 `measurements.csv`

UTF-8, separator `,`, punct zecimal, antet obligatoriu, metri și m² cu 2 zecimale, ha cu 4 zecimale. Web-ul exportă exact același format.

```
level,vineyard_id,row_id,block_count,row_count,row_length_m,canopy_area_m2,canopy_area_ha,interrow_area_m2,interrow_area_ha,plant_count,row_structure
survey,,,12,640,51234.50,18234.10,1.8234,40123.40,4.0123,21034,
block,V01,,,25,910.10,237.10,0.0237,2068.00,0.2068,399,
row,V01,V01-R01,,,1.10,,,,,3,regular
```

Semantica măsurătorilor (identică la pipeline, API și static):
- **aria coroanelor** = aria reuniunii poligoanelor `canopy`;
- **aria inter-rândurilor** = aria reuniunii poligoanelor `interrow`;
- **lungimea unui rând** = suma lungimilor pieselor lui;
- **lungimea totală** = suma lungimilor tuturor rândurilor;
- toate se calculează plan, în EPSG:32635, fără corecție de teren.

### 6.5 Ieșirea statică (`static/`, generată)

- `manifest.json`: manifestul de intrare, plus `bbox_4326`, `layers[]`, `ortho_url`, `build_at`.
- `summary.json`: KPI, adică exact valorile rândului `survey` și ale rândurilor `block` din CSV.
- `blocks|rows|interrows|waste|targets|route|passages|forbidden|study_area|start.geojson`, în **EPSG:4326** (RFC 7946, 7 zecimale). Proprietățile de măsurare sunt precalculate în UTM, deci frontend-ul nu calculează nimic.
- `canopies.pmtiles`: vector tiles (tippecanoe, z15–z21, strat `canopy`).
- `measurements.csv`.

Implementarea curentă (`scripts/build-survey.mjs` → `frontend/public/data/<survey>/`), pentru straturile opționale din §6.3:
- `tiles.geojson` în EPSG:4326 (7 zecimale, `id` numeric = i+1, toate proprietățile păstrate); `summary.json` primește atunci blocul `tiles`: `{total, vineyard, no_vineyard, to_complete}` (`to_complete` = tile-uri cu `review_status` setat). Panoul general afișează „311 tile-uri: X cu vie · Y fără vie · Z de completat în Marcaj”.
- `masks/<tile>.png`: PNG 1 bit cu paletă de 2 intrări (index 0 complet transparent, index 1 = `vegMask` din `src/theme/rasterColors.json`, cu alfa parțial), 1024 px, scris de `scripts/survey/png.mjs` (octeți exacți, fără cuantizarea sharp); `masks/index.json` = `{tile: [[lon,lat] × 4]}` în ordinea sursei imagine MapLibre (TL, TR, BR, BL).
- Culoarea măștii e „coaptă” în PNG-uri de un script Node, deci nu poate citi `tokens.ts`: stă în `src/theme/rasterColors.json` (tot în `src/theme/`, singura sursă a culorilor), pe care `tokens.ts` îl re-exportă (`color.map.vegMask`, `vegMaskAlpha`) pentru legendă. Schimbi culoarea într-un singur loc, apoi rulezi din nou `pnpm data:survey`.
- Hartă: comutatoarele „Tile-uri” (contur + umplere pe status: cu vie / fără vie / de completat în Marcaj; click → tile, status, contoare, % vegetație, nota de revizuire) și „Mască vegetație” (surse imagine doar pentru tile-urile din cadru, la zoom ≥ `ZOOM.orthoDetail`, plafonate ca ortofoto de detaliu), ambele oprite implicit și descărcate doar la prima pornire (`src/components/map/overlays/`).
- `farms.geojson` în EPSG:4326 (toate proprietățile păstrate, `n_blocks` = lungimea `vineyard_ids`, `area_m2` măsurat în UTM dacă lipsește, plus `label_point` = `[lon, lat]` în interiorul conturului, pentru etichetă) și `roads.geojson` (chei opționale prezente cu `null`, `length_m` cu 2 zecimale). `blocks.geojson` primește `farm_id` din `vineyard_ids` ale fermelor (ele câștigă; altfel valoarea blocului, altfel `null`).
- `summary.json` primește atunci `farms[]`: `{farm_id, vineyard_ids, n_blocks, area_m2, row_count, row_length_m, canopy_area_m2, interrow_area_m2, plant_count, target_count}` (+ `n_parcels` dacă fermele îl au), **agregat din `blocks[]` (liniile `block` din CSV)**, deci suma fermelor = rândul `survey` exact: reziduul de rotunjire al liniilor de bloc (≤ 0,5 cenți per linie) se dă, cent cu cent, celor mai mari ferme, doar când fiecare bloc e într-o fermă; `target_count` = țintele scrise (respectă `--targets`), după `vineyard_id`. Plus `totals.farm_count` și `roads: {public_m, field_m, internal_m}` (suma lungimilor rotunjite, pe clasă).
- Hartă: comutatoarele „Ferme”, „Drumuri” (publice + de câmp, aceeași culoare `map.roadNetwork`, publicul mai gros, lățimi interpolate pe zoom) și „Drumuri interne” (`map.roadInternal`, întrerupt), **pornite implicit** când fișierul există (altfel dezactivate cu explicație, fără cereri), descărcate o dată la montare (`useFarmsRoads`). Drumurile stau deasupra ortofoto și a tile-urilor, sub blocuri, coroane și rânduri; fermele = contur + umplere foarte ușoară + eticheta „F01” (marker DOM de la zoom 14,5: stilul de bază nu are glyphs și rămâne offline). Click fermă → blocuri, suprafață (m² · ha), rânduri, lungime, coroane, inter-rânduri, plante, ținte, instantaneu cadastral dacă există; click drum → clasă, tip OSM, nume, suprafață, lungime, fermă, sursă, „confirmat în cadastru”. KPI „N ferme” în banda de sus; atribuirea „© OpenStreetMap contributors” vine cu sursa `roads`.
- Cadastru live (`src/lib/cadastre.ts`, singurul loc cu URL-ul, stratul și limitele): WMS-ul public AGCC `https://geodata.gov.md/geoserver/wms`, stratul `cadastru_data:terenuri` (fără login, CORS `*`, Fees=NONE, date informative). Comutatorul „Cadastru (AGCC, live)” e **oprit implicit** (nicio cerere externă până e pornit); rasterul se cere doar de la zoom 16 (`CADASTRE_MIN_ZOOM`) și stă sub toate straturile vectoriale. Click pe teren gol (niciun obiect al nostru sub click) cu stratul pornit → GetFeatureInfo (`info_format=application/json`, timeout 8 s) → panou „Parcelă cadastrală” cu număr cadastral, cod parcelă, destinație, suprafață, tip proprietate și conturul parcelei evidențiat; `description` (HTML) de la server nu se afișează niciodată. Atribuirea „Cadastru © AGCC (geodata.gov.md), informativ” vine cu sursa raster. e2e: `e2e/cadastre.spec.ts` (ambele cereri AGCC simulate).

### 6.6 Fapte de control (sunt teste, nu presupuneri)

- Originea tile-ului `siret3_r<r>_c<c>` (colțul stânga-sus, EPSG:32635): `x = 628992 + c·51.2`, `y = 5220966.4 − (r−5)·51.2`, pixel 0,025 m, 2048×2048 px. Verificat pe tag-urile GeoTIFF la r005_c004 și r018_c010. Tile-urile au r 5–39 și c 0–33.
- 311 tile-uri = 81,5268 ha. Compresie JPEG-in-TIFF.
- **Geofence-ul UAT Sireți** este relația OSM `19100171` (boundary=administrative, comuna Sireți, raionul Strășeni, MD-3731), ~2.729 ha, date © OpenStreetMap contributors, ODbL. Verificat: centrele tuturor celor 311 tile-uri și START-ul sunt în interiorul ei, deci survey-ul acoperă ~3% din comună. Relația `18967831` e doar vatra satului: nu e UAT-ul, nu se folosește.
- **UAT-ul de control pentru izolare:** comuna **Cojușna** (vecinul de la vest, raionul Strășeni), tot din OSM. Nu se suprapune cu survey-ul, deci un utilizator Cojușna vede 0 obiecte.
- START `629504.70, 5220250.75` ↔ `47.1230335 N, 28.7073776 E` (în tile-ul r018_c010).
- Exemple (verificate):

  | Tile | Coroane | Aria coroanelor | Rânduri | Lungime rânduri | Inter-rânduri | Aria inter-rândurilor |
  |---|---|---|---|---|---|---|
  | r021_c012 | 399 | 237,1 m² | 25 | 910,1 m | 24 | 2068,0 m² |
  | r006_c004 | 251 | 299,1 m² | 26 | 1031,5 m | 25 | 1996,4 m² |

## 7. Design — tokeni (sursa: marcaj.com și `docs/design/*.png`)

**Regula:** culorile, umbrele, razele și fonturile se definesc **o singură dată**, în `frontend/src/theme/tokens.ts`. Tema MUI (`theme.ts`, `cssVariables: true`, light și dark) și paleta hărții (`mapPalette.ts`) derivă din ei. **Nicio culoare hardcodată în componente.** ESLint interzice literalii `#rrggbb`, `rgb(`, `rgba(` în afara `src/theme/`.

```
font-family: Inter, -apple-system, "Segoe UI", sans-serif;  body 16px / 24px, weight 400
spacing unit: 4px
radius: sm 4px · md 6px (default) · lg 8px (butoane, input-uri, carduri) · xl 12px · pill 9999px

primary:   50 #EEF2FF · 100 #E0E7FF · 200 #C7D2FE · light #6366F1 · main #4F46E5 · dark #4338CA · contrast #FFFFFF
secondary: light #929292 · main #575757 · dark #202020
background: default #F6F6F6 · paper #FFFFFF
text: primary #202020 · secondary #575757 · disabled #B3B3B3
divider: #EAEAEA
grey: 50 #FAFAFA · 100 #F6F6F6 · 200 #EAEAEA · 300 #D2D2D2 · 400 #B3B3B3 · 500 #929292 · 600 #575757 · 700 #333333 · 800 #202020 · 900 #101010
success: 50 #F0FDFA · main #0D9488 · dark #09675F
warning: 50 #FEF3C7 · main #D97706 · dark #975304
error:   50 #FEF2F2 · main #C10007 · light #FF6467 · dark #870004
info:    50 #DFF2FE · main #00A6F4 · dark #0074AA
action: active rgba(0,0,0,.54) · hover rgba(0,0,0,.04) · selected rgba(0,0,0,.08)

elevation-sm:   0 1px 2px 0 rgba(0,0,0,.05)
elevation-base: 0 1px 2px 0 rgba(0,0,0,.06), 0 1px 3px 0 rgba(0,0,0,.10)
elevation-md:   0 2px 4px -1px rgba(0,0,0,.06), 0 4px 6px -1px rgba(0,0,0,.10)
elevation-lg:   0 4px 6px -2px rgba(0,0,0,.05), 0 10px 15px -3px rgba(0,0,0,.10)

dark theme: primary main #60A5FA · light #93C5FD · dark #3B82F6 · contrast #0F0E0C
            (restul, background/paper/text/grey pentru dark: de extras de pe marcaj.com la F0; până atunci MUI dark implicit)
```

### 7.1 Paleta hărții (`mapPalette.ts`, aceeași în light și dark)

Paleta hărții **nu se schimbă cu tema**: fundalul e mereu ortofoto. În dark se schimbă doar masca geofence-ului și chrome-ul UI. Pentru contrast pe sol maro, umbre și iarbă, fiecare linie are un contur alb (casing). Ajustările față de propunerea inițială sunt argumentate în `docs/DECISIONS.md` (ADR-012).

| Strat | Culoare | Stil |
|---|---|---|
| Geofence UAT (Sireți) | primary.main `#4F46E5` + casing alb | contur 2 px întrerupt [4,2]; exterior mascat cu grey-800 60% (în dark: grey-900 70%) |
| Coroane `vineyard` | fill `#2DD4BF` (teal-400) 40%, contur success.main `#0D9488` 1 px | vizibile de la z18 |
| Rând `regular` | primary.light `#6366F1` | 2 px + casing alb 3,5 px 70% |
| Rând `disrupted` | `#F59E0B` (amber-500) | 3 px + casing alb |
| Rând `unassessable` | grey-300 `#D2D2D2` | 2 px, punctat [1,2] |
| Inter-rând `bare_soil` · `mixed` · `vegetation` · `unassessable` | `#D6C3A1` · `#A7D7C5` · `#3DA99F` · grey-300 | fill 25%, contur 1 px 60% din aceeași culoare |
| Deșeu `waste` | error.main `#C10007` | casetă 2 px + pin; halo alb |
| Țintă inspecție | info.main `#00A6F4` | cerc 14 px, contur alb 2 px, număr alb = `route_order` |
| Ruta | primary.main `#4F46E5` | 4 px, halo alb 7 px, săgeți de direcție la fiecare 80 px |
| START/FINISH | grey-900 `#101010` | marker pătrat 18 px, contur alb, eticheta „S/F” |
| Pasaje autorizate | grey-400 `#B3B3B3` | fill 30% |
| Zone interzise | error.50 `#FEF2F2` 35% + hașură error.main | contur error.main 1 px |
| Tile `vineyard` · `no_vineyard` · de completat în Marcaj | success.main `#0D9488` · grey-400 `#B3B3B3` · `#F97316` (`map.tileToComplete`) | fill 14% · 5% · 24%, contur 1,5 · 0,75 · 3 px; „de completat” (`review_status` setat) are prioritate față de status |
| Drum public · de câmp | `#EA580C` (`map.roadNetwork`, orange-600: mai roșu decât ambra rândurilor întrerupte) | linie + casing alb 65%; public mai gros; lățime interpolată z13 → z19 (public 1,4 → 9 px, câmp 0,7 → 5 px) |
| Drum intern al fermei | `#FDE047` (`map.roadInternal`, galben lămâie) | linie întreruptă [2; 1,2] + casing grey-900 35%; 0,8 → 4,5 px |
| Fermă | contur `#EC4899` (`map.farmLine`) · etichetă `#831843` (`map.farmLabel`) cu halou alb | contur 1,2 → 3 px + casing alb, fill 6%; eticheta „F01” de la zoom 14,5 |
| Mască vegetație (pipeline) | `#D946EF` 55% (`rasterColors.json` → `map.vegMask`) | copt în PNG-urile `masks/`, raster sub toate straturile vectoriale |

Culorile noi (`#2DD4BF`, `#F59E0B`, cele trei culori de inter-rând) intră în `tokens.ts`, în grupul `map`, nu în componente. Validare: `scripts/check_palette.py` calculează contrastul fiecărei culori față de culoarea medie a solului și a vegetației din cele 2 tile-uri exemplu și cere un raport ≥ 3:1 (cu casing).

## 8. Convenții

- **Limbă:** UI în RO (implicit), EN și RU prin `next-intl`, fără texte hardcodate în componente. Cod, identificatori, commit-uri și nume de fișiere sunt în engleză.
- **Unități:** fiecare cifră afișată are unitate (m, m², ha, km, min) și sursă (survey + dată: „Sireț3 · 20.05.2025”). Formatarea se face doar prin `lib/format/units.ts`, cu `Intl.NumberFormat` pe locale (`ro`: `1.234,5 m²`, `0,82 ha`). Sub 10.000 m² se afișează m², iar ha se afișează mereu alături.
- **Măsurători:** sursa adevărului e `api/app/geo/measure.py` (UTM). Frontend-ul **nu** calculează arii sau lungimi oficiale (fără turf). Singura excepție e unealta ad-hoc de măsurare pe hartă: proiectează în EPSG:32635 cu proj4 și calculează plan (ADR-010).
- **Coordonate:** în DB și în bundle, EPSG:32635. Către hartă, EPSG:4326. Web Mercator există doar în randarea MapLibre, niciodată în calcule.
- **Securitate:**
  - `service_role` / conexiunea de admin a DB există doar în scripturi și în rutele `/admin`, niciodată în frontend sau în `NEXT_PUBLIC_*`;
  - fiecare request FastAPI cu utilizator rulează într-o tranzacție cu `SET LOCAL ROLE authenticated` și `request.jwt.claims`, deci RLS rămâne activ;
  - nu se pun chei în git; variabilele sunt în `.env.example`.
- **Date în git:** nimic din `src/Web/data/` în afară de `README.md`. Fișierele mari (PMTiles, bundle) stau doar local, pe laptopul de demo; se regenerează cu `make ortho` și `make data`.
- **Client API:** `schema.d.ts` nu se editează de mână (`make gen-api`).
- **Stil de cod:**
  - TS: Prettier implicit, importuri absolute `@/`, componente server implicit, `"use client"` doar pentru hartă și formulare;
  - Python: ruff (line 100), mypy strict pe `app/`.
- **Moduri de date:** `NEXT_PUBLIC_DATA_MODE=static|api`. Trecerea mock → real înseamnă doar `NEXT_PUBLIC_SURVEY_ID=siret3-mock` → `siret3`.
