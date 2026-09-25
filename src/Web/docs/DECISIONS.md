# DECISIONS — ADR-uri scurte (web)

Format: **Context · Decizie · Respins (și de ce) · Consecințe.** Starea tuturor: *propus*, până la aprobarea planului. v1.1 (25.09): actualizat după deciziile echipei — `src/Web/`, un singur om, doar Supabase, brandul Solemtrix, geofence Sireți, CI permis. Versiunile au fost verificate pe npm/PyPI/GitHub la 25.09.2026 (`src/Web/CLAUDE.md` §3).

---

### ADR-001 — Totul în `src/Web/`, livrare static-first
- **Context:** interfața e criteriu de admitere, Wi-Fi-ul de la eveniment nu e de încredere, iar datele AI vin târziu.
- **Decizie:**
  - tot ce ține de web stă în `src/Web/` (decizia echipei: locația existentă din repo, nu `web/`);
  - cerințele juriului se livrează întâi în modul static: fișiere pre-generate de Python, servite de Next, fără API și fără DB;
  - modul API și Supabase se adaugă peste, prin aceeași interfață `DataSource`.
- **Respins:**
  - API-first: demo-ul ar depinde de 3 servicii cloud;
  - două aplicații separate: cod dublat.
- **Consecințe:** un singur `measure.py` alimentează ambele moduri; fișierele statice nu au geofence, deci servesc doar un survey public (CC BY 4.0).

### ADR-002 — Versiuni fixate; TypeScript 5.9.3 și ESLint 9.39.5, nu ultimele majore
- **Context:** au apărut TypeScript 7.0.2 și ESLint 10.11.0.
- **Decizie:** TS 5.9.3 și ESLint 9.39.5; restul pe ultimele versiuni stabile, fixate exact, cu lockfile.
- **Respins:**
  - TS 7: `openapi-typescript@7.13.0` cere `typescript ^5.x`, iar `typescript-eslint@8.70.1` cere `<6.1.0`;
  - ESLint 10: `eslint-plugin-react` (dependență a `eslint-config-next`) acceptă doar `^9.7`.
- **Consecințe:** revizuim după hackathon.

### ADR-003 — pnpm 10 pentru frontend, uv pentru Python
- **Decizie:** pnpm 10.33.0 (deja instalat), uv 0.12.19 (`uv.lock`).
- **Respins:**
  - npm: mai lent, lockfile mai zgomotos;
  - poetry: mai lent, lock mai greu de reprodus în Docker;
  - pip-tools: fără gestionarea Python-ului.
- **Consecințe:** uv trebuie instalat (`brew install uv`). Dockerfile-ul API folosește `uv sync --frozen`.

### ADR-004 — UI kit: MUI 9 cu CSS variables și Emotion
- **Context:** Marcaj e construit pe MUI cu CSS variables, iar aplicația trebuie să arate ca o extensie a lui.
- **Decizie:**
  - `@mui/material` 9.4.0 cu `createTheme({ cssVariables: true, colorSchemes: { light, dark } })`;
  - `@mui/material-nextjs` (`AppRouterCacheProvider`) pentru SSR;
  - MUI X (DataGrid, Charts) în varianta MIT.
- **Respins:**
  - Tailwind + shadcn/ui: am reface manual fiecare componentă ca să semene cu Marcaj;
  - Mantine / Chakra: alt limbaj vizual;
  - Pigment CSS: încă opțional în MUI 9 și cu risc de integrare cu Turbopack.
- **Consecințe:** toate culorile vin din `tokens.ts` prin temă; `sx` folosește doar chei de temă (`primary.main`, `grey.200`).

### ADR-005 — Ortofoto: PMTiles raster pre-generat (WebP, z14–z21), servit local
- **Context:** 311 GeoTIFF JPEG-in-TIFF (~450 MB), un singur zbor; demo-ul se face de pe laptop și trebuie să meargă offline; nu avem CDN (doar Supabase).
- **Decizie:**
  - GDAL 3.13.3 în Docker face VRT → 3857 → MBTiles WebP q75 cu overview-uri;
  - go-pmtiles face `convert` → `.pmtiles`;
  - fișierul stă local în `src/Web/data/ortho/` și e servit de ruta Next `/data` (cu Range).
- **Respins:**
  - COG + titiler în FastAPI: CPU la fiecare tile și nu merge în modul static fără API;
  - folder XYZ: zeci de mii de fișiere;
  - Supabase Storage: limita per fișier pe planul free, inutilă pentru un demo local.
- **Consecințe:** o generare de ~30–60 min peste noapte. z22 intră doar dacă fișierul rămâne sub 1,5 GB. Dacă mai târziu apare hosting, fișierul se poate muta pe orice storage cu Range, fără cod nou (doar `STATIC_BASE_URL`).

### ADR-006 — Vectori mari: PMTiles în modul static, MVT PostGIS în modul API
- **Context:** estimăm 40–80k poligoane de coroane.
- **Decizie:**
  - modul static: `canopies.pmtiles` (tippecanoe 2.79.0);
  - modul API: `ST_AsMVT` în FastAPI, cu RLS activ și decupaj la geofence;
  - straturile mici (≤ ~8k obiecte) sunt GeoJSON în ambele moduri.
- **Respins:**
  - GeoJSON complet în browser: zeci de MB, blochează firul principal;
  - endpoint GeoJSON cu bbox pentru coroane: payload mare la zoom 18–19;
  - server de tile-uri separat (Martin / pg_tileserv): ar ocoli RLS-ul per utilizator sau ar cere încă un serviciu.
- **Consecințe:** izolarea pe UAT în modul API se aplică și tile-urilor.

### ADR-007 — Harta: MapLibre GL JS 6 + @vis.gl/react-maplibre + pmtiles
- **Decizie:** `maplibre-gl` 6.11.2, `@vis.gl/react-maplibre` 8.1.3 (wrapper-ul doar pentru MapLibre, fără dependența de mapbox-gl), `pmtiles` 4.5.0 prin `addProtocol`.
- **Respins:**
  - Leaflet: fără randare WebGL și suport slab pentru vector tiles la 60k poligoane;
  - OpenLayers: bun, dar mai greu de integrat idiomatic în React și mai mult cod pentru stiluri;
  - deck.gl: nu e nevoie la volumul ăsta, iar straturile MapLibre ajung;
  - Mapbox GL: licență comercială și token.
- **Consecințe:** MapLibre 6 e un major nou. Verificăm schimbările față de 5 la F0-7. Stilurile straturilor se generează din `mapPalette.ts`.

### ADR-008 — Acces DB din FastAPI: psycopg 3 + SQL explicit + `rls_tx`
- **Context:** API-ul trebuie să interogheze **în numele utilizatorului**, cu RLS activ, și are nevoie de funcții PostGIS (MVT, decupaj).
- **Decizie:**
  - `psycopg[binary]` 3.3.6 + `psycopg-pool`, cu rolul de login `web_api` (fără BYPASSRLS, `GRANT authenticated TO web_api`);
  - per request: `BEGIN; SET LOCAL ROLE authenticated; select set_config('request.jwt.claims', …, true)`;
  - SQL scris explicit, în `app/sql/*.sql`;
  - conexiunea admin separată, doar în `/admin` și în scripturi.
- **Respins:**
  - `supabase-py` / PostgREST: fără acces comod la `ST_AsMVT` și agregări, și un salt HTTP în plus;
  - SQLAlchemy + GeoAlchemy2: overhead și magie ORM fără beneficiu la ~20 de query-uri;
  - conexiunea ca `postgres`/service cu filtrare în cod: RLS n-ar mai fi granița.
- **Consecințe:** transaction pooler-ul Supabase (port 6543) e compatibil, pentru că `SET LOCAL` e per tranzacție. Prepared statements dezactivate (`prepare_threshold=None`).

### ADR-009 — Verificarea JWT: JWKS local cu PyJWT
- **Decizie:**
  - PyJWT 2.15.0, cu `PyJWKClient` pe `<SUPABASE_URL>/auth/v1/.well-known/jwks.json` (cache);
  - `aud=authenticated`, `iss=<SUPABASE_URL>/auth/v1`, algoritmii din `kid`;
  - fallback HS256 cu secretul legacy, prin config.
- **Respins:** apelarea `/auth/v1/user` la fiecare request adaugă latență și face ca API-ul să cadă când cade Auth.
- **Consecințe:** un token revocat rămâne valid până expiră (implicit 1 h), ceea ce e acceptabil.

### ADR-010 — Măsurătorile: sursa adevărului e Python/PostGIS în UTM
- **Decizie:**
  - ariile și lungimile oficiale se calculează doar în `api/app/geo/measure.py` (shapely, EPSG:32635) și PostGIS (`ST_Area`/`ST_Length` pe 32635);
  - frontend-ul doar afișează;
  - unealta ad-hoc de măsurare de pe hartă proiectează punctele în EPSG:32635 cu `proj4` 2.22.0 și calculează plan (shoelace), exact ca backend-ul, deci merge și în modul static; în modul API poate apela `/v1/measure`.
- **Respins:**
  - turf pe 4326: aproximări geodezice diferite de referința plană UTM a organizatorilor;
  - calculul în Web Mercator: eroare de scară de ~1/cos(47°), adică +47% pe arii.
- **Consecințe:** un test de paritate PostGIS ↔ shapely ↔ proj4 pe mock.

### ADR-011 — Shell-ul după capturile Marcaj (sidebar), brand Solemtrix
- **Context:** brief-ul propunea top bar + sidebar îngust. Capturile reale (`docs/design/marcaj_projects_empty.png`) arată un sidebar alb lat, cu logo, navigație cu item activ plin indigo, card de utilizator jos și selectorul EN/RO/RU în subsol, iar login-ul split (`marcaj_login.png`) are un hero indigo închis.
- **Decizie:**
  - reproducem structura și paleta din capturi: sidebar 240 px, pliabil la 72 px (pliat implicit pe hartă), cu selectorul de UAT sub logo; pe mobil, bară jos;
  - brandul e **Solemtrix** (decizia echipei), cu logo și nume proprii;
  - de la Marcaj nu preluăm logo-ul și nici textele; subsolul spune „adnotări corectate în Marcaj”.
- **Respins:**
  - top bar-ul din brief: diferă vizibil de Marcaj;
  - brandul Marcaj: nu ne aparține.
- **Consecințe:** logo-ul Solemtrix (SVG simplu) trebuie creat la F1-3; motivul cu casete punctate de pe login e redesenat, nu copiat.

### ADR-012 — Paleta hărții: ajustări pentru contrast pe ortofoto
- **Context:** fundalul e sol maro arat, umbre închise și iarbă verde. Unele culori propuse dispar pe el.
- **Decizie** (valorile finale sunt în `src/Web/CLAUDE.md` §7.1):
  - rând `regular`: `#4338CA` → primary.light `#6366F1`, cu casing alb, pentru că indigo-ul închis se pierde pe solul închis și în umbre;
  - rând `disrupted`: `#D97706` → `#F59E0B`, cu 3 px și casing, pentru că portocaliul închis e prea apropiat de sol;
  - rând `unassessable`: grey-500 → grey-300 punctat, pentru că gri-mediu e invizibil pe sol;
  - coroane: umplere `#2DD4BF` 40% + contur success.main `#0D9488`, pentru că teal-ul închis 45% peste frunze verzi se confundă cu ele;
  - fiecare linie are casing alb; paleta hărții e aceeași în light și dark, iar în dark se schimbă doar masca geofence-ului.
- **Respins:** o paletă separată pentru dark mode, pentru că ortofoto nu se schimbă cu tema.
- **Consecințe:**
  - culorile noi intră în `tokens.ts` (grupul `map`);
  - validarea automată (`scripts/check_palette.py`, raport ≥ 3:1 față de culorile medii de sol și vegetație din exemple) și vizuală pe r006_c004, r021_c012 și un tile din sat.

### ADR-013 — Infrastructură: doar Supabase (existent); aplicația rulează pe laptop
- **Context:** decizia echipei: fără Vercel, Railway sau Cloudflare. Proiectul și baza de date Supabase există deja. Regulamentul acceptă demo-ul de pe laptop.
- **Decizie:**
  - frontend-ul și API-ul rulează în Docker pe laptopul de demo (`make up`);
  - modul static (`make demo`) e fallback-ul offline;
  - Supabase cloud (existent) pentru DB, Auth și Storage;
  - README-ul din rădăcină conține instrucțiunile de rulare cu o singură comandă.
- **Respins:**
  - hosting cloud pentru frontend și API: exclus de echipă;
  - o instanță Supabase locală (CLI) pentru dezvoltare: adaugă setup și sincronizare fără câștig pentru un singur om; pgTAP rulează pe proiectul legat, în tranzacții cu rollback.
- **Consecințe:**
  - nu avem un URL public permanent (întrebarea Q2 din plan);
  - latența laptop → Supabase apare în modul API; se măsoară în `PERF.md`.

### ADR-014 — Orchestrare: Makefile + docker compose + Supabase CLI doar pentru migrații
- **Decizie:**
  - `make up` = `docker compose up` (frontend + api, conectate la Supabase cloud);
  - `make demo` = profilul `static` (doar frontend-ul, offline);
  - Supabase CLI se folosește doar pentru `link`, `db push` și `test db --linked`;
  - Makefile-ul e singurul punct de intrare documentat.
- **Respins:**
  - stack-ul Supabase local în compose: fragil și inutil fără dezvoltare locală pe DB;
  - scripturi npm și uv separate, fără Makefile: nu există o comandă unică pentru juriu.
- **Consecințe:** o comandă pentru fiecare mod (CLAUDE.md §5).

### ADR-015 — Mod teren: doar dacă e timp, fără PWA/offline
- **Context:** un singur om; modul teren nu e cerut de juriu.
- **Decizie:** dacă rămâne timp (F3-3): pagina `/field`, cu `watchPosition`, ținta următoare și check-in la cel mult 2 m. Fără service worker și fără offline.
- **Respins:**
  - `@serwist/next`: risc cu Turbopack;
  - service worker scris manual: timp pe care nu îl avem.
- **Consecințe:** în pitch, offline-ul pe teren apare ca roadmap.

### ADR-016 — i18n: next-intl 4, RO implicit
- **Decizie:**
  - `next-intl` 4.14.7, cu segmentul `[locale]` și `localePrefix: 'as-needed'` (RO fără prefix);
  - mesajele în `messages/{ro,en,ru}.json`;
  - numerele prin `Intl.NumberFormat` în `units.ts`.
- **Respins:**
  - `next-i18next`: gândit pentru Pages Router;
  - Lingui: are nevoie de un pas de compilare în plus.
- **Consecințe:** `proxy.ts` combină next-intl cu reîmprospătarea sesiunii Supabase.

### ADR-017 — Date în client: TanStack Query + zod; client API generat
- **Decizie:**
  - TanStack Query 5.103.2 pentru cache și stări de încărcare;
  - zod 4.6.5 validează fișierele statice (nu au OpenAPI);
  - `openapi-typescript` 7.13.0 + `openapi-fetch` 0.17.0 pentru API.
- **Respins:**
  - Orval / hey-api: generează mult cod și hook-uri pe care nu le vrem;
  - SWR: mai puțin control la invalidare (starea țintelor).
- **Consecințe:** `make gen-api` e parte din `make check`.

### ADR-018 — PDF de audit: WeasyPrint în API
- **Decizie:**
  - WeasyPrint 70.0, din șablon Jinja2 HTML, cu stilurile din aceleași tokeni exportați ca CSS;
  - harta vine ca PNG din canvas-ul MapLibre (`preserveDrawingBuffer` doar pe pagina de audit), trimis la `POST /v1/audits`.
- **Respins:**
  - react-pdf în browser: raportul nu ar fi reproductibil pe server;
  - ReportLab: layout manual, lent de scris;
  - randarea hărții pe server: încă un pipeline.
- **Consecințe:** imaginea Docker a API-ului include pango (apt).

### ADR-019 — Semantica geofence-ului: vizibilitate prin intersecție, cifre prin decupaj
- **Decizie:**
  - RLS face un obiect vizibil dacă `ST_Intersects(geom, geofence-urile mele)`;
  - API-ul decupează geometriile și **toate** măsurătorile UAT la geofence-ul activ (`ST_Intersection`);
  - obiectele invizibile dau 404, nu 403.
- **Respins:**
  - `ST_Within`: rândurile tăiate de limită ar dispărea complet;
  - doar mascare vizuală: nu e izolare reală.
- **Consecințe:** suma ariilor pe UAT-uri adiacente = aria totală; testul o verifică.

### ADR-020 — CI în `.github/workflows/web.yml` (excepție aprobată)
- **Context:** regula „nimic în afara `src/Web/`”. Echipa a aprobat excepția pentru CI.
- **Decizie:**
  - un singur workflow, `web.yml`, cu `paths: src/Web/**`, care rulează `make ci`: lint, tsc, vitest, pytest fără DB, e2e Playwright static pe `siret3-mock`, și verificarea că `schema.d.ts` e la zi;
  - fără secrete și fără DB în CI.
- **Respins:**
  - doar `make check` local: fără garanție la push;
  - teste pe DB în CI: ar cere secrete Supabase în GitHub.
- **Consecințe:** RLS-ul (pgTAP, pytest `db`) se verifică local, prin `make test`, înainte de push.

### ADR-021 — Geofence-ul demo: limitele administrative reale din OpenStreetMap
- **Context:** decizia echipei: geofence-ul este comuna Sireți, Republica Moldova.
- **Decizie:**
  - UAT Sireți = relația OSM `19100171` (boundary=administrative, rang de comună), ~2.729 ha. Verificat: conține toate cele 311 tile-uri și START-ul;
  - UAT de control = comuna Cojușna (vecina de la vest), pentru demo-ul și testele de izolare;
  - `seed_demo.py` le descarcă prin Overpass, le reproiectează în 32635 și le salvează și ca fixture (`src/Web/supabase/fixtures/osm/`), pentru reproductibilitate;
  - atribuirea „© OpenStreetMap contributors, ODbL” apare pe hartă și în PDF.
- **Respins:**
  - relația `18967831`: e doar vatra satului, nu UAT-ul;
  - geofence derivat din `study_area`: nu e limita reală;
  - UAT-uri sintetice în seed: rămân doar ca fixture de test (split A/B), pentru verificarea decupajului.
- **Consecințe:** geofence-ul e de ~33 de ori mai mare decât survey-ul, deci dashboard-ul arată acoperirea (81,5 ha din ~2.729 ha).

### ADR-022 — Schema separată `vine` în proiectul Supabase existent
- **Context:** proiectul Supabase există deja și poate conține tabele ale altor părți ale echipei.
- **Decizie:**
  - toate tabelele, tipurile și politicile noastre stau în schema `vine`, iar helper-ele în `app`;
  - `vine` se expune în PostgREST doar pentru citirile directe din Next (ținte, membri, vizite);
  - migrațiile nu ating `public`.
- **Respins:** `public`, din cauza riscului de coliziune cu tabelele existente și a expunerii implicite.
- **Consecințe:** clienții Supabase folosesc `db: { schema: 'vine' }`.

### ADR-023 — Primăriile și survey-urile în Postgres (`public`), administrare în `/super-admin`
- **Context:** primăriile erau o listă în cod; administratorul platformei trebuie să le înroleze fără deploy. Proiectul Supabase e nou și dedicat aplicației.
- **Decizie:**
  - tabele `public.uat`, `public.survey`, `public.admin_audit_log` (în `public`, nu în schema `vine` din ADR-022: proiectul e dedicat, iar `public` e expus implicit în Data API), cu RLS pe toate și acces zero pentru `anon`;
  - geometrii în EPSG:4326 (o primărie din România poate fi în UTM 34N), arii pe `geography`; măsurătorile viei rămân în EPSG:32635 (ADR-010);
  - limitele vin doar din OpenStreetMap (Nominatim `lookup`), descărcate din nou pe server la salvare — clientul nu trimite niciodată geometrie;
  - scrierile pentru UAT/survey trec prin sesiunea utilizatorului (RLS `platform_admin`); gestionarea conturilor prin Admin API cu cheia secretă, doar pe server, cu re-verificarea rolului direct din Auth.
- **Respins:**
  - funcții `SECURITY DEFINER` care scriu în `auth.users`: ocolesc Auth și sunt riscante;
  - upload manual de GeoJSON: fără sursă verificabilă a limitei.
- **Consecințe:** fără `SUPABASE_SECRET_KEY` tab-ul Utilizatori e dezactivat (cu instrucțiuni); schimbările de rol se aplică la reîmprospătarea JWT (≤ 1 h).

