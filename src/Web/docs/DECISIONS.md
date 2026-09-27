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

### ADR-024 — Roluri pe primărie, echipă și sarcini de teren
- **Context:** primarul nu administrează aplicația; un **administrator UAT** o face pentru primărie și își coordonează echipa de teren, care primește sarcini din analiza AI (goluri, plante lipsă, deșeuri).
- **Decizie:**
  - roluri în `app_metadata`: `platform_admin` (Solemtrix) → înrolează primării și administratori UAT; `uat_admin` → își gestionează echipa (`inspector`, `viewer`) doar în primăria lui; `inspector` → execută sarcini; `viewer` → doar citește;
  - echipa se gestionează prin Admin API pe server, cu rolul și primăria apelantului re-citite din Auth la fiecare acțiune;
  - sarcinile stau în `public.task`, cu RLS pe primărie și un trigger care limitează inspectorul la stare + notă și ține locația în limita primăriei;
  - sarcinile se generează din țintele pachetului AI (`targets.geojson`), o singură dată pe țintă (`unique (uat_key, survey_id, target_id)`).
- **Respins:**
  - primarul ca administrator: nu e rolul lui în practică;
  - administratorul UAT care creează alți administratori: rămâne privilegiul super-adminului;
  - sarcini doar în interfață (fără RLS): inspectorul ar putea modifica orice prin API.
- **Consecințe:** fără `SUPABASE_SECRET_KEY` pagina Echipă și atribuirea arată un avertisment; notificările (email / push) către inspectori vin după configurarea SMTP.


### ADR-025 — Vedere 3D oblică: relief sintetic din coroane (terrain MapLibre), nu extrudare
- **Context:** tile-urile ortofoto sunt doar RGB (fără altitudine, fără DSM), dar juriul și primăria înțeleg mai repede rândurile de vie într-o vedere oblică, cu plantele „ridicate” din sol.
- **Decizie:**
  - `scripts/build-terrain.mjs` rasterizează poligoanele `canopies.geojson` (EPSG:4326) în tile-uri XYZ raster-dem (256 px, z14–z19, ~0,2 m/px la z19, codare **terrarium**, pas 1/256 m): acoperire exactă pe pixel → blur gaussian σ 0,45 m → profil smoothstep cu vârf plat, înălțime 1,1 m; zoom-urile mici sunt media copiilor. Se scrie **fiecare** tile din `bounds` (cele fără coroane: un PNG plat comun), deci MapLibre nu primește niciun 404; `terrain.json` descrie setul. Sireț3: 11 153 coroane → 1 460 tile-uri (245 cu relief), 3,6 MB, ~5 s;
  - `/harta` citește `terrain.json` pe server; lipsă sau invalid → butonul 3D e dezactivat, cu tooltip. Harta 2D rămâne neschimbată: sursele se adaugă abia la prima comutare pe „Vedere oblică”;
  - în 3D: `setTerrain` (exagerarea = înălțimea din slider / 1,1 m, 0–2 m; „Plat” = fără terrain, doar înclinare), pitch 60°, ortofoto și vectorii drapați peste relief, plus un hillshade `igor` (sursă separată, culori din `mapPalette.relief`) care umbrește doar flancurile, solul plat păstrează culorile ortofoto;
  - nota „Relief sintetic: tile-urile au doar RGB, fără altitudine; înălțimea vine din adnotări (coroane)” e afișată lângă controale.
- **Respins:**
  - `fill-extrusion` pe coroane: prisme colorate cu vârf plat care acoperă chiar ortofoto-ul coroanei (extrudarea nu se texturează) și desenează 11–12 k poligoane ca geometrie 3D;
  - codarea `mapbox` (pas 0,1 m): ~11 trepte vizibile pe 1,1 m;
  - tile-uri doar unde sunt coroane: MapLibre ar cere și restul din `bounds` → 404 în consolă.
- **Consecințe:** `build-survey.mjs` înlocuiește tot folderul survey-ului, deci după `pnpm data:survey` se rulează din nou `pnpm data:terrain --survey <id>`; `pnpm data` / `data:fast` generează relieful mock-ului (inclusiv în CI). Performanță măsurată pe siret3 (M3 Pro, Chromium): 60 fps la pan în 2D și în 3D, 3D gata în ~4 s cu ~45 tile-uri de relief. Fără dependențe noi (`sharp` exista deja).

### ADR-026 — Notificări în aplicație (sarcină alocată), live prin Supabase Realtime
- **Context:** inspectorul trebuie să afle imediat că a primit o sarcină, fără să reîncarce pagina și fără să depindă de email (SMTP-ul nu e configurat, iar serverul implicit Supabase are o limită mică).
- **Decizie:**
  - tabel `public.notification` (`user_id`, `kind`, `task_id`, `payload` jsonb, `read_at`), cu RLS: fiecare își citește, marchează ca citite (doar coloana `read_at`) și șterge doar notificările lui; clienții nu pot insera;
  - rândurile le scrie doar trigger-ul `task_notify_assignee` (`after insert or update of assignee` pe `public.task`), cu o funcție `security definer` în schema neexpusă `private`, neexecutabilă de rolurile API. Nu notifică auto-atribuirea; la realocare șterge notificarea necitită a fostului responsabil;
  - `payload` îngheață datele de afișare (tipul sarcinii, rând, bloc, lungimea golului, termen, cine a alocat), iar titlul se traduce în client (RO / EN / RU);
  - tabelul e în publicația `supabase_realtime`: clopoțelul din meniu (`NotificationBell`) ascultă `INSERT` / `UPDATE` filtrat pe `user_id` (Realtime aplică RLS-ul), arată un toast, crește contorul și reîncarcă lista pe `/sarcini`; se reîncarcă și la revenirea în tab;
  - notificarea deschide `/sarcini?sarcina=<id>`: filtrele pornesc astfel încât sarcina să fie vizibilă, rândul e evidențiat și adus în centrul ecranului.
- **Respins:**
  - polling la câteva secunde: cereri inutile și întârziere;
  - notificări scrise din server actions: o atribuire făcută altfel (API, SQL, alt ecran) n-ar notifica pe nimeni;
  - Realtime Broadcast cu canale private: mai multă configurare (politici pe `realtime.messages`) pentru același rezultat.
- **Consecințe:** tipuri noi (ex. „sarcină rezolvată” către administrator) = o valoare nouă în `kind` + un trigger; email / push pot citi același tabel mai târziu. Migrația: `supabase/migrations/20260926000400_notifications.sql`.

### ADR-027 — Asistent în aplicație (Gemini), cu prompt construit din etichetele interfeței
- **Context:** utilizatorii primăriei trebuie să afle repede cum se face ceva („cum dau o sarcină unui inspector?”) și să ajungă direct pe pagina potrivită, fără documentație separată.
- **Decizie:**
  - un buton rotund fix în dreapta jos (în aplicație și în consola super-admin; pe hartă mutat la stânga zoom-ului, pe mobil deasupra barei de navigare) deschide un panou non-modal; pagina rămâne utilizabilă, iar linkurile din răspuns navighează în aplicație fără să închidă panoul (layout-ul persistă; conversația stă într-un store extern + `sessionStorage`, doar pentru tab-ul curent);
  - `POST /api/assistant` (Route Handler; cheia doar pe server: `OPENROUTER_API_KEY` + `OPENROUTER_MODEL`, implicit `google/gemini-3.8-flash`, sau direct `GEMINI_API_KEY` + `GEMINI_MODEL`, implicit `gemini-3.8-flash`; `src/lib/assistant/llm.ts`) trimite la Gemini istoricul scurt (≤ 12 mesaje) și un prompt de sistem construit de `src/lib/assistant/prompt.ts`: paginile, pașii pe rol și etichetele **exacte** ale butoanelor, citite din `messages/<limbă>.json` («tasks.createTasks» → „Creează N sarcini”). Răspunsul vine ca text în flux;
  - promptul știe rolul și primăria din sesiune (nu din client): consola de administrare apare doar pentru `platform_admin`; cifrele zborului se adaugă doar dacă primăria are survey-ul servit;
  - în client, doar căile interne devin linkuri; linkurile externe, HTML-ul și imaginile din răspuns sunt ignorate;
  - acces: cont sau vizitator demo (aceeași regulă ca paginile; proxy-ul nu rulează pe `/api`), limită de 30 de întrebări / 10 min per cont sau adresă, întrebare ≤ 1000 de caractere.
- **Respins:**
  - cheia în browser (`NEXT_PUBLIC_`): ar fi publică;
  - text de ajutor scris de mână, separat de interfață: s-ar desincroniza de butoane (testul `scripts/assistant/labels.test.mjs` verifică fiecare etichetă în RO / EN / RU);
  - unelte (function calling) care execută acțiuni: asistentul doar explică și trimite la pagină; acțiunile rămân în UI, sub RLS.
- **Consecințe:** fără `GEMINI_API_KEY` panoul spune că nu e configurat. Întrebările, rolul, numele primăriei și cifrele agregate ale zborului ajung la Google (Gemini API); nu se trimit emailuri, parole sau date personale ale altor utilizatori. O funcție nouă în aplicație trebuie descrisă și în `prompt.ts`.

### ADR-028 — Trasee proprii calculate în browser: prin o fermă sau prin toate fermele, doar pe culoarele dintre rânduri
- **Context:** inspectorul vrea să aleagă punctul de plecare (= sosire) și să primească cel mai scurt traseu închis prin țintele unei ferme (mergând doar prin ea) sau ale tuturor fermelor, și să vadă cât câștigă față de varianta normală. Regulamentul: *„Use passable inter-row areas and authorised passages, not paths through canopies, fences or forbidden areas.”* Ruta oficială (`route.geojson`) e una singură, precalculată de pipeline, și vizitează doar țintele obligatorii plus opționalele care merită ocolul (492 din 1.199 pe siret3).
- **Decizie — UI:** două butoane pe hartă, dreapta jos, deasupra riglei:
  - **„Traseu fermă”**, în 3 pași: 1) ferma, cu click pe hartă sau din listă; harta face zoom pe ea, iar restul se estompează; 2) startul, cu click în fermă (un click în afară se mută pe conturul fermei); 3) traseul. Panoul fermei are o scurtătură la pasul 2;
  - **„Toate fermele”**, în 2 pași: 1) startul, cu click sau „Din START-ul oficial”; 2) un singur tur prin toate țintele tuturor fermelor;
  - **la final,** ambele arată lungimea, durata (4 km/h), GPX-ul și **cu cât % e mai scurt decât varianta normală**;
  - cât timp o unealtă e activă, linia și numerele rutei oficiale se ascund, iar rigla se ascunde și ea. Pe telefon, cardul stă sus, cu înălțime limitată.
- **Decizie — calcul** (browser, Web Worker, `src/lib/farmRoute/`, TypeScript pur, fără dependențe noi):
  - **culoare, niciodată rânduri** (`lanes.ts`):
    - se merge pe axa culoarului dintre două rânduri vecine (media axelor lor, ca `centerlines.py` din pipeline) și pe câte un culoar la jumătate de interval în afara rândurilor de margine;
    - vecinul se caută punct cu punct, doar printre rândurile care merg alături în acel loc, deci rândurile aflate unul în continuarea altuia nu se împerechează;
    - culoarele se prelungesc 1,5 m după capătul rândurilor;
    - axele rândurilor nu intră în graf; sunt obstacole;
  - **graf** (`graph.ts`):
    - legăturile (capăt de culoar → alt culoar ≤ 8 m, adică întoarcerea pe la capăt; → drum ≤ 60 m; startul → rețea) nu au voie să traverseze axa unui rând (`SegmentGrid.crossedBy`);
    - un drum care taie un culoar devine nod, adică o trecere autorizată prin rânduri;
    - nodurile se unesc pe o grilă de 0,5 m;
  - **ținte cu doi candidați:** o țintă dintr-un rând poate fi văzută din culoarul din stânga sau din dreapta ei, iar turul alege (GTSP, ca `solve_gtsp`/`candidates.py`). Deșeurile au un singur candidat, pe cea mai apropiată linie;
  - **o fermă:** doar culoarele fermei și drumurile tăiate la conturul ei, cu o marjă de 10 m (`area.ts`);
  - **toate fermele:** culoarele tuturor fermelor și drumurile într-un bbox cu marjă de 300 m, inclusiv cele publice dintre ferme (`allFarms.ts`);
  - **tur** (`compress.ts`, `solve.ts`, `tour.ts`):
    - graful se comprimă (lanțurile de noduri de trecere devin o singură muchie; siret3: 41k → 10k noduri), apoi rulează un Dijkstra pe nod distinct, cu matrice float32;
    - pentru grupuri: nearest neighbour, apoi 2-opt, Or-opt și realegerea candidatului, în treceri pe loc, până nu mai apare nicio îmbunătățire sau se termină bugetul (1,5 s pentru o fermă, 4 s pentru toate);
    - Held–Karp exact până la 10 ținte cu câte un singur candidat;
  - **varianta normală** (`baseline.ts`): serpentina prin toate culoarele dintre două rânduri, bloc cu bloc, cel mai apropiat bloc întâi, de la același start și înapoi (ca `serpentine_est_m` din pipeline). Salturile sunt socotite în linie dreaptă, ceea ce favorizează referința, deci economia afișată e prudentă.
- **Rezultate pe siret3** (M3 Pro, Node):
  - **toate fermele:** 1.194 de ținte, 26,7 km față de 77,5 km, adică **cu 66% mai scurt**; în browser, aproximativ 15 s. Serpentina pipeline-ului, pentru comparație: 69,2 km;
  - **pe fermă:** F01 7,3 km (−70%), F07 0,57 km (−54%); 2 ms–1,6 s pe fermă;
  - **pe axa unui rând** (sub 0,5 m): 1–2,5% din lungime, aproape numai traversări perpendiculare prin treceri; de-a lungul rândurilor, 2–27 m din câțiva kilometri.
- **Respins:**
  - **mersul pe axa rândului** (prima versiune): trece peste coroane, contrar regulamentului;
  - **culoarele doar după ordinea laterală:** împerechea rânduri aflate unul în continuarea altuia, iar legăturile tăiau rândurile;
  - **toate fermele ca tururi închise, fermă după fermă:** 34 km, pentru că fiecare fermă se închidea la poarta ei;
  - **endpoint Python cu solver-ul pipeline-ului:** API-ul nu există, iar site-ul trebuie să meargă static și offline;
  - **turf / graphology:** câteva sute de linii ajung.
- **Consecințe:**
  - fără `farms.geojson` + `roads.geojson` (mock-ul), butoanele nu apar;
  - `forbidden.geojson` nu e încă ocolit;
  - calcul plan în UTM 35N (ADR-010), orientativ;
  - teste: unitare în `src/lib/farmRoute/farmRoute.test.ts`, din `pnpm test:scripts` (Node ≥ 22.18 elimină tipurile, deci `tsconfig` are `allowImportingTsExtensions`, iar modulele se importă cu extensia `.ts`, fără parameter properties); e2e în `e2e/farm-route.spec.ts`.

### ADR-029 — Pagina „Robot de teren”: plăcile ESP32 comandate printr-un proxy din serverul Next
- **Context:** robotul echipei are trei plăci ESP32 pe Wi-Fi-ul „Ghile” (firmware-ul e deja pe plăci), identificate pe 27.09.2026:

  | Adresă | Placă | Rute |
  |---|---|---|
  | `10.12.241.207` | ESP32-CAM | `/stream` (MJPEG), `/capture`, `/flash/*`, `/json` |
  | `10.12.241.87` | camera pan/tilt (2 motoare pas cu pas) + senzor HC-SR04 | `/move?motor&dir&speed&steps`, `/toggle_en`, `/distance` |
  | `10.12.241.233` | roțile: 4 motoare DC | `/dir?m=<1..4 \| all>&val=±1`, `/speed?m=&val=0..255`, `/start?m=`, `/stop?m=`, `/status` (JSON cu `running`, `rpm`, `direction`) |

  Plăcile nu trimit antete CORS. **ESP32-CAM servește o singură conexiune odată:** cât timp e deschis un stream MJPEG, orice altă cerere (flash, captură) așteaptă la nesfârșit. Măsurat: se eliberează în 0,18 s de la închiderea clientului direct, dar în 2–4 s printr-un relay Node.
- **Decizie:**
  - **pagina `/robot`** (meniu: „Robot de teren”) conține:
    - **imaginea live ca cadre `/capture` succesive** (2–7 cadre/s), nu MJPEG, deci camera e liberă între cadre, iar flash-ul și poza trec oricând;
    - **„Fă poză”:** ultimul cadru, instantaneu, în galerie cu descărcare JPEG; numele fișierului conține unghiurile camerei, pregătit pentru un mod street view;
    - **pad pentru cameră:** motorul 1 = stânga/dreapta, motorul 2 = sus/jos, 15° pe apăsare; unghiul e estimat din pași, cu „poziția curentă = 0°”;
    - **distanța** HC-SR04, citită o dată pe secundă;
    - **pad pentru deplasare:** înainte / înapoi = toate roțile în același sens; stânga / dreapta = întoarcere pe loc; plus stop;
    - **setări:** adresele (valori implicite din `NEXT_PUBLIC_ROBOT_*_URL` în `.env.local`), partea stânga/dreapta și inversarea fiecărui motor de roată, vitezele și durata unei mișcări. Se țin în `localStorage` prin `useSyncExternalStore`;
    - **taste:** săgeți = cameră, W A S D = deplasare, X / Esc = stop roți, spațiu = poză;
  - **`GET /api/robot`** trimite mai departe doar comenzi din lista albă (`src/lib/robot/commands.ts`), cu parametri validați. Acceptă doar adrese IPv4 private (10/8, 172.16/12, 192.168/16), fără cale, deci nu poate fi folosit ca SSRF spre internet sau spre server. Are aceeași regulă de acces ca paginile;
  - **o mișcare a roților** e o secvență rulată pe server: `stop`, apoi `speed` (`m=all` când toate se mișcă), apoi `dir` (`m=all` când au același sens), apoi `start` (`m=all`), apoi așteptarea duratei, apoi **`stop?m=all` trimis mereu în `finally`**, chiar dacă un pas a eșuat;
  - **`scripts/robot/fake-robot.mjs`:** simulează cele trei plăci cu aceleași rute, pe adresa LAN a laptopului; e2e-ul `e2e/robot.spec.ts` îl pornește singur, pe câte un port pentru fiecare worker.
- **Verificat pe plăcile reale:** imagine live 3–7 cadre/s; flash pornit/oprit în 0,4 s cu imaginea live pornită; poză; motoarele camerei în ambele sensuri; distanța; cele 4 mișcări ale roților, cu direcțiile corecte pe fiecare motor și oprire completă. Encoderul motorului 2 al roților raportează valori `rpm` haotice (0 … 1795), ceea ce pare o problemă hardware.
- **Respins:**
  - **MJPEG direct în browser:** blochează flash-ul și captura cât timp e deschis;
  - **relay MJPEG prin Next:** camera rămâne blocată 2–4 s după închidere;
  - **poza desenată dintr-un `<img>` cross-origin pe canvas:** canvasul devine „tainted” și nu mai poate fi exportat;
  - **comenzi direct din browser:** CORS, fără validare;
  - **adrese fixe în cod.**
- **Consecințe:**
  - laptopul care rulează site-ul trebuie să fie pe același Wi-Fi cu robotul;
  - pozele trăiesc doar în tab până sunt descărcate;
  - **modul street view** (mers 50 cm, apoi poze la 0°, 90° și 180°) se poate construi peste aceleași comenzi, după calibrarea distanței parcurse pe secundă la o anumită viteză.
