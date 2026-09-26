# PLAN — Solemtrix Vineyard, aplicația web pentru UAT-uri

GigaHack 2026 · Vineyard AI Field Challenge (challenge Marcaj) · deadline **duminică 27.09.2026, 15:00** (Chișinău)
Versiunea 1.1 · 25.09.2026, vineri seara · Stare: **propunere, așteaptă aprobare**

Documente legate:
- `src/Web/CLAUDE.md`: stack cu versiuni, structura, comenzile, **Data contracts** (§6), tokenii de design (§7);
- `DECISIONS.md`: ADR-uri;
- `DEMO.md`: scenariul de pitch.

**Decizii ale echipei (25.09, v1.1):**
- totul stă în `src/Web/`;
- un singur om pe web;
- infrastructura e **doar Supabase** (proiect existent), fără Vercel, Railway sau Cloudflare;
- brandul e **Solemtrix**;
- geofence-ul e **comuna Sireți, Republica Moldova**;
- butonul „Demo fără cont” e aprobat;
- CI-ul în `.github/workflows/web.yml` e aprobat;
- contractul cu echipa AI se stabilește când vin datele.

---

## 0. Rezumat (o pagină)

**Ce construim.** **Solemtrix Vineyard** transformă output-ul AI pe ortofotografia Sireț3 într-o hartă măsurată a viilor și într-un plan de inspecție pe teren. Aplicația are:
- un frontend Next.js cu aspect Marcaj (MUI, Inter, indigo), dar cu brand propriu;
- un API FastAPI care face tot calculul geospațial în EPSG:32635;
- Supabase (PostGIS, Auth, Storage, RLS, proiectul existent), care izolează datele pe UAT.

Frontend-ul și API-ul rulează pe laptop, cu Docker. Demo-ul de pe laptop e acceptat de regulament.

**Pentru cine.**
1. **Juriul GigaHack:** trebuie să vadă harta, obiectele, ID-urile, măsurătorile și ruta. Interfața web este criteriu de admitere.
2. **Primăriile (UAT)**, începând cu **Primăria Sireți** (raionul Strășeni, ~2.729 ha, limita reală din OpenStreetMap):
   - inventarul plantațiilor, starea rândurilor și deșeurile din comună;
   - rute scurte de inspecție;
   - audit de suprafață pentru subvenții;
   - izolare reală față de alte primării (ex. Cojușna, vecina de la vest).

**Ce arătăm juriului (3 minute, `DEMO.md`):**
1. login Primăria Sireți;
2. dashboard: 2.729 ha de comună, din care 81,5 ha zburate, plus KPI-urile viei;
3. hartă cu ortofoto;
4. click pe un rând;
5. măsurători;
6. ruta, cu lungimea și economia;
7. export GPX;
8. contul Cojușna, care nu vede nimic din Sireți;
9. modul teren.

**Principiul de livrare: static-first.**
- Cerințele juriului se livrează întâi în **modul static**: Next.js citește fișiere pre-generate de același cod Python de măsurare pe care îl folosește API-ul. Nu depinde de FastAPI, de Supabase sau de Wi-Fi.
- Stratul UAT (auth, RLS, geofence, dashboard pe UAT) vine peste, în **modul API**, prin `NEXT_PUBLIC_DATA_MODE=static|api`.
- Butonul „Demo fără cont” de pe login deschide modul static.

**Real la deadline:** harta, toate straturile, măsurătorile, ruta, exporturile (CSV/GeoJSON/GPX/KML), modul static offline, auth, RLS cu 2 UAT-uri reale (Sireți, Cojușna), masca geofence, dashboard-ul, tabelul de blocuri și rânduri, CI-ul.

**Seed/demo:** utilizatorii, vizitele de teren, suprafețele „declarate” la audit, istoricul de survey-uri (tăiat).

**Doar dacă rămâne timp** (un singur om): modul teren, PDF-ul de audit, pagina de admin. Detalii în §12–13.

**Riscul principal** e timpul: un om, ~23 h de lucru efectiv. Răspunsul:
- ordine strictă: MVP-ul juriului întâi, apoi stratul UAT, apoi restul;
- o listă de tăieri decisă de acum (§13).

---

## 1. Arhitectura

```
      laptop (Docker: make up / make demo)                                   cloud
┌───────────────────────────────────────────────────────────────┐    ┌──────────────────────────────┐
│ browser: Next.js 16 · MUI 9 · MapLibre 6 + PMTiles · next-intl │    │ Supabase (proiect existent)   │
│            │ static                 │ api (Bearer JWT)    │ supabase-js (RLS)                  │
│            ▼                        ▼                     └──────────────────▶│  Auth (email+parolă)        │
│  Next route /data/[...path]   FastAPI :8000 (Python 3.12)                     │  Postgres + PostGIS          │
│  (Range) ◀─ src/Web/data/…    auth.py  JWKS → claims  ─────────────────────▶│   geometrii EPSG:32635, GiST │
│   summary.json · *.geojson    db.py rls_tx(): SET LOCAL ROLE authenticated   │   RLS: membru ∧ ST_Intersects│
│   canopies.pmtiles            geo/measure.py (UTM) · ST_AsMVT · exporturi    │  Storage: field-photos,      │
│   ortho/siret3.pmtiles                     ▲                                  │   reports (policies pe UAT)  │
│            ▲                               │ import_bundle (conexiune admin)  └──────────────▲───────────────┘
│            │ build_static.py (același measure.py)                                            │
│  src/Web/scripts ◀── src/Web/data/surveys/<id>/pipeline/ ◀── pipeline AI (bundle EPSG:32635)  │
│                  ◀── data & info/01_tiles (311 GeoTIFF) → build_ortho.sh → ortho.pmtiles       │
│                  ◀── OSM (Sireți 19100171, Cojușna) → seed_demo.py ─────────────────────────────┘
└───────────────────────────────────────────────────────────────┘
```

### Cine face ce

| Rol | Next.js | FastAPI | Supabase |
|---|---|---|---|
| UI, routing, i18n, hartă | ✔ | | |
| Sesiune utilizator | ✔ (`@supabase/ssr`, `proxy.ts` reîmprospătează cookie-ul) | verifică JWT-ul | emite JWT-ul |
| Date simple (UAT-urile mele, membri, starea țintelor, vizite) | ✔ direct din Supabase, sub RLS | | RLS |
| Geometrii, MVT, măsurători, agregări, căutare spațială | | ✔ | PostGIS |
| Exporturi (CSV, GeoJSON, GPX, KML, PDF) | | ✔ | Storage pentru PDF |
| Import bundle AI, validare, recalcul | | ✔ (+ scripturi) | |
| Izolarea pe UAT | afișează masca | aplică `uat_id` și decupează la geofence | **RLS = granița dură** |

Stack-ul și versiunile: `src/Web/CLAUDE.md` §3. Structura: §4. Deciziile: `DECISIONS.md`.

---

## 2. Cele două moduri de date

`frontend/src/lib/data/DataSource.ts`, o singură interfață, cu două implementări:

```ts
interface DataSource {
  mode: 'static' | 'api'
  getManifest(surveyId): Promise<SurveyManifest>            // nume, dată, sursă, licență, bbox, ortho_url
  getSummary(surveyId, uatId?): Promise<SurveySummary>      // KPI survey + per bloc (rândurile survey/block din CSV)
  getBlocks(surveyId, q?): Promise<BlockRow[]>
  getRows(surveyId, q?: {vineyardId?, structure?, sort?, page?}): Promise<Page<RowRow>>
  getObject(surveyId, layer, id): Promise<ObjectDetail>     // panoul de atribute
  search(surveyId, text): Promise<SearchHit[]>              // vineyard_id / row_id → bbox
  getRoute(surveyId): Promise<RouteDetail>                  // geometrie + ținte ordonate + lungimi + economii
  mapSources(surveyId): MapSources                          // spec-uri MapLibre: geojson URL | pmtiles:// | MVT URL
  exportUrl(surveyId, kind: 'csv'|'geojson'|'gpx'|'kml'): string
}
```

**Modul static:**
- filtrarea și sortarea se fac în browser, pe proprietățile deja calculate; nu se face niciun calcul geometric;
- căutarea folosește un index `{id → bbox}` din `summary.json`;
- GPX și KML sunt pre-generate;
- nu există auth; afișează survey-ul public (CC BY 4.0) cu geofence-ul Sireți desenat doar ca strat de context.

**Modul API:** totul trece prin FastAPI, cu JWT. Starea țintelor și vizitele se citesc și se scriu direct în Supabase, sub RLS.

**Servirea fișierelor:** în ambele moduri, ortofoto și fișierele statice vin din ruta Next `app/data/[...path]/route.ts`. Ea citește din `WEB_DATA_DIR` și suportă `Range` (necesar pentru PMTiles). Nu există CDN.

---

## 3. Contractul API FastAPI (`/v1`)

**Autentificare:**
- header `Authorization: Bearer <Supabase access token>`;
- `auth.py` verifică semnătura prin JWKS (`<SUPABASE_URL>/auth/v1/.well-known/jwks.json`, cu cache), cu `aud=authenticated` și `iss=<SUPABASE_URL>/auth/v1`;
- fallback HS256 cu `SUPABASE_JWT_SECRET`, dacă proiectul existent folosește încă chei legacy (`SUPABASE_JWT_MODE=jwks|hs256`, de verificat la F2);
- rolul și UAT-urile utilizatorului vin din DB, nu din token.

Fiecare handler cu utilizator rulează în:

```python
async with rls_tx(claims) as cur:    # BEGIN; SET LOCAL ROLE authenticated;
    ...                              # SELECT set_config('request.jwt.claims', :claims, true); … COMMIT
```

- Conexiunea folosește rolul de login `web_api`, fără `BYPASSRLS`, cu `GRANT authenticated TO web_api`.
- Conexiunea admin (`DATABASE_ADMIN_URL`) apare doar în `/v1/admin/*`, după `is_platform_admin`, și în scripturi.

**Parametrul `uat_id`:** e obligatoriu pentru utilizatorii cu mai multe UAT-uri. Restrânge și decupează rezultatele la geofence-ul acelui UAT (`ST_Intersection`). RLS rămâne limita superioară.

**Coduri de răspuns:** 401 = token invalid; 403 = rol insuficient; 404 = inexistent **sau invizibil**.

**Prioritate:** endpoint-urile marcate ★ sunt minimul pentru modul API la demo. Restul se fac doar dacă e timp.

| # | Metodă | Cale | Parametri | Răspuns | Acces |
|---|---|---|---|---|---|
| 1 ★ | GET | `/health` | — | `{status, db, version}` | public |
| 2 ★ | GET | `/v1/me` | — | user, `platform_admin`, `uats[{id,name,role}]` | autentificat |
| 3 ★ | GET | `/v1/uats/{uat_id}` | — | nume, geofence (4326), aria (ha), survey-uri, % acoperit de survey | membru |
| 4 ★ | GET | `/v1/surveys/{sid}/summary` | `uat_id` | `SurveySummary` (blocuri, rânduri, lungimi, arii în m² și ha, % disrupted, deșeuri, ținte deschise, ha geofence, ha zburate) | membru |
| 5 ★ | GET | `/v1/surveys/{sid}/features/{layer}` | `uat_id, bbox (4326), zoom` | GeoJSON 4326; `layer ∈ blocks,rows,interrows,waste,targets,route` | membru |
| 6 | GET | `/v1/surveys/{sid}/tiles/canopies/{z}/{x}/{y}.mvt` | `uat_id` | MVT (`ST_AsMVT`), decupat la geofence | membru |
| 7 ★ | GET | `/v1/surveys/{sid}/objects/{layer}/{id}` | `uat_id` | atribute, măsurători, bbox 4326 | membru |
| 8 | GET | `/v1/surveys/{sid}/rows` | `uat_id, vineyard_id, structure, sort, page, size≤500` | `Page<RowRow>` | membru |
| 9 | GET | `/v1/surveys/{sid}/blocks` | `uat_id, sort` | `BlockRow[]` | membru |
| 10 | GET | `/v1/surveys/{sid}/search` | `q, uat_id` | `[{layer,id,label,bbox}]` (max 20) | membru |
| 11 | GET | `/v1/surveys/{sid}/route` | `uat_id` | geometrie 4326, `length_m`, `duration_min`, `baseline_length_m`, economii, ținte ordonate | membru |
| 12 | GET | `/v1/surveys/{sid}/route/export` | `format=geojson\|gpx\|kml, uat_id` | fișier (GeoJSON 32635 cu `crs`; GPX/KML 4326) | membru |
| 13 | GET | `/v1/surveys/{sid}/measurements.csv` | `uat_id` | CSV (CLAUDE.md §6.4), decupat la UAT | membru |
| 14 | POST | `/v1/measure` | `{geometry (4326)}` | `{length_m, area_m2, area_ha}` în UTM | autentificat |
| 15 | PATCH | `/v1/targets/{id}` | `{status}` | țintă | inspector, uat_admin |
| 16 | POST | `/v1/targets/{id}/visits` | `{position (4326), note, photo_path?}` | vizită + `distance_m`; `visited` dacă distanța e ≤ 2 m | inspector, uat_admin |
| 17 | POST | `/v1/uploads/sign` | `{kind: photo\|report, uat_id}` | URL semnat Supabase Storage | inspector, uat_admin |
| 18 | POST | `/v1/audits` | `{survey_id, uat_id, vineyard_ids[], declared_area_m2?, map_png}` | aria măsurată, plante/ha, % goluri, diferența față de declarat | uat_admin |
| 19 | GET | `/v1/audits/{id}.pdf` | — | PDF (WeasyPrint): hartă, cifre, data survey-ului, sursa, atribuirile CC BY 4.0 și ODbL | uat_admin, viewer |
| 20 | POST | `/v1/admin/uats` | multipart: nume, țară, geofence (GeoJSON / Shapefile .zip / id relație OSM) | UAT | platform_admin |
| 21 | POST | `/v1/admin/uats/{id}/members` | `{email, role}` | invitație (Supabase Admin API) | platform_admin, uat_admin |
| 22 | POST | `/v1/admin/surveys/{sid}/import` | `{source: "pipeline"\|"cvat"}` | raport de import | platform_admin |
| 23 | GET | `/v1/admin/audit-log` | `uat_id, from, to` | evenimente | platform_admin, uat_admin |

În modul API, straturile fără endpoint dedicat (rute, ținte, rânduri pentru tabel) pot fi servite la început din `/features/{layer}` (★ 5). Tabelul de rânduri filtrează atunci în browser, ca în modul static, deci #8–#10 devin optimizări, nu blocaje.

**Clientul TypeScript:**
- `make gen-api` rulează `uv run python -m app.export_openapi > api/openapi.json`, apoi `pnpm openapi-typescript api/openapi.json -o frontend/src/lib/api/schema.d.ts`;
- `client.ts` = `createClient<paths>()` (`openapi-fetch`), cu un middleware pentru bearer;
- CI-ul eșuează dacă `schema.d.ts` nu e la zi.

---

## 4. Contractele de date

Schema propusă, locațiile și faptele de control sunt în **`src/Web/CLAUDE.md` §6**.

**Starea contractului:** e o **propunere**, pe care o confirmăm cu echipa AI când vin datele (decizia echipei). Ca să nu blocăm nimic:
- UI-ul se construiește pe **mock** (`make mock`), care respectă exact contractul;
- dacă formatul AI diferă, adaptarea se face **într-un singur loc**, un adaptor în `scripts/adapters/<nume>.py`, care transformă output-ul AI în bundle. Frontend-ul, API-ul și `build_static` nu se schimbă;
- plasa de siguranță: `scripts/cvat_to_bundle.py` construiește bundle-ul și din **exportul CVAT din Marcaj**, pe care îl poate descărca oricine din echipă. Pixel → UTM se face prin formula de origine a tile-urilor; `row_id`/`vineyard_id` vin din atributele Marcaj. Doar ruta și țintele trebuie să vină de la AI.

**Validarea la import** (`api/app/geo/bundle.py`, pydantic):
- CRS 32635; bbox-ul zonei de studiu ± 100 m;
- enum-uri exacte; ID-uri unice; `row.vineyard_id` există în blocuri;
- geometrii valide (`make_valid` dă o avertizare);
- ținte cu `route_order` la cel mult 2 m de rută;
- ruta închisă la cel mult 5 m de START;
- `outside_share` ≤ 2%, recalculat pe `passages ∪ interrows`.

**Mock:**
- `make mock` citește `data & info/05_examples/siret3_examples_cvat/annotations.xml`, adică 2 tile-uri, 750 de obiecte reale;
- scrie bundle-ul `siret3-mock`;
- blocurile sunt reuniunea coridoarelor ±0,3 m ale rândurilor;
- țintele sunt golurile ≥ 5 m, plus 3 deșeuri fictive marcate `MOCK`;
- ruta sintetică merge START → tile-uri → START; UI-ul arată banner-ul „Date de test”.

Logica se portează din `tools/view_labels.py` al organizatorilor, fără rasterio.

---

## 5. Modelul de date Supabase/PostGIS

Proiectul Supabase **există deja**. Migrațiile se aplică prin `supabase link --project-ref <ref>` + `make db-push`.
- Toate obiectele noastre stau în schema **`vine`**, nu în `public`, ca să nu intre în conflict cu ce există deja în DB (întrebarea Q1).
- Helper-ele stau în schema `app`.
- Geometriile sunt `geometry(<tip>, 32635)`, cu index GiST. Afișarea se face prin `ST_Transform(geom, 4326)`.

Migrațiile: `0001_extensions` (postgis, schemele), `0002_core`, `0003_survey_layers`, `0004_field`, `0005_rls`, `0006_storage`, `0007_audit_log`.

```sql
-- 0001
create extension if not exists postgis with schema extensions;
create schema vine; create schema app;

-- 0002 core
create type vine.uat_role as enum ('uat_admin','inspector','viewer');
create table vine.uat (
  id uuid pk default gen_random_uuid(), name text not null, country text check (country in ('MD','RO')),
  code text unique, osm_relation_id bigint, geofence geometry(MultiPolygon,32635) not null,
  area_m2 double precision generated always as (ST_Area(geofence)) stored, created_at timestamptz default now());
create table vine.uat_member (uat_id uuid references vine.uat on delete cascade,
  user_id uuid references auth.users on delete cascade, role vine.uat_role not null, primary key (uat_id,user_id));
create table vine.platform_admin (user_id uuid primary key references auth.users);
create table vine.survey (id text pk, name text, captured_at date, gsd_m real, source text, license text,
  footprint geometry(MultiPolygon,32635), stage text check (stage in ('model','marcaj_corrected')),
  status text check (status in ('importing','ready','failed')), pipeline_version text, imported_at timestamptz);

-- 0003 straturi (survey_id text references vine.survey on delete cascade în toate)
create table vine.block    (id bigserial pk, survey_id, vineyard_id text, geom geometry(MultiPolygon,32635), area_m2 float8,
                            unique (survey_id, vineyard_id));
create table vine.vine_row (id bigserial pk, survey_id, row_id text, vineyard_id text, geom geometry(MultiLineString,32635),
                            row_structure vine.row_structure_t, length_m float8, plant_count int, max_gap_m float8,
                            tile_structures jsonb, unique (survey_id,row_id));        -- „row” e cuvânt rezervat în SQL
create table vine.canopy   (id bigserial pk, survey_id, canopy_id text, vineyard_id text, row_id text, tile text,
                            geom geometry(Polygon,32635), area_m2 float8);
create table vine.interrow (id bigserial pk, survey_id, interrow_id text, vineyard_id text,
                            interrow_cover vine.interrow_cover_t, tile text, row_ids text[],
                            geom geometry(Polygon,32635), area_m2 float8);
create table vine.waste    (id bigserial pk, survey_id, waste_id text, vineyard_id text null, tile text, confidence real,
                            geom geometry(Polygon,32635));
create table vine.target   (id bigserial pk, survey_id, target_id text, type vine.target_type_t, vineyard_id text,
                            row_id text, waste_id text, route_order int, reachable bool, gap_length_m float8,
                            status vine.target_status_t default 'open', geom geometry(Point,32635));
create table vine.route    (id bigserial pk, survey_id, name text, geom geometry(LineString,32635), length_m float8,
                            duration_min float8, baseline_length_m float8, outside_share float8, source text,
                            created_at timestamptz);
create table vine.ref_layer(id bigserial pk, survey_id, kind text check (kind in ('passage','forbidden','study_area','start')),
                            geom geometry(Geometry,32635));
-- indexuri: GiST(geom) pe toate; btree (survey_id, vineyard_id), (survey_id, row_id), (survey_id, status)

-- 0004 teren și audit
create table vine.inspection_visit (id bigserial pk, target_id bigint references vine.target, user_id uuid references auth.users,
  uat_id uuid references vine.uat, visited_at timestamptz default now(), position geometry(Point,32635), distance_m float8,
  note text, photo_path text);
create table vine.audit_report (id uuid pk, uat_id uuid, survey_id text, vineyard_ids text[], declared_area_m2 float8,
  measured_area_m2 float8, plants_per_ha float8, gap_share float8, pdf_path text, created_by uuid, created_at timestamptz);
create table vine.audit_log (id bigserial pk, at timestamptz default now(), actor uuid, uat_id uuid, action text,
  entity text, entity_id text, payload jsonb);
```

Pentru ca Next.js să citească direct din Supabase (starea țintelor, membri, vizite), schema `vine` se adaugă la „Exposed schemas” în setările API ale proiectului. Straturile geometrice nu se citesc prin PostgREST, ci prin FastAPI.

### 5.1 RLS (pseudo-SQL, `0005_rls`)

```sql
create function app.is_platform_admin() returns bool stable security definer set search_path = '' language sql as
  $$ select exists (select 1 from vine.platform_admin where user_id = auth.uid()) $$;

-- reuniunea geofence-urilor UAT-urilor în care utilizatorul e membru; NULL dacă nu e membru nicăieri
create function app.my_geofence() returns geometry stable security definer set search_path = '' language sql as
  $$ select extensions.ST_Union(u.geofence) from vine.uat u join vine.uat_member m on m.uat_id = u.id
     where m.user_id = auth.uid() $$;

create function app.my_role(p_uat uuid) returns vine.uat_role stable security definer set search_path = '' language sql as
  $$ select role from vine.uat_member where uat_id = p_uat and user_id = auth.uid() $$;

-- pentru fiecare strat: block, vine_row, canopy, interrow, waste, target, route, ref_layer
alter table vine.canopy enable row level security;
create policy canopy_select on vine.canopy for select to authenticated using (
  (select app.is_platform_admin())
  or ST_Intersects(geom, (select app.my_geofence()))     -- (select …) = evaluat o dată/query → index GiST folosit
);
-- fără politici insert/update/delete pe straturi → scrie doar conexiunea admin (import)

create policy target_update on vine.target for update to authenticated
  using (ST_Intersects(geom, (select app.my_geofence())) and exists (
     select 1 from vine.uat_member m join vine.uat u on u.id = m.uat_id
     where m.user_id = auth.uid() and m.role in ('uat_admin','inspector') and ST_Intersects(target.geom, u.geofence)))
  with check (true);                                   -- coloane editabile limitate prin GRANT UPDATE (status)

create policy uat_select on vine.uat for select to authenticated using (
  (select app.is_platform_admin()) or id in (select uat_id from vine.uat_member where user_id = auth.uid()));
create policy member_select on vine.uat_member for select to authenticated using (
  user_id = auth.uid() or (select app.my_role(uat_id)) = 'uat_admin' or (select app.is_platform_admin()));
create policy visit_insert on vine.inspection_visit for insert to authenticated with check (
  user_id = auth.uid() and (select app.my_role(uat_id)) in ('uat_admin','inspector')
  and exists (select 1 from vine.target t where t.id = target_id));
create policy visit_select on vine.inspection_visit for select to authenticated using (
  (select app.my_role(uat_id)) is not null);
-- survey: select dacă ST_Intersects(footprint, my_geofence()); audit_report: membri; audit_log: uat_admin/platform_admin

-- 0006 storage: bucket-uri private field-photos, reports; cale = '<uat_id>/…'
create policy photos_rw on storage.objects for all to authenticated using (
  bucket_id in ('field-photos','reports') and (select app.my_role(((storage.foldername(name))[1])::uuid)) is not null);
```

**Obiectele care traversează geofence-ul:** RLS lasă vizibil obiectul întreg. API-ul îl **decupează** la geofence-ul UAT-ului activ, în MVT, GeoJSON și în toate măsurătorile UAT. Pentru Sireți nu schimbă nimic (tot survey-ul e în comună), dar regula e corectă pentru orice alt UAT.

**Testul RLS:**
1. **pgTAP** (`src/Web/supabase/tests/rls_isolation.test.sql`, `supabase test db --linked`).
   - Fiecare test rulează în `BEGIN … ROLLBACK`, deci nu lasă date în proiectul existent.
   - Fixture-urile de test creează în tranzacție:
     - utilizatorii `alice` (Sireți) și `bob` (Cojușna);
     - un UAT sintetic `split-A`/`split-B` care împarte zona de studiu în două.
   - Verificări:
     - Alice vede N coroane, iar Bob 0;
     - `split-A` + `split-B` văd mulțimi disjuncte, cu reuniunea = totalul;
     - un utilizator fără membership vede 0;
     - `update target` făcut de un `viewer` eșuează.
2. **pytest prin FastAPI** (`api/tests/test_rls_api.py`, marcat `@pytest.mark.db`, rulează local, nu în CI), cu token-uri emise de proiect pentru conturile de test:
   - `/summary` Bob = 0 obiecte;
   - `/objects/{id Sireți}` cu Bob → 404;
   - `/features/rows` gol pentru Bob (și MVT gol, dacă `/tiles` e implementat);
   - pe UAT-urile sintetice, suma ariilor split-A + split-B = totalul (± 0,1%).

---

## 6. Fluxul de import al datelor AI

```
pipeline AI ──(adaptor, dacă e nevoie)──▶ src/Web/data/surveys/siret3/pipeline/ (bundle, EPSG:32635)
                                 │
                   make data SURVEY=siret3
                                 │
   ┌─────────────────────────────┼──────────────────────────────────────────┐
   ▼                             ▼                                          ▼
bundle.validate()            build_static.py                          import_bundle.py (conexiune admin)
 (pydantic + reguli)          ├ measure.py → summary.json, CSV         ├ delete survey layers → COPY (WKB)
   │ raport                   ├ reproiectare 4326 → *.geojson          ├ recalcul PostGIS vs measure.py (± 0,1%)
   │ (erorile opresc,         ├ tippecanoe → canopies.pmtiles          └ survey.status = ready, audit_log
   │  avertizările nu)        └ GPX/KML pre-generate
   ▼
 src/Web/data/surveys/<id>/import_report.json
```

- `measure.py` e unic: `build_static`, API-ul și testele îl folosesc. Agregările decupate la geofence folosesc PostGIS. Un test verifică paritatea.
- Coroanele (~40–80k) se importă cu `COPY`. Ținta e sub 60 s, de pe laptop spre Supabase cloud.
- Ortofoto: o singură generare, independentă de AI (§7).
- Actualizarea de duminică (Marcaj corectat): `make data SURVEY=siret3` + import. Ținta e sub 10 minute. Repetăm sâmbătă seara pe bundle-ul `model`.

---

## 7. Ortofoto și vectori mari

**Ortofoto** (ADR-005):
1. `scripts/build_ortho.sh` rulează în Docker, cu GDAL 3.13.3:
   - `gdalbuildvrt` → `gdalwarp -t_srs EPSG:3857`;
   - MBTiles WebP q75 + `gdaladdo`, z14–z21;
   - `pmtiles convert` → `src/Web/data/ortho/siret3.pmtiles`.
2. z22 intră doar dacă fișierul rămâne sub 1,5 GB.
3. Fișierul stă **doar local** și e servit de ruta Next `/data`. Nu se urcă nicăieri.
4. Pentru mock: aceeași comandă pe cele 2 tile-uri exemplu.

**Vectori:**

| Strat | Volum estimat | Mod static | Mod API |
|---|---|---|---|
| blocuri, rânduri, ținte, deșeuri, rută, pasaje, zone interzise, geofence | ≤ ~5.000 | GeoJSON | `/features/{layer}` |
| inter-rânduri | ~5.000–8.000 | GeoJSON (~3 MB) | `/features/interrows` (bbox) |
| coroane | ~40.000–80.000 | `canopies.pmtiles` (tippecanoe `-Z15 -z21`) | MVT `ST_AsMVT` |

**Zoom:** blocuri ≤ z16; rânduri, inter-rânduri și ținte ≥ z17; coroane ≥ z18. MVT-urile au `Cache-Control: private, max-age=3600`.

Geofence-ul Sireți (~2.729 ha) e mult mai mare decât survey-ul (81,5 ha). Harta deschide pe bbox-ul survey-ului, cu geofence-ul întreg vizibil la zoom-out, iar masca gri acoperă exteriorul comunei.

---

## 8. Pagini, componente, wireframe-uri

**Shell-ul** urmează capturile Marcaj din `docs/design/` (ADR-011):
- sidebar alb, ~240 px, pliabil la 72 px, cu logo-ul **Solemtrix**, selectorul de UAT și navigația (itemul activ plin indigo);
- în subsol, cardul utilizatorului, `EN RO RU` și clopoțelul;
- conținutul stă pe `#F6F6F6`, cu carduri albe `radius-lg` `elevation-base`;
- pe hartă, sidebar-ul e pliat implicit;
- pe mobil, navigația devine o bară de jos.

Coloana „Prio” spune ce e obligatoriu pentru juriu (J), ce e stratul UAT (U) și ce se face doar dacă e timp (T).

| Rută (`/[locale]/…`) | Componente | Prio |
|---|---|---|
| `login` | `AuthSplitLayout`, `LoginForm`, `DemoButton` („Demo fără cont” → mod static), `LanguageSelect` | U (demo J) |
| `map` | `MapView`, `LayerPanel`, `Legend`, `KpiStrip`, `AttributePanel`, `SearchBox`, `GeofenceLayer`/`GeofenceMask`, `RouteLayer`, `MeasureTool` (T), `FarmRoute*` (traseu pe fermă cu start ales pe hartă, ADR-028; T), `Attribution` | **J** |
| `dashboard` | `KpiGrid`, `KpiCard`, `BlocksTable` (top 10), `CoverDonut`, `StructureBar`, `SurveyBadge`, `SavingsCard`, `UatCoverageCard` | **J** (KPI) + U |
| `blocks` | `BlocksGrid` / `RowsGrid` (DataGrid), `StructureChip`, `ExportMenu` | **J** |
| `routes` | `RouteSummary`, `TargetList`, `TargetStatusChip`, `ExportMenu`, mini-hartă | **J** (lungime) + U (stare) |
| `field` | `FieldMap`, `NextTargetCard`, `CheckInSheet`, `GpsStatus` | T |
| `audit` | `AuditForm`, `AuditResult`, PDF | T |
| `admin` | `UatTable`, `GeofenceUpload`, `MembersTable`, `SurveyImportCard` | T (Supabase Studio ajunge la demo) |

### Wireframe-uri

**Login** (ca `docs/design/marcaj_login.png`, brand Solemtrix)
```
┌────────────────────────────────────────────┬──────────────────────────────────┐
│ ◈ Solemtrix               (fundal indigo)  │ 🌐 Română ▾                       │
│                                            │                                  │
│ Viile comunei, măsurate                    │  Bine ați revenit!               │
│ și gata de inspecție                       │  Autentificați-vă pentru a continua│
│ Inventar, goluri, deșeuri, rute scurte.    │  Email    [____________________] │
│   ┌ ROW_V03-R017 ┐   (motiv casete         │  Parolă   [________________ 👁]  │
│   └──────────────┘    punctate)            │  Ați uitat parola?               │
│ ⤓ Date AI   👥 Primării   ✓ Audit          │  [        Autentificare        ] │
│                                            │  ── sau ──  [ Demo fără cont → ] │
└────────────────────────────────────────────┴──────────────────────────────────┘
```

**Dashboard UAT**
```
┌──────────┬───────────────────────────────────────────────────────────────────┐
│◈Solemtrix│ Primăria Sireți · r. Strășeni                Sireț3 · 20.05.2025 ▾ │
│ UAT ▾    │ Inventar viticol în limita comunei               [Export CSV]     │
│──────────│ ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐     │
│▣Dashboard│ │2.729 ha │ 81,5 ha │ {n}     │ {n}     │ {km}    │ {ha}    │     │
│ Hartă    │ │ comună  │zburat 3%│ blocuri │ rânduri │ rânduri │ coroane │     │
│ Blocuri  │ ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤     │
│ Rute     │ │ {ha}    │ {%}     │ {n}     │ {n}     │ {km}    │ {−min}  │     │
│ Teren    │ │inter-rând│disrupted│ deșeuri │ ținte  │ ruta    │ vs naiv │     │
│ Audit    │ └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘     │
│          │ ┌ Blocuri ──────────────────────────┐ ┌ Acoperire inter-rând ┐    │
│──────────│ │ V01  25 r  910 m  0,02 ha  4% ⚠ │ │  (donut)              │    │
│ SA user  │ │ …                     [Toate →] │ ├ Structura rândurilor ┤    │
│ EN RO RU🔔│ └───────────────────────────────────┘ └───────────────────────┘    │
└──────────┴───────────────────────────────────────────────────────────────────┘
```

**Hartă**
```
┌─┬──────────────────────────────────────────────────────────────────────────────┐
│≡│ {n} blocuri · {n} rânduri · {km} · coroane {ha} · inter-rând {ha} · ruta {km}  │
│▣├──────────────┬───────────────────────────────────────────────┬───────────────┤
│◧│ 🔍 V03-R017   │  (ortofoto; contur Sireți; exterior comunei gri)│ Rând V03-R017 │
│⌖│ Straturi      │          ═══ rânduri  ▒ inter-rânduri          │ bloc   V03    │
│☰│ ☑ Ortofoto    │          ● ținte numerotate  ■ deșeuri         │ ⚠ disrupted   │
│ │ ☑ Coroane     │          ━━▶ ruta (indigo, halo alb)           │ lungime {m}   │
│ │ ☑ Rânduri     │          ■ S/F                                 │ plante  {n}   │
│ │ ☑ Inter-rând  │                                               │ gol max {m}   │
│ │ ☑ Deșeuri     │                                  [📏][+][−]   │ [Zoom]        │
│ │ ☑ Ținte/Rută  │                                               │               │
│ │ ☐ Pasaje/Interz│ © 3DATA COLLECT / OpenAerialMap CC BY 4.0 · © OSM contributors ODbL │
└─┴──────────────┴───────────────────────────────────────────────┴───────────────┘
```

**Blocuri & rânduri**
```
┌──────────┬───────────────────────────────────────────────────────────────────┐
│ sidebar  │ Blocuri și rânduri                       [Export measurements.csv]│
│          │ [Blocuri | Rânduri]  Bloc: [Toate ▾] Structură: [Toate ▾] 🔍____   │
│          │ │ row_id   │ bloc│ lungime m│ structură│ plante │ gol max│  🗺  │   │
│          │ │ V03-R017 │ V03 │   142,30 │disrupted │     96 │   7,4 m│  →   │   │
│          │ Total: {n} rânduri · {m}                          ‹ 1 2 3 … ›     │
└──────────┴───────────────────────────────────────────────────────────────────┘
```

**Rută & inspecții**
```
┌──────────┬──────────────────────────────┬────────────────────────────────────┐
│ sidebar  │ Ruta de inspecție             │        mini-hartă cu ruta și       │
│          │ {km} · {min} la 4 km/h        │        țintele numerotate          │
│          │ −{km} (−{%}) vs naiv          │                                    │
│          │ {n} ținte · {n} accesibile    │                                    │
│          │ [GPX] [KML] [GeoJSON]         │                                    │
│          │ # │ țintă │ tip  │ rând │ stare│                                    │
└──────────┴──────────────────────────────┴────────────────────────────────────┘
```

**Mod teren (mobil)** (T)
```
┌──────────────────────────┐
│ ● GPS ±4 m               │
│   (hartă full-screen,    │
│    poziția ◉, ruta)      │
├──────────────────────────┤
│ Următoarea: #4 T021      │
│ gol 6,1 m · V05-R003     │
│ la 38 m ↗                │
│ [ Check-in ] (activ ≤2 m)│
└──────────────────────────┘
```

**Audit subvenții** (T)
```
┌──────────┬───────────────────────────────────────────────────────────────────┐
│ sidebar  │ Audit de suprafață   Blocuri: [V03 ✕][+]  Declarat: [2,40] ha     │
│          │ plantat măsurat {ha} ({%})   densitate {n}/ha   goluri {%}        │
│          │ sursa: Sireț3 · 20.05.2025 · CC BY 4.0        [Generează PDF]     │
└──────────┴───────────────────────────────────────────────────────────────────┘
```

**Admin** (T)
```
┌──────────┬───────────────────────────────────────────────────────────────────┐
│ sidebar  │ Administrare         [UAT-uri | Membri | Survey-uri]             │
│          │ │ Primăria Sireți │ MD │ 2.729 ha │ OSM 19100171 │ Sireț3 ✓ │      │
│          │ │ Primăria Cojușna│ MD │ {ha}     │ OSM {id}     │ —        │      │
└──────────┴───────────────────────────────────────────────────────────────────┘
```

---

## 9. Produsul UAT — module și ierarhie

| # | Modul | Valoare pentru UAT | La deadline |
|---|---|---|---|
| 1 | Hartă (geofence Sireți, straturi, căutare, legendă) | inventarul vizual | **Real** |
| 2 | Dashboard KPI + `UatCoverageCard` (comună vs zburat) | „câtă vie am, în ce stare și cât din comună e acoperit” | **Real** |
| 3 | Blocuri & rânduri + export CSV | lista de lucru a agronomului | **Real** |
| 4 | Rute & inspecții (ordine, lungime, GPX/KML, stare) | economia de timp pe teren | **Real** (ruta din pipeline, fără replanificare) |
| 5 | Mod teren (check-in ≤ 2 m) | inspectorul are ținta următoare | **Doar dacă e timp** (F3); fără offline |
| 6 | Audit subvenții (PDF) | măsurat vs declarat | **Doar dacă e timp**; suprafața declarată e seed |
| 7 | Istoric survey-uri | evoluția golurilor | **Tăiat**: un singur zbor; apare doar în roadmap-ul de pitch |
| 8 | Admin | onboarding-ul unei primării | **Tăiat din UI**: se face din `seed_demo.py` + Supabase Studio; endpoint-ul `/admin/uats` cu id OSM, doar dacă e timp |

**Ce adăugăm:**
- cardul de economii: se afișează doar dacă AI-ul dă `baseline_length_m`;
- cardul de acoperire: ha comună, ha zburate, %;
- filtrul „Rânduri de verificat” (`disrupted`, sortate după golul maxim);
- jurnalul de audit pe scrieri.

---

## 10. Calitate, teste, performanță, CI

**Frontend:**
- ESLint (+ regula anti-culori în afara `src/theme/`), Prettier, `tsc --noEmit`;
- Vitest:
  - `units.ts` (ro/en/ru, pragul m²/ha);
  - parserele zod de manifest și summary;
  - `DataSource.static`;
  - paleta (fiecare enum are culoare);
- Playwright `demo-flow.spec.ts`, static, offline, pe `siret3-mock`:
  1. KPI = 2 blocuri, 51 de rânduri, 1941,6 m, 536,2 m² de coroane, 4064,4 m² de inter-rânduri;
  2. click pe `V01-R05` → `V01`, `regular`;
  3. ruta are `length_m`;
  4. schimbarea limbii EN și RU;
  5. export CSV.

  Pe desktop și pe mobil.

**Backend:**
- ruff, mypy (strict pe `app/`);
- pytest:
  - `test_measure_examples` (cifrele din CLAUDE.md §6.6, ± 0,05);
  - `test_crs_control`:
    - START ↔ lat/lon ≤ 1e-7°;
    - colțurile tile-urilor după formulă;
    - toate cele 311 GeoTIFF, doar dacă există local;
    - geofence-ul Sireți conține toate centrele de tile și START-ul;
  - `test_contract`;
  - `test_exports`;
  - `test_rls_api` (marcat `db`, doar local);
- pgTAP: izolarea RLS.

**CI (`.github/workflows/web.yml`, excepție aprobată):**
- declanșare pe push și PR, cu `paths: src/Web/**`;
- job-uri:
  - `frontend`: pnpm install, lint, tsc, vitest, Playwright static pe `siret3-mock`;
  - `api`: `uv sync`, ruff, mypy, `pytest -m "not db"`, plus verificarea că `schema.d.ts` e la zi;
- mock-ul se generează în CI din `data & info/05_examples`, care e în git.
- Fără secrete, fără DB în CI. RLS-ul se testează local, prin `make test`.

**Performanța** (15% din scor): `scripts/bench.py` scrie `docs/PERF.md` (hardware-ul laptopului, versiunile). Măsurăm:
- `build_static` pe 311 tile-uri;
- `build_ortho`;
- importul în Supabase;
- API p50/p95 pe `/summary`, `/features/rows`, `/tiles/canopies` (200 de request-uri);
- Lighthouse pe `/map`.

Ținte: build static sub 3 min; p95 MVT sub 300 ms (laptop → Supabase cloud); LCP pe hartă sub 2,5 s.

**Docker:** `docker-compose.yml` pornește `frontend` (Next standalone) și `api` (uvicorn), iar profilul `static` pornește doar frontend-ul. Supabase e cloud (ADR-014).

---

## 11. Infrastructură și configurare

| Componentă | Unde |
|---|---|
| Frontend + API | laptopul de demo, Docker (`make up`); modul static (`make demo`) ca fallback offline |
| DB / Auth / Storage | **proiectul Supabase existent** (`supabase link`) |
| Ortofoto și fișierele statice | local, `src/Web/data/` (ignorat de git), servite de ruta Next `/data` |

**Linkul din README:**
- regulamentul cere „link to the working web interface” în README, dar acceptă demo-ul de pe laptop;
- în README punem instrucțiunile de rulare (`make demo`, un singur comand) și demo-ul de pe laptop;
- dacă vreți și un URL public temporar, e nevoie de un serviciu de tunel sau de hosting, pe care l-ați exclus (întrebarea Q2).

**`.env.example`:**

| Variabilă | Unde | Note |
|---|---|---|
| `NEXT_PUBLIC_DATA_MODE` | fe | `static` \| `api` |
| `NEXT_PUBLIC_SURVEY_ID` | fe | `siret3-mock` \| `siret3` |
| `NEXT_PUBLIC_API_URL` | fe | `http://localhost:8000` |
| `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | fe | nu sunt secrete |
| `WEB_DATA_DIR`, `ORGANIZER_DATA_DIR` | fe (ruta `/data`), scripts | |
| `SUPABASE_URL`, `SUPABASE_JWT_MODE`, `SUPABASE_JWT_SECRET` | api | `SUPABASE_JWT_SECRET` e secret, doar pentru hs256 |
| `DATABASE_URL` | api | rolul `web_api`, pooler în modul tranzacție; secret |
| `DATABASE_ADMIN_URL`, `SUPABASE_SECRET_KEY` | api admin, scripts | secret, niciodată în fe |

---

## 12. Ce e real și ce e seed

| Element | Real | Seed / demo |
|---|---|---|
| Ortofoto, coroane, rânduri, inter-rânduri, deșeuri, ținte, rută | ✔ pipeline / Marcaj (mock marcat până vin datele) | |
| Măsurători, KPI, CSV, GPX/KML | ✔ | |
| **UAT Sireți**: geofence-ul | ✔ limita administrativă OSM 19100171 (ODbL) | |
| **UAT Cojușna**: geofence-ul | ✔ OSM | există pentru demo-ul de izolare (0 obiecte) |
| Utilizatori | | `primar@sireti.demo` (uat_admin), `inspector@sireti.demo`, `primar@cojusna.demo`, `admin@solemtrix.demo` (platform_admin) |
| Vizite de teren, stări ale țintelor | check-in-ul, dacă F3 se face | 5 vizite seed |
| Suprafețe declarate la audit | | 3 valori „declarat (exemplu)” |

Seed-ul: `scripts/seed_demo.py` descarcă limitele din OSM (Overpass, cu fallback la un fișier salvat în `src/Web/supabase/fixtures/osm/`), le reproiectează în 32635 și creează UAT-urile, utilizatorii (Admin API) și membership-urile. Scriptul e idempotent.

---

## 13. Faze și backlog (un singur om)

**Buget realist:** ~23 h de lucru efectiv:
- vineri 22:00–02:00 (4 h);
- sâmbătă 09:00–13:00 + 14:00–19:45 + 20:15–23:15 (12,75 h);
- duminică 07:30–14:00 (6,5 h).

Sumele din tabele se încadrează exact în acest buget. Nu există rezervă, de aceea contează ordinea tăierilor de mai jos.

Task-urile sunt în ordinea execuției; fiecare durează cel mult 2 h.

### F0 — vineri 22:00 → 02:00: fundația de date și scheletul

| ID | Task | h | Dep. | Criteriu de acceptare |
|---|---|---|---|---|
| F0-1 | `src/Web/api` scaffold (uv, FastAPI, ruff, mypy, pytest), `geo/crs.py`, `geo/measure.py`, `test_measure_examples`, `test_crs_control` | 1,5 | — | pytest verde pe cifrele de control |
| F0-2 | `bundle.py` (pydantic) + `cvat_to_bundle.py` + `build_static.py` → `siret3-mock` | 1,5 | F0-1 | `make mock && make data SURVEY=siret3-mock` produce `static/` valid |
| F0-3 | Scaffold `src/Web/frontend` (Next 16.3.6, TS 5.9.3, pnpm, versiuni exacte), ESLint/Prettier/Vitest, `.gitignore`, `Makefile` | 0,75 | — | `pnpm dev` pornește; `make lint` trece |
| F0-4 | `build_ortho.sh` pe cele 2 tile-uri; **pornește ortofoto-ul complet** peste noapte | 0,25 + rulare | — | `siret3-mock` și `siret3` generate până dimineață |

### F1 — sâmbătă 09:00 → 19:45 (9,75 h): toate cerințele juriului, în mod static

| ID | Task | h | Dep. | Criteriu de acceptare |
|---|---|---|---|---|
| F1-1 | `tokens.ts`, `theme.ts` (MUI `cssVariables`, light+dark, Inter), `mapPalette.ts`, regula ESLint anti-culori | 1 | F0-3 | carduri și butoane ca Marcaj; lint-ul prinde un `#fff` |
| F1-2 | i18n (ro/en/ru), `proxy.ts`, `units.ts` + teste | 1 | F0-3 | `/`, `/en`, `/ru`; testele de formatare trec |
| F1-3 | Shell (sidebar Solemtrix, pliabil, bară jos pe mobil) + `app/data/[...path]` cu Range + `DataSource.static` | 1,5 | F1-1, F1-2 | navigarea merge; `Range` → 206 |
| F1-4 | `MapView` + protocolul pmtiles + ortofoto + straturile GeoJSON + stilurile din `mapPalette` + `LayerPanel` + `Legend` | 2 | F1-3, F0-2 | straturile aliniate cu ortofoto (control: START, colțurile) |
| F1-5 | Coroane din `canopies.pmtiles` (tippecanoe în `build_static`) + regula de zoom | 0,75 | F1-4 | z18+ fluid |
| F1-6 | `AttributePanel` + `SearchBox` (căutare simplă după ID) | 1 | F1-4 | click și căutare pe `V01-R05` → valori corecte |
| F1-7 | `KpiStrip` + Dashboard (KPI + blocuri + 2 grafice) | 1,25 | F1-3 | cifrele = `measurements.csv`, cu unitate și sursă |
| F1-8 | `RouteLayer` + `RouteSummary` + export GPX/KML/GeoJSON (pre-generate) | 1,25 | F1-4 | lungimea afișată = `length_m` |

### F1b — sâmbătă 20:15 → 23:15 (3 h): tabel, date reale, e2e, CI

| ID | Task | h | Dep. | Criteriu de acceptare |
|---|---|---|---|---|
| F1-9 | Blocuri & rânduri (DataGrid, filtre, zoom, export CSV) | 1 | F1-7 | CSV identic cu `static/measurements.csv` |
| F1-10 | **Date reale `siret3`** (bundle `model` sau adaptor) → `make data` | 1 | bundle AI | harta reală completă; KPI reale |
| F1-11 | Playwright `demo-flow` + `.github/workflows/web.yml` + poarta offline (Wi-Fi oprit) | 1 | F1-9 | CI verde; demo-ul static merge offline |

### F2 — duminică 07:30 → 12:15 (4,75 h): Supabase, API, login, geofence (stratul UAT minim)

| ID | Task | h | Dep. | Criteriu de acceptare |
|---|---|---|---|---|
| F2-1 | Migrațiile 0001–0005 în schema `vine` + `make db-push` + pgTAP pe RLS | 1,25 | — | `supabase test db --linked` verde; DB-ul existent neatins în afara `vine`/`app` |
| F2-2 | `seed_demo.py` (OSM Sireți + Cojușna, utilizatori) + `import_bundle.py` (COPY) | 0,75 | F2-1, F1-10 | 2 UAT-uri; importul `siret3` sub 60 s; paritate de măsurători |
| F2-3 | `auth.py` + `rls_tx` + endpoint-urile ★ (`/me`, `/uats`, `/summary`, `/features`, `/objects`) + `test_rls_api`. `/tiles/canopies` trece la „dacă e timp”: coroanele vin din PMTiles static, fără scurgere, pentru că tot survey-ul e în Sireți | 1,5 | F2-2 | Bob (Cojușna) primește 0 și 404; Sireți primește cifrele complete |
| F2-4 | Login (split, Solemtrix, „Demo fără cont”) + `@supabase/ssr` + `DataSource.api` + `make gen-api` + `GeofenceMask` (UAT-ul se schimbă prin logout/login; selectorul, dacă e timp) | 1,25 | F2-3 | login Sireți → dashboard și hartă din API; Cojușna → hartă goală, cu propria limită |

Exportul corectat din Marcaj vine duminică între 11 și 13. Reimportul lui e F3-1.

### F3 — duminică 12:15 → 14:00 (1,75 h): date finale, polish, repetiție, freeze

| ID | Task | h | Dep. | Criteriu de acceptare |
|---|---|---|---|---|
| F3-1 | Reimportul datelor finale (`marcaj_corrected`) + completarea cifrelor în `DEMO.md` | 0,5 | F2-2 | cifrele finale în UI și în scenariu |
| F3-2 | `bench.py` → `docs/PERF.md` + stări de încărcare și eroare + verificarea dark mode | 0,75 | toate | PERF completat |
| F3-3 | **Doar dacă a rămas timp din fazele anterioare**, în ordinea asta: `/tiles/canopies` MVT → selector UAT → mod teren (check-in ≤ 2 m, fără offline) → PDF audit → `/admin/uats` | 0 (în buget) | F2-3 | — |
| F3-4 | Repetiția `DEMO.md` ×2 (API + static offline); **freeze la 14:00** | 0,5 | toate | două rulări curate |

### 14:00 → 15:00: doar fix-uri

`make check`, instrucțiunile de rulare în `README.md`-ul din rădăcină, commit final. Nicio funcționalitate nouă.

**Drumul critic:** F0-1 → F0-2 → F1-3 → F1-4 → F1-6/F1-7/F1-8 → F1-10 (depinde de AI) → F1-11.

**Ordinea tăierilor dacă rămânem în urmă** (se aplică de sus în jos):
1. graficele de pe dashboard (rămân KPI-urile);
2. `SearchBox`;
3. dark mode (rămâne doar light);
4. în F2: pytest-ul prin API (rămâne pgTAP);
5. **ultima soluție:** F2 întreg. Demo-ul juriului rămâne complet în mod static, iar stratul UAT se prezintă din `PLAN.md` și din migrații.

---

## 14. Riscuri

| Risc | Prob. | Impact | Reducere |
|---|---|---|---|
| Un singur om, ~23 h | sigur | mare | ordine strictă J → U → T; tăieri decise dinainte (§13); freeze la 14:00 |
| Formatul sau ora datelor AI necunoscute | mare | mare | mock pe contract; adaptor într-un singur loc; plasa `cvat_to_bundle` din exportul Marcaj |
| DB-ul Supabase existent are alte tabele sau alte setări (chei legacy, PostGIS lipsă) | medie | medie | schema separată `vine`; `create extension if not exists`; JWT cu fallback HS256; verificare la F2-1 |
| Testele pgTAP rulate pe DB-ul de producție | medie | medie | fiecare test în `BEGIN … ROLLBACK`; fixture-uri doar în tranzacție; fără `supabase db reset` pe cloud |
| Wi-Fi-ul cade la demo | mare | critic | `make demo` complet offline, repetat cu Wi-Fi oprit (F1-11, F3-4) |
| Volumul de coroane încetinește harta | medie | mare | PMTiles/MVT, coroane doar de la z18 |
| Latența laptop → Supabase cloud în modul API | medie | medie | MVT cu cache; la demo, fallback pe modul static fără comentarii |
| Reproiectare greșită | medie | critic | teste de control (START, colțurile tile-urilor, geofence ⊇ tile-uri) |
| Măsurători diferite între static, API și CSV-ul AI | medie | mare | un singur `measure.py`; paritate PostGIS; avertizare peste 0,5% |
| Overpass/OSM indisponibil la seed | mică | mică | limitele salvate ca fixture în repo (ODbL, cu atribuire) |
| Versiuni noi (Next 16, MUI 9, MapLibre 6) | medie | medie | versiuni exacte; codemod-uri; fallback `--webpack` |
| Fără URL public pentru README | medie | mică | demo de pe laptop + instrucțiuni `make demo` (Q2) |

---

## 15. Întrebări rămase

1. **Proiectul Supabase existent:**
   - care e `project ref`-ul?
   - cine îmi dă acces la CLI (`supabase login` + `link`)?
   - DB-ul e folosit și de altceva, de exemplu de echipa AI? Planul presupune că da, de aceea schema separată `vine`.
   - Auth-ul are email+parolă activat?
2. **URL public în README:** vă ajunge demo-ul de pe laptop + instrucțiunile `make demo`, sau vreți totuși un link public temporar? Asta ar cere un serviciu de tunel sau de hosting, pe care l-ați exclus.
3. **Limba pitch-ului:** RO sau EN? De ea depinde limba implicită a UI-ului la demo și textul din `DEMO.md`.
