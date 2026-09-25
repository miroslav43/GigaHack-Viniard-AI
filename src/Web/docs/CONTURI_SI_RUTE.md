# Solemtrix Vineyard — conturi și rute

> Versiunea din repo **nu conține parole**: repo-ul `miroslav43/GigaHack-Viniard-AI` e public, iar contul
> `platform_admin` poate crea și șterge conturi. Parola conturilor demo se primește direct de la echipă
> (sau din fișierul local `src/Web/CONTURI_SI_RUTE.local.md`, ignorat de git).

Actualizat: 26.09.2026 · App local: `http://localhost:3000` · Supabase: proiect `htdnuwmnztenevnfahie` (eu-west-1)

---

## 1. Conturi

Toate conturile demo au aceeași parolă, comunicată privat (nu e în repo).

| Email | Rol (`uat_role`) | Primărie (`uat`) | Ce vede |
|---|---|---|---|
| `admin@solemtrix.demo` | `platform_admin` — administrator platformă | — (toată platforma) | după login ajunge direct în **`/super-admin`** (panou general, UAT-uri, utilizatori, survey-uri, sistem, jurnal); paginile de primărie îl redirecționează acolo |
| `primar@sireti.demo` | `uat_admin` — administrator UAT | Sireți | panou, hartă, blocuri, rută pentru Sireți |
| `inspector@sireti.demo` | `inspector` — inspector de teren | Sireți | panou, hartă, blocuri, rută pentru Sireți |
| `primar@cojusna.demo` | `uat_admin` — administrator UAT | Cojușna | doar limita comunei Cojușna + „niciun zbor” (nu vede nimic din Sireți) |

**Fără cont:** butonul **„Demo fără cont”** de pe `/login` → date publice Sireț3, doar vizualizare (cookie `solemtrix_demo=1`, 7 zile).

**Unde se gestionează:**
- din aplicație: `/super-admin?tab=users` (logat ca `admin@solemtrix.demo`) — creare, primărie + rol, resetare parolă, dezactivare, ștergere;
- din Supabase: [Auth → Users](https://supabase.com/dashboard/project/htdnuwmnztenevnfahie/auth/users); rolul și primăria sunt în `raw_app_meta_data` (`uat`, `uat_role`);
- re-creare de la zero: `src/Web/supabase/seed/demo_users.sql` (înlocuiește `__DEMO_PASSWORD__` cu parola demo și rulează în SQL Editor).

**Chei (în `src/Web/frontend/.env.local`, ignorat de git):**

| Variabilă | Tip | Unde e folosită |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | public | `https://htdnuwmnztenevnfahie.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | public (browser) | `sb_publishable_nJ1b-A4knJ_h7deNIcyK1g_U1JdBUVk` |
| `SUPABASE_SECRET_KEY` | **secret, doar server** | Admin API (tab-ul Utilizatori). Valoarea e doar în `.env.local` / [API Keys](https://supabase.com/dashboard/project/htdnuwmnztenevnfahie/settings/api-keys); nu o copia în fișiere partajate |

---

## 2. Pagini

Româna nu are prefix; engleza și rusa au prefix: `/en/…`, `/ru/…` (ex. `/en/harta`, `/ru/ruta`). Limba se schimbă din butoanele RO / EN / RU și se ține minte în cookie.

| Rută | Pagină | Cine are acces | Parametri |
|---|---|---|---|
| `/login` | Autentificare (+ „Demo fără cont”) | oricine; un utilizator logat e trimis la `/` | `?next=/harta` — unde revine după login |
| `/` | Panou general (KPI comună, zbor, plantații, inspecție, blocuri) | cont logat sau demo | — |
| `/harta` | Hartă: ortofoto, straturi, atribute, căutare, rută, unealtă de măsurare | cont logat sau demo | `?rand=V02-R16` — selectează rândul; `?bloc=V01` — zoom pe bloc |
| `/blocuri` | Blocuri și rânduri (tabel, filtre, export CSV) | cont logat sau demo | — |
| `/ruta` | Rută de inspecție (lungime, durată, ordine ținte, GPX / GeoJSON) | cont logat sau demo | — |
| `/super-admin` | Consola platformei (shell propriu, fără meniul de primărie) | **doar `platform_admin`**, **doar prin URL** (nu e în meniu); oricine altcineva: **HTTP 403**. Administratorul e trimis aici automat după login și de pe `/`, `/harta`, `/blocuri`, `/ruta` | `?tab=overview` (implicit) · `uat` · `users` · `surveys` · `system` · `audit` |
| `/acces-interzis` | Pagina 403 „Acces interzis” | afișată automat de `proxy.ts` | — |

Comportament fără cont și fără demo: orice pagină (în afară de `/login` și `/super-admin`) → redirect la `/login?next=…`; `/super-admin` → 403 direct.

O primărie fără zbor în limita ei (ex. Cojușna) vede pe `/`, `/harta`, `/blocuri`, `/ruta` harta limitei + mesajul „Niciun zbor de dronă în limita acestei primării”.

---

## 3. Fișiere de date (publice, fără autentificare)

Generate de `pnpm data` în `src/Web/frontend/public/` (ignorate de git). Sunt date deschise Sireț3 (CC BY 4.0) și limite OSM (ODbL).

| Rută | Conținut |
|---|---|
| `/data/siret3-mock/summary.json` | KPI (blocuri, rânduri, lungimi, arii, rută) |
| `/data/siret3-mock/rows.json` | rândurile (tabelul) |
| `/data/siret3-mock/rows.geojson` · `canopies.geojson` · `interrows.geojson` · `blocks.geojson` · `waste.geojson` · `targets.geojson` · `route.geojson` | straturile hărții (EPSG:4326) |
| `/data/siret3-mock/route_EPSG32635.geojson` | ruta în EPSG:32635, format livrabil |
| `/data/siret3-mock/route.gpx` | ruta pentru GPS |
| `/data/siret3-mock/measurements.csv` | măsurători (format `src/Web/CLAUDE.md` §6.4) |
| `/data/tiles.json` | indexul celor 311 tile-uri (colțuri lon/lat) |
| `/data/uats.json` | ariile primăriilor (fixture) |
| `/data/ref/study_area.geojson` · `passages.geojson` · `forbidden.geojson` · `start.geojson` | date organizatori (EPSG:4326) |
| `/data/ref/geofence_sireti.geojson` · `geofence_cojusna.geojson` | limite administrative OSM |
| `/ortho/overview.webp` | mozaic ortofoto de ansamblu (~0,4 m/px) |
| `/ortho/tiles/siret3_rXXX_cYYY.jpg` | ortofoto pe tile, 1024 px (5 cm/px) |
| `/maplibre/maplibre-gl-worker.mjs` · `maplibre-gl-shared.mjs` | worker-ul MapLibre (copiat de `predev` / `prebuild`) |

---

## 4. Supabase (baza de date)

| Obiect | Tip | Acces |
|---|---|---|
| `public.uat` | tabel — primării (limită OSM, arie, activă) | citire: propria primărie; scriere: `platform_admin` |
| `public.survey` | tabel — zboruri (amprentă, cale date) | citire: survey-urile care intersectează propria primărie; scriere: `platform_admin` |
| `public.admin_audit_log` | tabel — jurnal (append-only) | citire și inserare: `platform_admin` |
| `public.uat_public` · `survey_public` · `uat_survey` | view-uri (security_invoker, GeoJSON) | ca tabelele de mai sus |
| `public.enroll_uat(...)` · `public.upsert_survey(...)` | RPC | `authenticated`, RLS decide (practic doar `platform_admin`) |
| `anon` | rol | **niciun acces** la tabele |

Link-uri: [Proiect](https://supabase.com/dashboard/project/htdnuwmnztenevnfahie) · [SQL Editor](https://supabase.com/dashboard/project/htdnuwmnztenevnfahie/sql/new) · [Table Editor](https://supabase.com/dashboard/project/htdnuwmnztenevnfahie/editor) · [Auth Users](https://supabase.com/dashboard/project/htdnuwmnztenevnfahie/auth/users) · [API Keys](https://supabase.com/dashboard/project/htdnuwmnztenevnfahie/settings/api-keys) · [Security Advisors](https://supabase.com/dashboard/project/htdnuwmnztenevnfahie/advisors/security)

Schema: `src/Web/supabase/migrations/20260926000100_platform_admin.sql` · Seed: `src/Web/supabase/seed/uats_surveys.sql`, `demo_users.sql`.

---

## 5. Pornire rapidă

```bash
cd src/Web/frontend
pnpm install
pnpm data        # ortofoto + date (~35 s; cere tile-urile în "data & info/01_tiles")
pnpm dev         # http://localhost:3000
```

Alte comenzi: `pnpm build` · `pnpm start` · `pnpm lint` · `pnpm typecheck` · `pnpm e2e` (Playwright, 24 teste) · `node scripts/gen-seed-sql.mjs`.
