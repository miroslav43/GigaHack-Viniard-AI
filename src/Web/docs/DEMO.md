# DEMO — Solemtrix Vineyard, scenariul de 3 minute (pitch 5 min + 5 min întrebări)

Ordinea cerută de juriu: **hartă → obiecte → ID-uri → măsurători → rută.** Stratul UAT vine după, ca valoare de produs.

Cifrele dintre `{acolade}` se citesc din `summary.json` / `/summary` pe datele finale (`marcaj_corrected`) și se completează duminică la F3-6. **Nu se spun cifre inventate.**

## Pregătire (înainte de a urca pe scenă)

| Ce | Cum |
|---|---|
| Laptop | Chrome pe ecran complet, zoom 100%, notificările oprite, încărcătorul în priză |
| Tab 1 | laptop, modul API (`make up`): `http://localhost:3000/login`, deja completat cu `primar@sireti.demo` |
| Tab 2 (fallback) | laptop, `make demo` (static, offline, port 3001) → `http://localhost:3001/map` |
| Tab 3 | laptop, fereastră incognito, sesiunea `primar@cojusna.demo` (UAT Cojușna), pentru izolare |
| Telefon (doar dacă F3-3 s-a făcut) | `/field` prin IP-ul laptopului în aceeași rețea, logat ca `inspector@sireti.demo` |
| Pre-încălzire | cu 2 minute înainte: `GET /health` (API + Supabase), harta la zoom 19 pe blocul demo, ruta |
| Poarta de fallback | dacă Supabase nu răspunde (Wi-Fi) sau tab 1 nu încarcă în 5 s → tab 2 (static, offline), fără comentarii: „aceleași date, pre-calculate” |

## Scenariul (≈ 3:00)

| t | Ecran / acțiune | Ce se spune |
|---|---|---|
| 0:00–0:15 | **Login** Solemtrix → autentificare ca Primăria Sireți | „Solemtrix, contul Primăriei Sireți din raionul Strășeni. Aplicația vede doar ce e în limita administrativă a comunei, iar izolarea e în baza de date, nu doar pe ecran.” |
| 0:15–0:40 | **Dashboard**: acoperirea + banda KPI | „Comuna are ~2.729 ha, iar drona a zburat 81,5 ha din ea. Din ortofotografia Sireț3 din 20 mai 2025: `{blocks}` blocuri, `{rows}` rânduri cu `{row_length_km}` km, coroane pe `{canopy_ha}` ha și inter-rânduri pe `{interrow_ha}` ha. Totul e măsurat în metri, în UTM 35N, nu pe hartă.” Arătăm cu mouse-ul unitatea și sursa de sub fiecare cifră. |
| 0:40–1:05 | **Hartă**: ortofoto + geofence, zoom pe blocul `{demo_block}` | „Ortofoto la 5 cm. Conturul indigo e limita comunei, iar masca gri e în afara ei. Comut straturile: coroane (una per plantă), axe de rând colorate după structură, inter-rânduri după acoperire, deșeuri.” Comutăm 2–3 straturi din `LayerPanel`. |
| 1:05–1:30 | **Obiecte + ID-uri**: click pe rândul portocaliu `{demo_row}` | „Rândul `{demo_row}` din blocul `{demo_block}`: *disrupted*, gol de `{gap_m}` m, `{row_len_m}` m lungime, `{plants}` plante. ID-ul e același peste marginile tile-urilor, fiindcă e un singur rând fizic.” Apoi click pe un inter-rând: „*mixed*, `{ir_area}` m².” |
| 1:30–1:50 | **Blocuri & rânduri**: tabel, sortare după golul maxim | „Lista de lucru a agronomului: rândurile cu goluri, sortate. Exportul e identic cu `measurements.csv` din repo.” Click pe export CSV. |
| 1:50–2:25 | **Rută & inspecții** | „Ruta pleacă din punctul dat de organizatori, trece pe la `{targets}` ținte (goluri și deșeuri), doar prin inter-rânduri și pasaje autorizate, și revine la start: `{route_km}` km, `{route_min}` minute la 4 km/h.” Dacă avem baseline: „față de ruta naivă economisim `{saved_km}` km, adică `{saved_pct}` %, `{saved_min}` minute pe tură.” Export GPX. |
| 2:25–2:45 | **Izolare**: tab 3 (Cojușna) | „Primăria vecină, Cojușna: își vede doar limita ei. Nimic din Sireți nu vine prin API, nici măcar un rând. E testat automat, și în baza de date, și prin API.” |
| 2:45–3:00 | **Roadmap** (sau telefon/PDF, dacă F3-3 s-a făcut) | „Următorul pas: modul de teren, cu ținta următoare și check-in la 2 m, și raportul de audit pentru subvenții, adică suprafața măsurată față de cea declarată. Modelul de date și securitatea pentru ele există deja.” |

**Închidere (o frază):** „Solemtrix rulează același cod pe orice zbor nou și orice primărie. Adnotările au fost corectate în Marcaj, iar tot calculul e reproductibil din repo, cu o singură comandă.”

## Întrebări probabile și răspunsuri scurte

| Întrebare | Răspuns |
|---|---|
| Cum calculați ariile? | Python (shapely) și PostGIS, plan, în EPSG:32635, fără corecție de teren. Aria coroanelor = aria reuniunii poligoanelor. Testat pe tile-urile exemplu: 237,1 m² și 299,1 m², exact ca referința. |
| De ce merge fără internet? | Modul static: aceleași cifre, pre-generate de același cod, ortofoto în PMTiles local. |
| Cum e izolată o primărie? | RLS în Postgres: un rând e vizibil doar dacă utilizatorul e membru al UAT-ului și geometria intersectează geofence-ul. API-ul rulează query-urile ca utilizatorul, iar cifrele se decupează la limită. |
| Cât de repede merge? | Cifrele din `docs/PERF.md`: build static `{t_build}`, p95 tile-uri `{p95_mvt}` ms, LCP `{lcp}` s, pe `{hardware}`. |
| Ce e real și ce e demo? | Real: harta, obiectele, măsurătorile, ruta, auth, izolarea și limitele UAT (OSM, Sireți și Cojușna). Demo: conturile și suprafețele declarate de la audit. |
| De ce nu e online? | Demo-ul rulează pe laptop, cum permite regulamentul. Baza de date și autentificarea sunt în Supabase cloud; aplicația pornește cu `make up` sau, complet offline, cu `make demo`. |

## Checklist F3-6 (repetiție, de 2 ori)

- [ ] Rulare completă în modul API sub 3:00, fără erori în consolă.
- [ ] Rulare offline (Wi-Fi oprit), `make demo`, până la pasul „Rută” inclusiv.
- [ ] Toate `{acoladele}` au fost înlocuite cu cifrele finale.
- [ ] Contul Cojușna nu vede nimic din Sireți (verificat vizual și în Network).
- [ ] Instrucțiunile de rulare (`make up`, `make demo`) sunt în `README.md`-ul din rădăcină.
- [ ] Limba pitch-ului e stabilită. Dacă e EN: UI pe `/en` și textul de mai sus tradus.
