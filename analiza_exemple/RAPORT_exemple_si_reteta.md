# Analiza exemplelor Sireț3 și rețeta de pre-adnotare

Date: `05_examples/siret3_examples_cvat/annotations.xml` (2 tile-uri, 750 obiecte), regulile v1.0 și criteriile de scor.
Toate cifrele de mai jos sunt măsurate de scripturile din `scripts/` (ieșiri brute în `analysis.txt`, `fit.txt`).

## 1. Concluzia principală

Referința este **semi-automată și foarte regulată**. Dacă o reproducem, primim scor aproape maxim pe geometrie:

| Ce am măsurat | r021_c012 (V01) | r006_c004 (V02) |
|---|---|---|
| Vârfuri canopy la distanța 0,275–0,30 m de axă (marginea coridorului) | 27% din toate vârfurile | 25% |
| Distanța maximă a vreunui vârf canopy de axă | 0,34 m | 0,35 m |
| Vârfurile inter-rândurilor față de axa cea mai apropiată | 0,298–0,301 m | 0,298–0,302 m |
| Aria canopy minimă | 0,190 m² | 0,189 m² |

Deci: **canopy = vegetație ∩ coridor de ±0,30 m în jurul axei**, componente conexe, filtrate la ≥ 0,19 m² (pragul de 0,2 m² din reguli).
**Inter-rând = banda dintre axa_i + 0,30 m și axa_{i+1} − 0,30 m**, poligon cu 4 vârfuri (5 dacă atinge un colț), tăiat la marginea tile-ului.

Ipoteza „segmente de 3–4 m” **nu se confirmă**: e o plantă (sau un pâlc de vegetație continuă) per poligon.

## 2. Ce arată exemplele, per clasă

### vineyard (canopy): 650 poligoane
- Arie mediană 0,47 / 0,57 m²; lungime pe rând mediană 1,44 / 1,77 m; lățime peste rând ≤ 0,60 m (plafonată de coridor).
- 14–15 vârfuri în mediană (latură mediană 0,23 m), deci contur simplificat, nu la nivel de pixel.
- Distanța dintre canopy-uri pe rând: mediana 0,30 / 0,45 m; 13% se ating (≤ 5 cm). Canopy-urile atinse nu sunt despărțite geometric „la plantă”: separarea vine din golurile din masca de vegetație.
- **Atenție, contrar regulii „never one polygon over a row”:** în r006, pe rândurile cu iarbă (R16–R18), referința conține poligoane de 10–31 m² și până la 55 m lungime (tot rândul). Iarba din coridor e inclusă în canopy. 25 de poligoane > 2 m² dau 52% din aria canopy a tile-ului. Nu le „reparați” împărțindu-le dacă vrem să semănăm cu referința; merită totuși o întrebare pe Slack la Marcaj dacă referința ascunsă face la fel.
- Canopy-urile tăiate de marginea tile-ului sunt trase până la margine (15–16 per tile).
- Suprapuneri canopy–canopy practic nule (< 0,01 m²).

### row: 51 polilinii
- **Toate au exact 2 puncte** (drepte), toate au ambele capete pe marginea tile-ului (în aceste tile-uri rândurile traversează tot).
- Axa trece prin centrul plantelor: centroidul canopy-ului e la 3,5 cm mediană de axă.
- Distanța între rânduri: 2,78 m mediană (2,49–3,73; rândurile din V01 se deschid în evantai, unghiuri 46,7–53,2°), 2,53 m în V02.
- row_id numerotat consecutiv, ordonat pe perpendiculară (`V01-R01`…`R25`). Rândurile de colț de 1–5 m sunt și ele adnotate.
- row_structure: `disrupted` dacă golul maxim ≥ 5 m, **inclusiv golul de la capăt până la marginea tile-ului** (R07 are gol interior 2,3 m dar 11,3 m la capăt → disrupted; R23: 6,8 m la capăt). Regula pe canopy-urile de referință reproduce toate cele 51 etichete (un caz la limită: 5,0 m regular vs 5,0 m disrupted).

### interrow_area: 49 poligoane
- Lățime medie 2,03 / 1,84 m = distanța dintre axe − 0,60 m. Arie 7–179 m².
- Toate au vârfurile pe marginea tile-ului. Nu există inter-rând în afara rândurilor extreme.
- interrow_cover: fracția de vegetație (aceeași mască a*) este 0,00–0,16 pentru `bare_soil` și 0,51–0,68 pentru `mixed`, deci pragurile din reguli (25% / 75%) funcționează direct pe masca noastră.

### waste: 0 obiecte
- Exemplele nu au niciun gunoi. În schimb au **327 și 648 de pete albe** (tuburi de protecție, țăruși) de ≥ 0,01 m², din care 68 / 308 la sub 0,6 m de axă. Un detector pe culoare (alb/luminos) ar produce sute de fals-pozitive per tile.

## 3. Test: cât de aproape ajunge rețeta

Scor canopy = 0,6·IoU + 0,4·F1@0,5 (formula oficială), pe cele două tile-uri:

| Variantă | r021 IoU / F1 | r006 IoU / F1 | Scor mediu |
|---|---|---|---|
| ExG simplu, fără coridor (cel mai bun prag) | IoU 0,49 / F1 nemăsurat | IoU 0,45 / F1 nemăsurat | n/a |
| **Lab a\* + coridor, cu axele de referință** | 0,815 / 0,836 | 0,878 / 0,899 | **0,855** |
| **Totul automat (axe detectate + canopy + inter-rânduri)** | 0,774 / 0,795 | 0,841 / 0,890 | **0,82** |

Axe detectate automat: F1 = 0,98 pe ambele tile-uri (24/25 și 25/26; ratate doar două cioturi de colț de 1,1 m și 5,1 m).
Inter-rânduri din axe detectate: IoU 0,955 pe ambele; aria totală în ±2% de referință.

Sensibilitate (cu axe de referință): lățimea coridorului 0,28/0,30/0,32 m → 0,851/0,855/0,843; aria minimă 0,15/0,19/0,25 m² → 0,842/0,855/0,847.
**Lab a\*** bate ExG și ExG-ExR: a\* separă verdele de solul roșu-maroniu și de umbră mai bine.

## 4. Rețeta per clasă (ce recomand)

### Pasul 0: mască de vegetație
`na = -(a* - 128)` din `cv2.cvtColor(RGB, COLOR_RGB2LAB)`, blur Gaussian σ = 2,5 px, prag `na > 4`. Fără închidere morfologică (închiderea unește plante și scade F1).

### row (axe): de făcut pe mozaic sau pe grupuri de tile-uri, apoi tăiat pe tile
1. Unghi dominant per bloc: unghiul care maximizează varianța histogramei proiecției pe perpendiculară (pas 0,5°, bin 0,1 m). FFT dă același lucru.
2. Vârfuri în profilul perpendicular (netezit σ = 3 bin, distanță minimă ~0,9 m, prominență 5%).
3. Per vârf: `cv2.fitLine` Huber pe pixelii de vegetație din banda ±0,35 m, iterat de 3 ori; se respinge dacă deviază > 3° de unghiul blocului.
4. Capete: percentila 0,5 / 99,5 a pixelilor de-a lungul liniei; dacă un capăt e la < 3 m de marginea tile-ului, se prelungește până la margine (referința merge margine-la-margine prin goluri).
5. 2 puncte per rând; rândurile curbe apar rar, dar la nevoie se adaugă un punct.
6. row_id: se numerotează pe tot blocul în ordinea perpendiculară, **pe mozaic**, nu per tile. Rândul din tile-ul vecin se leagă prin offset-ul perpendicular (toleranță ~0,5 m).

Pentru orientare pe mozaic: FFT/Hough recomandate de celălalt thread sunt echivalente cu pașii 1–2. Hough direct pe pixeli e mai zgomotos în iarbă.

### vineyard (canopy)
`mask ∩ coridor(axă, ±0,30 m)` → componente conexe 8 → păstrează ≥ 0,19 m² → `cv2.findContours` + `approxPolyDP` cu ε ≈ 2–3 px (țintă ~15 vârfuri / plantă) → tăiat la tile. Nu împărțiți forțat la 1–1,5 m: referința nu o face, iar split-ul ar strica F1 la pâlcurile lungi.
Blocuri fără vineyard: nu produceți canopy fără axă (coridorul elimină automat iarba, livezile și grădinile).

### interrow_area
Quad între axa_i deplasată +0,30 m și axa_{i+1} deplasată −0,30 m, doar între rânduri vecine din **același** bloc, tăiat la tile și la capătul rândului mai scurt; găuri pentru copaci/clădiri. Cover: fracția măștii a\* din poligon: < 0,25 `bare_soil`, 0,25–0,75 `mixed`, > 0,75 `vegetation`; `unassessable` dacă > ~50% umbră adâncă (V mic).

### row_structure
Golul maxim de-a lungul axei în coridorul ±0,30 m, **inclusiv capetele până la marginea tile-ului**; ≥ 5 m → `disrupted`. Cu canopy-uri prezise pragul de 5 m a dat o eroare falsă (6,7 m); de verificat manual rândurile între 4,5 și 7 m.

### vineyard_id
Blocuri = componente conexe ale uniunii coridoarelor dilatate cu 2,5 m (regula de 5 m), tăiate de `passages.geojson` (drumurile separă blocuri). Pe mozaic, nu per tile.

### waste
Referința nu dă exemple, iar un fals-pozitiv costă cât o ratare. Recomand:
1. SAM 3 cu prompt text + câteva box-uri exemplar (confirmat de lucrarea SAM 3: „concept prompts” cu exemplare pozitive și negative). Folosiți tuburile albe ca **exemplare negative**.
2. Filtre dure: eliminați tot ce e la < 0,6 m de o axă, obiecte alungite de tip tub (raport laturi > 2,5, arie < 0,12 m²) și arii < 0,05 m².
3. Pre-adnotați doar candidații cu încredere mare; restul se caută vizual în Marcaj. Probabil puține obiecte pe toată zona.

## 5. Riscuri și ce nu pot verifica din exemple
- Ambele exemple sunt vii tinere cu rânduri margine-la-margine. Nu am exemple de: capete de rând în interiorul tile-ului, vii bătrâne cu coroane unite, grădini din sat, livezi, tile-uri fără vie.
- Tile-urile fără vie sunt penalizate la canopy: rândurile false (livezi, culturi în rânduri) trebuie filtrate. Criterii utile: distanța între rânduri 2–3,5 m, lățimea vegetației < 1 m, pas plante 1–1,5 m (livezile au coroane 2–4 m la 4–6 m).
- Pragul a\* > 4 e potrivit pe acest zbor (20 mai 2025); de verificat vizual pe 5–10 tile-uri diverse înainte de exportul final.

## Scripturi
- `scripts/analyze.py`: statistici per clasă.
- `scripts/fit.py`, `fit2.py`: căutarea parametrilor măștii vs referință.
- `scripts/axes.py`: detector de axe + scor cap-coadă (row F1, canopy IoU/F1).
- `scripts/attrs.py`: inter-rânduri, cover și row_structure vs referință.
Rulează cu Python 3 + numpy, opencv, scipy, tifffile (deja instalate). Scripturile citesc doar din `data & info`.
