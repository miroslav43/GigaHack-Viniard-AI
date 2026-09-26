# Site-ul de prezentare — cercetare (26.09.2026)

Baza textelor din `/prezentare` (ce apare la `/` pentru vizitatorii fără cont). Fiecare afirmație de pe site vine de aici.
Ce nu e verificat e marcat **[de verificat]** și **nu** apare pe site.

## 1. Cumpărătorii și problema lor

| Cine | Ce are de făcut | Unde ajută Solemtrix |
|---|---|---|
| **Primării (APL I)** | cadastrul funciar anual (inginerul cadastral), terenuri abandonate (Codul funciar nr. 22/2024 + HG 688/2024: verificare la cerere, somație, arendare 1–5 ani), impozitul funciar, deșeuri (Legea 209/2016) | inventarul viilor din comună, dovezi pentru terenuri neîngrijite, deșeuri reperate, sarcini pentru echipă |
| **Consilii raionale (APL II)** | centralizarea cadastrului funciar, planificare | aceeași hartă pentru mai multe comune |
| **ONVV** — Registrul vitivinicol (Legea nr. 57/2006, HG 292/2017; declarații pe propria răspundere) | registrul are **45.500 ha**, BNS **66.400 ha** de soiuri de vin (aprilie 2024) — „o echipă de experți analizează de unde a apărut decalajul” | verificare independentă, pe parcelă |
| **AIPA / MAIA** — subvenții FNDAMR (2,3 mld lei în 2026), plantare vii 70–80 mii lei/ha (vin), până la 300 mii lei/ha (soiuri autohtone) | Curtea de Conturi (ian. 2026): dosare incomplete, lipsa trasabilității, risc de dublă finanțare, „lipsa SIAC limitează monitorizarea”; CE 2025: LPIS pilotat doar în 5 raioane; UNDP: LPIS „nefuncțional”, controale „limitate și fragmentate” | control țintit pe teren, dovezi pe plantă și rând, strat de detaliu (2,5 cm) complementar sateliților (10 m) |

Sectorul: ~100.000 ha de vii (ONVV, 24.09.2026), 7 ha defrișate la 1 ha plantat în ultimii 8 ani; 40% din suprafață la gospodării
(2021); 25,5% abandonată la recensământul din 2011; 4 regiuni IGP (Codru, Ștefan Vodă, Valul lui Traian, Divin); export 2025: 117 mil. l, 212 mil. $.

Surse:
- ONVV, suprafața și defrișările: https://agrobiznes.md/viile-moldovei-se-restrang-pentru-fiecare-hectar-plantat-alte-sapte-sunt-defrisate.html
- Registrul vs BNS: https://agroexpert.md/rom/preturi-si-tendinte/suprafata-vitei-de-vie-din-rm-descreste-la-1-hectar-plantat-5-sunt-defrisate
- Registrul vitivinicol la ONVV: https://www.maia.gov.md/ro/content/1611
- FNDAMR 2026–2030: https://gov.md/ro/comunicate-de-presa/subventionarea-agriculturii-urmatorii-5-ani-mai-multa-competitivitate
- Subvenții plantare: https://agrobiznes.md/aipa-ultimele-zile-pentru-solicitarea-subventiilor-pe-etape-si-postinvestitionale.html · https://telegraph.md/statul-va-oferi-pana-la-300-000-de-lei-per-hectar-pentru-soiuri-de-vita-de-vie-autohtone/
- Curtea de Conturi (ian. 2026): https://bani.md/cutremur-financiar-in-agricultura-curtea-de-conturi-statul-a-promis-subventii-de-3-miliarde-fara-bani-in-buget/
- Raportul CE 2025 (cap. 11): https://enlargement.ec.europa.eu/document/download/23fa6af0-89b3-4532-a3d9-d1638727d14c_en?filename=moldova-report-2025.pdf
- UNDP/UE, LPIS: https://www.undp.org/sites/g/files/zskgke326/files/2025-11/prodoc-promoting_agricultural_development_and_quality_employment_11.25.pdf
- Terenuri abandonate, Codul funciar: https://maia.gov.md/ro/content/6008
- Gunoiști neautorizate 2025 (534 > 200 m²): https://ziar.md/video-gunoistile-care-sufoca-localitatile-ar-urma-sa-dispara-dar-cu-o-conditie/
- Recensământul agricol 2011, capitolul viticol: https://statistica.gov.md/files/files/publicatii_electronice/Recensamint_agricol/Studiu1_viniviticol_ro.pdf

## 2. Cum cumpără

- Legea nr. 131/2015 (până la 31.12.2026), apoi Legea nr. 325/2025 (de la 01.01.2027). Achiziții de valoare mică: bunuri/servicii ≤ 300.000 lei fără TVA; regim simplificat ≤ 50.000 lei; MTender. Regulamentul nou (12.06.2026) introduce regimuri simplificate pentru licențe software și servicii pe abonament. Surse: https://tender.gov.md/ro/content/%C3%AEntreb%C4%83ri-adresate-frecvent · https://mf.gov.md/en/node/134707
- Bugetul local: primarul propune, consiliul local aprobă până la **10 decembrie** (Legea 397/2003) → discuțiile se poartă în septembrie–noiembrie.
- Pilot fără cost printr-un acord de colaborare (fără achiziție), apoi abonament anual per UAT; la nivel central: MTender, eventual co-finanțare UE/UNDP/Banca Mondială. USAID nu mai e o sursă (programe oprite în 2025).
- Pagini care conving instituțiile (Planet pentru agenții de plăți, Agremo, OneSoil, EOSDA, Esri, Cybernetica): problemă concretă în titlu, cifre de la instituții numite, „cum funcționează” în 3–5 pași, securitatea ca secțiune separată, „pilot / discuție cu experții” în loc de cumpărare directă.

## 3. Ce putem afirma (adevărat azi)

- Date găzduite în UE: proiectul Supabase e în **eu-west-1 (Irlanda)**; criptare în tranzit și în repaus asigurată de furnizor (https://supabase.com/security). Certificările (SOC 2, ISO 27001) sunt ale furnizorului, nu ale Solemtrix.
- Roluri (administrator UAT, inspector, vizualizator), izolarea datelor pe primărie (RLS în baza de date), jurnal al acțiunilor administrative.
- Interfață în limba română (limba de stat), plus engleză și rusă; export deschis CSV, GPX, GeoJSON.
- Cifrele pilotului Sireți (zbor 20.05.2025, 311 tile-uri, 81,5 ha), măsurate plan în EPSG:32635.
- Datele personale: Legea nr. 195/2024 (în vigoare din 23.08.2026, aliniată GDPR, a abrogat Legea 133/2011).

**Nu afirmăm** (încă): integrare MPass/MSign, găzduire MCloud, conformitate WCAG 2.1 AA, certificări, precizie în procente, recomandări sau
parteneriate cu AGE/MAIA/AIPA/ONVV, valoare juridică a măsurătorilor (nu înlocuiesc cadastrul).

**[de verificat de echipă]** înainte de a le adăuga pe site: acordul de prelucrare a datelor (model de contract, Legea 195/2024, art. 44–50
transfer în UE), persoana juridică și facturarea în lei, un telefon și un email de contact, operatorul de drone partener (HG nr. 949/2022,
aprobări, zona de 10 km de la frontieră), prețul.

## 4. Obiecții și răspunsul de pe site

| Obiecție | Răspuns |
|---|---|
| „Primăria nu are mandat de control al viilor” | inventar pentru cadastrul funciar, terenuri abandonate, deșeuri; datele pot fi puse la dispoziția ONVV/AIPA |
| „Nu avem buget pentru IT” | pilot gratuit; abonament sub pragul achizițiilor de valoare mică; planificat în bugetul pe anul următor |
| „Are valoare juridică?” | nu înlocuiește cadastrul; sunt date de lucru care arată unde merită verificat pe teren |
| „De ce nu satelit?” | complementar: satelitul (10 m) vede parcela; drona (2,5 cm) vede fiecare plantă, golurile din rând și deșeurile |
| „Zborurile cer aprobări” | operator autorizat conform HG nr. 949/2022; aprobările se obțin înaintea zborului |
| „Unde sunt datele și ale cui sunt?” | în UE; datele sunt ale instituției, export oricând |
