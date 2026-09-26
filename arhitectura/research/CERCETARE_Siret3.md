# Sireț3: cercetare pentru actualizarea arhitecturii

Vineri 25.09.2026, seara. Rezultatele a 7 fire de cercetare rulate în paralel, pe secțiunile planului `../ARHITECTURA_Siret3.md`. Fiecare constatare are: secțiunea din plan, încrederea (high / medium), ce s-a găsit, recomandarea, impactul și sursele.

Planul a preluat doar constatările high și medium. Riscurile și incertitudinile rămân aici. Rezumatul schimbărilor e în §11 din plan.

**Verificări locale făcute la integrare:**

- Dimensiunile ZIP-urilor originale (`ls -la 01_tiles/`): part1 93 823 066 B, part2 93 624 576 B, part3 93 207 449 B, part4 94 108 721 B, part5 10 414 422 B. Confirmă constatarea de la §6.1.
- Regula de vizitare a țintelor e confirmată în descrierea challenge-ului: o țintă e vizitată dacă traseul trece la ≤ 2 m de ea. Riscul „regula de 2 m neverificată” de la §4 e deci închis.
- Ghidul Marcaj spune la §2 că „fiecare dintre cele cinci părți rămâne sub limita de 90 MB”, dar se referă la părțile fără `annotations.xml`.

Cuprins:

1. Detecția deșeurilor (waste), plan §4.9
2. Rânduri de vie, vie vs. livadă, blocuri și goluri, plan §4.2–4.4 și §4.8
3. Segmentarea canopy-ului și rețeaua neuronală, plan §4.5–4.6
4. Traseul pedestru, plan §4.10–4.11
5. Interfața web, plan §4.14
6. CVAT for images 1.1 și ZIP-urile Marcaj, plan §4.13
7. Stack-ul Python pe M3 Pro și Docker, plan §4.1 și §4.15

---

## 1. Detecția deșeurilor (waste) în ortofoto UAV la ~2,5 cm GSD (plan §4.9)

### 1.1 SAM 3: prompturi și exemplare (high, §4.9 pasul 3)

- **Constatare:** SAM 3 (`facebook/sam3`) acceptă prompturi text, box-uri exemplar sau ambele, cu etichete pozitive și negative. În HF transformers se apelează `Sam3Processor(images, text=..., input_boxes=[[xyxy,...]], input_boxes_labels=[[1,0,...]])`. Valorile etichetelor: 1 = pozitiv, 0 = negativ (exclude), -10 = padding. Post-procesarea: `processor.post_process_instance_segmentation(threshold=0.5, mask_threshold=0.5)`.
- **Important:** exemplarele sunt box-uri din ACEEAȘI imagine. API-ul nu primește exemplare luate din altă imagine.
- **Recomandare (SCHIMBĂ):**
  - SAM 3 primește prompturi text, fiecare rulat separat: `trash`, `plastic bag`, `plastic bottle`, `tire`, `rubbish heap`.
  - În același crop se adaugă 1–3 box-uri negative (label 0) pe tuburi sau țăruși albi, detectați automat pe axa rândului.
  - Nu se lipesc exemplare din alte imagini.
  - Pragul e 0,4 la generarea candidaților (pentru sweep manual) și 0,6 la pre-adnotarea automată.
  - *Notă la integrare:* regula combinată din §1.9 folosește SAM 3 ≥ 0,5, dar numai împreună cu linear probe ≥ 0,9. Planul a preluat regula combinată (`waste.auto_sam3_min: 0.5`).
- **Impact:** planul presupunea „box-uri exemplar pozitive/negative”. În Sireț3 nu avem exemplare pozitive de waste: exemplele au 0 gunoaie, iar un exemplar nu se poate aduce dintr-un alt tile. Varianta realistă e text pozitiv plus box-uri NEGATIVE pe tuburile albe din același crop.
- **Surse:** https://huggingface.co/docs/transformers/main/en/model_doc/sam3, https://github.com/facebookresearch/sam3

### 1.2 SAM 3: rezoluția internă (high, §4.9 pașii 1 și 3)

- **Constatare:** SAM 3 redimensionează intern imaginea la 1008×1008. Un tile de 2048 px ar fi micșorat de ~2×, adică la ~5 cm/px, iar o sticlă de 25 cm ar ajunge la ~5 px.
- **Recomandare (ADAUGĂ):**
  - Varianta completă: SAM 3 rulează pe crop-uri de 1024 px cu overlap de 64 px (4–9 crop-uri pe tile, ~1 300–2 800 în total), la rezoluție nativă sau ușor mărită.
  - Varianta rapidă: SAM 3 doar ca verificator, pe crop-uri de 512 px (12,8 m) centrate pe candidații de culoare din pasul 1.
- **Impact:** rulat pe tile întreg, SAM 3 ratează obiectele mici, deci trebuie rulat pe crop-uri.
- **Surse:** https://github.com/benreichman/sam3-mac

### 1.3 SAM 3 pe Mac (medium, §4.9 pasul 3 și riscul „Acces SAM 3”)

- **Constatare:**
  - Repo-ul oficial cere CUDA 12.6+, Python 3.12+ și PyTorch 2.7+. Nu există fallback oficial pentru MPS/CPU (issue #164, iar PR-ul #627 e încă o contribuție a comunității).
  - `benreichman/sam3-mac` (`pip install sam3-mac`, peste pachetul oficial) rulează pe Mac, dar DOAR cu prompturi text. Pe M4 Pro, la cald: ~3 s/imagine pe CPU și ~7–8 s pe MPS, deci CPU e default-ul recomandat. Primul apel durează 10–15 s.
  - Există și un port MLX (`mlx-community/sam3-image`, Python 3.13+, ~3,5 GB) care suportă box-uri include/exclude. Card-ul lui declară licența apache-2.0, deși greutățile originale sunt sub SAM License.
- **Recomandare (SCHIMBĂ):**
  - Se folosește transformers `Sam3Model` pe `device='cpu'`, iar portul MLX doar dacă transformers dă erori.
  - Rularea se face doar pe crop-urile-candidat, nu pe toate tile-urile, cu un buget de ≤ 30 min.
  - Nu ne bazăm pe MPS.
- **Impact:**
  - Pe M3 Pro (18 GB) SAM 3 merge, dar lent: ~2 800 crop-uri × ~3–4 s înseamnă ~3 h pe CPU. Varianta cu verificator doar pe candidați ia câteva minute.
  - Pentru box-uri negative e nevoie de HF transformers pe CPU sau de portul MLX, nu de sam3-mac.
- **Surse:** https://github.com/benreichman/sam3-mac, https://github.com/facebookresearch/sam3/issues/164, https://github.com/facebookresearch/sam3/pull/627, https://huggingface.co/mlx-community/sam3-image

### 1.4 Accesul la `facebook/sam3` (medium, §4.9 pasul 3 și riscuri)

- **Constatare:**
  - Modelul e gated. Pașii pentru acces: login HF, accepți condițiile și îți partajezi datele de contact pe pagina modelului, apoi rulezi `hf auth login` cu token.
  - Are 0,9B parametri (F32, ~3,5 GB).
  - Pagina sugerează aprobare automată, dar pentru repo-urile Meta aprobarea a fost uneori manuală sau întârziată.
  - Pe 27.03.2026 a apărut SAM 3.1: checkpoint-uri actualizate, cu îmbunătățiri mai ales pentru tracking video.
- **Recomandare (PĂSTREAZĂ fallback-ul și CERE ACCESUL ACUM):**
  - Accesul se cere de pe 2 conturi HF diferite.
  - SAM 3.1 nu e necesar pentru imagini statice.
  - Dacă accesul nu vine în 12 h, trecem pe CLIP + sweep manual.
- **Impact:** riscul „Acces SAM 3 întârziat” rămâne real.
- **Surse:** https://huggingface.co/facebook/sam3, https://github.com/facebookresearch/sam3

### 1.5 DroneWaste (high, §4.9 pasul 3, opțiunea YOLO)

- **Constatare:**
  - DroneWaste (Zenodo 17045559) e deschis, sub CC BY 4.0: `images.tar.gz` de 3,9 GB și `dronewaste_v1.0.json` de 8,8 MB, în format COCO (măști și bbox).
  - Conține 4 993 de imagini din ortomozaicurile a 17 depozite ilegale de deșeuri, 5 135 de adnotări și 20 de materiale mapate pe coduri EWC.
  - Pagina nu publică GSD-ul și altitudinea.
  - Chiar in-domain, detectorii ating doar mAP@50 ≈ 38%: YOLOv8x 38,2%, YOLOv12x 38,5%, Faster R-CNN 36,5%.
  - Repo-ul de cod (`lucamora/dronewaste`) e sub MIT și nu publică greutăți.
- **Recomandare (SCHIMBĂ):**
  - Nu antrenăm YOLO pe DroneWaste în hackathon.
  - Setul se folosește cel mult ca sursă de crop-uri pozitive pentru un linear probe: bbox-uri redimensionate la ~2,5 cm/px, dacă GSD-ul estimat din dimensiunea obiectelor e comparabil.
  - Descărcăm doar dacă rămâne timp (3,9 GB).
  - În README: atribuire CC BY 4.0.
- **Impact:** domeniul diferă mult. DroneWaste are grămezi în depozite, clasificate pe material, pe când noi căutăm obiecte izolate în vie. Un YOLO fine-tunat pe DroneWaste ar da multe FP pe sol sau pietre, iar la F1 fiecare FP contează cât un ratat.
- **Surse:** https://zenodo.org/records/17045559, https://github.com/lucamora/dronewaste

### 1.6 UAVVaste, AerialWaste, TACO (high, §4.9 pasul 3 și secțiunea NN)

- **Constatare:**
  - UAVVaste are licența Apache-2.0: 772 de imagini UAV la altitudine joasă și 3 718 adnotări COCO-like (bbox și segmentare), cu o singură clasă, `rubbish`, în mediu urban și natural (străzi, parcuri, gazon). Se descarcă prin `python3 main.py` din repo sau de pe Zenodo 8214061.
  - AerialWaste conține depozite și landfill-uri la nivel de scenă (aerian și satelit), nu obiecte individuale, deci e irelevant.
  - TACO e fotografiat de la sol, deci e nepotrivit.
- **Recomandare (ADAUGĂ):**
  - UAVVaste devine sursa principală de pozitive pentru clasificatorul de crop-uri (câteva sute de crop-uri de 64–128 px).
  - Ca negative se extrag sute de tuburi albe, țăruși, pietre și bucăți de sol deschis din cele 2 tile-uri exemplu Sireț3.
- **Impact:** UAVVaste e mai apropiat de țintă (bucăți individuale de gunoi pe iarbă sau sol) decât DroneWaste.
- **Surse:** https://github.com/PUTvision/UAVVaste, https://zenodo.org/record/8214061, https://www.nature.com/articles/s41597-023-01976-9

### 1.7 CLIP / RemoteCLIP pe obiecte mici (medium, §4.9 pașii 3 și 4)

- **Constatare:**
  - CLIP și RemoteCLIP sunt evaluate pe clasificarea de SCENE întregi (RemoteCLIP: RESISC45 ~80%, AID ~91%), nu pe obiecte de 10–40 px.
  - Nu am găsit dovezi de acuratețe zero-shot pe crop-uri mici de gunoi aerian.
  - Pe obiecte mici cu fundal dominant, scorurile zero-shot sunt slab calibrate, pentru că crop-ul e dominat de sol sau vegetație.
- **Recomandare (SCHIMBĂ):**
  - (a) Crop strâns (box plus 50% context, minim 64 px), mărit la 224 px.
  - (b) Zero-shot nu se folosește ca decizie finală. Antrenăm un linear probe (logistic regression) pe embedding-uri OpenCLIP ViT-B/32 sau DINOv2-S. Pozitivele vin din UAVVaste (opțional și DroneWaste), iar negativele din tile-urile exemplu. Pragul se alege pe exemple astfel încât FP = 0.
  - (c) Zero-shot CLIP rămâne doar pentru ordonarea din `waste_candidates.csv`. Semnalul e marja softmax (pozitiv max − negativ max) > 0,2, nu un prag absolut.
- **Impact:** un prag absolut CLIP ≥ 0,85 folosit ca decizie automată e nefundamentat și riscant pentru FP.
- **Surse:** https://arxiv.org/html/2306.11029v4, https://link.springer.com/article/10.1007/s11042-026-21440-1

### 1.8 Ce spun regulile despre NOT WASTE (high, §4.9 pasul 2)

- **Constatare:**
  - Regulile listează explicit ca NOT WASTE: tuburi, țăruși, stâlpi și sârme de spalier, furtunuri de irigație, pietre, sol deschis, arbuști înfloriți, resturi de tăiere și vehicule.
  - Principiul e „When in doubt, leave it out”, iar scorarea e F1 la IoU ≥ 0,3, deci FP-urile și duplicatele penalizează.
  - Tuburile sunt distanțate regulat pe axa rândului, cu pasul de plantare. Din nadir au formă mică, rotundă sau alungită.
- **Recomandare (PĂSTREAZĂ filtrele și ADAUGĂ):**
  1. Eliminăm orice candidat alb aflat la ≤ 0,6 m de axă ȘI la ±25% de o poziție de plantă prezisă (proiecția pe axă, modulo pasul de plantare).
  2. Eliminăm candidații albi și nesaturați mai mici de 0,03 m². Un tub din nadir are ~0,01–0,02 m², adică 16–32 px.
  3. Aplicăm NMS între candidați la IoU > 0,3, cu o singură cutie pe cluster, ca să evităm duplicatele.
  4. Mască de furtun: eliminăm candidații liniari lungi (> 3 m, lățime < 5 cm).
  5. Box-ul poate fi puțin lax, pentru că IoU ≥ 0,3 e tolerant: padding de 10%.
- **Impact:** filtrele din plan (< 0,6 m de axă; aspect > 2,5 și arie < 0,12 m²) sunt corecte. Periodicitatea le completează și e un semnal mai sigur.
- **Surse (locale):** regulile de adnotare și descrierea challenge-ului, extrase text în scratchpad:
  - `/private/tmp/claude-501/-Users-maleticimiroslav-Vin-Gigahack-data---info/62a47fe7-0155-4b5f-a470-d6f1a44f9238/scratchpad/Vineyard_AI_annotation_rules.txt`
  - `/private/tmp/claude-501/-Users-maleticimiroslav-Vin-Gigahack-data---info/62a47fe7-0155-4b5f-a470-d6f1a44f9238/scratchpad/Vineyard_AI_Field_Challenge_description.txt`
  - Originalele sunt în `../data & info/03_docs/`.

### 1.9 Pipeline-ul recomandat (medium, §4.9 pașii 3–6)

- **Constatare:** se așteaptă puține obiecte reale, iar scorarea e F1 la IoU 0,3. Pipeline-ul recomandat:
  1. Candidați HSV/luminozitate + filtre geometrice și de periodicitate, cu recall mare.
  2. Verificator: linear probe CLIP/DINOv2 cu `p ≥` un prag calibrat pentru 0 FP pe exemple. Dacă avem acces, se adaugă SAM 3 cu text + box-uri negative pe crop de 512 px (score ≥ 0,5, cu masca suprapusă pe candidat).
  3. TOȚI candidații cu scor ≥ 0,3 intră în `waste_candidates.csv` pentru sweep uman.
  4. În export ajung doar box-urile confirmate de om, plus cele auto cu `p ≥ 0,9` și acord între ambele modele.
- **Recomandare (SCHIMBĂ pasul 4, „doar scor ≥ 0,85”):**
  - Auto-accept doar dacă linear probe ≥ 0,9 ȘI (SAM 3 ≥ 0,5 sau CLIP cu marjă > 0,2).
  - Restul trece prin sweep uman, cu un timp estimat de ~1 min pe 20 de candidați.
- **Impact:** waste valorează 10%, iar un FP anulează un TP. Confirmarea umană a unei liste scurte, ordonate după scor, e cea mai ieftină cale spre precizie mare.
- **Surse:** https://huggingface.co/docs/transformers/main/en/model_doc/sam3, https://github.com/PUTvision/UAVVaste

### 1.10 Schimbări propuse (waste)

- **§4.9 pasul 3:** „SAM 3 cu box-uri exemplar pozitive/negative” devine „SAM 3 cu text prompt (trash / plastic bag / plastic bottle / tire / rubbish heap) + box-uri NEGATIVE (label 0) pe tuburi albe din același crop”. Exemplarele pot fi doar in-image.
- **§4.9 pasul 3:** SAM 3 rulează pe CPU (transformers `Sam3Model`, `device='cpu'`) sau prin portul MLX. Rulează doar pe crop-uri de 512 px centrate pe candidați, nu pe tile întreg: acesta ar fi micșorat la 1008 px, adică la ~5 cm/px. Buget: ≤ 30 min.
- **§4.9 pasul 3:** CLIP zero-shot nu mai decide. Decizia o ia un linear probe (logistic regression) pe embedding-uri OpenCLIP ViT-B/32 sau DINOv2-S:
  - pozitive: crop-uri UAVVaste (Apache-2.0), opțional DroneWaste (CC BY 4.0);
  - negative: tuburi, țăruși, pietre și sol din cele 2 exemple;
  - pragul se calibrează pentru 0 FP pe exemple.
- **§4.9 pasul 3:** eliminăm opțiunea „YOLO fin-tunat pe DroneWaste”. Motivele: domeniul e de depozite, mAP50 e ~38% chiar in-domain, nu există greutăți publicate și setul are 3,9 GB. Licența e verificată: CC BY 4.0.
- **§4.9 pasul 2:** adăugăm:
  - filtrul de periodicitate: un candidat alb la ≤ 0,6 m de axă și la ±25% de o poziție de plantă se elimină;
  - filtrul de furtun: liniar > 3 m, lățime < 5 cm;
  - NMS la IoU > 0,3 și padding de 10% al box-ului.
- **§4.9 pasul 4:** „doar scor ≥ 0,85” devine auto-accept doar dacă linear probe ≥ 0,9 ȘI (SAM 3 ≥ 0,5 sau marjă CLIP > 0,2). Toți candidații cu scor ≥ 0,3 intră în `waste_candidates.csv` pentru confirmare umană.
- **Riscuri:** cerem ACUM acces la `facebook/sam3` (HF login → accept terms → `hf auth login`), de pe 2 conturi. SAM 3.1 nu e necesar pentru imagini statice.
- **README:** atribuire CC BY 4.0 pentru DroneWaste și Apache-2.0 pentru UAVVaste, dacă le folosim.

### 1.11 Riscuri (waste)

- GSD-ul și altitudinea pentru DroneWaste și UAVVaste nu apar în paginile consultate. Rescalarea crop-urilor la 2,5 cm/px trebuie estimată empiric, de exemplu după dimensiunea sticlelor.
- Nu e confirmat oficial că transformers `Sam3Model` merge pe CPU/MPS. Dacă rulează doar cu CUDA, trecem pe portul MLX, a cărui licență declarată (apache-2.0) e ambiguă față de SAM License.
- Aprobarea accesului HF pentru `facebook/sam3` poate fi manuală sau întârziată.
- Filtrul „< 0,6 m de axă” elimină și gunoiul prins în vie, pe rând. E acceptat pentru precizie, dar costă recall.
- În exemplele Sireț3 nu există pozitive, deci pragul poate fi calibrat doar pentru FP = 0, nu și pentru recall.

---

## 2. Rânduri de vie, vie vs. livadă, blocuri și goluri (plan §4.2–4.4, §4.8)

### 2.1 Orientarea și distanța dintre rânduri (high, §4.3 pașii 2–3; §4.2 punctul 4)

- **Constatare:** analiza de frecvență locală (FFT/Gabor) e metoda de referință pentru orientarea și distanța dintre rânduri în imagini VHR. Delenne et al. raportează o eroare medie de 2° pe orientare și de 6 cm pe lățimea inter-rândului, cu 88% din 114 parcele clasificate corect. Metoda din plan (unghiul care maximizează varianța histogramei offset-ului) e echivalentă cu o transformată Radon și e corectă.
- **Recomandare:**
  - KEEP: proiecția/Radon pentru unghi.
  - ADD: distanța `s` se estimează per tile din autocorelația profilului perpendicular (primul vârf în intervalul 1,8–3,8 m). Distanța minimă între vârfuri devine `0,6·s`, în locul valorii fixe de 0,9 m.
  - ADD: scorul de periodicitate = energia vârfului FFT la `1/s`, împărțită la mediana spectrului radial. Sub ~3, tile-ul se consideră „fără vie”. Scorul înlocuiește regula cu prominența de 5% din §4.2.4.
- **Impact:** confirmă pașii 2–3 din §4.3. Distanța minimă fixă de 0,9 m și prominența de 5% sunt însă arbitrare: pe inter-rândurile înierbate apar vârfuri false la jumătatea distanței.
- **Surse:** https://hal.inrae.fr/hal-02587901, https://hal.inrae.fr/hal-02587805, https://link.springer.com/chapter/10.1007/978-3-540-77058-9_24

### 2.2 Refit robust și verificarea on-lattice (medium, §4.3 pasul 4)

- **Constatare:** pipeline-ul clasic pentru UAV (Comba et al. 2015) are trei pași: segmentare dinamică (prag adaptiv local), clustering în spațiul Hough și ajustare Total Least Squares pe fiecare cluster. Hough pe masca de 2,5 cm produce multe linii false. Clustering-ul în spațiul parametrilor, urmat de refit robust, e cel care îl face stabil.
- **Recomandare:**
  - KEEP: `fitLine` Huber.
  - ADD criteriu de rezidual: dacă p95 al distanței pixelilor din bandă față de linie depășește 0,15 m, piesa se împarte în 2 subsegmente sau se fit-uiește pătratic (vezi §2.5).
  - ADD verificarea „on-lattice”: vârfurile al căror offset se abate cu mai mult de `0,25·s` de grila `o0 + i·s` se marchează suspecte (smocuri de iarbă, urme de roți). Ele se păstrează doar dacă trec testul de lățime și ocupare.
- **Impact:** pasul 4 din §4.3 (`fitLine` Huber pe banda ±0,35 m, iterat) face deja același lucru ca Hough-clustering + TLS. Nu merită trecut pe Hough clasic.
- **Surse:** https://dl.acm.org/doi/abs/10.1016/j.compag.2015.03.011, https://www.researchgate.net/publication/274898413_Vineyard_detection_from_unmanned_aerial_systems_images

### 2.3 Inter-rânduri înierbate și rânduri neregulate (medium, §4.2 și §4.3 pașii 1–3)

- **Constatare:** literatura recentă despre rânduri curbe și neregulate („Versatile method”, 2024) observă că metodele anterioare au fost testate aproape numai pe vii comerciale regulate, cu puține plante lipsă. Pentru rânduri curbe, terase și goluri, autorii combină indicii vizibili cu elevația (DSM), care separă și iarba dintre rânduri de vie. Noi avem doar RGB, fără DSM.
- **Recomandare (ADD):**
  - Când contrastul a\* e slab, se folosește un profil de rezervă pe luminanță sau textură. Canopy-ul viței e mai închis la culoare și mai texturat decât iarba.
  - Semnalul de rezervă e deviația standard locală a lui V (fereastră de 0,25 m) sau profilul lui V inversat, iar FFT-ul rulează pe acest canal.
  - Pe exemplele cu `interrow_cover = vegetation` verificăm dacă profilul a\* își pierde vârfurile. Dacă da, perechea (a\*, textură) devine intrarea standard.
  - Masca NN (§4.6), dacă e antrenată pe canopy vs. iarbă, e cea mai robustă intrare pentru axe.
- **Impact:** principalul mod de eșec al măștii a\* (§4.2/§4.3) sunt inter-rândurile complet înierbate, unde profilul perpendicular devine aproape plat. Viile de grădină din Sireț3 sunt probabil neregulate.
- **Surse:** https://www.sciencedirect.com/science/article/abs/pii/S0168169924007634, https://www.researchgate.net/publication/383908565_Versatile_method_for_grapevine_row_detection_in_challenging_vineyard_terrains_using_aerial_imagery

### 2.4 Vie vs. livadă (high, §4.2 punctele 2–3)

- **Constatare:**
  - Regulile competiției dau criteriile exacte. Livada are coroane rotunde de 2–4 m, cu rânduri la 4–6 m. Via are sub 1 m lățime și plante la 1–1,5 m pe rând.
  - Literatura (Warner/Aksoy, OBIA pe textură) confirmă că spectrul singur nu separă livada de vie, pe când structura spațială le separă.
  - Capcană: o livadă cu rânduri la 5 m are armonica a 2-a la 2,5 m, adică în intervalul de căutare 1,8–3,8 m.
- **Recomandare:**
  - KEEP: lățimea p80 < 1,2 m.
  - ADD trei trăsături ieftine, calculate pe rând sau pe bloc:
    - (a) Raportul lățime canopy / distanță între rânduri: via are ~0,15–0,45, livada ~0,5–0,8.
    - (b) Periodicitatea de-a lungul rândului, din FFT pe profilul ocupării în lungul axei. Un vârf clar la 3–6 m cu duty cycle < 60% înseamnă pomi discreți, deci respingere. Via matură dă o ocupare aproape continuă, iar via tânără dă pete la 1–1,5 m.
    - (c) Circularitatea componentelor conexe din coridor: o componentă cu arie > 2 m² și raport al axelor < 1,5 se respinge ca pom.
  - ADD verificarea armonicii: dacă spectrul are un vârf la `2s` mai puternic decât cel la `s`, s-a prins armonica. În acest caz distanța reală e `2s`, probabil o livadă.
  - Trăsătura (c) filtrează și pomii izolați din vie (regula NOT A CANOPY).
- **Impact:** filtrul din §4.2 (lățime p80 < 1,2 m, ≥ 2 vecini la 1,8–3,8 m) poate lăsa să treacă o livadă tânără cu coroane mici, prin armonică, sau un rând de pomi tineri.
- **Surse:** https://www.tandfonline.com/doi/full/10.1080/13658816.2010.510839, https://www.researchgate.net/publication/298332673_Vineyard_Detection_and_Vine_Variety_Discrimination_from_Very_High_Resolution_Satellite_Data, https://hal.inrae.fr/hal-02587901

### 2.5 Rânduri curbe (medium, §4.3 pașii 4–5 și legarea globală, pasul 2)

- **Constatare:**
  - Un tile are 51,2 m, iar săgeata unei linii drepte pe o coardă `L` este `L²/(8R)`.
  - Pentru toleranța de 0,15 m (pragul de refit din plan), o linie dreaptă pe tot tile-ul cere R ≥ ~2 200 m.
  - Pentru metrica de scor (80% din lungime la ≤ 0,4 m), o linie dreaptă per tile trece până la R ≈ 800 m.
  - Metodele pentru rânduri curbe (Nolan 2015, cu skeletonizare; Versatile 2024) urmăresc rândul local, nu îl fit-uiesc global.
- **Recomandare (ADD):** un fallback de tracking per tile, aplicat doar benzilor cu rezidualul p95 > 0,15 m:
  - ferestre de 256 px (6,4 m) de-a lungul rândului;
  - în fiecare fereastră se calculează centroidul pixelilor de vegetație din banda ±0,35 m;
  - fiecare fereastră pornește de la poziția celei precedente (tracking);
  - rezultatul e o polilinie cu vertex la 5–10 m, simplificată Douglas-Peucker cu 0,1 m.

  Ieșirea în CVAT rămâne polyline, deci e permisă. Aceeași funcție de tracking servește și la refit-ul global din pasul 2 al legării.
- **Impact:** linia dreaptă per tile e sigură pentru majoritatea rândurilor. La rânduri curbe (R < 800 m), metrica de potrivire la 0,4 m pică pe tot rândul.
- **Surse:** https://research.monash.edu/en/publications/automated-detection-and-segmentation-of-vine-rows-using-high-reso/, https://www.sciencedirect.com/science/article/abs/pii/S0168169924007634

### 2.6 Legarea între tile-uri și `row_id` (high, §4.3 legarea globală pașii 1–2; §4.4 pasul 4)

- **Constatare:**
  - Legarea prin capete la ≤ 0,4 m e fragilă. Dacă un rând are un gol lângă marginea tile-ului, capătul se oprește la > 3 m de margine, nu se mai prelungește și lanțul se rupe.
  - Rezultatul e un `row_id` dublu, care scade scorul pe numărul de rânduri (2%) și pe consistență.
  - Toate tile-urile provin din același ortofoto, deci la margini nu există decalaj de mozaicare. Se poate lega geometric pe linii, nu pe capete.
- **Recomandare:**
  - CHANGE criteriul de legare. Două piese din tile-uri adiacente se leagă dacă `|Δunghi| ≤ 3°` și offset-ul perpendicular al piesei B față de dreapta-suport prelungită a piesei A e ≤ 0,3 m. Distanța dintre capete nu mai e condiție; se permite un gol coliniar de ≤ 1 tile.
  - ADD indexarea pe grilă per bloc. După formarea blocului se estimează `o0` și `s` (mediana), iar `row_index = round((offset − o0)/s)`. Rândurile din tile-uri diferite cu același index și `|Δoffset| < 0,3·s` primesc același `row_id`. Asta rezolvă și golurile de la margine.
  - La blocurile în evantai (46–53°), grila se estimează local, pe o fereastră de 2×2 tile-uri.
- **Impact:** afectează §4.3 (legarea globală) și §4.4 (`row_id` stabil). Regula oficială cere ca rândul să fie desenat de la prima la ultima viță, prin goluri, iar ID-urile trebuie să rămână aceleași peste margini.
- **Surse:** https://dl.acm.org/doi/abs/10.1016/j.compag.2015.03.011, https://hal.inrae.fr/hal-02587901

### 2.7 Separarea blocurilor (medium, §4.4 pașii 1–3)

- **Constatare:**
  - Graful din plan (paralel ≤ 5°, la 1,8–4,0 m, suprapunere ≥ 20%, tăiat de pasaje) poate uni două parcele vecine cu aceeași orientare, separate doar de o zonă de întoarcere de 4–6 m care nu apare ca drum în `passages.geojson`.
  - Parcelele vecine diferă adesea prin distanța dintre rânduri sau printr-un decalaj de fază al grilei. Delenne folosește tocmai perechea (orientare, inter-rând) pentru segmentarea parcelelor.
- **Recomandare (ADD) două condiții pe muchii:**
  1. Distanța dintre cele două rânduri trebuie să difere cu ≤ 0,2 m (≤ 8%) de mediana locală `s` a fiecăruia. Muchia se taie la un salt de spacing sau de fază (offset nemultiplu de `s`, cu abatere > `0,3·s`).
  2. Două rânduri coliniare cu gol ≤ 5 m se leagă doar dacă golul nu e o bandă transversală fără vegetație continuă, prezentă simultan pe ≥ 3 rânduri vecine. O astfel de bandă e o zonă de întoarcere sau un drum: muchia se taie și se creează un bloc nou.

  Se păstrează eliminarea blocurilor cu < 3 rânduri (regula viei de grădină).
- **Impact:** scorul pe numărul de blocuri (2%, toleranță 15%) și consistența `vineyard_id` (2%) depind direct de asta.
- **Surse:** https://hal.inrae.fr/hal-02587901, https://hal.inrae.fr/hal-02596316

### 2.8 Goluri și plante lipsă (medium, §4.8 și legătura cu §4.10)

- **Constatare:**
  - Abordarea standard pe ortofoto (Primicerio et al. 2017) e profilul de ocupare de-a lungul rândului, împărțit în celule de plantă (1–1,5 m), cu un prag pe aria de canopy din fiecare celulă.
  - Compromisul e explicit: pragurile mici prind mai multe goluri, dar dau și mai multe false positive.
  - Metodele 3D (trunchi/DSM) ating F1 76–91% la trunchiuri, dar eșuează la canopy dens, care ascunde trunchiul.
- **Recomandare:**
  - KEEP: pragul de 5,0 m și lista de revizie manuală pentru golurile de 4,5–7 m.
  - ADD înainte de măsurare: o închidere morfologică 1D de 0,4 m pe profilul de ocupare (acoperă găurile din frunziș). Apoi o celulă se consideră ocupată doar dacă ocuparea pe ±0,30 m e ≥ 20%, pe o fereastră de 0,5 m.
  - Vegetația care umple golul se acceptă doar dacă e mai îngustă de 1,2 m și nu e iarbă (aceeași trăsătură de textură ca la inter-rânduri).
  - Golurile ≥ 1,5 × distanța dintre plante (≈ 2 m) alimentează direct țintele de inspecție din §4.10 („potentially missing vines”).
- **Impact:** regula din §4.8 (gol maxim ≥ 5 m → disrupted) reproduce toate cele 51 de etichete, deci pragul e calibrat. Riscul e în altă parte: găurile mici din frunziș și umbrele mici fragmentează golurile, iar vegetația înaltă din inter-rând umple fals golul.
- **Surse:** https://www.tandfonline.com/doi/full/10.1080/22797254.2017.1308234, https://pmc.ncbi.nlm.nih.gov/articles/PMC10682715/, https://www.mdpi.com/2504-446X/7/6/349

### 2.9 Cod open source (high, §4.3)

- **Constatare:** codul open source pentru detecția rândurilor din UAV e puțin și generic:
  - `ibaldoncini/crop-row-detector`: nesupervizat, pentru horticultură, ca notebook;
  - `petern3/crop_row_detection`: Hough cu OpenCV, pentru roboți;
  - `maikbasso/plant-line-detection`: ghidare UAV.

  Niciunul nu tratează legarea între tile-uri, blocurile sau rândurile curbe pe ortofoto georeferențiat.
- **Recomandare (KEEP):** păstrăm implementarea proprie și nu pierdem timp integrând aceste repo-uri. Cel mult consultăm notebook-ul ibaldoncini pentru pasul de rectificare/rotire per parcelă.
- **Impact:** nu există un schelet de clonat care să bată prototipul existent (F1 0,98 pe exemple).
- **Surse:** https://github.com/ibaldoncini/crop-row-detector, https://github.com/petern3/crop_row_detection, https://github.com/maikbasso/plant-line-detection

### 2.10 Schimbări propuse (rânduri, blocuri, goluri)

- **§3.6 `rows`:** se adaugă `spacing_estimate: autocorr`, `peak_min_dist_factor: 0.6`, `periodicity_min_snr: 3.0`, `onlattice_tol_factor: 0.25`, `residual_split_m: 0.15`, `track_window_px: 256`.
- **§4.2 punctul 4:** prominența ≥ 5% se înlocuiește cu SNR-ul vârfului FFT la `1/s` ≥ ~3, calibrat pe exemple.
- **§4.2 punctele 2–3:** se adaugă raportul lățime/spacing (vie ≤ 0,45), periodicitatea în lungul rândului (vârf la 3–6 m cu duty < 60% → livadă), circularitatea componentelor > 2 m² (pom) și verificarea armonicii `2s` vs. `s`.
- **§4.3 pasul 1:** profil de rezervă pe textură/luminanță (std locală a lui V pe 0,25 m) când contrastul a\* e slab pe inter-rânduri înierbate. Masca NN se folosește ca intrare dacă e disponibilă.
- **§4.3 pasul 4:** dacă p95 al rezidualului > 0,15 m, se aplică tracking cu ferestre de 6,4 m, cu ieșire polilinie simplificată DP cu 0,1 m.
- **§4.3 legarea:** se renunță la condiția capetelor la ≤ 0,4 m. Se leagă pe dreapta-suport prelungită (`Δunghi ≤ 3°`, `Δoffset ≤ 0,3 m`), cu gol coliniar permis.
- **§4.4:**
  - `row_id` vine din indexarea pe grilă per bloc: `round((offset − o0)/s)`;
  - muchia din graf se taie la un salt de spacing > 0,2 m sau la un defazaj > `0,3·s`;
  - golul coliniar se leagă doar dacă nu e o bandă transversală goală pe ≥ 3 rânduri vecine.
- **§4.8:** înainte de măsurarea golului se aplică o închidere 1D de 0,4 m pe profilul de ocupare. Umplerea golului se acceptă doar pentru vegetație mai îngustă de 1,2 m și care nu e iarbă. Golurile ≥ ~2 m se exportă ca ținte de inspecție (§4.10).

### 2.11 Riscuri (rânduri, blocuri, goluri)

- Articolele principale nu au putut fi citite integral: ScienceDirect, T&F și HAL au răspuns cu 403. Parametri precum SNR ≥ 3, duty < 60% și raportul ≤ 0,45 sunt estimări inginerești și trebuie calibrați pe tile-urile de exemplu.
- Nu avem DSM. Separarea vie/iarbă pe inter-rândurile complet înierbate rămâne cel mai mare risc pentru axe și pentru `interrow_cover`.
- Viile de grădină neregulate (rânduri scurte, spacing variabil, bolți) pot strica grila per bloc. Pentru blocurile mici, indexarea pe grilă are nevoie de fallback pe legarea simplă.
- La blocurile în evantai spacing-ul variază în lungul rândului, deci grila se estimează local (fereastră de 2×2 tile-uri), nu pe tot blocul.
- O livadă tânără cu coroane < 1,2 m și rânduri la 3,5–4 m e la limita intervalului 1,8–3,8 m. Dacă există în zonă, se verifică manual la QA.

---

## 3. Segmentarea canopy-ului și rețeaua neuronală obligatorie (plan §4.5–4.6)

### 3.1 Setul Riseholme (high, §4.6)

- **Constatare:**
  - Setul Riseholme (Lincoln, UK) de pe Zenodo are licența CC-BY-4.0. Conține 855 de imagini RGB din UAV și 40 215 adnotări COCO, cu clasele pole, trunk, vine_row și vineyard (canopy).
  - Acoperă 3 sezoane: aug 2024, mar 2025 și iul 2025. Arhiva `riseholme-vineyard.zip` are 3,3 GB.
  - Pagina NU dă GSD-ul, altitudinea sau camera, iar articolul asociat nu e încă publicat. Nu reiese nici dacă canopy-ul e adnotat per plantă sau per rând.
  - Sunt cadre UAV brute (split train/val/test pentru YOLOv11), nu ortofoto georeferențiat.
- **Recomandare (KEEP):**
  - Nu descărcăm Riseholme în timpul hackathonului.
  - În README poate apărea ca „dataset extern evaluat și respins (scară/sezon diferit)”.
  - Dacă rămâne timp, îl folosim doar ca „test out-of-domain”, cu subseturile aug-2024 și iul-2025 și masca `vineyard` rasterizată. Nu îl folosim pentru antrenare.
- **Impact:** §4.6 nu depinde de el. Ca pretraining la 2,5 cm/px e riscant: scara e necunoscută, clima și sistemul de conducere sunt din UK, iar etichetele sunt de tip instanță YOLO. Setul din martie 2025 e fără frunze și ar strica învățarea canopy-ului.
- **Surse:** https://zenodo.org/records/19234907

### 3.2 Setul Brescia (groundcover) (high, §4.6)

- **Constatare:**
  - „UAV RGB Image Dataset for Vineyard Groundcover Semantic Segmentation” (Univ. Brescia, Italia) are licența CC-BY-4.0.
  - Conține 24 de imagini de 8192×5460 luate de la 8 m cu DJI P1, deci cu un GSD de ordinul milimetrilor, mult sub 2,5 cm.
  - Măștile au 9 clase: vine canopy, bare soil și 7 familii de cover crop.
  - Fișierele sunt restricționate (cer login sau aprobare).
- **Recomandare:**
  - Nu îl adăugăm în pipeline.
  - Îl citez în pitch sau README ca dovadă că distincția viță/iarbă e o problemă recunoscută.
  - Dacă vrem totuși, cerem accesul vineri seara, fără să depindem de el.
- **Impact:** ar fi singurul set cu o clasă explicită viță vs. iarbă între rânduri. Accesul restricționat și GSD-ul de ~10× mai fin îl fac nepractic pentru un weekend.
- **Surse:** https://zenodo.org/records/17701564

### 3.3 Indici spectrali vs. clasificatori (medium, §4.5)

- **Constatare:**
  - Poblete-Echeverría et al. 2017 (Remote Sensing 9(3):268) au comparat pe o vie comercială ANN, Random Forest, indici spectrali RGB + Otsu și k-means. Cele mai bune au fost ANN și „indice simplu + prag Otsu”, iar pragul Otsu a rămas stabil în timp.
  - Studiile arată că indicii singuri NU separă solul de umbră pixel cu pixel.
  - ExGR e raportat ca stabil la iluminare variabilă, iar ExG e sensibil la lumina variabilă sau înnorată.
- **Recomandare:**
  - KEEP: a\* (σ 2,5) ca mască principală.
  - ADD un fallback per tile: dacă fracția de vegetație din coridor iese anormală (< 10% sau > 90%), pragul a\* se înlocuiește cu Otsu pe a\*, calculat doar pe pixelii din coridoare.
  - ADD ExGR = (2G−R−B) − (1,4R−G), pe RGB normalizat, doar ca a doua opinie în QA, nu în export.
  - Umbra se tratează ca non-canopy prin V < 50 (HSV), cum e deja în §4.7.
- **Impact:** confirmă alegerea din plan: Lab a\* cu prag fix 4 face parte din familia „indice + prag”. Niciun indice nu separă via de iarba verde dintre rânduri, pentru că ambele sunt verzi. Separarea vine din geometrie (coridorul ±0,30 m) sau din textură, adică din NN.
- **Surse:** https://doi.org/10.3390/rs9030268, https://www.nature.com/articles/s41598-025-23868-1

### 3.4 U-Net vs. Otsu (medium, §4.6)

- **Constatare:** literatura recentă pe vie arată că U-Net pe RGB (plus DSM, când există) bate Otsu/RANSAC la separarea canopy-ului de sol și de cover crop. Avantajul vine din textură și context, pe care indicii calculați pixel cu pixel nu le au.
- **Recomandare (CHANGE §4.6):**
  - Țintele de antrenare: pseudo-canopy (a\* ∩ coridor, după filtrul de pomi) = 1, iar TOATĂ vegetația din afara coridoarelor (iarbă, pomi) = 0, explicit.
  - Rezultatul e o rețea „vine-vs-grass”, nu o copie a măștii a\*.
  - La inferență se păstrează intersecția cu coridorul.
- **Impact:** aici e valoarea NN-ului: a\* nu separă iarba dintre rânduri de viță, un U-Net poate. Asta contează doar dacă NN-ul e antrenat pe TOT tile-ul, cu iarba etichetată ca „rest”. Antrenat numai în coridor, nu învață diferența.
- **Surse:** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11217331/, https://doi.org/10.3390/rs11091023

### 3.5 Antrenarea pe MPS (medium, §4.6)

- **Constatare:**
  - Pe MPS unele operații nu sunt implementate și cer `PYTORCH_ENABLE_MPS_FALLBACK=1`, adică rulează pe CPU, mai lent.
  - Nu există antrenare distribuită, doar un singur GPU.
  - Mixed precision/AMP pe MPS a fost istoric instabilă sau nesuportată, iar checkpoint-urile bf16 au produs segfault.
  - Există raportări recente de probleme la segmentarea semantică pe MPS (ultralytics #26237).
  - Throughput de referință: o epocă ResNet-50 de ~15 s pe RTX 4090 durează ~45–50 s pe M3/M4 Max. M3 Pro e mai lent decât Max.
- **Recomandare:**
  - KEEP: U-Net + ResNet18 (smp, `encoder_weights='imagenet'`), cu ResNet34 ca alternativă.
  - NU folosim SegFormer/MiT: e mai lent, attention pe MPS are operații cu fallback, iar câștigul la 2 clase e mic.
  - Setări: fp32 (fără autocast), `export PYTORCH_ENABLE_MPS_FALLBACK=1`, batch 8 la 512², AdamW cu lr 3e-4 și cosine decay, 15–20 de epoci.
  - Patch-urile se pre-extrag în `.npy`/uint8, ca DataLoader-ul (`num_workers` 4) să nu devină gâtul de sticlă.
  - Cu 18 GB de memorie unificată nu trecem de batch 16.
  - Vineri facem un smoke-run de 1 epocă pe MPS, înainte de rularea mare. Dacă apar erori sau NaN, trecem pe `device='cpu'` pentru debug.
- **Impact:** estimarea din plan („1–2 h pe MPS, rulează noaptea”) e probabil pesimistă.
  - 100 de tile-uri × 4 patch-uri de 512 (la 0,05 m/px) = ~400 de patch-uri pe epocă.
  - U-Net-ResNet18 în fp32 ar trebui să antreneze ~10–20 img/s pe M3 Pro (estimare proprie, nemăsurată), adică ~30–60 s pe epocă.
  - 15 epoci ar dura deci ~15 min. Timpul real trebuie măsurat.
- **Surse:** https://huggingface.co/docs/transformers/v4.49.0/perf_train_special, https://github.com/ultralytics/ultralytics/issues/26237, https://github.com/huggingface/trl/issues/7220, https://tillcode.com/apple-silicon-pytorch-mps-setup-and-speed-expectations/, https://arxiv.org/pdf/2508.18206

### 3.6 Etichete zgomotoase (medium, §4.6)

- **Constatare:** bunele practici pentru etichete zgomotoase în segmentarea din teledetecție:
  - antrenarea pe un subset mai curat bate adesea antrenarea pe tot setul zgomotos;
  - filtrarea iterativă a eșantioanelor cu loss mare (Self-Filtered Learning) ajută;
  - label smoothing și early stopping limitează memorarea zgomotului: rețelele învață întâi pattern-urile curate și memorează zgomotul abia în epocile târzii;
  - pierderile simetrice sau „normalizate” (NCE+RCE, Ma et al. 2020) sunt demonstrat robuste la zgomot.
- **Recomandare (ADD în §4.6):**
  1. Antrenăm doar pe tile-urile validate la QA (e deja în plan).
  2. Bandă „ignore” de 3 px (la 0,05 m/px) în jurul marginilor pseudo-măștii și ale coridorului, prin `ignore_index=255` sau o mască de pierdere.
  3. Loss = Dice + BCE cu label smoothing 0,05.
  4. Early stopping pe cele 2 tile-uri de referință, cu patience 3.
  5. Opțional, o rundă de self-filtering: după epoca 5 se aruncă cele mai zgomotoase 10% din patch-uri (după loss) și antrenarea se reia.

  NU facem self-training iterativ cu predicțiile NN: n-avem timp și riscăm confirmation bias.
- **Impact:** pseudo-etichetele a\* ∩ coridor au zgomot sistematic la margini (±1–2 px de umbră sau halo) și pe rândurile cu iarbă înaltă. NN-ul nu trebuie să copieze aceste erori.
- **Surse:** https://impact.ornl.gov/en/publications/self-filtered-learning-for-semantic-segmentation-of-buildings-in-/, https://arxiv.org/html/2402.16164, http://proceedings.mlr.press/v119/ma20c/ma20c.pdf, https://arxiv.org/html/2603.00604v1

### 3.7 Ablația (high, §4.6)

- **Constatare:** cu doar 2 tile-uri de referință, orice prag sau fuziune aleasă pe ele e supraajustare. O ablație credibilă raportează scorul per tile și arată calitativ unde NN-ul diferă de a\*.
- **Recomandare (ADD):** tabel de ablație cu metrica challenge-ului (0,6·IoU + 0,4·F1@0,5), per tile și în medie:

  | Variantă | Descriere |
  |---|---|
  | A | a\* ∩ coridor (baseline) |
  | B | NN > 0,5 ∩ coridor |
  | C | a\* AND NN |
  | D | a\* OR NN |
  | E | NN fără coridor (arată dacă NN-ul a învățat singur viță vs. iarbă) |
  | F | NN antrenat fără bandă ignore și fără label smoothing |

  - Pragul NN rămâne fix la 0,5, fără tuning pe referință.
  - Pe lângă tabel: 3–4 panouri pe tile-uri grele (iarbă verde între rânduri, umbră, pomi în rând), cu RGB | a\* | NN | diferență, plus timpul de inferență per tile.
  - Fuziunea intră în export doar dacă B, C sau D câștigă pe AMBELE tile-uri.
- **Impact:** livrabilul NN e obligatoriu, deci valoarea lui trebuie demonstrată clar, chiar dacă nu intră în export.
- **Surse:** https://doi.org/10.3390/rs9030268

### 3.8 Schimbări propuse (canopy, NN)

- **§4.6 Date:** pseudo-etichetele se construiesc pe TOT tile-ul: viță = a\* ∩ coridor după filtrul de pomi; iarba și pomii din afara coridorului = 0, explicit. Scopul e ca NN-ul să învețe diferența viță vs. iarbă.
- **§4.6 Antrenare:** bandă ignore de 3 px (`ignore_index=255`) la marginile pseudo-măștii și ale coridorului, label smoothing 0,05 pe BCE și early stopping (patience 3) pe cele 2 tile-uri de referință. Opțional: după epoca 5 se aruncă cele mai zgomotoase 10% din patch-uri.
- **§4.6 Setări MPS:** fp32 fără autocast, `PYTORCH_ENABLE_MPS_FALLBACK=1`, batch 8 la 512², AdamW cu lr 3e-4 și cosine decay, patch-uri pre-extrase în `.npy`. Encoder ResNet18 (alternativă ResNet34), fără SegFormer.
- **§4.6 Estimare de timp:** „1–2 h, rulează noaptea” devine „estimat ~15–30 min pentru 15 epoci, de confirmat prin smoke-run de 1 epocă vineri”. Orele eliberate pot merge în ablație.
- **§4.6:** tabelul de ablație A–F (a\* baseline / NN / AND / OR / NN fără coridor / NN fără tehnicile anti-zgomot), per tile, cu prag fix 0,5, plus 3–4 panouri calitative pe tile-uri grele.
- **§4.5:** fallback Otsu pe a\* (calculat doar pe pixelii din coridoare) când fracția de vegetație din coridor e anormală (< 10% sau > 90%). ExGR se calculează doar ca verificare QA.
- **§10 Surse / README:** Riseholme (CC-BY-4.0, GSD necunoscut, cadre brute, include un sezon fără frunze) și Brescia groundcover (CC-BY-4.0, acces restricționat, GSD de ordinul mm) apar ca evaluate și nefolosite, cu motivul.

### 3.9 Riscuri (canopy, NN)

- Throughput-ul pe M3 Pro (10–20 img/s pentru U-Net-R18 la 512², fp32) e o estimare proprie și trebuie măsurat cu smoke-run.
- Cu doar 2 tile-uri de referință, diferențele din ablație pot fi zgomot. Raportăm per tile și nu tunăm praguri pe ele.
- Dacă la 5 cm/px iarba dintre rânduri are o textură foarte apropiată de cea a viței, NN-ul nu va bate a\* ∩ coridor și rămâne livrabil doar documentat. Soluția e antrenarea la 0,025 m/px nativ (16 patch-uri pe tile).
- Poblete-Echeverría 2017 nu a putut fi citit integral (MDPI a răspuns cu 403). Concluziile vin din abstract și din rezumatele din căutare.
- Pentru Riseholme nu sunt publicate GSD-ul și altitudinea și nici modul de adnotare a canopy-ului (per plantă sau per rând).

---

## 4. Traseul pedestru în domeniul poligonal permis (plan §4.10–4.11)

### 4.1 OR-Tools: instalare și setări (high, §4.11 pasul 5)

- **Constatare:** OR-Tools 9.15.6755 (14 ian 2026) are wheel-uri oficiale macOS arm64 pentru cp311, cp312, cp313 și cp314 (`macosx_11_0_arm64`). Pe M3 Pro se instalează cu pip, fără compilare.
- **Recomandare (KEEP):**
  - Fixăm `ortools==9.15.6755` în requirements.
  - Model: `RoutingIndexManager(n, 1, 0)`, cu depot = START la indexul 0.
  - Costul e o matrice de întregi în centimetri, `int(round(d_m*100))`, pentru că callback-ul de tranzit acceptă doar int.
  - Căutare: `first_solution_strategy=PATH_CHEAPEST_ARC`, `local_search_metaheuristic=GUIDED_LOCAL_SEARCH`, `time_limit.seconds=30` (60 s doar la rularea finală), `log_search=True` în debug.
  - GLS nu se oprește singur, deci limita de timp e obligatorie.
  - Pentru 100–500 de noduri, matricea n×n (max. 250k intrări) încape ușor în memorie, ca listă Python sau `np.int64`.
- **Impact:** confirmă alegerea din §4.11 pasul 5. Pe Python 3.11/3.12 nu există risc de instalare.
- **Surse:** https://pypi.org/project/ortools/, https://developers.google.com/optimization/routing/routing_options

### 4.2 OR-Tools: strategii și benchmark (high, §4.11 pasul 5)

- **Constatare:** opțiunile documentate de OR-Tools:
  - first solution: PATH_CHEAPEST_ARC, SAVINGS, CHRISTOFIDES, PARALLEL_CHEAPEST_INSERTION;
  - metaeuristici: GREEDY_DESCENT, GUIDED_LOCAL_SEARCH (descrisă ca în general cea mai eficientă pentru rutare), SIMULATED_ANNEALING, TABU_SEARCH;
  - limite: `time_limit`, `solution_limit`, `lns_time_limit`.
- **Recomandare (ADD):** un benchmark mic: GLS cu 10 s, 30 s și 60 s, combinat cu 2 strategii inițiale (PATH_CHEAPEST_ARC, SAVINGS). Dacă diferența de lungime e < 0,5%, rămânem la 30 s. Raportăm cea mai bună soluție și gap-ul față de elkai (§4.3).
- **Impact:** setarea „GLS, 60 s” e rezonabilă, dar nu verifică dacă soluția e aproape de optim.
- **Surse:** https://developers.google.com/optimization/routing/routing_options

### 4.3 elkai / LKH-3 (high, §4.11 pasul 5)

- **Constatare:**
  - elkai 2.0.1 (wrapper peste LKH-3) are wheel-uri macOS arm64 pentru cp37–cp313 și un API într-un singur apel: `elkai.DistanceMatrix(M).solve_tsp(runs=...)`, care întoarce turul închis `[0, ..., 0]`. Suportă TSP simetric și asimetric.
  - Codul LKH e licențiat doar pentru uz necomercial, iar elkai ține GIL-ul în timpul rezolvării.
- **Recomandare (ADD ca opțional/benchmark, nu ca dependență obligatorie):**
  - `elkai==2.0.1`, cu `solve_tsp(runs=10)` pe aceeași matrice în cm.
  - Păstrăm turul mai scurt dintre OR-Tools și elkai și notăm sursa în metadate.
  - Nu folosim pachetul `lkh` (PyLKH 2.0.0): cere binarul LKH-3 compilat manual și declară suport doar până la Python 3.11.
- **Impact:** elkai dă o a doua opinie aproape optimă (LKH) pentru ~500 de noduri în câteva secunde. Licența e o problemă dacă repo-ul trimis trebuie să fie utilizabil comercial.
- **Surse:** https://github.com/fikisipi/elkai, https://pypi.org/project/elkai/, https://pypi.org/project/lkh/

### 4.4 python-tsp ca fallback (medium, §4.11 pasul 5)

- **Constatare:** python-tsp 0.5.0 (Python >= 3.9, compatibil NumPy 2) e scris în Python pur. Oferă local search (2-opt și PS1–PS6), simulated annealing, Lin-Kernighan și record-to-record, cu `max_processing_time`.
- **Recomandare (CHANGE fallback-ul):**
  - În loc de 2-opt scris manual: `python_tsp.heuristics.solve_tsp_local_search(D, x0=nn_tour, perturbation_scheme='two_opt', max_processing_time=20)`, urmat de `solve_tsp_lin_kernighan`.
  - Fixăm `python-tsp==0.5.0`.
  - Îl folosim doar dacă OR-Tools aruncă excepție sau nu întoarce soluție.
- **Impact:** fallback-ul „nearest neighbour + 2-opt” din plan presupune cod scris de noi.
- **Surse:** https://github.com/fillipe-gsm/python-tsp

### 4.5 Close-Enough TSP ca GTSP (medium, §4.10–4.11 pașii 3–4)

- **Constatare:** Close-Enough TSP (o țintă e vizitată dacă traseul trece la ≤ r de ea) se rezolvă practic prin discretizare. Fiecare țintă primește un set de puncte candidate din vecinătatea ei, iar problema devine un Generalized TSP: se vizitează exact un candidat din fiecare set. E schema Carrabs et al., cu rafinare SOCP.
- **Recomandare (CHANGE pasul 3 în GTSP cu OR-Tools):**
  - Candidații unei ținte sunt nodurile grafului la ≤ 1,9 m (marjă de 0,1 m sub 2 m), cel mult 2–3 pe fiecare parte.
  - Fiecare set intră într-o disjuncție obligatorie, `routing.AddDisjunction([idx_candidati])`, fără penalitate, deci se vizitează exact un nod.
  - Țintele cu `reachable=false` nu intră în model.
  - Numărul total de noduri rămâne ≤ ~1 500, iar Dijkstra multi-sursă rămâne rapid cu `scipy.sparse.csgraph.dijkstra` pe indici.
- **Impact:** planul leagă fiecare țintă de UN singur nod (cel mai apropiat, la ≤ 2 m). O țintă pe axa rândului e însă la 1,25–1,40 m de mediana AMBELOR inter-rânduri vecine, deci are cel puțin 2 candidați reali. Fixarea unuia singur forțează ocoluri, chiar dacă traseul trece deja prin inter-rândul de pe cealaltă parte.
- **Surse:** https://arxiv.org/pdf/2507.03775, https://www.sciencedirect.com/science/article/abs/pii/S0305054816302179, https://optimization-online.org/wp-content/uploads/2014/02/4248.pdf

### 4.6 Ținte acoperite „gratuit” (medium, §4.11 pașii 3, 5, 6)

- **Constatare:** în CETSP pe o rețea de coridoare, multe ținte sunt acoperite gratuit de segmentele parcurse spre alte ținte. Acesta e principalul câștig față de un TSP pe puncte.
- **Recomandare (ADD):** o buclă iterativă după desfășurarea LineString-ului:
  1. calculăm acoperirea tuturor țintelor cu `shapely.dwithin(targets, route, 2.0)`;
  2. eliminăm din model țintele deja acoperite de căi (nu prin noduri proprii) și re-rezolvăm;
  3. repetăm de 2–3 ori sau până nu mai scade lungimea.

  La final verificăm obligatoriu că acoperirea e 100% din țintele reachable. Țintele care cad pe același nod se deduplică.
- **Impact:** matricea și turul se pot reduce semnificativ, iar lungimea scade.
- **Surse:** https://arxiv.org/html/2310.04257v2, https://www.sciencedirect.com/science/article/abs/pii/S037722172300293X

### 4.7 Graful: linii mediane și schelet (medium, §4.11 pasul 2)

- **Constatare:** pachetul `centerline` 1.1.1 (MIT, mai 2024) calculează linia mediană prin Voronoi pe conturul densificat al poligonului, echivalent cu `shapely.voronoi_polygons(..., only_edges=True)` filtrat cu `covered_by`. Pe poligoane complexe cu multe găuri produce ramuri parazite (spurs) și e lent la densificare fină.
- **Recomandare (CHANGE):**
  - (a) Inter-rânduri: mediana = media geometrică a două axe consecutive din §4.3, tăiată cu `intersection(domain_eroded)`. `centerline` rămâne doar ca fallback, cu `interpolation_distance=0.5`.
  - (b) Pasaje: păstrăm `skimage.morphology.skeletonize` pe un raster de 0,25 m, după `buffer(-0.3)` al domeniului, ca scheletul să nu atingă marginile.
    - Graful se construiește direct cu `sknw.build_sknw(ske, multi=False, iso=False)`: muchii cu `pts` și `weight` în pixeli.
    - Pixelii se convertesc în UTM cu transformul affine al rasterului, iar ramurile terminale < 2 m se elimină.
    - Alternativa vectorială, `shapely.voronoi_polygons(shapely.segmentize(boundary, 0.5), only_edges=True)` + `covered_by(domain)`, se folosește doar dacă scheletul dă probleme.
- **Impact:** pentru pasaje (2 poligoane, 13 găuri) Voronoi vectorial e fragil. Pentru inter-rânduri e inutil, pentru că avem deja axele rândurilor (§4.3).
- **Surse:** https://pypi.org/project/centerline/, https://shapely.readthedocs.io/en/stable/reference/shapely.voronoi_polygons.html, https://github.com/Image-Py/sknw

### 4.8 String pulling (medium, §4.11 pașii 2 și 5)

- **Constatare:** scheletul raster e în zig-zag (pași de pixel, noduri la joncțiuni), deci supraestimează lungimea și produce trasee neoptime în zonele largi ale pasajelor.
- **Recomandare (ADD):** „string pulling” după desfășurare.
  - Se parcurg vârfurile rutei greedy, iar subsecvența `i..j` se înlocuiește cu segmentul drept `[i, j]` dacă `shapely.covered_by(LineString([p_i, p_j]), domain_eroded)` e True.
  - `domain_eroded = domain.buffer(-0.3)`, pregătit cu `shapely.prepare`.
  - Aceeași idee se aplică pe muchiile grafului înainte de Dijkstra: `shapely.simplify(edge, 0.25)`, urmat de verificarea `covered_by`.
- **Impact:** afectează metrica principală (lungimea traseului) și comparația din pitch.
- **Surse:** https://shapely.readthedocs.io/en/stable/reference/shapely.prepare.html

### 4.9 Validarea rapidă și exactă (high, §4.11 pasul 6)

- **Constatare:** `shapely.prepare` (in-place) accelerează doar predicatele (`contains`, `covered_by`, `covers`, `intersects`, `within` etc.), nu și overlay-urile (`difference`/`intersection`).
- **Recomandare (CHANGE) validarea în 2 niveluri:**
  1. Rapid: `shapely.prepare(domain_in)`; `ok = shapely.covered_by(route, domain_in)`.
  2. Exact, doar la final: `outside = shapely.difference(route, domain_in, grid_size=0.001).length; share = outside / route.length`, cu pragul < 0,005.

  Domeniul se construiește cu `shapely.union_all(polys, grid_size=0.001)` + `shapely.make_valid`, apoi `domain_in = domain.buffer(-0.05)`. `grid_size` pe overlay elimină artefactele de virgulă mobilă (segmente de 1e-12 m în afara domeniului).
- **Impact:** validarea din pasul 6 folosește `difference().length`, care nu beneficiază de `prepare`. Pentru miile de verificări (string pulling, conectori) sunt necesare predicate.
- **Surse:** https://shapely.readthedocs.io/en/stable/reference/shapely.prepare.html

### 4.10 `is_simple` nu e un criteriu valid (high, §4.11 pasul 6)

- **Constatare:** un tur închis pe un graf de coridoare trece inevitabil de două ori prin aceleași segmente (dus-întors în inter-rândurile fundătură), deci LineString-ul NU va fi `is_simple`.
- **Recomandare (CHANGE criteriul):** nu testăm `is_simple`. Testăm în schimb:
  - `route.is_valid`;
  - geometria e LineString (nu Multi);
  - nu există segmente de lungime 0 (vârfurile consecutive duplicate se elimină);
  - primul și ultimul vârf sunt la ≤ 5 m de START (ideal exact START, adăugat explicit ca vârf);
  - `length_m = round(route.length, 2)` se scrie din aceeași geometrie serializată (se recitește GeoJSON-ul și se recalculează).
- **Impact:** criteriul „fără auto-intersecții degenerate”, implementat ca `is_simple`, ar putea bloca greșit commit-ul.
- **Surse:** https://shapely.readthedocs.io/en/stable/reference/shapely.prepare.html

### 4.11 Schimbări propuse (traseu)

- **§4.11 pasul 5:** fixăm `ortools==9.15.6755` (wheel arm64 cp311–cp314), costuri int în cm, PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH, `time_limit` 30 s (60 s la final), benchmark scurt 10/30/60 s.
- **§4.11 pasul 5:** adăugăm `elkai==2.0.1` cu `solve_tsp(runs=10)` ca verificare opțională (licența LKH e necomercială, deci nu e dependență obligatorie). Fallback-ul devine `python-tsp==0.5.0` (local search 2-opt + Lin-Kernighan) în loc de 2-opt scris manual.
- **§4.11 pasul 3:** „cel mai apropiat nod” devine GTSP: candidați la ≤ 1,9 m din ambele inter-rânduri vecine, `routing.AddDisjunction(candidati)` fără penalitate (exact unul vizitat).
- **§4.11:** bucla „drop covered and re-solve” (2–3 iterații) cu `shapely.dwithin(targets, route, 2.0)`. Verificare finală: 100% din țintele reachable sunt acoperite.
- **§4.11 pasul 2:**
  - mediana inter-rândurilor vine din axele §4.3 (media a două axe consecutive), nu din Voronoi;
  - pasaje: `buffer(-0.3)` → skeletonize la 0,25 m → `sknw.build_sknw` → UTM prin affine → se elimină ramurile < 2 m.
- **§4.11:** string pulling după desfășurarea traseului: scurtături drepte verificate cu `covered_by` pe `domain.buffer(-0.3)` pregătit.
- **§4.11 pasul 6:**
  - validare în 2 niveluri: `covered_by` pe domeniul pregătit, apoi `difference(..., grid_size=0.001).length`;
  - domeniul se construiește cu `union_all(grid_size=0.001)` + `make_valid`;
  - cerința `is_simple` se elimină și se înlocuiește cu `is_valid` + fără segmente de lungime zero + START ca prim și ultim vârf.

### 4.12 Riscuri (traseu)

- ~~Regula exactă de „vizită” (raza de 2 m) nu a fost verificată în textul regulilor.~~ **Închis la integrare:** descrierea challenge-ului spune că o țintă e vizitată dacă traseul trece la ≤ 2 m de ea.
- Licența LKH (elkai) e necomercială. Dacă organizatorii cer o licență deschisă pentru cod, elkai rămâne doar în scripturile de benchmark.
- Un schelet raster pe tot extentul de 81,5 ha la 0,25 m înseamnă ~13 M pixeli. Încape în 18 GB, dar skeletonize se face doar pe bbox-ul pasajelor.
- Dacă inter-rândurile corectate în Marcaj diferă de axele din §4.3, mediana calculată din axe poate ieși din poligon. Se taie mereu cu `domain_eroded`, iar capetele se reconectează.
- Numărul de noduri GTSP (ținte × candidați) poate trece de ~1 500 dacă țintele sparse sunt multe. Se limitează candidații la 2 pe parte sau se grupează țintele sparse.

---

## 5. Interfața web: MapLibre, raster, vectori, găzduire, tabele (plan §4.14)

### 5.1 Fondul de hartă (high, §4.14)

- **Constatare:**
  - Sursa `image` din MapLibre primește 4 colțuri lon/lat în ordinea top-left, top-right, bottom-right, bottom-left. Pentru un tile de 51,2 m, un quad rotit acoperă convergența grilei UTM (circa 1–2°), deci ideea din plan e corectă geometric.
  - Ghidul oficial de performanță avertizează însă că fiecare sursă are un overhead mare la randare. 150–311 surse `image` înseamnă 311 surse, texturi și draw call-uri separate, fără piramidă pentru zoom mic.
- **Recomandare (SCHIMBĂ):** fondul devine o piramidă XYZ statică, generată cu gdal2tiles. Sursa `image` rămâne doar ca fallback de 10 minute, dacă GDAL nu se instalează.
  - Comenzi:
    - `brew install gdal` (gdal2tiles cere GDAL ≥ 3.6 pentru WEBP)
    - `gdalbuildvrt siret3.vrt tiles/*.tif`
    - `gdal2tiles.py --xyz -z 14-20 -r average --tiledriver=WEBP --webp-quality=75 --processes=10 -x -w none siret3.vrt web/ortho`
  - În MapLibre: `{type:'raster', tiles:['ortho/{z}/{x}/{y}.webp'], tileSize:256, minzoom:14, maxzoom:20, bounds:[...]}`. Peste z20, MapLibre face overzoom.
  - Estimare pentru ~0,82 km²: ~1 500 tile-uri la z20 (≈ 0,1 m/px, ca în plan), ~2 000 de fișiere în total, ~40–60 MB.
  - Pentru detaliu la 0,05 m/px: `-z 14-21` dă ~6–8k fișiere și ~150–200 MB. E în limite, dar build-ul și upload-ul durează mai mult.
- **Impact:** fondul de hartă din §4.14 riscă să fie lent sau să pâlpâie la zoom mic, pe laptop și mai ales pe proiector. În plus, la zoom < 17 se încarcă tot mozaicul (~25 MB).
- **Surse:** https://maplibre.org/maplibre-style-spec/sources/, https://maplibre.org/maplibre-gl-js/docs/guides/large-data/, https://gdal.org/en/stable/programs/gdal2tiles.html

### 5.2 PMTiles pentru raster (high, §4.14)

- **Constatare:** driverul PMTiles din GDAL scrie DOAR vector tiles (MVT), nu raster. Un PMTiles raster cere un pas în plus: gdal → MBTiles → `pmtiles convert`.
- **Recomandare:** NU folosim PMTiles pentru raster, ci directorul XYZ de mai sus.
- **Impact:** varianta „PMTiles raster” ar adăuga un tool și un pas, fără câștig real pentru ~2k fișiere.
- **Surse:** https://gdal.org/en/stable/drivers/vector/pmtiles.html

### 5.3 PMTiles și HTTP Range pe hosting static (medium, §4.14)

- **Constatare:**
  - PMTiles depinde de HTTP Range, iar pe hosting static acesta a fost fragil.
  - GitHub Pages a întors răspunsuri Range invalide pentru fișiere gzip (Firefox: „Decoding failed”, 2025; raportat rezolvat în jurul lui martie 2026). Issue-ul PMTiles #584 raportează eșecuri intermitente și în Chrome.
  - Pe Cloudflare, răspunsurile Range pentru `.pmtiles` sunt corupte (maplibre/demotiles #35). Workaround-ul e R2 + Worker.
  - Fallback-ul de pe laptop, `python -m http.server` (SimpleHTTPRequestHandler), nu suportă deloc Range.
- **Recomandare:**
  - EVITĂM PMTiles la demo. Pentru vectori folosim tile-uri MVT într-un director static (§5.4), care merg pe orice host, inclusiv `python -m http.server`.
  - Dacă totuși folosim PMTiles: `pmtiles@4.5.0`, `maplibregl.addProtocol('pmtiles', new pmtiles.Protocol().tile)`, iar local servim cu `npx http-server` (suportă Range), nu cu python.
- **Impact:** un singur fișier `.pmtiles` pentru canopy poate strica demo-ul exact la pitch, iar fallback-ul local din plan nu l-ar servi.
- **Surse:** https://github.com/orgs/community/discussions/178318, https://github.com/protomaps/PMTiles/issues/584, https://github.com/maplibre/demotiles/issues/35, https://docs.protomaps.com/pmtiles/maplibre

### 5.4 Straturi vectoriale mari (high, §4.14)

- **Constatare:** pentru 50k–100k de poligoane de canopy, GeoJSON-ul brut (~0,5 KB/poligon, deci 25–50 MB) e prea mare pentru o singură sursă. Ghidul MapLibre recomandă 6 zecimale, proprietăți minime, chunking sau vector tiles. tippecanoe (`brew install tippecanoe`) cere input în WGS84 și are flag-uri care păstrează toate feature-urile.
- **Recomandare (SCHIMBĂ):**
  - Canopy (și waste) devin MVT într-un director:
    `tippecanoe -e web/vt -l canopy -Z16 -z20 --no-tile-compression --no-feature-limit --no-tile-size-limit --drop-densest-as-needed -y id -y vineyard_id -y row_id canopy_wgs84.geojson`
  - `--no-tile-compression` e OBLIGATORIU, pentru că GitHub Pages și python nu trimit `Content-Encoding: gzip` pentru `.pbf`.
  - În MapLibre: `{type:'vector', tiles:['vt/{z}/{x}/{y}.pbf'], minzoom:16, maxzoom:20, promoteId:{canopy:'id'}}`.
  - Rămân GeoJSON global (coordonate cu 6 zecimale, cu `promoteId`) rândurile, inter-rândurile, blocurile, țintele și traseul, care au doar câteva mii de feature-uri.
  - Fallback fără tippecanoe: un singur `canopy.geojson` cu 6 zecimale, doar `id` și `row_id`, la `minzoom:17`. Host-ul îl servește gzip, deci transferul e de ~5–10 MB.
- **Impact:** planul actual (câte un GeoJSON per tile, încărcat dinamic la zoom ≥ 18) cere cod JS de gestionare a vizibilității (add/remove pentru până la 311 surse) și are aceeași problemă de overhead per sursă.
- **Surse:** https://github.com/felt/tippecanoe, https://maplibre.org/maplibre-gl-js/docs/guides/large-data/

### 5.5 Limite de găzduire (high, §4.14 / §4.15)

- **Constatare:**
  - GitHub Pages: site publicat ≤ 1 GB, repo recomandat ≤ 1 GB, 100 GB/lună bandwidth (limită soft), timeout de deploy de 10 minute.
  - Limitele Git generale (din cunoștințe, neverificate acum): fișier ≤ 100 MB (limită hard), avertisment la 50 MB. Conținutul LFS nu e servit la deploy din branch, ci doar dacă e publicat prin Actions cu checkout `lfs: true`.
  - Cloudflare Pages Free: max. 20 000 de fișiere per site și max. 25 MiB per fișier.
- **Recomandare:**
  - PĂSTRĂM GitHub Pages, cu deploy prin GitHub Actions (`actions/upload-pages-artifact` + `actions/deploy-pages`) dintr-un folder `web/` generat local și comis.
  - Fără LFS: niciun fișier > 25 MB, iar bugetul total < 300 MB.
  - Cloudflare Pages doar ca backup (`npx wrangler pages deploy web`), cu maxzoom raster 20.
- **Impact:** piramida z14–20 (~2k fișiere) plus MVT (~2–5k fișiere) încap în ambele. Cu raster până la z21 (~6–8k) plus MVT, ne apropiem de limita de 20k de pe Cloudflare Free.
- **Surse:** https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits, https://developers.cloudflare.com/pages/platform/limits/, https://docs.protomaps.com/pmtiles/cloud-storage

### 5.6 Versiunea MapLibre (high, §4.14)

- **Constatare:**
  - MapLibre GL JS e acum la 6.11.2 (npm). v6 e doar ESM: nu mai are UMD `maplibre-gl.js`, fișierul e `maplibre-gl.mjs`. Cere WebGL2 și construiește worker-ul ca Blob URL.
  - `addProtocol`, feature-state, `queryRenderedFeatures` și `fitBounds` au rămas neschimbate.
  - Ultima versiune v5 (UMD) e 5.24.0.
- **Recomandare (ADAUGĂ):**
  - Fixăm `maplibre-gl@5.24.0` (UMD, compatibilă cu cele mai multe exemple) sau 6.11.2 cu `<script type=module>`.
  - Oricare ar fi, fișierul se pune LOCAL în `web/vendor/`, ca demo-ul să meargă offline de pe laptop, fără CDN.
  - Recomandare: 5.24.0, pentru viteza de dezvoltare.
- **Impact:** exemplele de pe internet cu `<script src=maplibre-gl.js>` nu mai merg pe v6. Alegerea versiunii decide cum scriem „un singur fișier JS”.
- **Surse:** https://maplibre.org/maplibre-gl-js/docs/guides/v5-to-v6-migration-guide/, https://www.npmjs.com/package/maplibre-gl, https://maplibre.org/news/

### 5.7 Tabele și interacțiune (medium, §4.14)

- **Constatare:**
  - Tabelele sortabile și interacțiunea cu harta nu cer framework. Pentru câteva mii de rânduri de vie, un `<table>` cu sortare în vanilla JS (~40 de linii) e suficient.
  - Evidențierea se face cu `map.setFeatureState({source,id},{sel:true})` (necesită `promoteId`), iar zoom-ul cu `map.fitBounds(bbox)` pe bbox-uri precalculate.
- **Recomandare (ADAUGĂ):**
  - La build se generează `rows.json`, `blocks.json` și `route.json` cu `id`, `vineyard_id`, `row_id`, `length_m`, `area_m2`, `bbox:[w,s,e,n]` și `row_structure`.
  - Tabelul se generează din JSON. Click pe header sortează cu `[...rows].sort()`, fără mutație.
  - Click pe un rând din tabel face `fitBounds` + feature-state. Click pe hartă derulează tabelul la rândul respectiv (`scrollIntoView`).
  - Dacă tabelele trec de 5k rânduri, se randează doar primele 500, plus un filtru text. Tabulator nu e necesar.
  - Buget total UI: HTML+CSS+JS < 50 KB, plus maplibre ~800 KB.
- **Impact:** panourile 1–4 se pot face într-un singur `app.js` de ~400 de linii, fără build step.
- **Surse:** https://maplibre.org/maplibre-gl-js/docs/guides/large-data/

### 5.8 Schimbări propuse (web)

- **§4.14 Stack:**
  - site static cu `maplibre-gl` 5.24.0 (UMD) pus local în `web/vendor/`, plus un `app.js` vanilla, fără framework și fără build step;
  - deploy pe GitHub Pages prin Actions (`upload-pages-artifact` + `deploy-pages`);
  - fallback local: `python -m http.server -d web`, care merge pentru că nu folosim Range sau PMTiles.
- **§4.14 Imagine de fond:** în loc de 311 surse `image`, o piramidă XYZ generată cu `gdalbuildvrt siret3.vrt tiles/*.tif && gdal2tiles.py --xyz -z 14-20 -r average --tiledriver=WEBP --webp-quality=75 --processes=10 -x -w none siret3.vrt web/ortho` (~2k fișiere, ~40–60 MB). Sursa `image` cu 4 colțuri rămâne doar ca fallback.
- **§4.14 Straturi vectoriale:**
  - canopy și waste ca MVT într-un director, generate cu `tippecanoe -e web/vt -l canopy -Z16 -z20 --no-tile-compression --no-feature-limit --no-tile-size-limit --drop-densest-as-needed -y id -y vineyard_id -y row_id`;
  - rândurile, inter-rândurile, blocurile, țintele și traseul rămân GeoJSON global, cu 6 zecimale și `promoteId`;
  - dispare logica „câte un GeoJSON per tile, încărcat dinamic”.
- **§4.14:** fără PMTiles la demo, din cauza riscului Range pe GitHub Pages/Cloudflare și a incompatibilității cu `python http.server`.
- **§4.14 Panouri:** build-ul exportă `rows.json`, `blocks.json` și `route.json` cu bbox precalculat. Tabelele sunt vanilla și sortabile: click în tabel = `fitBounds` + `setFeatureState`; click pe hartă = rândul evidențiat în tabel.
- **§4.15 Setup:** `brew install gdal tippecanoe` intră în `make setup` (sau e documentat în README), iar `make web` rulează gdal2tiles + tippecanoe + exportul JSON.
- **Buget de mărime:** `web/` < 300 MB, niciun fișier > 25 MB, < 20k fișiere (compatibil și cu Cloudflare Pages Free), fără Git LFS.

### 5.9 Riscuri (web)

- gdal2tiles cu `-x` pe un VRT fără alpha poate cere `gdalbuildvrt -addalpha` sau `-srcnodata 0`, ca marginile negre să devină transparente. Se verifică pe 5 tile-uri înainte de rularea completă.
- Suprafața totală (~0,82 km²) și numărul de fișiere sunt estimate din 311 × 51,2 m. Se recalculează după gdal2tiles.
- Limitele Git de 100 MB per fișier și comportamentul LFS pe Pages vin din cunoștințe generale și nu au fost reverificate în această sesiune.
- MapLibre v6 cere WebGL2. Dacă alegem v6, testăm pe laptopul și browserul de la pitch.
- Local lipsesc gdal și tippecanoe (verificat: nu sunt în PATH). Instalarea prin brew poate dura 5–15 minute.

---

## 6. CVAT for images 1.1: format XML, import/export, ZIP-urile Marcaj (plan §4.13)

### 6.1 CRITIC: 5 părți nu încap sub 90 MB (high, §4.13)

- **Constatare:**
  - ZIP-urile originale au 93,2–94,1 MB. Part4 are 94 108 721 B = 89,75 MiB, iar tile-urile sunt deja „Stored” (0% compresie).
  - Exemplul oficial are 2 tile-uri și 750 de obiecte. `annotations.xml` are 260 KB necomprimat (~52 KB deflate), adică ~26 KB comprimat pe tile.
  - Pentru 71–78 de tile-uri pe parte se adaugă ~2 MB, deci ~96 MB pe parte: peste 90 MB și peste 90 MiB (94 371 840 B).
- **Recomandare (CHANGE):**
  - Împărțim tile-urile în 7 părți de ~45 de tile-uri (≈ 60 MB + XML), fiecare cu un `annotations.xml` doar pentru tile-urile ei.
  - Tile-urile rămân „stored” (`zip -0`), iar `annotations.xml` se comprimă deflate -9 (de ex. `zip -9 partK.zip annotations.xml && zip -0 -r partK.zip images/`).
  - Validatorul impune o limită strictă de ≤ 85 000 000 B pe ZIP (marjă față de ambiguitatea MB/MiB).
  - Checklist-ul de Publish verifică suma fișierelor = 311, nu „5 părți”.
- **Impact:** pasul 4 din §4.13 („5 părți, identice cu împărțirea originală … < 90 MB”) nu poate trece verificarea pentru părțile 1–4, iar pasul 6 („se urcă toate 5 părțile”) trebuie adaptat.
- **Surse (locale):**
  - `/Users/maleticimiroslav/Vin Gigahack/data & info/01_tiles/*.zip` (`ls -la`, `unzip -v`), reverificat la integrare;
  - `05_examples/siret3_examples_cvat.zip` (`zipinfo`);
  - `Marcaj_quick_start_for_teams.txt` §2 („under the 90 MB upload limit”).

### 6.2 Structura ZIP-ului de upload (high, §4.13)

- **Constatare:**
  - Structura ZIP-ului de upload diferă de cea a ZIP-urilor furnizate. Părțile originale au tile-urile direct în rădăcină, fără folder, pe când upload-ul cere `annotations.xml` + `images/siret3_rXXX_cYYY.tif`.
  - În exemplul oficial, `<image name>` e DOAR basename-ul („siret3_r021_c012.tif”, fără „images/”), cu `id` secvențial 0..n-1 și width/height = 2048.
- **Recomandare (KEEP/ADD):**
  - Packer-ul pune tile-urile în `images/`, cu numele originale și bytes identici; validatorul compară sha256 cu sursa.
  - În XML: `name=basename` și `id` = 0..n-1 per ZIP, în ordinea sortată a numelor.
  - Validatorul verifică bijecția: set(nume `<image>`) == set(`images/*.tif`) din același ZIP. Reuniunea tuturor ZIP-urilor = cele 311 fișiere, fără duplicate între părți.
- **Impact:** writer-ul și packer-ul trebuie să producă exact acest layout, altfel tile-urile sunt „skipped” sau nu se potrivesc.
- **Surse:** local `05_examples/siret3_examples_cvat/annotations.xml`, `Vineyard_AI_annotation_rules.txt` Appendix A, https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/

### 6.3 Potrivirea frame-urilor la import (high, §4.13)

- **Constatare:**
  - În CVAT upstream (loader-ul `load_anno` din `cvat/apps/dataset_manager/formats/cvat.py`), frame-ul se potrivește după `osp.splitext(name)[0]`, cu sau fără extensie și cu potrivire și pe rădăcina căii.
  - Dacă numele NU se potrivește, CVAT face fallback la atributul `id` al `<image>`, folosit ca număr de frame (`match_dm_item`: `item.attributes.get("frame", item.id)`).
  - Eroarea apare doar dacă nici acel id nu există.
- **Recomandare (ADD):**
  - Validatorul cere potrivirea exactă, case-sensitive, a numelui cu fișierul din ZIP: fără spații, Unicode NFC, extensia exact `.tif`. Nu ne bazăm pe `id`.
  - La testul de vineri noaptea verificăm în raport numărul de frame-uri și de obiecte, apoi deschidem 1–2 tile-uri în editor și confirmăm că poligoanele stau pe imaginea corectă.
- **Impact:** un nume greșit poate atașa în tăcere adnotările altui tile, cel al cărui frame = id. Marcaj e un produs derivat și poate avea alt matcher, dar riscul trebuie eliminat din sursă.
- **Surse:** https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/bindings.py (`match_dm_item`, l. ~2352), https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/formats/cvat.py (`load_anno`)

### 6.4 Schema elementelor (high, §4.13)

- **Constatare:** schema pe elemente, cum o scrie exportatorul CVAT:
  - `<polygon|polyline label source occluded points z_order [group_id]>`, cu `points="x,y;x,y"`;
  - `<box label source occluded xtl ytl xbr ybr z_order>`; eticheta `waste` are tipul `rectangle`, deci se scrie ca `<box>`;
  - `<tag label source>`;
  - atributele sunt copii `<attribute name="...">valoare</attribute>`.

  La import:
  - loader-ul citește `occluded` și `z_order` cu default 0;
  - `source` din fișier e IGNORAT și suprascris mereu cu `"file"`. Valorile permise pentru `source` în DB sunt auto, semi-auto, manual, file, consensus.

  Exemplele:
  - exemplul oficial folosește `source="manual"` și coordonate cu 1 zecimală;
  - exemplul din regulament omite `source` și `z_order`, deci ambele sunt opționale.
- **Recomandare:**
  - KEEP: writer-ul scrie atributele în ordinea `label, source, occluded, points|xtl..ybr, z_order`, ca în exemplu.
  - Pentru `source` folosim `"auto"`. Putem folosi și `"manual"`, ca în exemplu, dacă vrem diff zero la round-trip; pentru import nu contează.
  - Coordonatele cu 1 zecimală sunt OK.
  - Waste: `<box label="waste" source="auto" occluded="0" xtl=.. ytl=.. xbr=.. ybr=.. z_order="0"><attribute name="vineyard_id">V01</attribute></box>`, cu `xtl<xbr` și `ytl<ybr`.
  - ADD: ZIP-ul de test de vineri conține exemplul plus un tile cu cel puțin un `<box>` waste, ca să validăm și acest tip.
  - Toate valorile de atribut se escapează XML (`xml.sax.saxutils` sau lxml).
- **Impact:** planul cere `source="auto"`, care e valid, dar irelevant pentru CVAT upstream (devine `"file"`). Exemplul nu conține niciun `<box>` (waste), deci formatul box nu e testat de exemplu.
- **Surse:** https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/formats/cvat.py (`dump_as_cvat_annotation`, `load_anno`: `shape["source"]="file"`), https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/engine/models.py (SourceType: auto, semi-auto, manual, file, consensus), https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/

### 6.5 Ce NU verifică importul CVAT (high, §4.13)

- **Constatare:** pe calea de import, CVAT upstream NU aplică validările din serializer-ul API:
  - numărul de puncte (polygon ≥ 6 valori, polyline ≥ 4, box = 4) nu se verifică; `_validate_input_annotations` verifică doar tracks/intervals;
  - valorile atributelor `select` nu se verifică față de lista permisă (există conversie doar pentru checkbox);
  - atributele cu nume necunoscut pentru etichetă sunt ARUNCATE în tăcere;
  - o etichetă necunoscută ridică `ValueError` („Label … is not registered”), care poate pica tot importul.
- **Recomandare (ADD la validator, blocant):**
  - (a) etichete ∈ {vineyard, waste, row, interrow_area}, cu tipul corect per etichetă (polygon / box / polyline / polygon);
  - (b) numele atributelor sunt exact cele din `<meta>` pentru eticheta respectivă, fără atribute în plus;
  - (c) `row_structure` ∈ {regular, disrupted, unassessable} și `interrow_cover` ∈ {bare_soil, vegetation, mixed, unassessable}, fără spații;
  - (d) polygon cu ≥ 3 vârfuri distincte și aria ≥ 0,01 m² (16 px²); polyline cu ≥ 2 puncte distincte și lungimea ≥ 0,1 m (4 px);
  - (e) `row_structure` și `interrow_cover` nu sunt goale;
  - `vineyard_id` gol e permis doar pentru waste aflat la peste 10 m de un bloc.
- **Impact:** un poligon degenerat sau o valoare „Regular” poate intra în proiect și apoi e notat greșit sau strică editorul. Un typo în numele atributului dispare fără eroare. Validatorul nostru e singura plasă de siguranță.
- **Surse:** https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/engine/serializers.py (`bad_num_points_unless`, l. ~4031), https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/task.py (`_validate_input_annotations`), https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/bindings.py (`_import_shape`, `_import_attribute`, `_get_label_id`)

### 6.6 Geometria nu e reparată la import (medium, §4.13)

- **Constatare:** la import, CVAT nu clipează și nu repară coordonatele din afara imaginii sau poligoanele care se auto-intersectează: stochează ce primește. Clipping-ul la marginile imaginii e un comportament al UI-ului la desenare. Editorul poate afișa sau tăia vizual, dar scorarea convertește în metri geometria stocată.
- **Recomandare (ADD în writer, înainte de serializare):**
  - clip la dreptunghiul [0,2048]×[0,2048] (shapely `intersection(box(0,0,2048,2048))`), apoi `make_valid`/`buffer(0)`;
  - MultiPolygon se sparge în poligoane separate cu aceleași atribute, iar componentele < 16 px² se elimină;
  - `simplify(0.5 px, preserve_topology=True)`, care reduce și dimensiunea XML-ului;
  - se elimină vârful de închidere duplicat (CVAT închide implicit poligonul);
  - polyline-urile se clipează cu aceeași cutie.

  Validatorul verifică `is_valid` și `0 ≤ x,y ≤ 2048`.
- **Impact:** poligoanele invalide (bowtie) produc arii greșite sau erori în scorare și în `from-marcaj`. Coordonatele < 0 sau > 2048 ies din tile.
- **Surse:** https://github.com/openvinotoolkit/cvat/issues/2992, https://github.com/cvat-ai/cvat/issues/2990, https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/formats/cvat.py (`load_anno` parsează `points` fără validare)

### 6.7 „No objects in this frame” (medium, §4.13 pasul 7)

- **Constatare:**
  - „No objects in this frame” nu există în CVAT upstream. Acolo, un frame gol e doar un `<image>` fără copii, iar exportatorul îl include cu `include_empty=True`.
  - E o funcție Marcaj, iar documentația Marcaj spune doar că se bifează manual în editor. Nu există un mecanism documentat prin care importul să seteze acest flag.
  - Un `<tag>` ar cere o etichetă de tip tag, care nu e în proiect, iar editarea etichetelor e interzisă.
- **Recomandare:**
  - KEEP: scriem `<image .../>` gol pentru tile-urile fără obiecte (nu le omitem din XML, ca raportul să arate numărul corect de frame-uri) și NU adăugăm `<tag>`.
  - ADD: pipeline-ul exportă `empty_tiles.csv`, iar tracker-ul de corectură are o coloană „No objects bifat”.
  - La testul de vineri verificăm în editor dacă un tile cu `<image/>` gol apare deja cu flag-ul bifat (improbabil).
- **Impact:** tile-urile fără vie nu pot fi „închise” prin pre-adnotare. Fiecare trebuie bifat manual, altfel job-ul nu se poate trimite (Submit).
- **Surse:** `Marcaj_quick_start_for_teams.txt` §5 (bifare manuală „No objects in this frame”), `Vineyard_AI_annotation_rules.txt` §1, https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/formats/cvat.py (`group_by_frame(include_empty=True)`), https://docs.cvat.ai/docs/annotation/manual-annotation/modes/annotation-with-tags/

### 6.8 Exportul din Marcaj (medium, §4.13 pașii 3 și 8)

- **Constatare:** exportul „CVAT for images 1.1” la nivel de proiect diferă de fișierul nostru:
  - adaugă pe `<image>` atributele `subset` și `task_id`, în ordinea id, name, subset, task_id, width, height;
  - coordonatele se scriu cu 2 zecimale (`{:.2f}`);
  - `source` e `"file"` pentru obiectele importate și `"manual"` pentru cele editate;
  - `group_id` apare doar dacă e nenul;
  - `id` e frame-ul global al proiectului, nu indicele nostru;
  - numele poate include o cale, în funcție de modul în care Marcaj a stocat fișierele.
- **Recomandare (CHANGE în from-marcaj):**
  - potrivirea se face după `basename(name)`, nu după id;
  - se ignoră atributele necunoscute (`subset`, `task_id`, `group_id`, `source`);
  - se acceptă `<box>`, `<polygon>`, `<polyline>`, `<tag>` și `<image>` gol, iar coordonatele se citesc ca float;
  - pe exportul real se verifică: 311 `<image>` unice și zero etichete sau valori în afara listei;
  - round-trip-ul se compară geometric (IoU ≥ 0,999, Hausdorff < 0,1 px) și pe atribute, nu textual.
- **Impact:** parser-ul `vineyard from-marcaj` nu trebuie să presupună structura exactă a exemplului, iar un round-trip „identic” nu e posibil pe exportul real.
- **Surse:** https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/formats/cvat.py (`dump_as_cvat_annotation`, `_export_images`), `Marcaj_quick_start_for_teams.txt` §6 (Export json_simple / CVAT XML)

### 6.9 Conținutul strict al ZIP-ului (high, §4.13 pașii 4–6)

- **Constatare:**
  - Importul CVAT dintr-un ZIP extrage arhiva și încarcă TOATE fișierele `**/*.xml` găsite recursiv (pentru task/job). Pentru proiect folosește Datumaro `import_from("cvat")`.
  - O etichetă inexistentă ridică excepție („Label … is not registered”).
  - Marcaj raportează separat „skipped files / dropped objects”, iar mesajul „class added from the label dictionary” e normal.
- **Recomandare (ADD la packer):**
  - ZIP-ul conține EXACT `annotations.xml` + `images/*.tif`, fără `__MACOSX/` sau `.DS_Store`. Se creează cu `zip -X` și/sau cu `zipfile` din Python, nu din Finder.
  - Validatorul listează conținutul ZIP-ului și respinge orice alt fișier.
  - Upload-ul se face secvențial, cu raportul fiecărei părți salvat (captură) în tracker.
- **Impact:** orice fișier XML în plus din ZIP (de exemplu un XML de debug; `.DS_Store` nu contează aici) ar fi citit ca adnotări. O etichetă greșită poate compromite toată partea.
- **Surse:** https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/formats/cvat.py (`_import`: `glob('**/*.xml', recursive=True)`), `Marcaj_quick_start_for_teams.txt` §3

### 6.10 Blocul `<meta>` (high, §4.13 pasul 1)

- **Constatare:**
  - Blocul `<meta>` din exemplu include `<task><name>`. Fiecare atribut are `mutable=False`, `input_type`, `default_value` (regular / bare_soil) și `values` separate prin newline.
  - Ordinea atributelor pe obiecte: vineyard_id, row_id, row_structure (pentru row) și vineyard_id, interrow_cover (pentru interrow_area).
  - La import, etichetele se leagă de proiect după nume, iar meta-ul din fișier nu modifică etichetele proiectului.
- **Recomandare (KEEP):**
  - Copiem byte-cu-byte blocul `<meta>` din exemplu (cu `<version>1.1</version>`) și scriem atributele în ordinea de mai sus.
  - Toate atributele sunt prezente pe fiecare obiect. `row_id` gol NU e permis pentru row; `vineyard_id` gol e permis doar pentru waste îndepărtat.
- **Impact:** confirmă pasul 1 din §4.13 (copiem exact meta-ul din exemplu).
- **Surse:** local `05_examples/siret3_examples_cvat/annotations.xml` (head), `Vineyard_AI_annotation_rules.txt` Appendix A

### 6.11 Schimbări propuse (CVAT, Marcaj)

- **§4.13 pasul 4:** „5 părți, identice cu împărțirea originală” devine „7 părți de ~45 de tile-uri (în ordinea sortată după nume, cu vecinii împreună), fiecare ZIP ≤ 85 000 000 B”. Motivul: părțile originale au deja 93–94 MB, iar XML-ul adaugă ~2 MB pe parte.
- **§4.13 pasul 4:** layout-ul ZIP e `annotations.xml` (deflate -9) + `images/<nume original>.tif` (stored, sha256 identic cu sursa). Nimic altceva: fără `__MACOSX`, `.DS_Store` sau alte `.xml`.
- **§4.13 pasul 1:**
  - `<image id=0..n-1 per ZIP name=basename width=2048 height=2048>`;
  - tile-urile fără obiecte apar ca `<image .../>` gol;
  - waste se scrie ca `<box ... xtl ytl xbr ybr z_order="0">`;
  - `source` poate fi auto sau manual, pentru că CVAT îl suprascrie oricum cu `"file"`.
- **§4.13 pasul 1:** în writer se adaugă clip la [0,2048]², `make_valid`, explode MultiPolygon, eliminarea bucăților < 16 px², simplify 0,5 px și eliminarea vârfului de închidere duplicat.
- **§4.13 pasul 2 (validator):**
  - bijecție nume XML ↔ `images/` per ZIP, cu 311 nume unice pe total;
  - tip geometric corect per etichetă;
  - set exact de atribute per etichetă, cu valorile select din listă;
  - polygon cu ≥ 3 vârfuri distincte și `is_valid`; polyline cu ≥ 2 puncte distincte; box cu `xtl<xbr` și `ytl<ybr`;
  - coordonate în [0,2048];
  - dimensiunea ZIP ≤ 85 MB, iar conținutul ZIP strict pe whitelist.
- **§4.13 pasul 3:** round-trip-ul se compară geometric și pe atribute, nu textual, pentru că exportul CVAT are 2 zecimale, `subset`/`task_id` și `source=file`.
- **§4.13 pasul 5:** ZIP-ul de test de vineri conține exemplul, cel puțin un `<box>` waste și un tile cu `<image/>` gol. Verificăm raportul (frame-uri și obiecte), poziția poligoanelor în editor și starea „No objects”.
- **§4.13 pasul 7:** se adaugă `empty_tiles.csv` și coloana „No objects bifat” în tracker. Flag-ul nu poate fi setat prin import.
- **§4.13 pasul 8:** `from-marcaj` potrivește după `basename(name)`, ignoră `subset`/`task_id`/`group_id`/`source`, acceptă `<tag>` și `<image>` goale și verifică 311 imagini unice.

### 6.12 Riscuri (CVAT, Marcaj)

- Marcaj e un fork sau produs derivat din CVAT. Matcher-ul de nume, validarea geometriei și limita de „90 MB” (MB sau MiB) pot diferi de upstream și se confirmă doar prin testul de vineri.
- Nu știm dacă Marcaj respinge poligoanele invalide, le raportează ca „dropped” sau le acceptă tacit. Validatorul nostru trebuie să fie strict indiferent de asta.
- Estimarea de ~26 KB comprimat pe tile vine din exemplu (375 de obiecte pe tile, cu canopy detaliat). Dacă modelul produce mai multe vârfuri, XML-ul crește. Măsurăm dimensiunea reală înainte de a fixa numărul de părți.
- Starea „No objects in this frame” nu poate fi pre-setată, deci tile-urile fără vie trebuie bifate manual înainte de Submit.

---

## 7. Stack-ul Python geospațial/ML pe Apple M3 Pro și Docker reproductibil (plan §4.1, §4.15)

### 7.1 Python 3.12 în loc de 3.11 (high, §4.1)

- **Constatare:**
  - Planul cere Python 3.11, dar versiunile curente ale pachetelor cheie cer Python >= 3.12. Pe PyPI: rasterio 1.5.1 (08.2026), pyproj 3.8.0, numpy 2.5.3 și scipy 1.18.1 au toate `requires_python >=3.12`.
  - Pe 3.11 am rămâne la rasterio 1.4.4 (ultimul cu >=3.10), pyproj 3.7 și numpy 2.4.
  - Wheel-uri cp312 pentru macosx arm64, manylinux aarch64 și manylinux x86_64 există pentru: shapely 2.1.2, scikit-image 0.26.0, ortools 9.15.6755, torch 2.14.0 și torchvision 0.29.0.
  - pyogrio 0.13.0 și opencv au wheel-uri abi3. geopandas 1.1.4, segmentation-models-pytorch 0.5.0 și timm sunt pure-python.
- **Recomandare (SCHIMBĂ):**
  - Python 3.12 peste tot: venv și `python:3.12-slim`.
  - Nu folosim 3.13 sau 3.14: câștigul e nul, iar riscul de wheel-uri lipsă e mai mare.
- **Impact:** pe 3.11 stack-ul ar fi fixat pe versiuni mai vechi, iar `python:3.11-slim` din Docker trebuie schimbat și el. Pe Mac există deja `/opt/homebrew/bin/python3.12` și uv 0.11.1.
- **Surse:** https://pypi.org/pypi/rasterio/json, https://pypi.org/pypi/pyproj/json, https://pypi.org/pypi/numpy/json, https://pypi.org/pypi/torch/json

### 7.2 Setul de versiuni fixate (medium, §4.1)

- **Constatare:** set fixat, compatibil pe macOS arm64 și pe Linux arm64/amd64, Python 3.12: `rasterio==1.5.1`, `shapely==2.1.2`, `geopandas==1.1.4`, `pyproj==3.8.0`, `pyogrio==0.13.0`, `opencv-python-headless==4.14.0.94`, `scikit-image==0.26.0`, `scipy==1.18.1`, `ortools==9.15.6755`, `torch==2.14.0`, `torchvision==0.29.0`, `segmentation-models-pytorch==0.5.0`, plus typer, pyyaml și lxml.
  - Restul îl rezolvă lock-ul: numpy 2.x, timm, huggingface-hub, safetensors, pandas.
  - opencv 5.0.0.93 a ieșit în 07.2026 și e un major nou. Linia 4.x continuă, iar ultima versiune e 4.14.0.94.
- **Recomandare (ADAUGĂ):**
  - Setul intră ca limite `==` în pyproject, iar `uv lock` rezolvă tranzitivele. OpenCV rămâne pe 4.x.
  - Weights-urile encoderului smp vin de pe HuggingFace, deci le salvăm în `weights/` și setăm `encoder_weights=None` la inferență, ca Docker să meargă offline.
  - Verificare rapidă după lock: `uv run python -c "import rasterio,pyogrio,pyproj,geopandas,cv2,torch;print(rasterio.__gdal_version__,pyproj.proj_version_str,torch.backends.mps.is_available())"`.
- **Impact:** planul spune doar „totul fixat în requirements.lock”, fără versiuni. Un major nou de OpenCV în mijlocul hackathonului e un risc inutil.
- **Surse:** https://pypi.org/pypi/opencv-python-headless/json, https://pypi.org/pypi/segmentation-models-pytorch/json, https://pypi.org/pypi/geopandas/json

### 7.3 uv în loc de conda (high, §4.1)

- **Constatare:**
  - uv e instalat local (0.11.1) și are suport documentat pentru torch CPU-only pe Linux, cu fallback pe PyPI (MPS) pe macOS.
  - Mecanismul: `[tool.uv.sources] torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]` plus `[[tool.uv.index]] name="pytorch-cpu" url="https://download.pytorch.org/whl/cpu" explicit=true`.
  - Pe aceeași mașină există și `/opt/anaconda3` cu python3.11. Documentația rasterio avertizează că wheel-urile nu sunt testate împreună cu pachete conda.
- **Recomandare (SCHIMBĂ):** doar uv, fără conda.
  - Comenzi, în ordine:
    1. `conda deactivate` (sau `unset PROJ_DATA PROJ_LIB GDAL_DATA`)
    2. `uv python pin 3.12`
    3. `uv add <pachetele de mai sus>`
    4. `uv lock`
    5. `uv sync --locked`
    6. `uv export --format requirements-txt --no-hashes -o requirements.lock`
  - Pentru README se comit `uv.lock` și `requirements.lock`.
  - În pyproject intră blocul `tool.uv.sources`/`tool.uv.index` pentru torch și torchvision.
- **Impact:** planul lasă alegerea „venv/conda” deschisă. Un conda base activ, care exportă `PROJ_DATA`/`GDAL_DATA`, poate strica pyproj și rasterio instalate din wheel-uri.
- **Surse:** https://docs.astral.sh/uv/guides/integration/pytorch/, https://rasterio.readthedocs.io/en/stable/installation.html

### 7.4 torch CPU în Docker (high, §4.15)

- **Constatare:**
  - Pe Linux, torch 2.14.0 de pe PyPI declară dependențe CUDA pentru orice arhitectură (`platform_system == "Linux"`): cuda-toolkit 13.0.3, nvidia-cudnn-cu13, nccl, nvshmem și triton.
  - Wheel-ul singur are 555 MB pe x86_64 și 454 MB pe aarch64; cu CUDA, totalul trece de câțiva GB.
  - Wheel-ul `+cpu` de pe `download.pytorch.org/whl/cpu` are 196 MB pe x86_64 și 159 MB pe aarch64.
- **Recomandare (ADAUGĂ):**
  - Sursele uv cu marker linux de mai sus. Alternativ, în Dockerfile, `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0` înaintea restului.
  - Verificare în container: `python -c "import torch;print(torch.__version__)"` trebuie să afișeze `2.14.0+cpu`.
  - Imaginea finală estimată: ~1,2–1,6 GB.
- **Impact:** „torch CPU” în Docker nu se obține automat. Cu un simplu `pip install -r requirements.lock` imaginea trece de 5 GB, iar build-ul durează foarte mult.
- **Surse:** https://pypi.org/pypi/torch/2.14.0/json, https://download.pytorch.org/whl/cpu/torch/, https://docs.astral.sh/uv/guides/integration/pytorch/

### 7.5 Dockerfile în două etape (high, §4.15)

- **Constatare:** pattern-ul Docker recomandat de uv are două etape.
  - Etapa builder: `python:3.12-slim` cu `COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/`, `UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy` și `uv sync --locked --no-install-project` cu cache mount. Apoi `COPY . /app` și `uv sync --locked`.
  - Etapa finală copiază doar `/app/.venv` și setează `PATH=/app/.venv/bin:$PATH`.
  - Toate wheel-urile din set există pentru manylinux aarch64 și x86_64, deci nu e nevoie de apt pentru GDAL, PROJ sau build-essential.
- **Recomandare (ADAUGĂ):**
  - Dockerfile-ul de mai sus, cu imaginea uv fixată la o versiune (de ex. `ghcr.io/astral-sh/uv:0.11.1`), nu `latest`.
  - Se adaugă `ENV OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1` și `CMD ["make","final"]`. Dacă Makefile-ul nu e necesar în runtime, entrypoint-ul devine `vineyard all`.
  - Tile-urile se montează ca volum (`-v $PWD/data:/app/data:ro`) și nu intră în imagine.
  - `.dockerignore` exclude `data/`, `work/`, `.venv/` și `*.tif`.
- **Impact:** planul spune doar „dependențe din lock”. Acest pattern dă cache de layere și o imagine fără uv și fără tool-uri de build.
- **Surse:** https://docs.astral.sh/uv/guides/integration/docker/

### 7.6 arm64 vs. amd64 pe Apple Silicon (medium, §4.15)

- **Constatare:**
  - Pe Apple Silicon, un build linux/arm64 rulează nativ. linux/amd64 rulează emulat: cu Rosetta e ~20% mai lent, cu QEMU ~85% mai lent.
  - Există bug-uri raportate de build amd64 foarte lent cu Rosetta.
  - În Docker pe Mac nu există MPS, deci NN-ul rulează pe CPU.
- **Recomandare (ADAUGĂ în Makefile):**
  - `docker-build`: `docker buildx build --platform linux/arm64 -t vineyard:arm64 --load .`, pentru test local rapid.
  - Build-ul multi-arch final: `docker buildx build --platform linux/amd64,linux/arm64 -t <user>/vineyard:1.0 --push .`, lansat o singură dată, sâmbătă noaptea, fiindcă amd64 e lent.
  - În README raportăm separat: (a) nativ macOS + MPS; (b) Docker arm64 CPU, cu limitele VM-ului din Docker Desktop (CPU/RAM setate, de ex. 8 CPU / 10 GB).
  - Nu raportăm timpi din amd64 emulat.
- **Impact:** juriul poate cere o re-rulare. Imaginea arm64 nu pornește pe un PC x86 al juriului fără emulare, iar timpii din Docker pe Mac nu sunt comparabili cu cei nativi pe MPS.
- **Surse:** https://oneuptime.com/blog/post/2026-01-16-docker-mac-apple-silicon/view, https://github.com/docker/for-mac/issues/7075, https://www.docker.com/blog/docker-desktop-4-25/

### 7.7 Benchmark și hardware (high, §4.1)

- **Constatare:**
  - M3 Pro 18 GB are 11 nuclee CPU: 5 de performanță și 6 de eficiență (`sysctl hw.perflevel0.physicalcpu=5`, `hw.perflevel1.physicalcpu=6`). Planul presupune 10 nuclee egale.
  - Regulamentul spune că performanța se judecă „from the processing time and hardware stated in the README” și că juriul poate cere o re-rulare.
- **Recomandare (SCHIMBĂ `vineyard benchmark`):**
  - `time.perf_counter()` per etapă și total, pe 3 rulări, cu raportarea medianei și a minimului.
  - Cold (primul run, cu citirea de pe disc și încărcarea modelului) separat de warm.
  - Pentru NN se apelează `torch.mps.synchronize()` înainte de oprirea cronometrului.
  - Un sweep scurt `--workers 5,8,10` pe 30 de tile-uri, din care se alege optimul.
  - `timing.json` conține:
    - hardware: `sysctl -n machdep.cpu.brand_string`, nucleele P/E, `hw.memsize`, `sw_vers -productVersion`;
    - software: versiunile Python, torch, rasterio și GDAL, plus device-ul (mps/cpu);
    - rulare: n_workers, n_tiles=311, wall-clock total, s/tile per etapă și peak RSS (`/usr/bin/time -l make final`).
  - În README: un tabel per etapă și o singură cifră end-to-end, cu condiția „laptop pe alimentare, fără alte aplicații grele”.
- **Impact:** `Pool(cores-2)` = 9 procese, din care cel puțin 4 pe E-cores, deci scalarea nu e liniară. Estimarea de 5–10 min trebuie măsurată, nu presupusă.
- **Surse:** `/Users/maleticimiroslav/Vin Gigahack/data & info` (`Vineyard_AI_Field_Challenge_description.txt`, criteriul Engineering quality), https://docs.pytorch.org/docs/2.14/mps_environment_variables.html

### 7.8 Multiprocessing: spawn vs. fork (high, §4.1)

- **Constatare:**
  - Pe macOS, start method-ul implicit e `spawn` (din Python 3.8), iar `fork` e considerat nesigur.
  - Pe Linux cu 3.12 implicitul e încă `fork`; din 3.14 devine `forkserver`.
  - Cu spawn, funcțiile trebuie să fie la nivel de modul și picklable, iar codul de pornire trebuie protejat cu `if __name__ == '__main__'`.
  - Lipsa guard-ului produce eroarea „bootstrapping phase”, inclusiv în DataLoader cu `num_workers>0` (pytorch #162612, M4 Pro, torch 2.8).
- **Recomandare (ADAUGĂ):** un singur helper `parallel_map(fn, items, workers)`.
  - Folosește `multiprocessing.get_context('spawn').Pool(workers, initializer=_init_worker, maxtasksperchild=50)` și `imap_unordered(fn, tile_paths, chunksize=2)`.
  - Workerii primesc doar căi și config, nu array-uri, handle-uri rasterio sau GeoDataFrame-uri. `rasterio.open` se apelează în worker.
  - Funcțiile sunt top-level, în modulele `vineyard.stages.*`.
  - Entry point-ul typer e deja protejat prin console script, dar orice script din `scripts/` are nevoie de guard.
- **Impact:** fără asta, același cod ar rula cu fork în Docker și cu spawn pe Mac. Asta înseamnă comportamente și timpi diferiți, plus bug-uri care apar doar pe o platformă.
- **Surse:** https://docs.python.org/3/library/multiprocessing.html, https://github.com/pytorch/pytorch/issues/162612

### 7.9 Oversubscription de thread-uri (medium, §4.1)

- **Constatare:** OpenCV, BLAS și OpenMP pornesc fiecare câte un pool de thread-uri per proces. Cu N procese × M thread-uri apare oversubscription, iar setarea doar a `OMP_NUM_THREADS` nu limitează pool-ul intern al OpenCV. Practica standard e `cv2.setNumThreads(0 sau 1)` în initializer-ul worker-ului, plus `OMP_NUM_THREADS=1`.
- **Recomandare (ADAUGĂ):**
  - `_init_worker()` apelează `cv2.setNumThreads(1)`, `cv2.ocl.setUseOpenCL(False)` și, dacă torch e importat, `torch.set_num_threads(1)`.
  - Variabilele de mediu se setează în CLI înaintea importului numpy/cv2, ca procesele spawn să le moștenească: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 GDAL_NUM_THREADS=1 GDAL_CACHEMAX=256`.
  - Aceleași variabile intră și în Dockerfile.
- **Impact:** cu 9 procese × 11 thread-uri pe un M3 Pro, etapele 2, 3, 6, 8, 9 și 10 pot fi mai lente decât cu mai puține procese.
- **Surse:** https://github.com/opencv/opencv/issues/15277, https://medium.com/@rachittayal7/a-note-on-opencv-threads-performance-in-prod-d10180716fba

### 7.10 MPS într-un singur proces (medium, §4.1)

- **Constatare:**
  - MPS se folosește dintr-un singur proces.
  - Dacă un op nu e implementat pe MPS, execuția pică. `PYTORCH_ENABLE_MPS_FALLBACK=1` îl trimite pe CPU.
  - `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (implicit 1.7) limitează memoria alocatorului Metal. Pe 18 GB aceasta e memorie unificată, împărțită cu cele 9 procese CPU.
- **Recomandare (ADAUGĂ):**
  - Inferența NN e o etapă separată, rulată în procesul principal pe `device='mps'` (cu fallback `cpu`).
  - DataLoader cu `num_workers=0` sau 2 (cu `persistent_workers=True` și guard), batch 4–8 × 512², `torch.inference_mode()`.
  - NN-ul nu rulează concurent cu Pool-ul CPU.
  - Se setează `PYTORCH_ENABLE_MPS_FALLBACK=1` și, ca plasă de siguranță, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8`.
  - Workerii CPU nu importă deloc torch.
- **Impact:** dacă inferența NN rulează în paralel cu Pool-ul CPU, sau în workeri spawn care inițializează fiecare MPS, apar contenție pe memoria unificată și timpi instabili.
- **Surse:** https://docs.pytorch.org/docs/2.14/mps_environment_variables.html

### 7.11 Schimbări propuse (stack, Docker)

- **§4.1 Mediu:** „Python 3.11 în venv/conda” devine „Python 3.12 cu uv, fără conda (`conda deactivate`, `unset PROJ_DATA PROJ_LIB GDAL_DATA`), cu `uv.lock` + `requirements.lock` exportat”.
- **§4.1:** lista fixată: `rasterio==1.5.1`, `shapely==2.1.2`, `geopandas==1.1.4`, `pyproj==3.8.0`, `pyogrio==0.13.0`, `opencv-python-headless==4.14.0.94` (nu 5.x), `scikit-image==0.26.0`, `scipy==1.18.1`, `ortools==9.15.6755`, `torch==2.14.0`, `torchvision==0.29.0`, `segmentation-models-pytorch==0.5.0`, plus typer, pyyaml și lxml. Restul îl rezolvă `uv lock`.
- **§4.1:** în pyproject intră `[tool.uv.sources]` pentru torch/torchvision → index `pytorch-cpu` cu marker `sys_platform == 'linux'`, plus `[[tool.uv.index]] url=https://download.pytorch.org/whl/cpu explicit=true`.
- **§4.1 Paralelism:**
  - `multiprocessing.Pool(n=cores−2)` devine `get_context('spawn').Pool(workers, initializer=_init_worker, maxtasksperchild=50)` + `imap_unordered(chunksize=2)`;
  - workerii primesc doar căi, iar `_init_worker` setează `cv2.setNumThreads(1)`;
  - variabile de mediu: `OMP`/`OPENBLAS`/`VECLIB`/`GDAL_NUM_THREADS=1` și `GDAL_CACHEMAX=256`;
  - numărul de workeri se alege printr-un sweep 5/8/10 (M3 Pro = 5P + 6E = 11 nuclee, nu 10).
- **§4.1 NN:** rulează pe MPS doar în procesul principal, ca etapă separată, nu concurent cu Pool-ul. `PYTORCH_ENABLE_MPS_FALLBACK=1`, `torch.mps.synchronize()` la cronometrare, weights smp salvate local (`encoder_weights=None` la inferență).
- **§4.1 Benchmark:** `timing.json` cu `perf_counter` per etapă, 3 rulări (mediană/min), cold vs. warm, s/tile, peak RSS (`/usr/bin/time -l`), hardware (`machdep.cpu.brand_string`, nuclee P/E, `hw.memsize`, macOS), versiuni (python, torch, `rasterio.__gdal_version__`), device și n_workers.
- **§4.15 Docker:** `python:3.12-slim`, Dockerfile uv în două etape (uv fixat la 0.11.1, `uv sync --locked --no-install-project` cu cache mount, se copiază doar `.venv`), torch `+cpu` verificat, `.dockerignore` pentru `data/`, `work/`, `.venv/` și `*.tif`, tile-urile montate ca volum read-only.
- **§4.15 Makefile:** `docker-build` (arm64 local, `--load`) și `docker-push` (buildx `--platform linux/amd64,linux/arm64 --push`, rulat o singură dată).
- **§4.15 README:** două rânduri de timp: (a) macOS nativ, M3 Pro + MPS; (b) Docker arm64 CPU, cu limitele VM declarate. Nu raportăm timpi din amd64 emulat.

### 7.12 Riscuri (stack, Docker)

- `opencv 4.14.0.94` și combinația numpy 2.5 + scikit-image 0.26 nu au fost verificate printr-o rezolvare efectivă. Rulăm imediat `uv lock` și testul de import. Dacă nu se rezolvă, fixăm `numpy<2.5`.
- rasterio, pyogrio și pyproj vin fiecare cu propriile copii GDAL/PROJ. De obicei merg împreună, dar testul de import și un round-trip GPKG trebuie rulate și în Docker, și pe Mac.
- torch 2.14.0 a fost lansat acum doar 3 săptămâni. Dacă apar regresii pe MPS, revenim la torch 2.13.x / torchvision 0.28.x.
- Build-ul amd64 emulat pe Mac poate dura mult sau se poate bloca (docker/for-mac #7075). Îl pornim devreme sau construim amd64 în GitHub Actions.
- Limitele de memorie ale Docker Desktop pe 18 GB pot omorî workerii (OOM). În container setăm un `--workers` mai mic, de exemplu 6.
- Nu am verificat dacă smp/timm descarcă weights la import sau la construirea modelului. Testăm cu `docker run --network none` înainte de livrare.
