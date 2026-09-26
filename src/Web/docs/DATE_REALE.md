# Datele reale Sireț3 pe site (din ZIP-urile de upload Marcaj)

Cum ajunge pe laptopul web ce vede echipa AI, pornind de la cele 5 ZIP-uri `siret3_upload_0Nof05.zip`
(CVAT 1.1, adnotările modelului pentru cele 311 tile-uri). Nimic din ce urmează nu intră în git:
`src/AI/work/`, `src/Web/data/`, `frontend/public/data/` și arhivele cu tile-uri sunt ignorate.

Cea mai simplă cale rămâne copierea bundle-ului `src/Web/data/surveys/siret3/pipeline/` de la colegul AI.
Când ai doar ZIP-urile:

```bash
# 0. o singură dată: uv (brew install uv), apoi din src/AI
uv python install 3.12
uv sync --locked --extra route            # fără nn / waste-ml: nu sunt necesare aici
unset PROJ_DATA PROJ_LIB GDAL_DATA
uv run --no-sync vineyard doctor

# 1. tile-urile: ingest cere arhivele organizatorilor în data & info/01_tiles/
#    (dacă există doar folderele extrase, se refac fără compresie, cu .tif la rădăcină)
cd "../../data & info/01_tiles"
for i in 1 2 3 4 5; do d="siret3_challenge_tiles_part${i}of5"; (cd "$d" && zip -q -0 -j "../$d.zip" *.tif); done
cd - && uv run --no-sync vineyard run --until tile_prep --workers 8

# 2. ZIP-urile de upload → un singur annotations.xml → AnnSet
#    (`vineyard from-marcaj` nu e încă implementat; import-reference face aceeași conversie CVAT → AnnSet)
mkdir -p work/marcaj_upload/merged && mv /cale/siret3_upload_0*of05.zip work/marcaj_upload/
uv run --no-sync python - <<'PY'
import glob, zipfile
from lxml import etree
root, n = None, 0
for z in sorted(glob.glob("work/marcaj_upload/siret3_upload_0*of05.zip")):
    doc = etree.fromstring(zipfile.ZipFile(z).read("annotations.xml"))
    if root is None:
        root = etree.Element("annotations"); [root.append(c) for c in doc if c.tag != "image"]
    for img in doc.findall("image"):
        img.set("id", str(n)); n += 1; root.append(img)
etree.ElementTree(root).write("work/marcaj_upload/merged/annotations.xml", xml_declaration=True, encoding="utf-8")
print(n, "images")
PY
uv run --no-sync vineyard import-reference --set "paths.examples_dir=$PWD/work/marcaj_upload/merged" --run-id upload-reference-v1

# 3. post: derive → passable → targets → route (~7 min) → measure → web_bundle
uv run --no-sync vineyard post --annset upload-reference-v1
#    → src/Web/data/surveys/siret3/pipeline/. Manifestul iese cu stage "marcaj_corrected" (sursa e
#    tratată ca referință); ZIP-urile de upload sunt ieșirea modelului, deci se pune "stage": "model".

# 4. site-ul (din src/Web/frontend)
pnpm data:survey --survey siret3 --emit-seed   # → public/data/siret3 + supabase/seed/survey_siret3.sql
pnpm data:terrain --survey siret3              # relieful vederii 3D
# .env.local: NEXT_PUBLIC_SURVEY_ID=siret3, apoi repornește `pnpm dev` (variabila se citește la pornire)
```

Survey-ul `siret3` e înregistrat în Supabase (`public.survey`, seed `supabase/seed/survey_siret3.sql`), deci
conturile Sireți îl primesc prin `uat_survey`. Rezultat pe 26.09.2026: 54 blocuri, 743 rânduri, 48,14 km de
rânduri, 1,50 ha coroane (12.256 plante), 9,31 ha inter-rânduri, 1.330 ținte, ruta 15,65 km.
