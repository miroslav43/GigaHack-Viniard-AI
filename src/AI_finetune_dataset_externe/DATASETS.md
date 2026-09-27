# Datasets used by `fte` (external-data fine-tuning)

Nothing below is committed to the repository. Every dataset is fetched by a script with a
`manifest.json` (source URL, licence, sizes, sha256 of the extracted files) under `work/raw/` or
`work/data/`. Licences were checked on the record pages on 2026-09-26.

## Used

| Dataset | Licence | Files used | Used for | Fetch |
|---|---|---|---|---|
| **ICAERUS AI4Leafhopper – input dataset for the vine segmentation model**. Marengo I., Sirsat M. (2025). doi:[10.5281/zenodo.14605849](https://doi.org/10.5281/zenodo.14605849). ICAERUS project, EU grant 101060643. | **CC BY-NC 4.0** (non-commercial) | `AI_model/INPUT/YOLODataset/{images,labels}/{train,val}`: 85 RGBA crops at 1.4 cm/px, 1,093 hand-drawn per-vine polygons (YOLO-seg). About 28.5 MB, read by HTTP Range out of the 3.1 GB zip. | Canopy heads c0 (vine) and c1 (contact between touching plants) | `python -m fte.data.icaerus` |
| **DroneWaste v1.0**. Morandini et al. (2025), Politecnico di Milano / CERTH. doi:[10.5281/zenodo.17045559](https://doi.org/10.5281/zenodo.17045559) | CC BY 4.0 | `dronewaste_v1.0.json` and the selected members of `images.tar.gz` (streamed; the archive is never stored). Resampled 2 cm/px (site 6: 2.8) → 2.5 cm/px. | Waste positives: litter-like item classes. Negatives: soil/rubble/wood/vehicle piles and empty tiles. | `python -m fte.data.dronewaste` |
| **UAVVaste**. Kraft et al., *Remote Sensing* 13(5):965 (2021). doi:[10.5281/zenodo.8214061](https://doi.org/10.5281/zenodo.8214061) | CC BY 4.0 | 772 images, 3,718 `rubbish` polygons. Rescaled to 2.5 cm/px (median object assumed 0.25 m). | Waste positives: small items | Already under `src/AI/work/external/uavvaste` (see its `SOURCE.json`) |
| **Sireț3** orthomosaic tiles. 3DATA COLLECT / OpenAerialMap. | CC BY 4.0 | The 311 challenge tiles. Pseudo-labels are derived automatically from the classical pipeline's run `complete-v4` (no manual labels). | In-domain canopy pseudo-labels, non-vineyard negatives (orchards, grass, village), waste hard negatives (vine tubes, stakes, pale soil) | `src/AI` pipeline |

**Class mapping for DroneWaste to `waste`:**
- **Kept:** Plastic packaging, Plastic, Tyres, Textile, Metal barrels, Paper, Appliances, Electronic equipment, Furniture, and Pallets (capped).
- **Background (label 0):** Rubble, Construction and demolition materials, Asphalt milling, Excavation materials, Foundry, Wood, Vehicles, Asbestos. The annotation rules say stones, soil, pruning residue and vehicles are not waste.
- **Ignored (255):** Scrap, Mixed items.

**Licence note (ICAERUS, CC BY-NC 4.0).** We use this data only for non-commercial research within GigaHack 2026. It is not redistributed here; the script downloads it. The canopy weights trained with it (`fte-canopy`) are shared for non-commercial use only. The waste weights (`fte-waste`) use only CC BY 4.0 data.

## Evaluated and not used

| Dataset | Why not |
|---|---|
| Riseholme multi-temporal UAV vineyard (doi:10.5281/zenodo.19234907, CC BY 4.0) | Only `pole`/`trunk` boxes and rotated `vine_row` strips. The `vineyard` category has 0 instances, so there is no canopy ground truth. |
| GAIA vineyard UAV dataset (HF `links-ads/gaia-vineyard-uav-dataset`) | No licence stated (no card, no LICENSE file) |
| GRowSeg (HF `links-ads/gaia-growseg`, MIT) | Weights are gated (manual approval); row model only |
| AGRIDS (doi:10.5281/zenodo.15211733) | CC BY-NC-ND (no derivatives) |
| UAV groundcover 9-class (doi:10.5281/zenodo.17701564) | Restricted access |
| UOPNOA (doi:10.5281/zenodo.4648002) | Parcel-level land-use masks at 25–50 cm/px, not per plant |
| ICAERUS YOLOv9 vine-seg weights (doi:10.5281/zenodo.14610756, CC BY 4.0) | Needs the GPL-3.0 WongKinYiu/yolov9 code; not needed for our heads |
| SODA, UAV-BD | No licence stated |
| UCWD (12 GB), AerialWaste (20–50 cm/px), TACO / PlastOPol / BePLi / ZeroWaste | Too large, wrong scale, or not aerial |
