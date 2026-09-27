# fte — fine-tuning on external open datasets (Sireț3)

This folder trains neural networks on open labelled datasets plus automatic Sireț3
pseudo-labels, and plugs their outputs into the `src/AI` pipeline without changing its code.

Three targets:

| Target | Model | What it changes in the pre-annotations |
|---|---|---|
| Per-plant canopy | `smp.Unet(resnet34)`, 3 heads at native 2.5 cm/px: vine canopy (c0), contact between touching plants (c1), ground vegetation (c2) | Cuts existing canopy polygons where they visibly narrow and the net sees a contact (`partition`). The union of the canopy mask stays the same, so class IoU, canopy area and inter-rows are unchanged. |
| Waste | `smp.Unet(resnet34)`, 1 head (litter heatmap), then connected-component boxes | Adds `waste` boxes (recall-oriented; the team deletes false positives in Marcaj) and a ranked review list for missed items. |
| Inter-row cover | c2 head (ground-vegetation fraction) | Re-decides only the borderline `interrow_cover` values. |

Datasets and licences: [DATASETS.md](DATASETS.md). ICAERUS is CC BY-NC 4.0, so the canopy weights are non-commercial.

## Setup

```bash
cd src/AI_finetune_dataset_externe
uv sync                       # own env; vineyard (../AI) is an editable path dependency
# or reuse the src/AI env: ../AI/.venv/bin/python -m ...
```

## Data

```bash
python -m fte.data.icaerus      # 28.5 MB by HTTP Range out of the 3.1 GB Zenodo zip
python -m fte.data.dronewaste   # streams images.tar.gz (3.9 GB) once, keeps ~1.5k images at 2.5 cm/px
```

## Pipeline

The exact commands and measured timings are filled in as each stage is finalised; see the sections below.
