"""Training and evaluation scripts for the vineyard detection models. Run them from the model folder, e.g.
    python -m tools.evaluate_escayard --vineyards B9

evaluate_boxes, evaluate_escayard, evaluate_mildew and tune_vine_disease score the full detector of the robot app
(package `vineyard`, with vineyard/detector.py), which is not in this folder: the nearest parent folder that contains it
is put first on the import path. The training scripts (trunk_v3, bunch_v3, leaf_daylight, trunk_adapt) do not need it.
"""
import sys
from pathlib import Path

for _folder in Path(__file__).resolve().parents:
    if (_folder / "vineyard" / "detector.py").exists():
        if str(_folder) not in sys.path:
            sys.path.insert(0, str(_folder))
        break
