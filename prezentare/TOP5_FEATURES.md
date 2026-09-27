# TOP-5 additional features · Solemtrix Hardware & Software

These are the features we built during the challenge on top of the scored outputs (canopies, waste, row axes and
attributes, measurements and the route). Each one runs in the web app and is shown on slide 5 of
`gigahack/gigahack.pdf`.

1. **A field robot and our own grape detector.**
   - The robot is a working ESP32 ground robot, driven from the web app.
   - It has a live camera with pan and tilt, one-press panoramas at 0°, 90° and 180°, and an obstacle stop at 20 cm.
   - Grapes, leaves and waste are marked with boxes on its photos.
   - It is built to drive our inspection route, so each photo is tied to a row.
   - It lets inspectors check a row without walking it.
   - Our own YOLO detector (YOLO11 / YOLO26) is fine-tuned on public vineyard datasets: WGISD and ViViD-5K for grape bunches, vineyard photos for trunks, and a Flavescence dorée set for diseased leaves.
   - It finds grape bunches and trunks in row photos. It is not yet validated on Sireț3.
   - The notebooks and tools are in `model_detectie_vita_de_vie/`.
2. **Cadastre and farm registry.**
   - Every vineyard block is linked to its official AGCC cadastre parcels, with their codes and land use.
   - The 44 blocks are grouped into 27 farms: blocks at most 10 m apart that no public road splits.
   - Public roads and internal roads are separate layers.
   - The result is ready for the vine registry and for AIPA subsidy checks.
3. **Inspector route and tasks.**
   - The inspector picks a farm and a start point. The browser solves the closed walk in 1.5 s and exports it as GPX.
   - For farm F09 the walk is 9.7 km, 65 % shorter than a serpentine.
   - The municipality assigns one target, or a whole farm, to an inspector. Tasks are ordered along the route.
   - Each municipality sees only its own data.
4. **Plant-health reports.**
   - Sireț3 has 929 missing plants, 228 row gaps (8.6 km in total), 85 missing plants per hectare and 22 % disrupted rows.
   - Targets are counted by type and priority, per farm, block and row.
   - Reports export as CSV, and the app works in RO, EN and RU.
5. **Live tile analysis.**
   - A user uploads a GeoTIFF tile in the browser.
   - The real pipeline runs on it and shows the row, canopy and waste masks as they appear.
   - The jury can ask for a re-run, and a new region can be tried the same way.
