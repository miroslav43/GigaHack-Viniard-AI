# Speaker script: 5 min pitch (about 4:20, with margin)

The main deck follows the organisers' plan: Title → Problem (business) → Solution (architecture + data pipeline)
→ Demo → Impact (economics + deployment time) → Scaling to industry. One or two sentences per slide. Bold marks the
number to say out loud. The technical detail slides are now backup (11–20), for the Q&A.

| # | Slide | s | Say |
|---|---|---|---|
| 1 | Title | 10 | "We are **Solemtrix Hardware & Software**: <names, team lead>. We turn one drone flight into a vineyard inventory and a **24 km** inspection walk." |
| 2 | Problem: roles and user stories | 35 | "The vine registry lists **45,500 ha**, the statistics **66,400 ha**: a **20,900 ha** gap, and subsidies of up to 300,000 lei per hectare depend on it. Four users: the state inspector wants every block measured from one flight; the farmer wants gaps and waste per row; the agronomist wants the shortest walk as GPX; the municipality wants farms, roads and waste on one map." |
| 3 | Business case | 20 | "One flight, one map, one walk: drone, AI pipeline, a human check in Marcaj, the map and reports per row ID, then the walk or the robot, then the decision. On Sireț3: **81.5 ha**, **27 farms**, **674 rows**, **13,702 plants**, **1,215 targets**." |
| 4 | Architecture | 35 | "Computer vision first. Rows are a 1-D signal: angle, projection, autocorrelation, SNR gate against orchards. Canopy is a colour threshold inside the row corridor, one polygon per plant. Attributes are explicit rules, for example a gap of 5 m makes a row disrupted. Waste is a cascade ending with a human. The route is a generalised TSP on the inter-rows. CV covers all 311 tiles in **3 minutes** with **no labels**; SAM 3 everywhere would take **18 hours**." |
| 5 | Data pipeline | 25 | "GeoTIFF tiles are ingested and checksummed in 2.5 s, perception takes 384 s, Marcaj adds the human check, and the post chain writes the route, measurements and web bundle in 603 s. **17 minutes** end to end on a laptop CPU, pinned and reproducible, no paid API." |
| 6 | Result: route | 25 | "For farm F09: **9.7 km** through all **455** targets, **65 % shorter** than a serpentine, solved in 1.5 s in the browser. For the whole survey, from the organizers' pre-test start: **24 km instead of 68** through **795** targets, every reachable must-visit one, **1.29 %** outside the allowed area." |
| 7 | Demo | 30 | "Here is the screencast [link]. The map shows every layer and the IDs on click, the blocks page the counts and areas, the route page the GPX. The organisers' pre-test point falls in farm F09, block V13, next to row V13-R020, and our route now starts and ends exactly there (**0.00 m**)." |
| 8 | Impact | 35 | "One inspection walk drops from **17 hours to 6**: about **11 hours** saved per survey, at almost zero compute cost. To deploy: about **11 to 20 hours** from zero to the first inspected region: a few hours of set-up and calibration, 17 minutes of AI, and half a day of human checking." |
| 9 | Scaling to industry | 30 | "About **4.7 s per hectare**: all of Moldova's vineyards in about **130 laptop-hours**, or 16 hours on one server. Next for the industry: our robot as Street View for every row with on-board disease detection, season-to-season change, registry and subsidy export, per-farm reports, and an offline Docker for government servers." |
| 10 | Summary | 8 | "Fast, accurate, short routes, deployable in a day or two. Thank you." |

Total ≈ 253 s ≈ 4:15. If time runs short, cut slide 6 to one sentence.

## Q&A: which backup slide to open

| Likely question | Slide | One-line answer |
|---|---|---|
| Details of the challenge input | 11 | 311 GeoTIFF tiles at 2.5 cm/px, 385 MB, 81.5 ha. |
| Why computer vision, not deep learning? | 12, 23 | Only 2 labelled tiles; on them the U-Net brings no gain and costs training and a GPU. |
| How are rows found? | 13 | 1-D projection, autocorrelation, Huber fit, lattice propagation; row F1 1.00 on the example tiles. |
| Canopy accuracy? | 14 | Canopy score 0.876, F1 0.905, inter-row IoU 0.96 on the organisers' 2 example tiles. |
| Waste? | 15 | Rules → OpenCLIP probe → SAM 3 → human approval; precision over recall. |
| Measurements? | 16 | measurements.csv by survey, block and row; counts score 0.973 on the example tiles. |
| Engineering, timings | 17, 28 | 17 min end to end; 2,378 tests, CI, pinned lock, offline Docker. |
| The robot | 18–20 | Concept: drives our route, side cameras, on-board disease detection, same row IDs. Answer specs from the team. |
| How do you know it is right? | 22 | Visual QA of all 311 tiles, lattice checks, cadastre agreement, validators; the 2 labelled tiles are a regression test. |
| Does it scale? | 24 | About 4.7 s/ha on a laptop, parallel per tile; ~130 laptop-hours for Moldova (extrapolation). |
| Shadows, grass, old vines, orchards? | 25 | Periodicity gates, shadow mask, relaxed pass plus the U-Net for old vines, humans in Marcaj. |
| How is the route built? | 26 | Domain, graph, target sets, GTSP with OR-Tools, validation. 99 must-visit targets are unreachable (76 disconnected from START, 23 too far from any allowed path). |
| Can I reproduce it? | 27 | Four commands, `uv.lock`, offline Docker, CI. |
| Farms, roads, cadastre? | 29 | 27 farms; the AGCC cadastre is a soft prior, never a clipping mask. |
| Who would pay for it? | 30 | Municipalities, the ONVV registry (45,500 vs 66,400 ha), AIPA subsidy checks. |
| Where do the deployment hours come from? | 8 | Compute times are measured (README §6); set-up, calibration and human-check hours are team estimates. |
