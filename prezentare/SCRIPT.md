# Speaker script: 5 min pitch (about 4:36, with margin)

One or two sentences per slide. Bold marks the number to say out loud. The outline slides are 3 s clicks.

| # | Slide | s | Say |
|---|---|---|---|
| 1 | Title | 8 | "We are Solemtrix. We turned 311 drone tiles into a **14 km** inspection walk, on one laptop, in **17 minutes**." |
| 2 | Outline | 3 | Just click on. |
| 3 | Challenge | 15 | "81.5 hectares at 2.5 cm per pixel: vines, orchards, meadows and yards mixed together. The hard part is not seeing a vine. It is seeing **13,702** of them the same way, then walking only to the ones that matter." |
| 4 | Architecture | 20 | "One offline pipeline: perception, then Marcaj with a human in the loop, then the route and measurements. Every tile runs in parallel, every threshold sits in one file, and every stage is timed." |
| 5 | Outline | 3 | Just click on. |
| 6 | Efficiency | 35 | "Our main decision: computer vision first. Only 2 tiles are labelled, so a network can be neither trained nor benchmarked seriously, and on those 2 tiles our U-Net brings no gain. CV covers all 311 tiles in **about 3 minutes** on a CPU with **no training labels**. A foundation model like SAM 3 on every tile would take about **18 hours**. So the network only runs where CV is blind: 88 tiles, 13 seconds." |
| 7 | Rows | 28 | "Vine rows are periodic, so we treat them as a 1-D signal, not an image. We find the angle, project onto the normal, and autocorrelation gives the spacing. The key trick: find a few good rows once, then propagate that lattice to neighbouring tiles, searching only ±2°. On the organisers' two example tiles: row F1 **1.00**, all 51 rows, grouping **1.00**, row length off by **0.05 %**." |
| 8 | Canopies | 23 | "Canopy is one colour channel, but only inside the row corridor, so grass between rows never counts. One polygon per plant: **13,702 plants**. On the organisers' example tiles: canopy score **0.876**, F1 **0.905**, inter-row IoU **0.96**, and every row and inter-row attribute right, **51/51** and **49/49**." |
| 9 | Waste | 18 | "For waste we run a cascade: cheap filters first, expensive models last. 1,331 candidates, a CLIP probe, SAM 3 only on the top crops, and a human approves every box. We choose precision over recall." |
| 10 | Outline | 3 | Just click on. |
| 11 | Route | 32 | "Routes walk the inter-rows only. Here is one farm, F09, with the start picked on the map: **9.7 km** through all **455** of its targets, **65 % shorter** than a serpentine, computed in 1.5 s in the browser and exported as GPX. For the whole survey the official route is **14 km instead of 68**, reaches **every reachable must-visit target**, and stays **1.28 %** outside the allowed area, under the 2 % limit." |
| 12 | Measures and web | 12 | "Every number is traceable to a block and a row ID. On the example tiles the counts score is **0.973**: row length off by **0.05 %**, 51 of 51 rows counted." |
| 13 | Engineering | 20 | "Built to be re-run: **17 minutes** end to end, **2,378 tests**, a pinned lock file, and Docker with no network. No paid API and no LLM at runtime." |
| 14 | Outline | 3 | Just click on. |
| 15 | Street View | 14 | "What comes next: the drone shows *where* something is wrong. Our robot shows *what*. It drives our route like a Street View car." |
| 16 | Robot perception | 18 | "Its side cameras and on-board ML detect disease symptoms. Every detection carries the same row ID as the drone map. With one camera per side, one pass covers two rows." |
| 17 | Loop | 14 | "Drone, pipeline, route, robot, map: one loop. The robot drives **14 km** instead of at least **41**, so there is battery left for more vineyards." |
| 18 | Summary | 7 | "Fast, accurate on the organisers' examples, short routes, validated. Thank you. Happy to take questions." |

## Q&A: which backup slide to open

| Likely question | Slide | One-line answer |
|---|---|---|
| How do you know it is right? | 20 | Survey-wide checks: visual QA of all 311 tiles, lattice checks, agreement with the cadastral strips, validators. The 2 labelled tiles are only a regression test; the real measure is the jury's Marcaj evaluation. |
| Why not deep learning end to end? | 21 | Only 2 labelled tiles exist; a network would learn from our own CV labels; on the 2 tiles it shows no gain, and it costs training and a GPU. |
| Does it scale? | 22 | About 4.7 s per hectare on a laptop, parallel per tile. About 130 laptop-hours for all of Moldova (an extrapolation). |
| Shadows, grass, old vines, orchards? | 23 | Periodicity gates, a shadow mask, a relaxed pass plus the U-Net for old vines, road cuts, and humans in Marcaj for the rest. |
| How is the route built? | 24 | Domain, graph, target sets, GTSP with OR-Tools, validation, and the baselines table. 99 must-visit targets are unreachable: 76 not connected to START by an authorised passage, 23 too far from any allowed path. |
| Can I reproduce it? | 25 | Four commands, `uv.lock`, offline Docker, CI. |
| Why is the route stage slow? | 26 | By design: the solver gets the time budget because that is where kilometres are saved. |
| Farms, roads, cadastre? | 27 | 27 farms; the AGCC cadastre is a soft prior, never a clipping mask. |
| Who would pay for it? | 28 | Municipalities, the ONVV registry (45,500 vs 66,400 ha), AIPA subsidy checks. |
| Robot details | — | Answer from the team's robot specs (model, disease classes, hardware). They are not in the deck yet. |
