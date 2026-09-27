# Statistici — pagina `/statistici` și animațiile (spec + plan, 27.09.2026)

Aprobat în discuție (27.09): varianta 1, o pagină nouă care se citește la scroll, plus numărătoare KPI și animații pe hartă în rest.

## Spec

**Date — `public/data/<id>/stats.json`** (`scripts/build-stats.mjs`, funcții pure în `scripts/stats/*.mjs`).
- Se calculează din fișierele deja publicate în `public/data/<id>/`. Nu atinge pipeline-ul, convertorul, `measurements.csv` sau Marcaj.
- Rulare: `pnpm data:stats --survey siret3`. `pnpm data` / `data:fast` îl rulează pentru mock. Se rulează din nou după `data:survey`, ca relieful.
- O secțiune lipsește din fișier (`null`) când datele ei lipsesc, de exemplu fermele și drumurile pe mock. Pagina nu o afișează.

Conținutul lui `stats.json`:
- ținte pe tip, prioritate și motiv de excludere;
- histograme pentru golul maxim pe rând și pentru lungimea rândurilor;
- rozeta orientării: lungimea rândurilor pe intervale de 10°, pe [0°, 180°);
- inter-rânduri după arie; acoperirea cu coroane, lățimea medie a inter-rândului, coroana medie;
- indicele de stare pe bloc;
- fermele: suprafață (Pareto), parcele, ținte/ha;
- drumuri pe clasă × suprafață;
- grila tile-urilor (rând/coloană din id, stare, vegetație);
- ruta: ținte pe rută vs. excluse, partea din afara zonei față de limita de 2%, economia.

**Indicele de stare (orientativ):**
- formula: `100 × (1 − 0,4·d − 0,3·min(1, m/M) − 0,3·min(1, g/G))`, unde:
  - `d` = partea de rânduri perturbate;
  - `m` = plante lipsă la 100 m de rând;
  - `g` = metri de gol la 100 m de rând;
  - `M`, `G` = percentila 95 dintre blocurile eligibile.
- Blocurile cu sub 100 m de rânduri nu primesc indice (`null`).
- Constantele sunt în `scripts/stats/health.mjs`, iar formula se afișează pe pagină.

**Pagina `/statistici`:** în meniu după „Panou”, pentru toate rolurile, ca `/blocuri`.
- Deschidere: 6 cifre care numără de la 0.
- Secțiunile, fiecare cu o frază-concluzie și un grafic:
  - A. Starea plantației;
  - B. Structură;
  - C. Ferme și cadastru;
  - D. Drumuri;
  - E. Rută;
  - F. Zbor și AI.
- Grafice: `@mui/x-charts` 9.14.0, din stack-ul aprobat. Rozeta și grila de tile-uri sunt SVG propriu. Culorile vin doar din `theme/`.
- Nu afișăm producție sau bani. `plant_count` = „coroane detectate”, nu vițe.

**Animații:**
- `useInView` + `CountUp` (rAF, ~1,2 s, formatare pe limbă). Le folosesc `KpiCard`, banda KPI de pe hartă și cifrele din prezentare.
- Graficele se montează sau se animă abia când intră în ecran.
- Pe hartă:
  - ruta oficială se desenează progresiv când stratul pornește;
  - țintele de prioritate 1 pulsează.
- La `prefers-reduced-motion` totul apare direct în starea finală.

**Teste:**
- `node:test` pentru `scripts/stats/`;
- Playwright pentru `/statistici`: secțiuni vizibile, cifrele finale, reduced-motion;
- `lint`, `typecheck`, `test:scripts`, `e2e`.

## Plan

1. `scripts/stats/*.mjs` (targets, rows, health, farms, roads, tiles, route) + teste, apoi `scripts/build-stats.mjs` și scripturile din `package.json`. Generat `stats.json` pentru `siret3` și `siret3-mock`.
2. `src/lib/types.ts` (`SurveyStats`) și `getStats()` în `src/lib/data.ts`.
3. Animații (în paralel):
   - `src/lib/motion/` (`useInView`, `useReducedMotion`) și `src/components/common/CountUp.tsx`;
   - `KpiCard` / `KpiStrip` / prezentare cu numărătoare;
   - pe hartă, ruta desenată progresiv și țintele P1 pulsând.
4. Pagina: `src/app/[locale]/(app)/statistici/page.tsx` și `src/components/stats/` (câte un fișier pe secțiune), meniul din `AppShell`, `messages/{ro,en,ru}.json` (namespace `stats`), promptul asistentului.
5. e2e `e2e/stats.spec.ts`; rulate `lint`, `typecheck`, `test:scripts`, `e2e`.
6. Documentație: `src/Web/CLAUDE.md` (pagini, comenzi) și un ADR în `docs/DECISIONS.md`.
