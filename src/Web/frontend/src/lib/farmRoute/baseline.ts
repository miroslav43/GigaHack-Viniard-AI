// The "normal" walk the planned route is compared with: without the AI's targets, an inspector sweeps every
// inter-row of every block to see every row — the pipeline's serpentine baseline (src/AI/vineyard/route/baseline.py,
// `serpentine_est_m`). Lanes between two rows are walked end to end in lateral order, alternating direction; blocks
// follow nearest first, from the start and back. Hops between lanes and blocks are straight lines, which flatters the
// baseline, so the saving shown is a prudent one.
import { dist, polylineLength, type XY } from "./geometry.ts";
import type { Lane } from "./lanes.ts";

/** The lanes a sweep walks: those between two rows (a block of one row: its edge lanes). */
function sweepLanes(block: readonly Lane[]): Lane[] {
  const inner = block.filter((l) => l.rowIds.length >= 2);
  return [...(inner.length ? inner : block)].sort((a, b) => a.offset - b.offset);
}

/** A block swept from `from`: enter at the nearer end of its first or last lane, then serpentine. */
function sweepBlock(block: readonly Lane[], from: XY): { m: number; exit: XY } {
  const lanes = sweepLanes(block);
  const ends = (l: Lane): [XY, XY] => [l.points[0], l.points[l.points.length - 1]];
  const [f0, f1] = ends(lanes[0]), [l0, l1] = ends(lanes[lanes.length - 1]);
  const ordered = Math.min(dist(from, f0), dist(from, f1)) <= Math.min(dist(from, l0), dist(from, l1)) ? lanes : [...lanes].reverse();
  return ordered.reduce(
    (acc, lane) => {
      const [a, b] = ends(lane);
      const [entry, exit] = dist(acc.exit, a) <= dist(acc.exit, b) ? [a, b] : [b, a];
      return { m: acc.m + dist(acc.exit, entry) + polylineLength(lane.points), exit };
    },
    { m: 0, exit: from },
  );
}

/** Length of the full sweep from `start` through every block of the lanes and back (m). */
export function sweepBaselineM(lanes: readonly Lane[], start: XY): number {
  const blocks = new Map<string, Lane[]>();
  for (const l of lanes) blocks.set(l.blockId, [...(blocks.get(l.blockId) ?? []), l]);
  let left = [...blocks.values()];
  let at = start, total = 0;
  while (left.length) {
    const near = (b: readonly Lane[]) => Math.min(...b.flatMap((l) => [dist(at, l.points[0]), dist(at, l.points[l.points.length - 1])]));
    const next = left.reduce((best, b) => (near(b) < near(best) ? b : best));
    const swept = sweepBlock(next, at);
    total += swept.m;
    at = swept.exit;
    left = left.filter((b) => b !== next);
  }
  return total + dist(at, start);
}
