// Closed tour through every terminal, starting and ending at terminal 0 (the start), on a symmetric distance matrix.
// Exact (Held–Karp) for small instances; nearest neighbour + 2-opt + Or-opt within a time budget otherwise.

export const TOUR_LIMITS = {
  /** at most this many stops besides the start are solved exactly */
  exactMaxStops: 10,
  /** local search budget for large instances (ms) */
  budgetMs: 1500,
} as const;

type Matrix = readonly ArrayLike<number>[];

export const tourLength = (d: Matrix, order: readonly number[]) =>
  order.reduce((s, v, i) => s + d[v][order[(i + 1) % order.length]], 0);

/** Held–Karp dynamic programme; order starts with 0. */
function exactTour(d: Matrix): number[] {
  const n = d.length;
  const m = n - 1; // stops 1..n-1 ↔ bits 0..m-1
  const full = 1 << m;
  const cost = Array.from({ length: full }, () => new Float64Array(m).fill(Infinity));
  const parent = Array.from({ length: full }, () => new Int8Array(m).fill(-1));
  for (let j = 0; j < m; j++) cost[1 << j][j] = d[0][j + 1];
  for (let set = 1; set < full; set++)
    for (let j = 0; j < m; j++) {
      if (!(set & (1 << j)) || cost[set][j] === Infinity) continue;
      for (let k = 0; k < m; k++) {
        if (set & (1 << k)) continue;
        const next = set | (1 << k), c = cost[set][j] + d[j + 1][k + 1];
        if (c < cost[next][k]) {
          cost[next][k] = c;
          parent[next][k] = j;
        }
      }
    }
  let last = 0;
  for (let j = 1; j < m; j++) if (cost[full - 1][j] + d[j + 1][0] < cost[full - 1][last] + d[last + 1][0]) last = j;
  const rev: number[] = [];
  for (let set = full - 1, j = last; j !== -1; ) {
    rev.push(j + 1);
    const p = parent[set][j];
    set &= ~(1 << j);
    j = p;
  }
  return [0, ...rev.reverse()];
}

function nearestNeighbour(d: Matrix): number[] {
  const left = new Set(d.map((_, i) => i).slice(1));
  const order = [0];
  while (left.size) {
    const cur = order[order.length - 1];
    let best = -1;
    for (const v of left) if (best === -1 || d[cur][v] < d[cur][best]) best = v;
    order.push(best);
    left.delete(best);
  }
  return order;
}

/** One sweep of 2-opt over the whole tour (improving reversals applied as found); true when something improved.
 *  Works on the caller's working copy. */
function twoOptSweep(d: Matrix, t: number[]): boolean {
  const n = t.length;
  let improved = false;
  for (let i = 1; i < n - 1; i++)
    for (let j = i + 1; j < n; j++) {
      const a = t[i - 1], b = t[i], c = t[j], e = t[(j + 1) % n];
      if (d[a][c] + d[b][e] < d[a][b] + d[c][e] - 1e-9) {
        for (let x = i, y = j; x < y; x++, y--) [t[x], t[y]] = [t[y], t[x]];
        improved = true;
      }
    }
  return improved;
}

/** One sweep of Or-opt (move a run of 1–3 stops to its best place, either way round); true when something moved. */
function orOptSweep(d: Matrix, t: number[]): boolean {
  let improved = false;
  for (let len = 1; len <= 3; len++)
    for (let i = 1; i + len <= t.length; i++) {
      const n = t.length;
      const prev = t[i - 1], first = t[i], last = t[i + len - 1], next = t[(i + len) % n];
      const removed = d[prev][first] + d[last][next] - d[prev][next];
      let best = { gain: 1e-9, k: -1, reverse: false };
      for (let k = 0; k < n; k++) {
        if (k >= i - 1 && k < i + len) continue; // inside the run or its own place
        const x = t[k], y = t[(k + 1) % n];
        if ((k + 1) % n >= i && (k + 1) % n < i + len) continue;
        const fwd = removed - (d[x][first] + d[last][y] - d[x][y]);
        const bwd = removed - (d[x][last] + d[first][y] - d[x][y]);
        if (fwd > best.gain) best = { gain: fwd, k, reverse: false };
        if (bwd > best.gain) best = { gain: bwd, k, reverse: true };
      }
      if (best.k < 0) continue;
      const run = t.splice(i, len);
      const at = best.k < i ? best.k + 1 : best.k + 1 - len;
      t.splice(at, 0, ...(best.reverse ? run.reverse() : run));
      improved = true;
    }
  return improved;
}

/** Visiting order of the terminals (a permutation of 0..n-1 starting with 0); the tour returns to 0. */
export function solveTour(d: Matrix, budgetMs: number = TOUR_LIMITS.budgetMs, now: () => number = () => Date.now()): number[] {
  const n = d.length;
  if (n <= 2) return d.map((_, i) => i);
  if (n - 1 <= TOUR_LIMITS.exactMaxStops) return exactTour(d);
  const deadline = now() + budgetMs;
  // local working copy, improved in place sweep after sweep; the start (0) never moves: index 0 is never touched
  const order = nearestNeighbour(d);
  while (now() < deadline) {
    const a = twoOptSweep(d, order);
    const b = now() < deadline && orOptSweep(d, order);
    if (!a && !b) break;
  }
  return order;
}

/** Re-picks, at every position, the candidate of that stop's group that shortens its two legs; true when one changed. */
function candidateSweep(d: Matrix, t: number[], groupOf: readonly number[], groups: readonly (readonly number[])[]): boolean {
  let improved = false;
  for (let i = 1; i < t.length; i++) {
    const prev = t[i - 1], next = t[(i + 1) % t.length];
    const cost = (c: number) => d[prev][c] + d[c][next];
    const best = groups[groupOf[t[i]]].reduce((b, c) => (cost(c) < cost(b) - 1e-9 ? c : b), t[i]);
    if (best !== t[i]) {
      t[i] = best;
      improved = true;
    }
  }
  return improved;
}

/**
 * Closed tour visiting one candidate of every group (generalised TSP), starting and ending at group 0's only
 * candidate. `d` is over all candidates. Nearest neighbour on groups, then 2-opt, Or-opt and candidate re-picks until
 * nothing improves or the budget runs out. Returns the chosen candidates in visiting order.
 */
export function solveGroupTour(d: Matrix, groups: readonly (readonly number[])[], budgetMs: number = TOUR_LIMITS.budgetMs, now: () => number = () => Date.now()): number[] {
  const groupOf: number[] = [];
  groups.forEach((g, gi) => g.forEach((c) => (groupOf[c] = gi)));
  if (groups.every((g) => g.length === 1) && groups.length - 1 <= TOUR_LIMITS.exactMaxStops) {
    const reps = groups.map((g) => g[0]);
    return exactTour(reps.map((a) => reps.map((b) => d[a][b]))).map((i) => reps[i]);
  }
  // nearest neighbour over groups, entering each through its nearest candidate (local working copies)
  const t = [groups[0][0]];
  const left = new Set(groups.map((_, i) => i).slice(1));
  while (left.size) {
    const cur = t[t.length - 1];
    let best = { g: -1, c: -1, d: Infinity };
    for (const g of left) for (const c of groups[g]) if (d[cur][c] < best.d) best = { g, c, d: d[cur][c] };
    if (best.g < 0) {
      // unreachable from here: append the rest as they come
      for (const g of left) t.push(groups[g][0]);
      break;
    }
    t.push(best.c);
    left.delete(best.g);
  }
  const deadline = now() + budgetMs;
  while (now() < deadline) {
    const a = twoOptSweep(d, t);
    const b = now() < deadline && orOptSweep(d, t);
    const c = candidateSweep(d, t, groupOf, groups);
    if (!a && !b && !c) break;
  }
  return t;
}
