// Closed tour through every terminal, starting and ending at terminal 0 (the start), on a symmetric distance matrix.
// Exact (Held–Karp) for small instances; nearest neighbour + 2-opt + Or-opt within a time budget otherwise.

export const TOUR_LIMITS = {
  /** at most this many stops besides the start are solved exactly */
  exactMaxStops: 10,
  /** local search budget for large instances (ms) */
  budgetMs: 1500,
} as const;

type Matrix = readonly (readonly number[])[];

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

/** One pass of 2-opt (reverse order[i..j]); returns an improved copy or null. */
function twoOptPass(d: Matrix, order: readonly number[]): number[] | null {
  const n = order.length;
  for (let i = 1; i < n - 1; i++)
    for (let j = i + 1; j < n; j++) {
      const a = order[i - 1], b = order[i], c = order[j], e = order[(j + 1) % n];
      if (d[a][c] + d[b][e] < d[a][b] + d[c][e] - 1e-9)
        return [...order.slice(0, i), ...order.slice(i, j + 1).reverse(), ...order.slice(j + 1)];
    }
  return null;
}

/** One pass of Or-opt (move a run of 1–3 stops elsewhere); returns an improved copy or null. */
function orOptPass(d: Matrix, order: readonly number[]): number[] | null {
  const n = order.length;
  for (let len = 1; len <= 3; len++)
    for (let i = 1; i + len <= n; i++) {
      const prev = order[i - 1], first = order[i], last = order[i + len - 1], next = order[(i + len) % n];
      const removed = d[prev][first] + d[last][next] - d[prev][next];
      const rest = [...order.slice(0, i), ...order.slice(i + len)];
      const run = order.slice(i, i + len);
      for (let k = 0; k < rest.length; k++) {
        const x = rest[k], y = rest[(k + 1) % rest.length];
        if (k === i - 1) continue; // same place
        const fwd = d[x][first] + d[last][y] - d[x][y];
        const bwd = d[x][last] + d[first][y] - d[x][y];
        if (Math.min(fwd, bwd) < removed - 1e-9)
          return [...rest.slice(0, k + 1), ...(fwd <= bwd ? run : [...run].reverse()), ...rest.slice(k + 1)];
      }
    }
  return null;
}

/** Visiting order of the terminals (a permutation of 0..n-1 starting with 0); the tour returns to 0. */
export function solveTour(d: Matrix, now: () => number = () => Date.now()): number[] {
  const n = d.length;
  if (n <= 2) return d.map((_, i) => i);
  if (n - 1 <= TOUR_LIMITS.exactMaxStops) return exactTour(d);
  const deadline = now() + TOUR_LIMITS.budgetMs;
  let order = nearestNeighbour(d);
  while (now() < deadline) {
    const next = twoOptPass(d, order) ?? orOptPass(d, order);
    if (!next) break;
    order = next;
  }
  return order;
}
