// Chain compression of the walking graph: lanes carry a vertex every 2 m, so most nodes just continue a line. Only
// junctions (degree ≠ 2) and the kept nodes (terminals) stay; each run of pass-through nodes becomes one edge that
// remembers its chain. Shortest paths on the result are the same, on a graph ten times smaller.
import type { WalkGraph } from "./graph.ts";

export interface CompressedGraph {
  /** CSR adjacency over the kept nodes (indices into `orig`) */
  adjStart: Int32Array;
  adjTo: Int32Array;
  adjW: Float64Array;
  /** per adjacency entry: the original nodes from the entry's source to its target, inclusive */
  adjChain: (readonly number[])[];
  /** compressed node → original node */
  orig: Int32Array;
  /** original node → compressed node (-1 when it was folded into an edge) */
  index: Int32Array;
}

export function compress(g: Pick<WalkGraph, "adjStart" | "adjTo" | "adjW">, keep: Iterable<number>): CompressedGraph {
  const n = g.adjStart.length - 1;
  const degree = (v: number) => g.adjStart[v + 1] - g.adjStart[v];
  const kept = new Uint8Array(n);
  for (const v of keep) kept[v] = 1;
  for (let v = 0; v < n; v++) if (degree(v) !== 2) kept[v] = 1;
  const orig = Int32Array.from({ length: n }, (_, v) => v).filter((v) => kept[v] === 1);
  const index = new Int32Array(n).fill(-1);
  orig.forEach((v, i) => (index[v] = i));

  // from every kept node, along each of its edges, to the next kept node
  const lists = Array.from(orig, (u) => {
    const out: { to: number; w: number; chain: number[] }[] = [];
    for (let k = g.adjStart[u]; k < g.adjStart[u + 1]; k++) {
      const chain = [u];
      let prev = u, cur = g.adjTo[k], w = g.adjW[k];
      while (!kept[cur]) {
        chain.push(cur);
        // a pass-through node has two edges: take the one not leading back
        const a = g.adjStart[cur];
        const back = g.adjTo[a] === prev ? 0 : 1;
        const next = a + (1 - back);
        prev = cur;
        w += g.adjW[next];
        cur = g.adjTo[next];
      }
      chain.push(cur);
      if (cur !== u) out.push({ to: index[cur], w, chain });
    }
    return out;
  });

  const adjStart = new Int32Array(orig.length + 1);
  lists.forEach((l, i) => (adjStart[i + 1] = adjStart[i] + l.length));
  const flat = lists.flat();
  return {
    adjStart,
    adjTo: Int32Array.from(flat, (x) => x.to),
    adjW: Float64Array.from(flat, (x) => x.w),
    adjChain: flat.map((x) => x.chain),
    orig,
    index,
  };
}
