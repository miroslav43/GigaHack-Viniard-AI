// Single-source shortest paths on the CSR walking graph (binary heap).
import type { WalkGraph } from "./graph.ts";

export interface ShortestPaths {
  dist: Float64Array;
  /** previous node on the shortest path from the source, -1 for the source and unreached nodes */
  prev: Int32Array;
  /** adjacency entry used to reach each node (its chain, on a compressed graph), -1 as for prev */
  prevEdge: Int32Array;
}

/** @param target stop once this node is settled (its distance and path are then final) */
export function shortestPaths(g: Pick<WalkGraph, "adjStart" | "adjTo" | "adjW">, source: number, target = -1): ShortestPaths {
  const n = g.adjStart.length - 1;
  const dist = new Float64Array(n).fill(Infinity);
  const prev = new Int32Array(n).fill(-1);
  const prevEdge = new Int32Array(n).fill(-1);
  const heapNode: number[] = [];
  const heapKey: number[] = [];
  const push = (v: number, k: number) => {
    let i = heapNode.length;
    heapNode.push(v);
    heapKey.push(k);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heapKey[p] <= k) break;
      heapNode[i] = heapNode[p];
      heapKey[i] = heapKey[p];
      i = p;
    }
    heapNode[i] = v;
    heapKey[i] = k;
  };
  const pop = () => {
    const top = heapNode[0], topKey = heapKey[0];
    const lastNode = heapNode.pop()!, lastKey = heapKey.pop()!;
    const size = heapNode.length;
    if (size > 0) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i, mk = lastKey;
        if (l < size && heapKey[l] < mk) {
          m = l;
          mk = heapKey[l];
        }
        if (r < size && heapKey[r] < mk) {
          m = r;
          mk = heapKey[r];
        }
        if (m === i) break;
        heapNode[i] = heapNode[m];
        heapKey[i] = heapKey[m];
        i = m;
      }
      heapNode[i] = lastNode;
      heapKey[i] = lastKey;
    }
    return [top, topKey] as const;
  };

  dist[source] = 0;
  push(source, 0);
  while (heapNode.length) {
    const [v, d] = pop();
    if (d > dist[v]) continue;
    if (v === target) break;
    for (let k = g.adjStart[v]; k < g.adjStart[v + 1]; k++) {
      const u = g.adjTo[k], nd = d + g.adjW[k];
      if (nd < dist[u]) {
        dist[u] = nd;
        prev[u] = v;
        prevEdge[u] = k;
        push(u, nd);
      }
    }
  }
  return { dist, prev, prevEdge };
}

/** Nodes from the source of `sp` to `target`, inclusive; empty when unreachable. */
export function pathTo(sp: ShortestPaths, target: number): number[] {
  if (!Number.isFinite(sp.dist[target])) return [];
  const out: number[] = [];
  for (let v = target; v !== -1; v = sp.prev[v]) out.push(v);
  return out.reverse();
}

/** The adjacency entries from the source of `sp` to `target`, in order; empty when unreachable or the source. */
export function edgesTo(sp: ShortestPaths, target: number): number[] {
  if (!Number.isFinite(sp.dist[target])) return [];
  const out: number[] = [];
  for (let v = target; sp.prevEdge[v] !== -1; v = sp.prev[v]) out.push(sp.prevEdge[v]);
  return out.reverse();
}
