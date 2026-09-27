// Small numeric helpers shared by the stats builders (pure, no I/O).

export const round = (v, digits = 2) => {
  const k = 10 ** digits;
  return Math.round(v * k) / k;
};

export const sum = (xs) => xs.reduce((a, x) => a + x, 0);

/** Share a/b, 0 when b is 0 (never NaN in stats.json). */
export const share = (a, b, digits = 4) => (b > 0 ? round(a / b, digits) : 0);

/** { value: count }, keys in first-seen order; null/undefined values are counted under `nullKey` (skipped when null). */
export const countBy = (items, key, nullKey = null) => {
  const out = {};
  for (const it of items) {
    const v = it[key] ?? nullKey;
    if (v == null) continue;
    out[v] = (out[v] ?? 0) + 1;
  }
  return out;
};

/** Linear-interpolated quantile of an unsorted list; 0 for an empty list. */
export const quantile = (xs, q) => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
};

/**
 * Histogram over fixed edges: bin i holds edges[i] <= v < edges[i+1]; the last bin is open-ended (>= last edge).
 * Values below edges[0] go to the first bin. Optional weights (e.g. length) instead of counts.
 */
export const histogram = (values, edges, weights = null) => {
  const counts = edges.map(() => 0);
  values.forEach((v, i) => {
    let b = edges.length - 1;
    while (b > 0 && v < edges[b]) b -= 1;
    counts[b] += weights ? weights[i] : 1;
  });
  return { edges, counts: weights ? counts.map((c) => round(c)) : counts };
};
