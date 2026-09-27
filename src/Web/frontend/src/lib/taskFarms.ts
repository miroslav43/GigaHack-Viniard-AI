// Field tasks and AI targets grouped by farm (/sarcini). Pure: the block → farm map comes from the survey's
// blocks.geojson (lib/tasks.ts, server), the grouping runs in the browser. Tested with node --test.

/** vineyard_id (block) → farm_id; empty for a survey without farms. */
export type FarmOf = Record<string, string>;

export interface FarmGroup<T> {
  /** stable key for React state; NO_FARM for items whose block has no farm */
  key: string;
  farm: string | null;
  /** distinct blocks of the group's items, sorted */
  blocks: string[];
  items: T[];
}

export const NO_FARM = "__none";

const natural = new Intl.Collator("en", { numeric: true }).compare;

/** Groups items by the farm of their block; farms in natural order (F2 before F10), the no-farm group last. Item order is kept. */
export function groupByFarm<T extends { vineyard_id: string | null }>(items: T[], farmOf: FarmOf): FarmGroup<T>[] {
  const groups = new Map<string, FarmGroup<T>>();
  for (const item of items) {
    const farm = (item.vineyard_id && farmOf[item.vineyard_id]) || null;
    const key = farm ?? NO_FARM;
    let g = groups.get(key);
    if (!g) groups.set(key, (g = { key, farm, blocks: [], items: [] }));
    g.items.push(item);
    if (item.vineyard_id && !g.blocks.includes(item.vineyard_id)) g.blocks.push(item.vineyard_id);
  }
  for (const g of groups.values()) g.blocks.sort(natural);
  return [...groups.values()].sort((a, b) => (a.farm === null ? 1 : b.farm === null ? -1 : natural(a.farm, b.farm)));
}

/** Tick state of a group's checkbox given the current selection. */
export function groupCheck<K>(keys: K[], selected: Set<K>): { checked: boolean; indeterminate: boolean } {
  const n = keys.filter((k) => selected.has(k)).length;
  return {
    checked: keys.length > 0 && n === keys.length,
    indeterminate: n > 0 && n < keys.length,
  };
}

/** Selection with every key of a group added (on = true) or removed. */
export function setGroup<K>(selected: Set<K>, keys: K[], on: boolean): Set<K> {
  const next = new Set(selected);
  for (const k of keys) {
    if (on) next.add(k);
    else next.delete(k);
  }
  return next;
}
