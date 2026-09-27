// node --test (Node ≥ 22.18 strips the types): pnpm test:scripts
import { test } from "node:test";
import assert from "node:assert/strict";
import { NO_FARM, groupByFarm, groupCheck, setGroup } from "./taskFarms.ts";

const farmOf = { V01: "F01", V02: "F01", V03: "F10", V04: "F2" };
const item = (id: number, vineyard_id: string | null) => ({ id, vineyard_id });

test("items are grouped by the farm of their block, farms in natural order, no-farm group last", () => {
  const groups = groupByFarm([item(1, "V03"), item(2, null), item(3, "V02"), item(4, "V04"), item(5, "V01"), item(6, "V99")], farmOf);
  assert.deepEqual(
    groups.map((g) => [g.key, g.farm, g.blocks, g.items.map((x) => x.id)]),
    [
      ["F01", "F01", ["V01", "V02"], [3, 5]],
      ["F2", "F2", ["V04"], [4]],
      ["F10", "F10", ["V03"], [1]],
      [NO_FARM, null, ["V99"], [2, 6]],
    ],
  );
});

test("a survey without farms gives a single no-farm group; no items give no groups", () => {
  const groups = groupByFarm([item(1, "V01"), item(2, "V02")], {});
  assert.equal(groups.length, 1);
  assert.equal(groups[0].farm, null);
  assert.deepEqual(
    groups[0].items.map((x) => x.id),
    [1, 2],
  );
  assert.deepEqual(groupByFarm([], farmOf), []);
});

test("group checkbox state and toggling a whole group keep other selections", () => {
  assert.deepEqual(groupCheck([1, 2], new Set()), {
    checked: false,
    indeterminate: false,
  });
  assert.deepEqual(groupCheck([1, 2], new Set([2, 9])), {
    checked: false,
    indeterminate: true,
  });
  assert.deepEqual(groupCheck([1, 2], new Set([1, 2])), {
    checked: true,
    indeterminate: false,
  });
  assert.deepEqual(groupCheck([], new Set([1])), {
    checked: false,
    indeterminate: false,
  });
  assert.deepEqual([...setGroup(new Set([9]), [1, 2], true)].sort(), [1, 2, 9]);
  assert.deepEqual([...setGroup(new Set([1, 2, 9]), [1, 2], false)], [9]);
});
