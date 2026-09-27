// node --test (Node ≥ 22.18 strips the types): pnpm test:scripts
import { test } from "node:test";
import assert from "node:assert/strict";
import { countByLabel, MAX_BOXES, parseDetections } from "./detections.ts";

test("the model's 0–1000 [ymin,xmin,ymax,xmax] boxes become [x0,y0,x1,y1] fractions", () => {
  const boxes = parseDetections('{"objects":[{"label":"grape","box_2d":[278,372,490,480],"score":0.96},{"label":"leaf","box_2d":[0,39,396,770]}]}')!;
  assert.deepEqual(boxes[0], { label: "grape", box: [0.372, 0.278, 0.48, 0.49], score: 0.96 });
  assert.deepEqual(boxes[1], { label: "leaf", box: [0.039, 0, 0.77, 0.396], score: null });
  assert.deepEqual(countByLabel(boxes), { grape: 1, leaf: 1, waste: 0, object: 0 });
});

test("fenced answers, label variants, swapped or out-of-range corners are handled; junk is dropped", () => {
  const text = "Here you go:\n```json\n" + JSON.stringify({
    objects: [
      { label: "Grape Cluster", box_2d: [500, 600, 400, 700], score: 1.4 }, // swapped y, score clamped
      { label: "vine leaf", box_2d: [-20, 0, 1200, 100] }, // clamped to the image
      { label: "car", box_2d: [1, 1, 50, 50] }, // not ours
      { label: "Bottle", box_2d: [100, 100, 200, 150] }, // litter → waste
      { label: "object", box_2d: [0, 900, 100, 1000], score: 0.5 },
      { label: "leaf", box_2d: [1, 2, 3] }, // three numbers
      { label: "leaf", box_2d: [10, 10, 10, 10] }, // a point
      "nonsense",
    ],
  }) + "\n```";
  const boxes = parseDetections(text)!;
  assert.deepEqual(boxes, [
    { label: "grape", box: [0.6, 0.4, 0.7, 0.5], score: 1 },
    { label: "leaf", box: [0, 0, 0.1, 1], score: null },
    { label: "waste", box: [0.1, 0.1, 0.15, 0.2], score: null },
    { label: "object", box: [0.9, 0, 1, 0.1], score: 0.5 },
  ]);
});

test("no JSON or no objects list: null; empty list: no boxes; never more than the cap", () => {
  assert.equal(parseDetections("sorry, I cannot"), null);
  assert.equal(parseDetections('{"items":[]}'), null);
  assert.deepEqual(parseDetections('{"objects":[]}'), []);
  const many = { objects: Array.from({ length: 100 }, () => ({ label: "leaf", box_2d: [0, 0, 100, 100] })) };
  assert.equal(parseDetections(JSON.stringify(many))!.length, MAX_BOXES);
});
