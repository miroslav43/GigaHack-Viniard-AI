// Grape, leaf and waste detections on the robot's photos: the prompt, and the model's answer turned into boxes. The model
// (Gemini) answers `{"objects":[{"label","box_2d":[ymin,xmin,ymax,xmax],"score"}]}` on a 0–1000 scale; anything else
// in the answer is dropped, never trusted. Pure: no network, no files.

/** grape = a grape cluster; leaf = a grapevine leaf; waste = litter (bottles, wood, plastic, cans…). Anything else is
 *  not detected (older saves may hold "object" boxes: they are ignored) */
export type DetectionLabel = "grape" | "leaf" | "waste";
export const DETECTION_LABELS: readonly DetectionLabel[] = ["grape", "leaf", "waste"];
export const isDetectionLabel = (l: string): l is DetectionLabel => (DETECTION_LABELS as readonly string[]).includes(l);

export interface DetectionBox {
  label: DetectionLabel;
  /** [x0, y0, x1, y1], fractions of the image (0..1), x to the right, y down */
  box: [number, number, number, number];
  /** the model's confidence 0..1, null when it gave none */
  score: number | null;
}

export const MAX_BOXES = 60;

export const DETECTION_PROMPT = `You are inspecting a photo taken by a field robot in a vineyard.
Detect, each one separately (never one box over a group):
- "grape": every grape cluster;
- "leaf": every grapevine leaf;
- "waste": every piece of litter or garbage — bottles, cans, pieces of wood, plastic, bags, paper, cardboard, rubble.
Box nothing else (no people, furniture, tools, posts, ground, sky or walls).
Return JSON only, no prose:
{"objects":[{"label":"grape"|"leaf"|"waste","box_2d":[ymin,xmin,ymax,xmax],"score":0..1}]}
box_2d is on a 0-1000 scale of the image height (y) and width (x). At most ${MAX_BOXES} objects, the most confident
first. If there is nothing, return {"objects":[]}.`;

const LABEL_OF: Record<string, DetectionLabel> = {
  grape: "grape", grapes: "grape", "grape cluster": "grape", grape_cluster: "grape", bunch: "grape",
  leaf: "leaf", leaves: "leaf", "grape leaf": "leaf", grape_leaf: "leaf", "vine leaf": "leaf",
  waste: "waste", garbage: "waste", trash: "waste", litter: "waste", rubbish: "waste", bottle: "waste", can: "waste", wood: "waste", plastic: "waste",
};

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** The JSON object in the answer (bare, or inside a ```json fence), or null. */
function jsonIn(text: string): unknown {
  const fenced = /```(?:json)?\s*([\s\S]*?)```/.exec(text);
  const body = (fenced ? fenced[1] : text).trim();
  const start = body.indexOf("{"), end = body.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    return JSON.parse(body.slice(start, end + 1));
  } catch {
    return null;
  }
}

/** The boxes of a model answer; malformed entries are skipped, the answer as a whole never throws. */
export function parseDetections(text: string): DetectionBox[] | null {
  const json = jsonIn(text) as { objects?: unknown } | null;
  if (!json || !Array.isArray(json.objects)) return null;
  return json.objects
    .flatMap((o): DetectionBox[] => {
      const obj = o as { label?: unknown; box_2d?: unknown; score?: unknown };
      const label = typeof obj.label === "string" ? LABEL_OF[obj.label.trim().toLowerCase()] : undefined;
      const b = obj.box_2d;
      if (!label || !Array.isArray(b) || b.length !== 4 || !b.every((v) => typeof v === "number" && Number.isFinite(v))) return [];
      const [ymin, xmin, ymax, xmax] = (b as number[]).map((v) => clamp01(v / 1000));
      const box: [number, number, number, number] = [Math.min(xmin, xmax), Math.min(ymin, ymax), Math.max(xmin, xmax), Math.max(ymin, ymax)];
      if (box[2] - box[0] < 0.002 || box[3] - box[1] < 0.002) return []; // a point, not a box
      const score = typeof obj.score === "number" && Number.isFinite(obj.score) ? clamp01(obj.score) : null;
      return [{ label, box, score }];
    })
    .slice(0, MAX_BOXES);
}

/** How many of each label. */
export const countByLabel = (boxes: readonly DetectionBox[]): Record<DetectionLabel, number> =>
  Object.fromEntries(DETECTION_LABELS.map((l) => [l, boxes.filter((b) => b.label === l).length])) as Record<DetectionLabel, number>;
