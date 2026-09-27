// Web Worker: plans the farm route (one farm) or the route through all farms off the main thread, so the map stays
// fluid while it runs (a few seconds for the whole survey).
import { planFarmRoute, type FarmRouteInput, type FarmRouteResult } from "./plan.ts";
import { planAllFarms, type AllFarmsInput } from "./allFarms.ts";

export type FarmRouteRequest = { id: number; kind: "farm"; input: FarmRouteInput } | { id: number; kind: "all"; input: AllFarmsInput };
export type FarmRouteResponse = { id: number; ok: true; result: FarmRouteResult } | { id: number; ok: false; error: string };

// the DOM lib types `self` as a Window; in a dedicated worker postMessage takes just the message
const scope = self as unknown as { onmessage: ((e: MessageEvent<FarmRouteRequest>) => void) | null; postMessage: (m: FarmRouteResponse) => void };

scope.onmessage = (e) => {
  const req = e.data;
  let reply: FarmRouteResponse;
  try {
    reply = { id: req.id, ok: true, result: req.kind === "all" ? planAllFarms(req.input) : planFarmRoute(req.input) };
  } catch (err) {
    reply = { id: req.id, ok: false, error: err instanceof Error ? err.message : String(err) };
  }
  scope.postMessage(reply);
};
