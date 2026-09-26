// Web Worker: plans the farm route off the main thread, so the map stays fluid on large farms.
import { planFarmRoute, type FarmRouteInput, type FarmRouteResult } from "./plan.ts";

export interface FarmRouteRequest {
  id: number;
  input: FarmRouteInput;
}
export type FarmRouteResponse = { id: number; ok: true; result: FarmRouteResult } | { id: number; ok: false; error: string };

// the DOM lib types `self` as a Window; in a dedicated worker postMessage takes just the message
const scope = self as unknown as { onmessage: ((e: MessageEvent<FarmRouteRequest>) => void) | null; postMessage: (m: FarmRouteResponse) => void };

scope.onmessage = (e) => {
  const { id, input } = e.data;
  let reply: FarmRouteResponse;
  try {
    reply = { id, ok: true, result: planFarmRoute(input) };
  } catch (err) {
    reply = { id, ok: false, error: err instanceof Error ? err.message : String(err) };
  }
  scope.postMessage(reply);
};
