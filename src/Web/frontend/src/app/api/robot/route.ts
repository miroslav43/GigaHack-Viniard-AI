// GET /api/robot?device=cam|motors|drive&base=<board address>&cmd=…: relays one whitelisted command to a robot board
// on the local Wi-Fi (src/lib/robot/commands.ts) and passes its answer back (a JPEG for a capture, text or JSON
// otherwise). The ESP32s send no CORS headers, so the page cannot read their answers directly. The live MJPEG
// stream is not relayed: the page shows it straight from the camera. A drive move is a sequence run here, ending
// with a stop that is sent even when a step failed.
// Errors: 401 sign_in · 400 bad_request | bad_address · 502 unreachable · 504 timeout.
import { mayAnalyse } from "@/lib/analiza/access";
import { parseBoardUrl, parseCommand, stepTimeoutMs, upstreamPlan } from "@/lib/robot/commands";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const json = (status: number, body: Record<string, unknown>) => Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function get(url: string, timeoutMs: number): Promise<Response> {
  const res = await fetch(url, { cache: "no-store", redirect: "error", signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}`), { name: "BoardError" });
  return res;
}

export async function GET(request: Request) {
  // same rule as the pages: a signed-in user or the demo visitor; everyone in open demo mode
  if (!(await mayAnalyse())) return json(401, { error: "sign_in" });
  const q = new URL(request.url).searchParams;
  const command = parseCommand(q);
  if (!command) return json(400, { error: "bad_request" });
  const base = parseBoardUrl(q.get("base") ?? "");
  if (!base) return json(400, { error: "bad_address" });

  const plan = upstreamPlan(command);
  const timeout = stepTimeoutMs(command);
  try {
    let last: Response | null = null;
    for (const step of plan.steps) last = await get(`${base}${step}`, timeout);
    if (plan.holdMs) await sleep(plan.holdMs);
    if (plan.always) return json(200, { ok: true });
    const type = last!.headers.get("content-type") ?? "text/plain; charset=utf-8";
    return new Response(await last!.arrayBuffer(), { headers: { "Content-Type": type, "Cache-Control": "no-store" } });
  } catch (e) {
    const timedOut = e instanceof Error && e.name === "TimeoutError";
    console.error(`[robot] ${command.device}/${command.cmd} → ${base}:`, e instanceof Error ? e.message : e);
    return json(timedOut ? 504 : 502, { error: timedOut ? "timeout" : "unreachable" });
  } finally {
    // the wheels always stop at the end of a move, even when a step failed or the wait was cut short
    if (plan.always) await get(`${base}${plan.always}`, timeout).catch((e) => console.error("[robot] final stop failed:", e));
  }
}
