// GET /api/robot?device=cam|motors|drive&base=<board address>&cmd=…: relays one whitelisted command to a robot board
// on the local Wi-Fi (src/lib/robot/commands.ts) and passes its answer back (a JPEG for a capture, text or JSON
// otherwise). The ESP32s send no CORS headers, so the page cannot read their answers directly.
// A timed drive move (`run`) is a sequence run here, ending with a stop that is sent even when a step failed; a held
// one (`go`) starts the wheels and arms a watchdog that stops them unless the page renews it (`keep`).
// Errors: 401 sign_in · 400 bad_request | bad_address · 502 unreachable · 504 timeout.
import { mayAnalyse } from "@/lib/analiza/access";
import { parseBoardUrl, parseCommand, stepTimeoutMs, upstreamPlan } from "@/lib/robot/commands";
import { arm, disarm, renew } from "@/lib/robot/watchdog";

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
  const stopWheels = () => void get(`${base}/stop?m=all`, timeout).catch((e) => console.error("[robot] watchdog stop failed:", e));
  if (command.device === "drive" && command.cmd === "keep") {
    return json(200, { ok: renew(base, stopWheels) });
  }
  if (command.device === "drive" && (command.cmd === "stop" || command.cmd === "go")) disarm(base);
  try {
    let last: Response | null = null;
    for (const step of plan.steps) last = await get(`${base}${step}`, timeout);
    if (plan.holdMs) await sleep(plan.holdMs);
    if (plan.always) return json(200, { ok: true });
    // the wheels now run until a stop, or until the page stops renewing the watchdog
    if (command.device === "drive" && command.cmd === "go") {
      arm(base, stopWheels);
      return json(200, { ok: true });
    }
    const type = last!.headers.get("content-type") ?? "text/plain; charset=utf-8";
    return new Response(await last!.arrayBuffer(), { headers: { "Content-Type": type, "Cache-Control": "no-store" } });
  } catch (e) {
    if (command.device === "drive" && command.cmd === "go") stopWheels();
    const timedOut = e instanceof Error && e.name === "TimeoutError";
    console.error(`[robot] ${command.device}/${command.cmd} → ${base}:`, e instanceof Error ? e.message : e);
    return json(timedOut ? 504 : 502, { error: timedOut ? "timeout" : "unreachable" });
  } finally {
    // the wheels always stop at the end of a move, even when a step failed or the wait was cut short
    if (plan.always) await get(`${base}${plan.always}`, timeout).catch((e) => console.error("[robot] final stop failed:", e));
  }
}
