// GET /api/robot?device=cam|motors|drive&base=<board address>&cmd=…: relays one whitelisted command to a robot board
// on the local Wi-Fi (src/lib/robot/commands.ts) and passes its answer back (a JPEG for a capture, text or JSON
// otherwise). The ESP32s send no CORS headers, so the page cannot read their answers directly. The live MJPEG
// stream is not relayed: the page shows it straight from the camera.
// Errors: 401 sign_in · 400 bad_request | bad_address · 502 unreachable (+ detail) · 504 timeout.
import { mayAnalyse } from "@/lib/analiza/access";
import { parseBoardUrl, parseCommand, timeoutMs, upstreamPath } from "@/lib/robot/commands";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const json = (status: number, body: Record<string, unknown>) => Response.json(body, { status, headers: { "Cache-Control": "no-store" } });

export async function GET(request: Request) {
  // same rule as the pages: a signed-in user or the demo visitor; everyone in open demo mode
  if (!(await mayAnalyse())) return json(401, { error: "sign_in" });
  const q = new URL(request.url).searchParams;
  const command = parseCommand(q);
  if (!command) return json(400, { error: "bad_request" });
  const base = parseBoardUrl(q.get("base") ?? "");
  if (!base) return json(400, { error: "bad_address" });

  const url = `${base}${upstreamPath(command)}`;
  try {
    const upstream = await fetch(url, { cache: "no-store", redirect: "error", signal: AbortSignal.timeout(timeoutMs(command)) });
    const type = upstream.headers.get("content-type") ?? "text/plain; charset=utf-8";
    return new Response(await upstream.arrayBuffer(), {
      status: upstream.ok ? 200 : 502,
      headers: { "Content-Type": type, "Cache-Control": "no-store" },
    });
  } catch (e) {
    const timedOut = e instanceof Error && e.name === "TimeoutError";
    console.error(`[robot] ${command.device}/${command.cmd} → ${base}:`, e instanceof Error ? e.message : e);
    return json(timedOut ? 504 : 502, { error: timedOut ? "timeout" : "unreachable" });
  }
}
