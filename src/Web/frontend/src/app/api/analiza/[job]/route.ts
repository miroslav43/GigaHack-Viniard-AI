// GET /api/analiza/<job>: {status, result} of a live tile analysis (result is null until the job is done).
import { mayAnalyse } from "@/lib/analiza/access";
import { readJob } from "@/lib/analiza/jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const NO_STORE = { "Cache-Control": "no-store" };

export async function GET(_request: Request, ctx: RouteContext<"/api/analiza/[job]">) {
  if (!(await mayAnalyse())) return Response.json({ error: "sign_in" }, { status: 401, headers: NO_STORE });
  const { job } = await ctx.params;
  const found = await readJob(job);
  if (!found) return Response.json({ error: "not_found" }, { status: 404, headers: NO_STORE });
  return Response.json(found, { headers: NO_STORE });
}
