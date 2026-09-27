// GET /api/analiza/<job>/<file>: one output of a live tile analysis, from a fixed list (contract.JOB_FILES):
// preview.jpg, veg_mask.png, canopies | rows | interrows | waste.geojson (EPSG:4326).
import { mayAnalyse } from "@/lib/analiza/access";
import { readJobFile } from "@/lib/analiza/jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_request: Request, ctx: RouteContext<"/api/analiza/[job]/[file]">) {
  if (!(await mayAnalyse())) return new Response(null, { status: 401 });
  const { job, file } = await ctx.params;
  const found = await readJobFile(job, file);
  if (!found) return new Response(null, { status: 404 });
  return new Response(new Uint8Array(found.body), {
    headers: { "Content-Type": found.type, "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" },
  });
}
