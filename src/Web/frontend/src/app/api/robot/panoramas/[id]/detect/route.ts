// POST /api/robot/panoramas/<id>/detect: runs the grape / leaf detection again (background; the page polls the list).
import { mayAnalyse } from "@/lib/analiza/access";
import { readMeta, runDetection, validId } from "@/lib/robot/panoramaStore";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(_request: Request, ctx: RouteContext<"/api/robot/panoramas/[id]/detect">) {
  if (!(await mayAnalyse())) return Response.json({ error: "sign_in" }, { status: 401 });
  const { id } = await ctx.params;
  if (!validId(id) || !(await readMeta(id))) return Response.json({ error: "not_found" }, { status: 404 });
  void runDetection(id);
  return Response.json({ ok: true }, { status: 202 });
}
