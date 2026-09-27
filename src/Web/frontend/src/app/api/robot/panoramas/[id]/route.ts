// DELETE /api/robot/panoramas/<id>: removes a saved panorama (its photos and detections).
import { mayAnalyse } from "@/lib/analiza/access";
import { deletePanorama, validId } from "@/lib/robot/panoramaStore";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function DELETE(_request: Request, ctx: RouteContext<"/api/robot/panoramas/[id]">) {
  if (!(await mayAnalyse())) return Response.json({ error: "sign_in" }, { status: 401 });
  const { id } = await ctx.params;
  if (!validId(id)) return Response.json({ error: "not_found" }, { status: 404 });
  await deletePanorama(id);
  return new Response(null, { status: 204 });
}
