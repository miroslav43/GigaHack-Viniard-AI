// GET /api/robot/panoramas/<id>/<0.jpg | 90.jpg | 180.jpg | strip.jpg>: a photo of a saved panorama.
import { mayAnalyse } from "@/lib/analiza/access";
import { PANORAMA_FILES, readPanoramaFile, validId, type PanoramaFile } from "@/lib/robot/panoramaStore";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_request: Request, ctx: RouteContext<"/api/robot/panoramas/[id]/[file]">) {
  if (!(await mayAnalyse())) return Response.json({ error: "sign_in" }, { status: 401 });
  const { id, file } = await ctx.params;
  if (!validId(id) || !PANORAMA_FILES.includes(file as PanoramaFile)) return Response.json({ error: "not_found" }, { status: 404 });
  const bytes = await readPanoramaFile(id, file as PanoramaFile);
  if (!bytes) return Response.json({ error: "not_found" }, { status: 404 });
  // a saved photo never changes: cached by the browser
  return new Response(new Uint8Array(bytes), { headers: { "Content-Type": "image/jpeg", "Cache-Control": "private, max-age=31536000, immutable" } });
}
