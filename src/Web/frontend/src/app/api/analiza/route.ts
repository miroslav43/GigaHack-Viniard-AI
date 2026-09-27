// POST /api/analiza: upload one Sireț3 tile (multipart field "file") and run the real pipeline on it
// (`vineyard demo tile`, src/AI). Answers 202 {job}; the page then polls GET /api/analiza/<job>.
// Errors: 401 sign_in · 400 bad_request | not_geotiff · 413 too_large · 409 busy (+ the running job, to follow it).
import { MAX_UPLOAD_BYTES, isTiffMagic, isTiffName } from "@/lib/analiza/contract";
import { mayAnalyse } from "@/lib/analiza/access";
import { startJob } from "@/lib/analiza/jobs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const json = (status: number, body: Record<string, unknown>) => Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
// multipart overhead on top of the file itself
const FORM_SLACK_BYTES = 64 * 1024;

export async function POST(request: Request) {
  if (!(await mayAnalyse())) return json(401, { error: "sign_in" });
  const declared = Number(request.headers.get("content-length") ?? 0);
  if (declared > MAX_UPLOAD_BYTES + FORM_SLACK_BYTES) return json(413, { error: "too_large" });

  const form = await request.formData().catch(() => null);
  const file = form?.get("file");
  if (!(file instanceof File) || file.size === 0) return json(400, { error: "bad_request" });
  if (file.size > MAX_UPLOAD_BYTES) return json(413, { error: "too_large" });
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (!isTiffName(file.name) || !isTiffMagic(bytes.subarray(0, 4))) return json(400, { error: "not_geotiff" });

  try {
    const started = await startJob(bytes);
    if (!started.ok) return json(409, { error: "busy", job: started.busyWith });
    return json(202, { job: started.id });
  } catch (e) {
    console.error("[analiza] cannot start a job", e);
    return json(500, { error: "pipeline_failed" });
  }
}
