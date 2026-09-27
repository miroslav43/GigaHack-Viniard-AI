// GET /api/robot/panoramas: the saved panoramas (newest first) with their grape / leaf detections.
// POST /api/robot/panoramas (multipart: station, atCm, takenAt, 0.jpg, 90.jpg, 180.jpg, strip.jpg): saves one on this
// laptop (src/lib/robot/panoramaStore.ts) and starts its detection in the background. 201 {panorama}.
// Errors: 401 sign_in · 400 bad_request | not_jpeg · 413 too_large.
import { mayAnalyse } from "@/lib/analiza/access";
import { listPanoramas, MAX_FILE_BYTES, PANORAMA_FILES, savePanorama, type PanoramaFile } from "@/lib/robot/panoramaStore";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const NO_STORE = { "Cache-Control": "no-store" };
const json = (status: number, body: unknown) => Response.json(body, { status, headers: NO_STORE });
const isJpeg = (b: Uint8Array) => b.length > 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff;
const intIn = (v: FormDataEntryValue | null, lo: number, hi: number) => {
  const n = Number(v);
  return typeof v === "string" && Number.isInteger(n) && n >= lo && n <= hi ? n : null;
};

export async function GET() {
  if (!(await mayAnalyse())) return json(401, { error: "sign_in" });
  return json(200, { panoramas: await listPanoramas() });
}

export async function POST(request: Request) {
  if (!(await mayAnalyse())) return json(401, { error: "sign_in" });
  if (Number(request.headers.get("content-length") ?? 0) > 4 * MAX_FILE_BYTES + 64 * 1024) return json(413, { error: "too_large" });
  const form = await request.formData().catch(() => null);
  if (!form) return json(400, { error: "bad_request" });
  const station = intIn(form.get("station"), 1, 100_000), atCm = intIn(form.get("atCm"), 0, 10_000_000);
  const takenAt = form.get("takenAt");
  if (station === null || atCm === null || typeof takenAt !== "string" || Number.isNaN(Date.parse(takenAt))) return json(400, { error: "bad_request" });
  const files = {} as Record<PanoramaFile, Uint8Array>;
  for (const name of PANORAMA_FILES) {
    const f = form.get(name);
    if (!(f instanceof File) || f.size === 0) return json(400, { error: "bad_request" });
    if (f.size > MAX_FILE_BYTES) return json(413, { error: "too_large" });
    const bytes = new Uint8Array(await f.arrayBuffer());
    if (!isJpeg(bytes)) return json(400, { error: "not_jpeg" });
    files[name] = bytes;
  }
  try {
    const panorama = await savePanorama({ station, atCm, takenAt: new Date(takenAt).toISOString(), files });
    return json(201, { panorama });
  } catch (e) {
    console.error("[robot] cannot save a panorama:", e);
    return json(500, { error: "save_failed" });
  }
}
