// Grape and leaf detection on one robot photo, by Gemini through OpenRouter (OPENROUTER_API_KEY / OPENROUTER_MODEL in
// frontend/.env.local, server only — the same key as the assistant). ROBOT_DETECT=fake answers fixed boxes without
// the network (tests, demos without credit).
import "server-only";
import { DETECTION_PROMPT, parseDetections, type DetectionBox } from "./detections";

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_MODEL = "google/gemini-3.8-flash";
/** one photo: the model takes 5–25 s */
const TIMEOUT_MS = 60_000;

export const detectionConfigured = () => process.env.ROBOT_DETECT === "fake" || Boolean(process.env.OPENROUTER_API_KEY);
export const detectionModel = () => (process.env.ROBOT_DETECT === "fake" ? "fake" : (process.env.OPENROUTER_MODEL ?? DEFAULT_MODEL));

const FAKE: DetectionBox[] = [
  { label: "grape", box: [0.35, 0.3, 0.5, 0.55], score: 0.9 },
  { label: "leaf", box: [0.05, 0.05, 0.3, 0.35], score: 0.8 },
  { label: "leaf", box: [0.6, 0.1, 0.85, 0.4], score: 0.7 },
  { label: "waste", box: [0.1, 0.7, 0.25, 0.95], score: 0.85 },
];

export async function detectObjects(jpeg: Uint8Array): Promise<DetectionBox[]> {
  if (process.env.ROBOT_DETECT === "fake") {
    await new Promise((r) => setTimeout(r, 300));
    return FAKE;
  }
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error("not_configured");
  const res = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "X-Title": "Solemtrix robot" },
    signal: AbortSignal.timeout(TIMEOUT_MS),
    body: JSON.stringify({
      model: detectionModel(),
      response_format: { type: "json_object" },
      // measured: "low" answers in ~9 s with scores; the default thinks ~25 s for the same boxes
      reasoning: { effort: "low" },
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: DETECTION_PROMPT },
            { type: "image_url", image_url: { url: `data:image/jpeg;base64,${Buffer.from(jpeg).toString("base64")}` } },
          ],
        },
      ],
    }),
  });
  const body = (await res.json().catch(() => null)) as { choices?: { message?: { content?: string } }[]; error?: { message?: string } } | null;
  if (!res.ok) throw new Error(`model HTTP ${res.status}: ${body?.error?.message ?? "no detail"}`);
  const boxes = parseDetections(body?.choices?.[0]?.message?.content ?? "");
  if (!boxes) throw new Error("the model's answer had no detections list");
  return boxes;
}
