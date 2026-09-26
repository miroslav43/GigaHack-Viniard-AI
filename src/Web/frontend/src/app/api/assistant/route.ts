// In-app assistant: questions about using Solemtrix, answered by Gemini (via OpenRouter or Google directly,
// src/lib/assistant/llm.ts) with a system prompt built from the app's own labels (src/lib/assistant/prompt.ts).
// The API keys stay on the server (never NEXT_PUBLIC_). The answer is streamed as plain text (markdown).
// The proxy does not run on /api, so access is checked here.
import { cookies } from "next/headers";
import { routing, type Locale } from "@/i18n/routing";
import { LlmError, assistantConfigured, streamAnswer, type Turn } from "@/lib/assistant/llm";
import { buildSystemPrompt } from "@/lib/assistant/prompt";
import { SURVEY_ID, getSummary } from "@/lib/data";
import { AUTH_ENABLED, DEMO_COOKIE } from "@/lib/supabase/config";
import { getViewer } from "@/lib/viewer";

const MAX_HISTORY = 12;
const MAX_MESSAGE = 2000;
const MAX_QUESTION = 1000;
// per signed-in user (or per demo visitor's address): enough for a conversation, not for scripted abuse
const WINDOW_MS = 10 * 60 * 1000;
const MAX_PER_WINDOW = 30;
const hits = new Map<string, number[]>();

const json = (status: number, error: string) => Response.json({ error }, { status });

function limited(key: string) {
  const now = Date.now();
  const recent = (hits.get(key) ?? []).filter((t) => now - t < WINDOW_MS);
  recent.push(now);
  hits.set(key, recent);
  if (hits.size > 5000) hits.clear();
  return recent.length > MAX_PER_WINDOW;
}

function parseBody(body: unknown): { turns: Turn[]; locale: Locale; path: string } | null {
  const b = body as { messages?: unknown; locale?: unknown; path?: unknown } | null;
  if (!b || !Array.isArray(b.messages) || b.messages.length === 0) return null;
  const turns = b.messages
    .slice(-MAX_HISTORY)
    .filter((m): m is Turn => (m?.role === "user" || m?.role === "assistant") && typeof m?.text === "string" && m.text.trim() !== "")
    .map((m) => ({ role: m.role, text: m.text.slice(0, MAX_MESSAGE) }));
  const last = turns.at(-1);
  if (!last || last.role !== "user" || last.text.length > MAX_QUESTION) return null;
  const locale = (routing.locales as readonly string[]).includes(b.locale as string) ? (b.locale as Locale) : routing.defaultLocale;
  const path = typeof b.path === "string" && /^\/[\w\-/?=&.%]*$/.test(b.path) ? b.path.slice(0, 200) : "/";
  return { turns, locale, path };
}

export async function POST(request: Request) {
  if (!assistantConfigured()) return json(503, "not_configured");

  const viewer = await getViewer();
  const jar = await cookies();
  // without a session only the demo visitor may ask (same rule as the pages)
  if (AUTH_ENABLED && viewer.kind === "demo" && jar.get(DEMO_COOKIE)?.value !== "1") return json(401, "sign_in");

  const who = viewer.userId ?? request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "anon";
  if (limited(who)) return json(429, "rate_limit");

  const input = parseBody(await request.json().catch(() => null));
  if (!input) return json(400, "bad_request");

  // flight numbers only for a municipality that has the served survey (never another UAT's data)
  const summary = viewer.uat?.surveys.includes(SURVEY_ID) ? await getSummary().catch(() => null) : null;
  const systemInstruction = buildSystemPrompt({ locale: input.locale, viewer, path: input.path, summary });

  let stream: AsyncIterable<string>;
  try {
    stream = await streamAnswer({ system: systemInstruction, turns: input.turns, signal: request.signal });
  } catch (e) {
    const status = e instanceof LlmError ? e.status : 502;
    console.error("[assistant]", status, e instanceof Error ? e.message : e);
    return json(status === 429 ? 429 : 502, status === 429 ? "rate_limit" : "upstream");
  }

  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      async start(controller) {
        try {
          for await (const text of stream) controller.enqueue(encoder.encode(text));
        } catch (e) {
          console.error("[assistant] stream", e instanceof Error ? e.message : e);
        } finally {
          controller.close();
        }
      },
    }),
    { headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" } },
  );
}
