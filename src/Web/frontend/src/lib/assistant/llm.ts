// The assistant's model call, streamed as text chunks. Two providers, both server-side only:
//   OPENROUTER_API_KEY (+ OPENROUTER_MODEL, default google/gemini-3.8-flash) — OpenAI-compatible chat completions;
//   GEMINI_API_KEY (+ GEMINI_MODEL, default gemini-3.8-flash) — Google Gen AI SDK.
// OpenRouter wins when both are set.
import "server-only";
import { GoogleGenAI } from "@google/genai";

export type Turn = { role: "user" | "assistant"; text: string };

export class LlmError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
const TEMPERATURE = 0.2;
const MAX_TOKENS = 1024;

export const assistantConfigured = () => Boolean(process.env.OPENROUTER_API_KEY || process.env.GEMINI_API_KEY);

/** Yields the answer's text as it arrives. Throws LlmError before the first chunk (HTTP status of the provider). */
export async function streamAnswer(opts: { system: string; turns: Turn[]; signal: AbortSignal }): Promise<AsyncIterable<string>> {
  if (process.env.OPENROUTER_API_KEY) return openRouter(process.env.OPENROUTER_API_KEY, opts);
  if (process.env.GEMINI_API_KEY) return gemini(process.env.GEMINI_API_KEY, opts);
  throw new LlmError(503, "not configured");
}

async function openRouter(key: string, { system, turns, signal }: { system: string; turns: Turn[]; signal: AbortSignal }) {
  const res = await fetch(OPENROUTER_URL, {
    method: "POST",
    signal,
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
      // app attribution on openrouter.ai (optional headers)
      "HTTP-Referer": process.env.SITE_URL || "http://localhost:3000",
      "X-Title": "Solemtrix",
    },
    body: JSON.stringify({
      model: process.env.OPENROUTER_MODEL || "google/gemini-3.8-flash",
      stream: true,
      temperature: TEMPERATURE,
      max_tokens: MAX_TOKENS,
      messages: [{ role: "system", content: system }, ...turns.map((t) => ({ role: t.role, content: t.text }))],
    }),
  });
  if (!res.ok || !res.body) throw new LlmError(res.status, (await res.text().catch(() => "")).slice(0, 300));
  const body = res.body;

  // server-sent events: "data: {json}" lines, ": comment" keep-alives, "data: [DONE]" at the end
  return (async function* () {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        try {
          const event = JSON.parse(data) as { choices?: { delta?: { content?: string } }[]; error?: { message?: string } };
          if (event.error) throw new LlmError(502, event.error.message ?? "stream error");
          const text = event.choices?.[0]?.delta?.content;
          if (text) yield text;
        } catch (e) {
          if (e instanceof LlmError) throw e;
        }
      }
    }
  })();
}

async function gemini(key: string, { system, turns, signal }: { system: string; turns: Turn[]; signal: AbortSignal }) {
  const ai = new GoogleGenAI({ apiKey: key });
  let stream: AsyncGenerator<{ text?: string }>;
  try {
    stream = await ai.models.generateContentStream({
      model: process.env.GEMINI_MODEL || "gemini-3.8-flash",
      contents: turns.map((t) => ({ role: t.role === "assistant" ? "model" : "user", parts: [{ text: t.text }] })),
      config: { systemInstruction: system, temperature: TEMPERATURE, maxOutputTokens: MAX_TOKENS, abortSignal: signal },
    });
  } catch (e) {
    throw new LlmError((e as { status?: number }).status ?? 502, e instanceof Error ? e.message : String(e));
  }
  return (async function* () {
    for await (const chunk of stream) if (chunk.text) yield chunk.text;
  })();
}
