"use server";

// "Request a demo / pilot" from the presentation site. Public, so: validation, a honeypot field, a minimum fill time
// and a per-address rate limit; the row is written with the secret key (public.lead has no public insert grant).
import { headers } from "next/headers";
import { INSTITUTION_TYPES } from "@/lib/leads";
import { ADMIN_API_ENABLED, createAdminClient } from "@/lib/supabase/admin";

export type LeadResult = { ok: true } | { ok: false; error: "invalid" | "unavailable" | "rate_limit" | "generic" };

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MIN_FILL_MS = 3000;
const WINDOW_MS = 60 * 60 * 1000;
const MAX_PER_WINDOW = 5;
const hits = new Map<string, number[]>();

const text = (v: FormDataEntryValue | null, max: number) => (typeof v === "string" ? v.trim().slice(0, max) : "");

export async function requestDemo(form: FormData): Promise<LeadResult> {
  // bots fill the hidden field or submit instantly: answer "ok" and store nothing
  if (text(form.get("website"), 200) !== "") return { ok: true };
  const startedAt = Number(form.get("started_at"));
  if (!Number.isFinite(startedAt) || Date.now() - startedAt < MIN_FILL_MS) return { ok: true };

  const lead = {
    name: text(form.get("name"), 120),
    institution: text(form.get("institution"), 200),
    institution_type: text(form.get("institution_type"), 40),
    position: text(form.get("position"), 120) || null,
    email: text(form.get("email"), 200).toLowerCase(),
    phone: text(form.get("phone"), 40) || null,
    message: text(form.get("message"), 2000) || null,
    locale: ["ro", "en", "ru"].includes(text(form.get("locale"), 2)) ? text(form.get("locale"), 2) : "ro",
    consent: form.get("consent") === "on",
  };
  if (
    lead.name.length < 2 ||
    lead.institution.length < 2 ||
    !(INSTITUTION_TYPES as readonly string[]).includes(lead.institution_type) ||
    !EMAIL_RE.test(lead.email) ||
    !lead.consent
  )
    return { ok: false, error: "invalid" };

  const h = await headers();
  const who = h.get("x-forwarded-for")?.split(",")[0]?.trim() || h.get("x-real-ip") || "local";
  const now = Date.now();
  const recent = (hits.get(who) ?? []).filter((t) => now - t < WINDOW_MS);
  if (recent.length >= MAX_PER_WINDOW) return { ok: false, error: "rate_limit" };
  hits.set(who, [...recent, now]);

  if (!ADMIN_API_ENABLED) return { ok: false, error: "unavailable" };
  const { error } = await createAdminClient().from("lead").insert(lead);
  if (error) {
    console.error("[lead]", error.message);
    return { ok: false, error: "generic" };
  }
  return { ok: true };
}
