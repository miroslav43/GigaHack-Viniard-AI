// Who is looking at the page: a signed-in municipality user, or the public demo.
import "server-only";
import { cache } from "react";
import { AUTH_ENABLED } from "./supabase/config";
import { createClient } from "./supabase/server";
import { fromRow, getDemoUat, type UatInfo, type UatRow } from "./uats";

export type UatRole = "platform_admin" | "uat_admin" | "inspector" | "viewer";

export interface Viewer {
  kind: "user" | "demo";
  userId: string | null;
  email: string | null;
  name: string | null;
  role: UatRole | null;
  /** null: signed in but not assigned to an active municipality */
  uat: UatInfo | null;
}

const demoViewer = async (): Promise<Viewer> => ({
  kind: "demo",
  userId: null,
  email: null,
  name: null,
  role: null,
  uat: await getDemoUat(),
});

/** Cached per request: the layout and the page share one lookup. */
export const getViewer = cache(async (): Promise<Viewer> => {
  if (!AUTH_ENABLED) return demoViewer();
  const supabase = await createClient();
  const { data } = await supabase.auth.getClaims();
  const claims = data?.claims;
  // no session: the proxy only lets such requests through with the demo cookie
  if (!claims) return demoViewer();

  // authorization data comes from app_metadata (not user-editable), never from user_metadata
  const app = (claims.app_metadata ?? {}) as { uat?: string; uat_role?: UatRole };
  const user = (claims.user_metadata ?? {}) as { full_name?: string };

  let uat: UatInfo | null = null;
  if (app.uat) {
    // RLS: a user can read only their own municipality; surveys only if they intersect its boundary
    const [{ data: row }, { data: links }] = await Promise.all([
      supabase.from("uat_public").select("*").eq("key", app.uat).eq("active", true).maybeSingle<UatRow>(),
      supabase.from("uat_survey").select("survey_id").eq("uat_key", app.uat),
    ]);
    if (row) uat = fromRow(row, (links ?? []).map((l) => l.survey_id as string));
  }

  return {
    kind: "user",
    userId: claims.sub ?? null,
    email: typeof claims.email === "string" ? claims.email : null,
    name: user.full_name ?? null,
    role: app.uat_role ?? "viewer",
    uat,
  };
});

export const isPlatformAdmin = (v: Viewer) => v.kind === "user" && v.role === "platform_admin";
