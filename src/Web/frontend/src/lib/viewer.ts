// Who is looking at the page: a signed-in municipality user, or the public demo.
import { AUTH_ENABLED } from "./supabase/config";
import { createClient } from "./supabase/server";
import { DEFAULT_UAT, isUatKey, UATS, type Uat } from "./uats";

export type UatRole = "platform_admin" | "uat_admin" | "inspector" | "viewer";

export interface Viewer {
  kind: "user" | "demo";
  email: string | null;
  name: string | null;
  role: UatRole | null;
  uat: Uat;
}

const DEMO: Viewer = { kind: "demo", email: null, name: null, role: null, uat: UATS[DEFAULT_UAT] };

export async function getViewer(): Promise<Viewer> {
  if (!AUTH_ENABLED) return DEMO;
  const supabase = await createClient();
  const { data } = await supabase.auth.getClaims();
  const claims = data?.claims;
  // no session: the proxy only lets such requests through with the demo cookie
  if (!claims) return DEMO;
  // authorization data comes from app_metadata (not user-editable), never from user_metadata
  const app = (claims.app_metadata ?? {}) as { uat?: string; uat_role?: UatRole };
  const user = (claims.user_metadata ?? {}) as { full_name?: string };
  return {
    kind: "user",
    email: typeof claims.email === "string" ? claims.email : null,
    name: user.full_name ?? null,
    role: app.uat_role ?? "viewer",
    uat: UATS[isUatKey(app.uat) ? app.uat : DEFAULT_UAT],
  };
}
