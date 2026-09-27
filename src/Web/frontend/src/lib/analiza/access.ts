// Who may use /api/analiza: the proxy does not run on /api, so each route checks it here. Same rule as the pages:
// a signed-in user or the demo visitor (cookie); without Supabase (open demo mode) everyone. Only the session claims
// are read (no database lookups): the page polls the job status every second.
import "server-only";
import { cookies } from "next/headers";
import { AUTH_ENABLED, DEMO_COOKIE } from "@/lib/supabase/config";
import { createClient } from "@/lib/supabase/server";

export async function mayAnalyse(): Promise<boolean> {
  if (!AUTH_ENABLED) return true;
  const jar = await cookies();
  if (jar.get(DEMO_COOKIE)?.value === "1") return true;
  const supabase = await createClient();
  const { data } = await supabase.auth.getClaims();
  return Boolean(data?.claims);
}
