// Supabase Admin API client (user management). Uses the SECRET key: server-only, never NEXT_PUBLIC_.
import "server-only";
import { createClient } from "@supabase/supabase-js";
import { SUPABASE_URL } from "./config";

const SECRET_KEY = process.env.SUPABASE_SECRET_KEY ?? "";

export const ADMIN_API_ENABLED = Boolean(SUPABASE_URL && SECRET_KEY);

export function createAdminClient() {
  if (!ADMIN_API_ENABLED) throw new Error("SUPABASE_SECRET_KEY is not configured");
  return createClient(SUPABASE_URL, SECRET_KEY, { auth: { persistSession: false, autoRefreshToken: false } });
}
