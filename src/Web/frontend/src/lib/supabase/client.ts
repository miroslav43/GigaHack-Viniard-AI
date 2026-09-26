"use client";

import { createBrowserClient } from "@supabase/ssr";
import { createClient as createPlainClient } from "@supabase/supabase-js";
import { SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL } from "./config";

export function createClient() {
  return createBrowserClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);
}

/**
 * Client for "forgot password" only. The SSR browser client always uses PKCE, whose link works only in the
 * browser that asked for it; the implicit flow returns the session in the link's fragment, so the email can be
 * opened on any device (e.g. the phone). /parola-noua reads it, as for invitations. Nothing is persisted here.
 */
export function createRecoveryClient() {
  return createPlainClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, {
    auth: { flowType: "implicit", persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
  });
}
