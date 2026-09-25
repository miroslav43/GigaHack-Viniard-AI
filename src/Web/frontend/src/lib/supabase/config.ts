// Supabase is optional at runtime: without the env vars (e.g. CI) the app runs in open demo mode.
export const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
export const SUPABASE_PUBLISHABLE_KEY = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY ?? "";
export const AUTH_ENABLED = Boolean(SUPABASE_URL && SUPABASE_PUBLISHABLE_KEY);

/** Cookie set by "Demo fără cont": read-only access to the public static survey, no Supabase session. */
export const DEMO_COOKIE = "solemtrix_demo";
