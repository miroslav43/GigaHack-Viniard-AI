import createMiddleware from "next-intl/middleware";
import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";
import { routing } from "./i18n/routing";
import { AUTH_ENABLED, DEMO_COOKIE, SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL } from "./lib/supabase/config";

const intl = createMiddleware(routing);

const PUBLIC_PATHS = ["/login", "/acces-interzis"];

/** "/en/harta" → { locale: "en", path: "/harta" }; Romanian has no prefix. */
function splitLocale(pathname: string) {
  const seg = pathname.split("/")[1];
  if ((routing.locales as readonly string[]).includes(seg) && seg !== routing.defaultLocale) {
    return { prefix: `/${seg}`, path: pathname.slice(seg.length + 1) || "/" };
  }
  return { prefix: "", path: pathname };
}

function redirectKeepingCookies(url: URL, from: NextResponse) {
  const res = NextResponse.redirect(url);
  from.cookies.getAll().forEach((c) => res.cookies.set(c));
  return res;
}

/** Paths only a platform administrator may open; everyone else gets a 403 (no login redirect, no hint). */
const ADMIN_ONLY_PATHS = ["/super-admin"];
const matches = (path: string, list: string[]) => list.some((p) => path === p || path.startsWith(`${p}/`));

/** Renders the localized 403 page in place of the requested URL, keeping any refreshed session cookies. */
function denyAccess(request: NextRequest, prefix: string, from?: NextResponse) {
  const url = request.nextUrl.clone();
  url.pathname = `/${prefix ? prefix.slice(1) : routing.defaultLocale}/acces-interzis`;
  url.search = "";
  const res = NextResponse.rewrite(url, { status: 403 });
  from?.cookies.getAll().forEach((c) => res.cookies.set(c));
  return res;
}

export default async function proxy(request: NextRequest) {
  const response = intl(request);
  const { prefix, path } = splitLocale(request.nextUrl.pathname);
  const adminOnly = matches(path, ADMIN_ONLY_PATHS);
  if (response.headers.get("location")) return response;
  // without Supabase (CI / open demo) nobody is a platform admin
  if (!AUTH_ENABLED) return adminOnly ? denyAccess(request, prefix) : response;

  // refresh the Supabase session cookie on the response next-intl produced
  const supabase = createServerClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, {
    cookies: {
      getAll() {
        return request.cookies.getAll();
      },
      setAll(cookiesToSet) {
        cookiesToSet.forEach(({ name, value, options }) => response.cookies.set(name, value, options));
      },
    },
  });
  // do not run code between createServerClient and getClaims()
  const { data } = await supabase.auth.getClaims();
  const signedIn = Boolean(data?.claims);
  const demo = request.cookies.get(DEMO_COOKIE)?.value === "1";

  const isPublic = matches(path, PUBLIC_PATHS);

  // role from app_metadata (server-set); pages and every server action check it again
  const role = (data?.claims?.app_metadata as { uat_role?: string } | undefined)?.uat_role;
  if (adminOnly) return role === "platform_admin" ? response : denyAccess(request, prefix, response);
  // the platform admin is not a municipality user: the town-hall pages lead to the platform console
  if (role === "platform_admin" && !isPublic) {
    const url = request.nextUrl.clone();
    url.pathname = `${prefix}/super-admin`;
    url.search = "";
    return redirectKeepingCookies(url, response);
  }

  if (!signedIn && !demo && !isPublic) {
    const url = request.nextUrl.clone();
    url.pathname = `${prefix}/login`;
    url.search = path === "/" ? "" : `?next=${encodeURIComponent(path + request.nextUrl.search)}`;
    return redirectKeepingCookies(url, response);
  }
  if (signedIn && isPublic) {
    const url = request.nextUrl.clone();
    url.pathname = role === "platform_admin" ? `${prefix}/super-admin` : prefix || "/";
    url.search = "";
    return redirectKeepingCookies(url, response);
  }
  return response;
}

export const config = {
  // everything except API/internal paths, generated static data and files with an extension
  matcher: ["/((?!api|_next|_vercel|data|ortho|maplibre|.*\\..*).*)"],
};
