import createMiddleware from "next-intl/middleware";
import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";
import { routing } from "./i18n/routing";
import { AUTH_ENABLED, DEMO_COOKIE, SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL } from "./lib/supabase/config";

const intl = createMiddleware(routing);

const PUBLIC_PATHS = ["/login"];

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

export default async function proxy(request: NextRequest) {
  const response = intl(request);
  if (!AUTH_ENABLED || response.headers.get("location")) return response;

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

  const { prefix, path } = splitLocale(request.nextUrl.pathname);
  const isPublic = PUBLIC_PATHS.some((p) => path === p || path.startsWith(`${p}/`));

  if (!signedIn && !demo && !isPublic) {
    const url = request.nextUrl.clone();
    url.pathname = `${prefix}/login`;
    url.search = path === "/" ? "" : `?next=${encodeURIComponent(path + request.nextUrl.search)}`;
    return redirectKeepingCookies(url, response);
  }
  if (signedIn && isPublic) {
    const url = request.nextUrl.clone();
    url.pathname = prefix || "/";
    url.search = "";
    return redirectKeepingCookies(url, response);
  }
  return response;
}

export const config = {
  // everything except API/internal paths, generated static data and files with an extension
  matcher: ["/((?!api|_next|_vercel|data|ortho|maplibre|.*\\..*).*)"],
};
