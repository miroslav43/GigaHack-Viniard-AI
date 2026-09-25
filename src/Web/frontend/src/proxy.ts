import createMiddleware from "next-intl/middleware";
import { routing } from "./i18n/routing";

export default createMiddleware(routing);

export const config = {
  // everything except API/internal paths, generated static data and files with an extension
  matcher: ["/((?!api|_next|_vercel|data|ortho|maplibre|.*\\..*).*)"],
};
