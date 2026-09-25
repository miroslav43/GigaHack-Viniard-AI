import { defineRouting } from "next-intl/routing";
import { createNavigation } from "next-intl/navigation";

export const routing = defineRouting({
  locales: ["ro", "en", "ru"],
  defaultLocale: "ro",
  // Romanian URLs stay unprefixed (/harta); /en/harta, /ru/harta
  localePrefix: "as-needed",
  // Romanian is the product default (and the pitch language) regardless of the browser's Accept-Language;
  // a language picked in the UI is still remembered in the NEXT_LOCALE cookie.
  localeDetection: false,
});

export type Locale = (typeof routing.locales)[number];

export const { Link, redirect, usePathname, useRouter, getPathname } = createNavigation(routing);
