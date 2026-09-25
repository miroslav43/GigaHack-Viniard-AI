import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { AccessDenied } from "@/components/common/AccessDenied";

// Rendered by proxy.ts (rewrite with HTTP 403) for protected paths the caller may not see.
export async function generateMetadata({ params }: PageProps<"/[locale]/acces-interzis">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("accessDenied.title")} · Solemtrix`, robots: { index: false } };
}

export default async function AccessDeniedPage({ params }: PageProps<"/[locale]/acces-interzis">) {
  const { locale } = await params;
  setRequestLocale(locale);
  return <AccessDenied />;
}
