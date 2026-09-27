import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { Page, PageHeader } from "@/components/common/PageHeader";
import { RobotConsole } from "@/components/robot/RobotConsole";

export async function generateMetadata({ params }: PageProps<"/[locale]/robot">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("robot.title")} · Solemtrix` };
}

/** Field robot console: live camera, photos, camera pan / tilt and the wheels, through /api/robot (src/lib/robot). */
export default async function RobotPage({ params }: PageProps<"/[locale]/robot">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("robot");
  return (
    <Page>
      <PageHeader title={t("title")} subtitle={t("subtitle")} />
      <RobotConsole />
    </Page>
  );
}
