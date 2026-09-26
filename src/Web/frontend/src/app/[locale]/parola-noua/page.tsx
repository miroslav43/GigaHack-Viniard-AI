import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { SetPasswordForm } from "@/components/auth/SetPasswordForm";
import { LanguageSwitch } from "@/components/shell/AppShell";
import { LogoMark } from "@/components/shell/Logo";

export async function generateMetadata({ params }: PageProps<"/[locale]/parola-noua">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("setPassword.title")} · Solemtrix`, robots: { index: false } };
}

/** Where the invitation email lands: the new member chooses a password (see SetPasswordForm). */
export default async function SetPasswordPage({ params }: PageProps<"/[locale]/parola-noua">) {
  const { locale } = await params;
  setRequestLocale(locale);

  return (
    <Box sx={{ minHeight: "100dvh", display: "flex", flexDirection: "column", px: { xs: 5, sm: 10 }, py: 6, bgcolor: "background.paper" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <LogoMark />
          <Typography sx={{ fontSize: 20, fontWeight: 700 }}>solemtrix</Typography>
        </Box>
        <LanguageSwitch />
      </Box>
      <Box sx={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", py: 8 }}>
        <Box sx={{ width: "100%", maxWidth: 440 }}>
          <SetPasswordForm />
        </Box>
      </Box>
    </Box>
  );
}
