import { Suspense } from "react";
import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import CloudDownloadOutlined from "@mui/icons-material/CloudDownloadOutlined";
import GroupAddOutlined from "@mui/icons-material/GroupAddOutlined";
import TaskAltOutlined from "@mui/icons-material/TaskAltOutlined";
import { LoginForm } from "@/components/auth/LoginForm";
import { LanguageSwitch } from "@/components/shell/AppShell";
import { LogoMark } from "@/components/shell/Logo";
import { LinkButton } from "@/components/common/links";

export async function generateMetadata({ params }: PageProps<"/[locale]/login">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale });
  return { title: `${t("auth.signIn")} · Solemtrix` };
}

/** Dashed annotation box, the Marcaj-style motif of the hero panel. */
function Tag({ label, sx }: { label: string; sx: object }) {
  return (
    <Box
      aria-hidden
      sx={{
        position: "absolute",
        border: "1.5px dashed",
        borderColor: "primary.light",
        opacity: 0.55,
        px: 2,
        pt: 1,
        fontSize: 12,
        fontWeight: 600,
        letterSpacing: 0.6,
        color: "primary.light",
        ...sx,
      }}
    >
      {label}
    </Box>
  );
}

export default async function LoginPage({ params }: PageProps<"/[locale]/login">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("auth");
  const features = [
    { icon: <CloudDownloadOutlined />, title: t("featData"), hint: t("featDataHint") },
    { icon: <GroupAddOutlined />, title: t("featTeams"), hint: t("featTeamsHint") },
    { icon: <TaskAltOutlined />, title: t("featAudit"), hint: t("featAuditHint") },
  ];

  return (
    <Box sx={{ minHeight: "100dvh", display: "grid", gridTemplateColumns: { xs: "1fr", md: "1.15fr 1fr" }, bgcolor: "background.paper" }}>
      <Box
        sx={{
          display: { xs: "none", md: "flex" },
          flexDirection: "column",
          justifyContent: "space-between",
          position: "relative",
          overflow: "hidden",
          m: 3,
          borderRadius: 4,
          p: 10,
          color: "common.white",
          bgcolor: "var(--solemtrix-night)",
          backgroundImage:
            "radial-gradient(circle at 70% 45%, var(--mui-palette-primary-dark) 0%, transparent 45%), radial-gradient(circle at 15% 90%, var(--mui-palette-primary-main) 0%, transparent 35%)",
        }}
      >
        <Tag label="ROW_V03-R017" sx={{ top: "22%", right: "8%", width: 190, height: 110 }} />
        <Tag label="CANOPY_0412" sx={{ top: "9%", left: "46%", width: 130, height: 90 }} />
        <Tag label="GAP_7.4M" sx={{ bottom: "20%", right: "14%", width: 170, height: 70 }} />
        <Box sx={{ display: "flex", alignItems: "center", gap: 3, position: "relative" }}>
          <LogoMark size={44} />
          <Typography sx={{ fontSize: 30, fontWeight: 700, letterSpacing: -0.5 }}>solemtrix</Typography>
        </Box>
        <Box sx={{ position: "relative", maxWidth: 560 }}>
          <Typography component="h2" sx={{ fontSize: 40, fontWeight: 700, lineHeight: 1.15, letterSpacing: -0.8 }}>
            {t("heroTitle")}
          </Typography>
          <Typography sx={{ mt: 4, fontSize: 18, lineHeight: 1.5, opacity: 0.75 }}>{t("heroSubtitle")}</Typography>
        </Box>
        <Box sx={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 6, position: "relative" }}>
          {features.map((f) => (
            <Box key={f.title}>
              <Box sx={{ color: "primary.light", "& svg": { fontSize: 36 } }}>{f.icon}</Box>
              <Typography sx={{ fontWeight: 600, fontSize: 17, mt: 2 }}>{f.title}</Typography>
              <Typography sx={{ fontSize: 14, opacity: 0.7, mt: 1 }}>{f.hint}</Typography>
            </Box>
          ))}
        </Box>
      </Box>

      <Box sx={{ display: "flex", flexDirection: "column", px: { xs: 5, sm: 10 }, py: 6 }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Box sx={{ display: { xs: "flex", md: "none" }, alignItems: "center", gap: 2 }}>
            <LogoMark />
            <Typography sx={{ fontSize: 20, fontWeight: 700 }}>solemtrix</Typography>
          </Box>
          <Box sx={{ ml: "auto", display: "flex", alignItems: "center", gap: 4 }}>
            <LinkButton href="/prezentare" size="small">
              {t("about")}
            </LinkButton>
            <LanguageSwitch />
          </Box>
        </Box>
        <Box sx={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", py: 8 }}>
          <Box sx={{ width: "100%", maxWidth: 440 }}>
            <Suspense>
              <LoginForm />
            </Suspense>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
