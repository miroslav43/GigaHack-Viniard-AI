import { getTranslations } from "next-intl/server";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import LockOutlined from "@mui/icons-material/LockOutlined";
import { LinkButton } from "./links";

/** 403 page body. Deliberately generic: it does not say what the protected page is. */
export async function AccessDenied() {
  const t = await getTranslations("accessDenied");
  return (
    <Box sx={{ minHeight: "70dvh", display: "grid", placeItems: "center", px: 5, textAlign: "center" }}>
      <Box sx={{ maxWidth: 460, display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
        <LockOutlined color="error" sx={{ fontSize: 56 }} />
        <Typography sx={{ fontSize: 64, fontWeight: 700, lineHeight: 1, color: "text.disabled" }}>403</Typography>
        <Typography variant="h1" component="h1">
          {t("title")}
        </Typography>
        <Typography color="text.secondary">{t("body")}</Typography>
        <LinkButton variant="contained" href="/" sx={{ mt: 2 }}>
          {t("home")}
        </LinkButton>
      </Box>
    </Box>
  );
}
