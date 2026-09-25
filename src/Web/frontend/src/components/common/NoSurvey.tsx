import { getLocale, getTranslations } from "next-intl/server";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Typography from "@mui/material/Typography";
import SatelliteAltOutlined from "@mui/icons-material/SatelliteAltOutlined";
import PersonOffOutlined from "@mui/icons-material/PersonOffOutlined";
import { BoundaryMap } from "@/components/map/BoundaryMap";
import { makeFormat } from "@/lib/format";
import type { Viewer } from "@/lib/viewer";
import { Page, PageHeader } from "./PageHeader";

/**
 * Shown when the viewer's municipality has no survey inside its boundary (or the account has no active
 * municipality). No other municipality's data is loaded.
 */
export async function NoSurvey({ viewer, fullHeight = false }: { viewer: Viewer; fullHeight?: boolean }) {
  const t = await getTranslations();
  const f = makeFormat(await getLocale());
  const uat = viewer.uat;

  if (!uat) {
    return (
      <Page>
        <PageHeader title={t("noSurvey.unassignedTitle")} />
        <Card sx={{ p: 6, display: "flex", gap: 4, alignItems: "flex-start" }}>
          <PersonOffOutlined color="warning" sx={{ fontSize: 36 }} />
          <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 720 }}>
            {t("noSurvey.unassignedBody")}
          </Typography>
        </Card>
      </Page>
    );
  }

  return (
    <Page>
      <PageHeader title={t("uat.townHall", { name: uat.name })} subtitle={[uat.district, t(`country.${uat.country}`)].filter(Boolean).join(" · ")} />
      <Card sx={{ overflow: "hidden" }}>
        <Box sx={{ p: 6, display: "flex", gap: 4, alignItems: "flex-start" }}>
          <SatelliteAltOutlined color="primary" sx={{ fontSize: 36 }} />
          <Box>
            <Typography variant="h3">{t("noSurvey.title")}</Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mt: 1, maxWidth: 720 }}>
              {t("noSurvey.body")}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              {t("noSurvey.boundary", { area: `${f.num(uat.areaHa, 0)} ${f.units.ha}`, id: uat.osmRelationId ?? "—" })}
            </Typography>
          </Box>
        </Box>
        <Box sx={{ height: fullHeight ? "calc(100dvh - 320px)" : 420, minHeight: 320, borderTop: 1, borderColor: "divider" }}>
          <BoundaryMap geofence={uat.geofence} />
        </Box>
      </Card>
    </Page>
  );
}
