import { readFile } from "node:fs/promises";
import path from "node:path";
import { getLocale, getTranslations } from "next-intl/server";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Typography from "@mui/material/Typography";
import SatelliteAltOutlined from "@mui/icons-material/SatelliteAltOutlined";
import { BoundaryMap } from "@/components/map/BoundaryMap";
import { makeFormat } from "@/lib/format";
import type { Viewer } from "@/lib/viewer";
import { Page, PageHeader } from "./PageHeader";

/** Shown to a municipality whose boundary contains no survey: no other municipality's data is loaded. */
export async function NoSurvey({ viewer, fullHeight = false }: { viewer: Viewer; fullHeight?: boolean }) {
  const t = await getTranslations();
  const f = makeFormat(await getLocale());
  const uats = JSON.parse(await readFile(path.join(process.cwd(), "public", "data", "uats.json"), "utf8")) as Record<
    string,
    { area_ha: number }
  >;
  const name = t(`uat.${viewer.uat.key}.name`);

  return (
    <Page>
      <PageHeader title={name} subtitle={t(`uat.${viewer.uat.key}.place`)} />
      <Card sx={{ overflow: "hidden" }}>
        <Box sx={{ p: 6, display: "flex", gap: 4, alignItems: "flex-start" }}>
          <SatelliteAltOutlined color="primary" sx={{ fontSize: 36 }} />
          <Box>
            <Typography variant="h3">{t("noSurvey.title", { uat: name })}</Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mt: 1, maxWidth: 720 }}>
              {t("noSurvey.body")}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              {t("noSurvey.boundary", {
                area: `${f.num(uats[viewer.uat.key]?.area_ha ?? 0, 0)} ${f.units.ha}`,
                id: viewer.uat.osmRelationId,
              })}
            </Typography>
          </Box>
        </Box>
        <Box sx={{ height: fullHeight ? "calc(100dvh - 320px)" : 420, minHeight: 320, borderTop: 1, borderColor: "divider" }}>
          <BoundaryMap geofenceUrl={viewer.uat.geofenceUrl} />
        </Box>
      </Card>
    </Page>
  );
}
