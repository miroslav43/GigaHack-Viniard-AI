"use client";

import { useMemo, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import MenuItem from "@mui/material/MenuItem";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import { enUS, roRO, ruRU } from "@mui/x-data-grid/locales";
import MapOutlined from "@mui/icons-material/MapOutlined";
import { mapPalette, type RowStructure } from "@/theme/mapPalette";
import { useFormat } from "@/lib/useFormat";
import type { BlockSummary, RowRecord } from "@/lib/types";
import { Link } from "@/i18n/routing";

const GRID_LOCALE = { ro: roRO, en: enUS, ru: ruRU } as const;
const STRUCTURES = Object.keys(mapPalette.row) as RowStructure[];

function MapLink({ href, title }: { href: string; title: string }) {
  return (
    <Tooltip title={title}>
      <IconButton size="small" component={Link} href={href} aria-label={title}>
        <MapOutlined fontSize="small" />
      </IconButton>
    </Tooltip>
  );
}

export function RowsTable({ rows, blocks }: { rows: RowRecord[]; blocks: BlockSummary[] }) {
  const t = useTranslations("blocks");
  const tc = useTranslations();
  const f = useFormat();
  const locale = useLocale() as keyof typeof GRID_LOCALE;
  const localeText = GRID_LOCALE[locale].components.MuiDataGrid.defaultProps.localeText;
  const [tab, setTab] = useState<"rows" | "blocks">("rows");
  const [block, setBlock] = useState("all");
  const [structure, setStructure] = useState("all");

  const rowColumns = useMemo<GridColDef<RowRecord>[]>(
    () => [
      { field: "row_id", headerName: "row_id", width: 130 },
      { field: "vineyard_id", headerName: t("colBlock"), width: 100 },
      { field: "length_m", headerName: t("colLength"), type: "number", width: 130, valueFormatter: (v: number) => f.m(v, 2) },
      {
        field: "row_structure",
        headerName: t("colStructure"),
        width: 150,
        renderCell: (p) => (
          <Chip
            size="small"
            label={tc(`structure.${p.value as RowStructure}`)}
            sx={{ bgcolor: mapPalette.row[p.value as RowStructure], color: "grey.900", fontWeight: 600 }}
          />
        ),
      },
      { field: "plant_count", headerName: t("colPlants"), type: "number", width: 100 },
      { field: "max_gap_m", headerName: t("colMaxGap"), type: "number", width: 140, valueFormatter: (v: number) => f.m(v, 1) },
      {
        field: "map",
        headerName: "",
        width: 70,
        sortable: false,
        filterable: false,
        renderCell: (p) => <MapLink href={`/harta?rand=${p.row.row_id}`} title={tc("common.showOnMap")} />,
      },
    ],
    [t, tc, f],
  );

  const blockColumns = useMemo<GridColDef<BlockSummary>[]>(
    () => [
      { field: "vineyard_id", headerName: t("colBlock"), width: 100 },
      { field: "row_count", headerName: t("colRows"), type: "number", width: 100 },
      { field: "row_length_m", headerName: t("colRowLength"), type: "number", width: 160, valueFormatter: (v: number) => f.m(v) },
      { field: "canopy_count", headerName: t("colPlants"), type: "number", width: 100 },
      { field: "canopy_area_m2", headerName: t("colCanopyArea"), type: "number", width: 160, valueFormatter: (v: number) => f.m2(v) },
      { field: "interrow_area_m2", headerName: t("colInterrowArea"), type: "number", width: 180, valueFormatter: (v: number) => f.m2(v) },
      { field: "disrupted_rows", headerName: t("colDisrupted"), type: "number", width: 120 },
      {
        field: "map",
        headerName: "",
        width: 70,
        sortable: false,
        renderCell: (p) => <MapLink href={`/harta?bloc=${p.row.vineyard_id}`} title={tc("common.showOnMap")} />,
      },
    ],
    [t, tc, f],
  );

  const filtered = useMemo(
    () => rows.filter((r) => (block === "all" || r.vineyard_id === block) && (structure === "all" || r.row_structure === structure)),
    [rows, block, structure],
  );
  const total = filtered.reduce((s, r) => s + r.length_m, 0);

  return (
    <Card>
      <Box sx={{ px: 5, pt: 2, borderBottom: 1, borderColor: "divider" }}>
        <Tabs value={tab} onChange={(_, v) => setTab(v)}>
          <Tab value="rows" label={t("tabRows", { n: rows.length })} />
          <Tab value="blocks" label={t("tabBlocks", { n: blocks.length })} />
        </Tabs>
      </Box>
      {tab === "rows" ? (
        <>
          <Box sx={{ display: "flex", gap: 3, p: 5, flexWrap: "wrap", alignItems: "center" }}>
            <TextField select size="small" label={t("filterBlock")} value={block} onChange={(e) => setBlock(e.target.value)} sx={{ minWidth: 140 }}>
              <MenuItem value="all">{tc("common.all")}</MenuItem>
              {blocks.map((b) => (
                <MenuItem key={b.vineyard_id} value={b.vineyard_id}>
                  {b.vineyard_id}
                </MenuItem>
              ))}
            </TextField>
            <TextField select size="small" label={t("filterStructure")} value={structure} onChange={(e) => setStructure(e.target.value)} sx={{ minWidth: 170 }}>
              <MenuItem value="all">{tc("common.all")}</MenuItem>
              {STRUCTURES.map((k) => (
                <MenuItem key={k} value={k}>
                  {tc(`structure.${k}`)}
                </MenuItem>
              ))}
            </TextField>
            <Typography variant="body2" color="text.secondary" sx={{ ml: "auto" }}>
              {t("count", { n: f.int(filtered.length), length: f.length(total) })}
            </Typography>
          </Box>
          <DataGrid
            rows={filtered}
            columns={rowColumns}
            getRowId={(r) => r.row_id}
            localeText={localeText}
            density="compact"
            disableRowSelectionOnClick
            initialState={{ sorting: { sortModel: [{ field: "max_gap_m", sort: "desc" }] }, pagination: { paginationModel: { pageSize: 25 } } }}
            pageSizeOptions={[25, 50, 100]}
            sx={{ border: 0, borderRadius: 0 }}
          />
        </>
      ) : (
        <>
          <DataGrid
            rows={blocks}
            columns={blockColumns}
            getRowId={(r) => r.vineyard_id}
            localeText={localeText}
            density="compact"
            disableRowSelectionOnClick
            hideFooter
            sx={{ border: 0, borderRadius: 0 }}
          />
          <Typography variant="caption" color="text.secondary" component="p" sx={{ px: 5, py: 3 }}>
            {t("canopyNote", { area: f.area(blocks.reduce((s, b) => s + b.canopy_area_m2, 0)) })}
          </Typography>
        </>
      )}
    </Card>
  );
}
