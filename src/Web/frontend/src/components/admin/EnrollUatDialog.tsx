"use client";

import { useState, type FormEvent } from "react";
import { useTranslations } from "next-intl";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import List from "@mui/material/List";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import MenuItem from "@mui/material/MenuItem";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";
import SearchOutlined from "@mui/icons-material/SearchOutlined";
import { BoundaryMap } from "@/components/map/BoundaryMap";
import { enrollUat, previewOsm, searchOsm } from "@/app/[locale]/(app)/super-admin/actions";
import type { OsmBoundary, OsmSearchHit } from "@/lib/osm";
import { slugify, useAdminAction } from "./common";

export function EnrollUatDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const t = useTranslations("superAdmin");
  const { run, pending, snackbar, message } = useAdminAction();
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<OsmSearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [boundary, setBoundary] = useState<OsmBoundary | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ key: "", name: "", district: "", country: "MD" as "MD" | "RO" });

  const reset = () => {
    setQuery("");
    setHits(null);
    setBoundary(null);
    setError(null);
  };

  const rankLabel = (rank: number) =>
    t.has(`uat.rank.${rank}`) ? t(`uat.rank.${rank}`) : t("uat.rank.other", { rank });

  const onSearch = async (e: FormEvent) => {
    e.preventDefault();
    setSearching(true);
    setError(null);
    setBoundary(null);
    const res = await searchOsm(query);
    setSearching(false);
    if (res.ok) setHits(res.data);
    else setError(message(res.error));
  };

  const onPick = async (hit: OsmSearchHit) => {
    setLoadingPreview(true);
    setError(null);
    const res = await previewOsm(hit.osmRelationId);
    setLoadingPreview(false);
    if (!res.ok) return setError(message(res.error));
    setBoundary(res.data);
    setForm({ key: slugify(res.data.name), name: res.data.name, district: res.data.district, country: res.data.country });
  };

  const onEnroll = () =>
    boundary &&
    run(() => enrollUat({ ...form, osmRelationId: boundary.osmRelationId }), () => {
      reset();
      onClose();
    });

  return (
    <>
      <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
        <DialogTitle>{t("uat.dialogTitle")}</DialogTitle>
        <DialogContent dividers>
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "380px 1fr" }, gap: 5, minHeight: 480 }}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
              <Box component="form" onSubmit={onSearch} sx={{ display: "flex", gap: 2 }}>
                <TextField
                  size="small"
                  fullWidth
                  label={t("uat.searchLabel")}
                  placeholder={t("uat.searchPlaceholder")}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  autoFocus
                />
                <Button type="submit" variant="contained" disabled={searching || query.trim().length < 2} startIcon={<SearchOutlined />}>
                  {t("uat.search")}
                </Button>
              </Box>
              <Typography variant="caption" color="text.secondary">
                {t("uat.searchHint")} {t("uat.rankHint")}
              </Typography>
              {error && <Alert severity="error">{error}</Alert>}
              {searching && <CircularProgress size={24} sx={{ alignSelf: "center" }} />}
              {hits && !searching && hits.length === 0 && <Alert severity="info">{t("uat.noResults")}</Alert>}
              {hits && hits.length > 0 && (
                <List dense sx={{ border: 1, borderColor: "divider", borderRadius: 2, maxHeight: 360, overflow: "auto", py: 0 }}>
                  {hits.map((h) => (
                    <ListItemButton
                      key={h.osmRelationId}
                      selected={boundary?.osmRelationId === h.osmRelationId}
                      onClick={() => onPick(h)}
                      // light selection here: the theme's filled-indigo selection hides the secondary text and chip
                      sx={{
                        borderRadius: 0,
                        "&.Mui-selected": { bgcolor: "action.selected", color: "text.primary" },
                        "&.Mui-selected:hover": { bgcolor: "action.hover" },
                      }}
                    >
                      <ListItemText
                        primary={
                          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                            <span>{h.name}</span>
                            <Chip size="small" label={rankLabel(h.placeRank)} color={h.placeRank === 16 ? "primary" : "default"} variant="outlined" />
                          </Box>
                        }
                        secondary={`${h.displayName} · OSM ${h.osmRelationId}`}
                      />
                    </ListItemButton>
                  ))}
                </List>
              )}
              {boundary && (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 3, pt: 2 }}>
                  <TextField size="small" label={t("uat.fieldName")} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
                  <TextField
                    size="small"
                    label={t("uat.fieldKey")}
                    value={form.key}
                    onChange={(e) => setForm({ ...form, key: slugify(e.target.value) })}
                  />
                  <TextField size="small" label={t("uat.fieldDistrict")} value={form.district} onChange={(e) => setForm({ ...form, district: e.target.value })} />
                  <TextField select size="small" label={t("uat.fieldCountry")} value={form.country} onChange={(e) => setForm({ ...form, country: e.target.value as "MD" | "RO" })}>
                    <MenuItem value="MD">MD</MenuItem>
                    <MenuItem value="RO">RO</MenuItem>
                  </TextField>
                  <Typography variant="caption" color="text.secondary">
                    {t("uat.boundarySource", { id: boundary.osmRelationId })}
                  </Typography>
                </Box>
              )}
            </Box>
            <Box sx={{ position: "relative", borderRadius: 2, overflow: "hidden", border: 1, borderColor: "divider", minHeight: 420 }}>
              <BoundaryMap geofence={boundary?.geometry ?? null} />
              {loadingPreview && (
                <Box sx={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", bgcolor: "action.hover" }}>
                  <CircularProgress />
                </Box>
              )}
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              reset();
              onClose();
            }}
          >
            {t("common.cancel")}
          </Button>
          <Button variant="contained" onClick={onEnroll} disabled={!boundary || pending || !form.key || form.name.trim().length < 2}>
            {t("uat.enrollSubmit")}
          </Button>
        </DialogActions>
      </Dialog>
      {snackbar}
    </>
  );
}
