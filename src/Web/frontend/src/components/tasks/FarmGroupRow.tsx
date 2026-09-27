"use client";

import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
import Box from "@mui/material/Box";
import Checkbox from "@mui/material/Checkbox";
import IconButton from "@mui/material/IconButton";
import TableCell from "@mui/material/TableCell";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import ExpandLess from "@mui/icons-material/ExpandLess";
import ExpandMore from "@mui/icons-material/ExpandMore";
import type { FarmGroup } from "@/lib/taskFarms";

/**
 * Header row of a farm in the /sarcini tables: click to show / hide the farm's rows; for the admin a checkbox that
 * ticks every row of the farm and an action (assign the whole farm). Sits in the same table as the rows it groups.
 */
export function FarmGroupRow({
  group,
  summary,
  colSpan,
  expanded,
  onToggle,
  check,
  action,
}: {
  group: FarmGroup<unknown>;
  summary: ReactNode;
  /** columns right of the checkbox column (or all columns when there is no checkbox) */
  colSpan: number;
  expanded: boolean;
  onToggle: () => void;
  check?: {
    checked: boolean;
    indeterminate: boolean;
    onChange: (on: boolean) => void;
  };
  action?: ReactNode;
}) {
  const t = useTranslations("tasks");
  const label = group.farm ? t("farm", { id: group.farm }) : t("noFarm");
  return (
    <TableRow hover onClick={onToggle} sx={{ cursor: "pointer", bgcolor: "action.hover" }} data-farm={group.key}>
      {check && (
        <TableCell padding="checkbox" onClick={(e) => e.stopPropagation()}>
          <Checkbox
            checked={check.checked}
            indeterminate={check.indeterminate}
            onChange={(e) => check.onChange(e.target.checked)}
            slotProps={{
              input: { "aria-label": t("farmSelect", { farm: label }) },
            }}
          />
        </TableCell>
      )}
      <TableCell colSpan={colSpan}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 2,
            flexWrap: "wrap",
          }}
        >
          <IconButton size="small" aria-expanded={expanded} aria-label={t("farmToggle", { farm: label })}>
            {expanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
          </IconButton>
          <Typography variant="body2" sx={{ fontWeight: 700 }}>
            {label}
          </Typography>
          {group.blocks.length > 0 && (
            <Typography variant="caption" color="text.secondary">
              {group.blocks.join(", ")}
            </Typography>
          )}
          <Typography variant="body2" color="text.secondary">
            {summary}
          </Typography>
          <Box sx={{ flex: 1 }} />
          {action && <Box onClick={(e) => e.stopPropagation()}>{action}</Box>}
        </Box>
      </TableCell>
    </TableRow>
  );
}
