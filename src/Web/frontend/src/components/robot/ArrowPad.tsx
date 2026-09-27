"use client";

// A cross of four arrow buttons with an optional centre button (stop / photo), used for the wheels and the camera.
import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import KeyboardArrowDown from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowLeft from "@mui/icons-material/KeyboardArrowLeft";
import KeyboardArrowRight from "@mui/icons-material/KeyboardArrowRight";
import KeyboardArrowUp from "@mui/icons-material/KeyboardArrowUp";

export type Arrow = "up" | "down" | "left" | "right";

const ICON: Record<Arrow, ReactNode> = {
  up: <KeyboardArrowUp fontSize="large" />,
  down: <KeyboardArrowDown fontSize="large" />,
  left: <KeyboardArrowLeft fontSize="large" />,
  right: <KeyboardArrowRight fontSize="large" />,
};
/** row / column in the 3 × 3 grid */
const AREA: Record<Arrow, string> = { up: "1 / 2", left: "2 / 1", right: "2 / 3", down: "3 / 2" };

export function ArrowPad({
  labels,
  disabled,
  busy,
  onPress,
  centre,
  testId,
}: {
  labels: Record<Arrow, string>;
  disabled: boolean;
  /** a command is running: the arrows wait */
  busy: boolean;
  onPress: (a: Arrow) => void;
  centre?: ReactNode;
  testId: string;
}) {
  return (
    <Box data-testid={testId} sx={{ display: "grid", gridTemplateColumns: "repeat(3, 64px)", gridTemplateRows: "repeat(3, 64px)", gap: 1, justifyContent: "center" }}>
      {(Object.keys(ICON) as Arrow[]).map((a) => (
        <Box key={a} sx={{ gridArea: AREA[a] }}>
          <Tooltip title={labels[a]}>
            <span>
              <IconButton
                aria-label={labels[a]}
                disabled={disabled || busy}
                onClick={() => onPress(a)}
                sx={{ width: 64, height: 64, bgcolor: "action.hover", borderRadius: 2 }}
              >
                {ICON[a]}
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      ))}
      {centre && <Box sx={{ gridArea: "2 / 2", display: "grid", placeItems: "center" }}>{centre}</Box>}
    </Box>
  );
}
