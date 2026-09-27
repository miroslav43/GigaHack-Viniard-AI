"use client";

// A cross of four hold-to-move arrow buttons with an optional centre button, used for the wheels and the camera: the
// move runs while the button is held (pointer captured, so a release outside the button still ends it).
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
  active,
  onHold,
  onRelease,
  centre,
  testId,
}: {
  labels: Record<Arrow, string>;
  disabled: boolean;
  /** the arrow being held (drawn pressed) */
  active: Arrow | null;
  onHold: (a: Arrow) => void;
  onRelease: () => void;
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
                aria-pressed={active === a}
                disabled={disabled}
                onPointerDown={(e) => {
                  e.currentTarget.setPointerCapture(e.pointerId);
                  onHold(a);
                }}
                onPointerUp={onRelease}
                onPointerCancel={onRelease}
                onLostPointerCapture={onRelease}
                onContextMenu={(e) => e.preventDefault()}
                sx={{
                  width: 64, height: 64, borderRadius: 2, touchAction: "none", userSelect: "none",
                  bgcolor: active === a ? "primary.main" : "action.hover",
                  color: active === a ? "primary.contrastText" : undefined,
                  "&:hover": { bgcolor: active === a ? "primary.dark" : undefined },
                }}
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
