"use client";

import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import { useInView } from "@/lib/motion/useInView";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";

const REVEAL_MS = 600;
const OFFSET_PX = 18;

/**
 * Fades and lifts its content in the first time it comes on screen. With `mount`, the content is only rendered then,
 * so charts play their own entry animation in view; `minHeight` keeps the layout from jumping meanwhile.
 * Reduced motion: shown straight away, no transition.
 */
export function Reveal({ children, delayMs = 0, mount = false, minHeight }: { children: ReactNode; delayMs?: number; mount?: boolean; minHeight?: number }) {
  const reduced = useReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>({ rootMargin: "0px 0px -48px 0px" });
  const shown = inView || reduced;
  return (
    <Box
      ref={ref}
      sx={{
        minHeight,
        opacity: shown ? 1 : 0,
        transform: shown ? "none" : `translateY(${OFFSET_PX}px)`,
        transition: reduced ? "none" : `opacity ${REVEAL_MS}ms ease-out ${delayMs}ms, transform ${REVEAL_MS}ms ease-out ${delayMs}ms`,
      }}
    >
      {!mount || shown ? children : null}
    </Box>
  );
}
