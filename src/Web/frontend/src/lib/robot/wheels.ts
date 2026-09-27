// Wheel directions of a drive move: each motor turns forward (+1) or back (-1) by its side, flipped for a motor
// wired the other way round. Left / right turn on the spot (the two sides in opposite directions).
import type { WheelDir } from "./commands.ts";

export type DriveMove = "forward" | "back" | "left" | "right";
export type Side = "left" | "right";

const SIDE_DIR: Record<DriveMove, Record<Side, 1 | -1>> = {
  forward: { left: 1, right: 1 },
  back: { left: -1, right: -1 },
  left: { left: -1, right: 1 },
  right: { left: 1, right: -1 },
};

export function wheelDirs(move: DriveMove, sides: readonly Side[], invert: readonly boolean[]): WheelDir[] {
  return sides.map((side, i) => (SIDE_DIR[move][side] * (invert[i] ? -1 : 1)) as WheelDir);
}
