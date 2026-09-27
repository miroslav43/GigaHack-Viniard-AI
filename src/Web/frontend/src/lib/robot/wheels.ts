// What each wheel motor does in each drive move: +1 forward, -1 back, 0 still. Set by the user in the page (which
// motor sits where, how a turn is made — e.g. two wheels on a diagonal, one forward and one back).
import type { WheelDir } from "./commands.ts";

export type DriveMove = "forward" | "back" | "left" | "right";
export const DRIVE_MOVES: readonly DriveMove[] = ["forward", "back", "left", "right"];
export type WheelMoves = Record<DriveMove, WheelDir[]>;

/** a start: motors 1–2 on the left, 3–4 on the right, turning on the spot */
export const DEFAULT_WHEEL_MOVES: WheelMoves = {
  forward: [1, 1, 1, 1],
  back: [-1, -1, -1, -1],
  left: [-1, -1, 1, 1],
  right: [1, 1, -1, -1],
};

/** A move the board accepts: four wheels, each -1 / 0 / 1, at least one turning. */
export const validMove = (dirs: readonly number[]) => dirs.length === 4 && dirs.every((d) => d === -1 || d === 0 || d === 1) && dirs.some((d) => d !== 0);

/** Saved wheel moves with anything broken replaced by the default. */
export function wheelMovesOr(saved: unknown): WheelMoves {
  const s = (saved ?? {}) as Partial<Record<DriveMove, number[]>>;
  return Object.fromEntries(DRIVE_MOVES.map((m) => [m, validMove(s[m] ?? []) ? (s[m] as WheelDir[]) : DEFAULT_WHEEL_MOVES[m]])) as WheelMoves;
}

/** Motors wired the other way round turn the opposite of what the table says. Checked on the robot: motor 2. */
export const DEFAULT_WHEEL_INVERT: readonly boolean[] = [false, true, false, false];

export const applyInvert = (dirs: readonly WheelDir[], invert: readonly boolean[]): WheelDir[] =>
  dirs.map((d, i) => (invert[i] ? (-d as WheelDir) : d)).map((d) => (d === 0 ? 0 : d)) as WheelDir[];

/** Saved inversions, or the default when missing / broken. */
export const wheelInvertOr = (saved: unknown): boolean[] =>
  Array.isArray(saved) && saved.length === 4 && saved.every((v) => typeof v === "boolean") ? [...saved] : [...DEFAULT_WHEEL_INVERT];
