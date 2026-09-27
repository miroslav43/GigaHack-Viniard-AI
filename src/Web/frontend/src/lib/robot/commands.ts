// The robot's boards and the only requests the site may send them (the /api/robot proxy builds every upstream URL
// here). The boards are ESP32s on the local Wi-Fi, their firmware in the team's Arduino sketches:
// - cam: ESP32-CAM (AI Thinker) — /stream (MJPEG, shown directly by the page), /capture, /flash/*, /json;
// - motors: ESP32 with two stepper drivers — the camera's pan (motor 1) and tilt (motor 2): /move, /toggle_en;
// - drive: the wheels' board — /drive?cmd=forward|back|left|right|stop&ms=<duration> (the contract the page expects).
// Only private IPv4 addresses are accepted, so the proxy cannot be pointed at the internet or at this server.

export type Device = "cam" | "motors" | "drive";
export const DEVICES: readonly Device[] = ["cam", "motors", "drive"];

export const LIMITS = {
  speedPps: [50, 5000],
  steps: [1, 20000],
  driveMs: [100, 10000],
} as const;

export const DRIVE_CMDS = ["forward", "back", "left", "right", "stop"] as const;
export type DriveCmd = (typeof DRIVE_CMDS)[number];

export type Command =
  | { device: "cam"; cmd: "capture" | "status" | "flash_on" | "flash_off" | "flash_toggle" }
  | { device: "motors"; cmd: "move"; motor: 1 | 2; dir: 0 | 1; speed: number; steps: number }
  | { device: "motors"; cmd: "toggle_en"; motor: 1 | 2 }
  | { device: "drive"; cmd: DriveCmd; ms: number };

/** `http://<private IPv4>[:port]` with nothing after it, normalised; null otherwise (a strict a.b.c.d: URL parsing
 *  would read "192.168.1" as 192.168.0.1). */
export function parseBoardUrl(raw: string): string | null {
  const m = /^(?:http:\/\/)?(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?::(\d{1,5}))?\/?$/.exec(raw.trim());
  if (!m) return null;
  const [a, b, c, d] = m.slice(1, 5).map(Number);
  const port = m[5] === undefined ? null : Number(m[5]);
  if ([a, b, c, d].some((p) => p > 255) || (port !== null && (port < 1 || port > 65535))) return null;
  const privateNet = a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168);
  return privateNet ? `http://${a}.${b}.${c}.${d}${port === null ? "" : `:${port}`}` : null;
}

const int = (v: string | null) => (v !== null && /^-?\d+$/.test(v) ? Number(v) : NaN);
const inRange = (n: number, [lo, hi]: readonly [number, number]) => Number.isInteger(n) && n >= lo && n <= hi;

/** A command from the proxy's query string, or null when anything is off. */
export function parseCommand(q: URLSearchParams): Command | null {
  const device = q.get("device"), cmd = q.get("cmd");
  if (device === "cam") {
    return cmd === "capture" || cmd === "status" || cmd === "flash_on" || cmd === "flash_off" || cmd === "flash_toggle" ? { device, cmd } : null;
  }
  if (device === "motors") {
    const motor = int(q.get("motor"));
    if (motor !== 1 && motor !== 2) return null;
    if (cmd === "toggle_en") return { device, cmd, motor };
    if (cmd !== "move") return null;
    const dir = int(q.get("dir")), speed = int(q.get("speed")), steps = int(q.get("steps"));
    if ((dir !== 0 && dir !== 1) || !inRange(speed, LIMITS.speedPps) || !inRange(steps, LIMITS.steps)) return null;
    return { device, cmd, motor, dir, speed, steps };
  }
  if (device === "drive") {
    const ms = int(q.get("ms"));
    if (!DRIVE_CMDS.includes(cmd as DriveCmd) || !inRange(ms, LIMITS.driveMs)) return null;
    return { device, cmd: cmd as DriveCmd, ms };
  }
  return null;
}

/** The board path (with query) of a command. */
export function upstreamPath(c: Command): string {
  switch (c.device) {
    case "cam":
      return { capture: "/capture", status: "/json", flash_on: "/flash/on", flash_off: "/flash/off", flash_toggle: "/flash/toggle" }[c.cmd];
    case "motors":
      return c.cmd === "toggle_en"
        ? `/toggle_en?motor=${c.motor}`
        : `/move?motor=${c.motor}&dir=${c.dir}&speed=${c.speed}&steps=${c.steps}`;
    case "drive":
      return `/drive?cmd=${c.cmd}&ms=${c.ms}`;
  }
}

/** How long the proxy waits for the board: a move answers only when the motor has stopped. */
export function timeoutMs(c: Command): number {
  if (c.device === "motors" && c.cmd === "move") return Math.min(60_000, 2_000 + Math.ceil((c.steps / c.speed) * 1000) * 2);
  if (c.device === "drive") return Math.min(20_000, 2_000 + c.ms * 2);
  return 8_000;
}
