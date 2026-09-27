// The robot's boards and the only requests the site may send them (the /api/robot proxy builds every upstream URL
// here). The boards are ESP32s on the local Wi-Fi (the team's Arduino sketches):
// - cam: ESP32-CAM (AI Thinker) — /stream (MJPEG, shown directly by the page), /capture, /flash/*, /json;
// - motors: ESP32 with two stepper drivers, the camera's pan (motor 1) and tilt (motor 2), and an HC-SR04 —
//   /move?motor&dir&speed&steps (answers when the motor has stopped), /toggle_en?motor, /distance;
// - drive: the wheels' board, four DC motors — /dir?m=<1..4>&val=±1, /speed?m=&val=<0..255>, /start?m=<k|all>,
//   /stop?m=<k|all>, /status. Two ways to drive: `run` (a timed move: the proxy sets the wheels, starts them, waits
//   and always sends the stop at the end) and `go` (hold to drive: the wheels start and keep going while the page
//   sends `keep` every few hundred ms; a server watchdog stops them when that stops, e.g. a dropped connection).
// Only private IPv4 addresses are accepted, so the proxy cannot be pointed at the internet or at this server.

export type Device = "cam" | "motors" | "drive";
export const DEVICES: readonly Device[] = ["cam", "motors", "drive"];

export const LIMITS = {
  speedPps: [50, 5000],
  steps: [1, 20000],
  driveMs: [100, 10000],
  wheelSpeed: [0, 255],
} as const;

export const WHEELS = 4;
/** per wheel motor: +1 / -1 = turn that way, 0 = stay still */
export type WheelDir = -1 | 0 | 1;

export type Command =
  | { device: "cam"; cmd: "capture" | "status" | "flash_on" | "flash_off" | "flash_toggle" }
  | { device: "motors"; cmd: "move"; motor: 1 | 2; dir: 0 | 1; speed: number; steps: number }
  | { device: "motors"; cmd: "toggle_en"; motor: 1 | 2 }
  | { device: "motors"; cmd: "distance" }
  | { device: "drive"; cmd: "run"; dirs: WheelDir[]; speed: number; ms: number }
  | { device: "drive"; cmd: "go"; dirs: WheelDir[]; speed: number }
  | { device: "drive"; cmd: "stop" | "status" | "keep" };

/** What the proxy sends for a command: these requests in order, then (for a drive move) a wait and a stop that is
 *  sent even when a step failed. */
export interface UpstreamPlan {
  steps: string[];
  holdMs?: number;
  always?: string;
}

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
    if (cmd === "distance") return { device, cmd };
    const motor = int(q.get("motor"));
    if (motor !== 1 && motor !== 2) return null;
    if (cmd === "toggle_en") return { device, cmd, motor };
    if (cmd !== "move") return null;
    const dir = int(q.get("dir")), speed = int(q.get("speed")), steps = int(q.get("steps"));
    if ((dir !== 0 && dir !== 1) || !inRange(speed, LIMITS.speedPps) || !inRange(steps, LIMITS.steps)) return null;
    return { device, cmd, motor, dir, speed, steps };
  }
  if (device === "drive") {
    if (cmd === "stop" || cmd === "status" || cmd === "keep") return { device, cmd };
    if (cmd !== "run" && cmd !== "go") return null;
    const dirs = (q.get("dirs") ?? "").split(",").map(int);
    const speed = int(q.get("speed"));
    if (dirs.length !== WHEELS || dirs.some((d) => d !== -1 && d !== 0 && d !== 1) || dirs.every((d) => d === 0)) return null;
    if (!inRange(speed, LIMITS.wheelSpeed)) return null;
    if (cmd === "go") return { device, cmd, dirs: dirs as WheelDir[], speed };
    const ms = int(q.get("ms"));
    if (!inRange(ms, LIMITS.driveMs)) return null;
    return { device, cmd, dirs: dirs as WheelDir[], speed, ms };
  }
  return null;
}

/** The board requests of a command. */
export function upstreamPlan(c: Command): UpstreamPlan {
  switch (c.device) {
    case "cam":
      return { steps: [{ capture: "/capture", status: "/json", flash_on: "/flash/on", flash_off: "/flash/off", flash_toggle: "/flash/toggle" }[c.cmd]] };
    case "motors":
      if (c.cmd === "distance") return { steps: ["/distance"] };
      if (c.cmd === "toggle_en") return { steps: [`/toggle_en?motor=${c.motor}`] };
      return { steps: [`/move?motor=${c.motor}&dir=${c.dir}&speed=${c.speed}&steps=${c.steps}`] };
    case "drive": {
      if (!("dirs" in c)) return { steps: c.cmd === "keep" ? [] : [c.cmd === "stop" ? "/stop?m=all" : "/status"] };
      const moving = c.dirs.flatMap((d, i) => (d === 0 ? [] : [{ m: i + 1, d }]));
      const all = moving.length === WHEELS;
      const sameDir = all && moving.every(({ d }) => d === moving[0].d);
      // few requests (each is a Wi-Fi round trip): one speed and, going straight, one direction for all four
      return {
        steps: [
          "/stop?m=all",
          ...(all ? [`/speed?m=all&val=${c.speed}`] : moving.map(({ m }) => `/speed?m=${m}&val=${c.speed}`)),
          ...(sameDir ? [`/dir?m=all&val=${moving[0].d}`] : moving.map(({ m, d }) => `/dir?m=${m}&val=${d}`)),
          // all four at once when they all turn, so the robot does not start crooked
          ...(all ? ["/start?m=all"] : moving.map(({ m }) => `/start?m=${m}`)),
        ],
        ...(c.cmd === "run" ? { holdMs: c.ms, always: "/stop?m=all" } : {}),
      };
    }
  }
}

/** How long the proxy waits for one board request: a camera move answers only when the motor has stopped. */
export function stepTimeoutMs(c: Command): number {
  if (c.device === "motors" && c.cmd === "move") return Math.min(60_000, 2_000 + Math.ceil((c.steps / c.speed) * 1000) * 2);
  return 5_000;
}
