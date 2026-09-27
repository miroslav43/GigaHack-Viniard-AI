// Client side of /api/robot: one call per command; errors come back as a short code the page translates.
import type { Device } from "@/lib/robot/commands";

export type RobotError = "no_address" | "bad_address" | "bad_request" | "sign_in" | "unreachable" | "timeout" | "failed";

export class RobotCallError extends Error {
  constructor(readonly code: RobotError) {
    super(code);
  }
}

export async function callRobot(device: Device, base: string, params: Record<string, string | number>): Promise<Response> {
  if (!base.trim()) throw new RobotCallError("no_address");
  const q = new URLSearchParams({ device, base, ...Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)])) });
  let res: Response;
  try {
    res = await fetch(`/api/robot?${q}`, { cache: "no-store" });
  } catch {
    throw new RobotCallError("unreachable");
  }
  if (res.ok) return res;
  const body = (await res.json().catch(() => null)) as { error?: RobotError } | null;
  throw new RobotCallError(body?.error ?? "failed");
}
