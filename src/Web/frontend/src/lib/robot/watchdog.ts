// Hold-to-drive safety: while the wheels run from a `go`, the page renews this every few hundred ms (`keep`); when
// the renewals stop (the button released without a stop reaching us, the tab closed, the Wi-Fi dropped) the wheels
// get a stop. One timer per wheel board, in this server's memory.
import "server-only";

/** no renewal for this long: stop the wheels (ms) */
export const WATCHDOG_MS = 800;

const timers = new Map<string, ReturnType<typeof setTimeout>>();

/** Arms (or renews) the watchdog of a board; `onExpire` sends the stop. */
export function arm(base: string, onExpire: () => void) {
  disarm(base);
  timers.set(
    base,
    setTimeout(() => {
      timers.delete(base);
      onExpire();
    }, WATCHDOG_MS),
  );
}

/** Renews a running watchdog; false when none runs (the wheels already stopped). */
export function renew(base: string, onExpire: () => void): boolean {
  if (!timers.has(base)) return false;
  arm(base, onExpire);
  return true;
}

export function disarm(base: string) {
  const t = timers.get(base);
  if (t) clearTimeout(t);
  timers.delete(base);
}
