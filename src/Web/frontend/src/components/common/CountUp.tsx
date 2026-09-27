"use client";

import { useEffect, useState } from "react";
import type { Format } from "@/lib/format";
import { useFormat } from "@/lib/useFormat";
import { easeOutCubic } from "@/lib/motion/easing";
import { useInView } from "@/lib/motion/useInView";
import { useReducedMotion } from "@/lib/motion/useReducedMotion";

// A figure that counts up from 0 the first time it comes on screen (docs/STATISTICI.md). The server, and anyone with
// "reduce motion" on, get the final value straight away; the final text is always what stays.
export type CountFormat = "int" | "num" | "length" | "km" | "m" | "m2" | "ha" | "pct" | "min";

export const COUNT_UP_MS = 1200;

/**
 * Formats `v` the way the final value `end` is formatted: a length that ends in km counts in km, a hectare figure
 * keeps the decimals of its final value, so the unit and the width do not jump mid-count.
 */
function formatAs(f: Format, format: CountFormat, v: number, end: number, digits?: number): string {
  switch (format) {
    case "int":
      return f.int(Math.round(v));
    case "num":
      return f.num(v, digits);
    case "length":
      return end >= 1000 ? f.km(v) : f.m(v);
    case "km":
      return f.km(v);
    case "m":
      return f.m(v, digits);
    case "m2":
      return f.m2(v, digits);
    case "ha":
      return f.ha(v, digits ?? (end / 1e4 < 1 ? 4 : 2));
    case "pct":
      return f.pct(v);
    case "min":
      return f.min(v);
  }
}

export function CountUp({
  value,
  format = "int",
  digits,
  prefix = "",
  suffix = "",
  durationMs = COUNT_UP_MS,
}: {
  value: number;
  format?: CountFormat;
  digits?: number;
  prefix?: string;
  suffix?: string;
  durationMs?: number;
}) {
  const f = useFormat();
  const reduced = useReducedMotion();
  const [ref, inView] = useInView<HTMLSpanElement>();
  // null = the final value (server render, before the element is seen, after the count)
  const [current, setCurrent] = useState<number | null>(null);

  useEffect(() => {
    if (!inView || reduced || !Number.isFinite(value) || value === 0 || durationMs <= 0) return;
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      if (t >= 1) {
        setCurrent(null);
        return;
      }
      setCurrent(value * easeOutCubic(t));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      setCurrent(null);
    };
  }, [inView, reduced, value, durationMs]);

  const text = (v: number) => `${prefix}${formatAs(f, format, v, value, digits)}${suffix}`;
  const finalText = text(value);
  return (
    <span ref={ref} aria-label={finalText} style={{ fontVariantNumeric: "tabular-nums" }}>
      {current == null ? finalText : text(current)}
    </span>
  );
}
