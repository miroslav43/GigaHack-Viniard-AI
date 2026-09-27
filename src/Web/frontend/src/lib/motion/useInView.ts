"use client";

import { useEffect, useState } from "react";

export interface InViewOptions {
  /** stay true after the first time the element is seen (default true) */
  once?: boolean;
  rootMargin?: string;
  threshold?: number;
}

const hasObserver = () => typeof window !== "undefined" && typeof window.IntersectionObserver === "function";

/**
 * Whether an element is on screen: `const [ref, inView] = useInView()` then `<div ref={ref}>`.
 * Without IntersectionObserver (old browsers) the element counts as seen, so nothing stays hidden.
 */
export function useInView<T extends Element = HTMLElement>(opts: InViewOptions = {}): readonly [(el: T | null) => void, boolean] {
  const { once = true, rootMargin = "0px", threshold = 0 } = opts;
  const [el, setEl] = useState<T | null>(null);
  const [seen, setSeen] = useState(false);

  useEffect(() => {
    if (!el || !hasObserver()) return;
    const io = new IntersectionObserver(
      (entries) => {
        const hit = entries.some((e) => e.isIntersecting);
        if (hit && once) io.disconnect();
        if (hit || !once) setSeen(hit);
      },
      { rootMargin, threshold },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [el, once, rootMargin, threshold]);

  return [setEl, seen || (el !== null && !hasObserver())] as const;
}
