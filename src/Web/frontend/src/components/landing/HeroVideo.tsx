"use client";

import { useEffect, useRef } from "react";
import Box from "@mui/material/Box";

/**
 * Background clip of the hero: muted, looping, inline (autoplay-safe on phones). With "reduce motion" the video
 * stays on its poster frame. Purely decorative (aria-hidden); the text over it carries the message.
 */
export function HeroVideo({ src, poster }: { src: string; poster: string }) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => (media.matches ? video.pause() : void video.play().catch(() => {}));
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, []);

  return (
    <Box
      component="video"
      ref={ref}
      src={src}
      poster={poster}
      muted
      loop
      playsInline
      autoPlay
      preload="auto"
      aria-hidden
      tabIndex={-1}
      sx={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
    />
  );
}
