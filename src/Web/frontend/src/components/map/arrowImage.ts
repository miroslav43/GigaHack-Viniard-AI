// Route direction arrow drawn on a canvas and registered with map.addImage (no sprite/glyph server needed, works offline).
import { mapPalette } from "@/theme/mapPalette";

export const ROUTE_ARROW = "route-arrow";

export function routeArrowImage(size = 32): ImageData {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  // chevron pointing right (+x = along the line in symbol-placement: line)
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const path = () => {
    ctx.beginPath();
    ctx.moveTo(size * 0.35, size * 0.25);
    ctx.lineTo(size * 0.65, size * 0.5);
    ctx.lineTo(size * 0.35, size * 0.75);
  };
  ctx.strokeStyle = mapPalette.casing;
  ctx.lineWidth = size * 0.22;
  path();
  ctx.stroke();
  ctx.strokeStyle = mapPalette.route;
  ctx.lineWidth = size * 0.11;
  path();
  ctx.stroke();
  return ctx.getImageData(0, 0, size, size);
}
