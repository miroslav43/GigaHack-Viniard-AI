// GPX 1.1 of a planned farm route: the track plus one waypoint per target, in visiting order.
import type { LonLat } from "../utm.ts";

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const pt = ([lon, lat]: LonLat) => `lat="${lat.toFixed(7)}" lon="${lon.toFixed(7)}"`;

export function farmRouteGpx(name: string, line: readonly LonLat[], stops: readonly { id: string; at: LonLat }[]): string {
  const wpts = stops.map((s, i) => `  <wpt ${pt(s.at)}><name>${i + 1}. ${esc(s.id)}</name></wpt>`);
  const trkpts = line.map((p) => `      <trkpt ${pt(p)}/>`);
  return [
    `<?xml version="1.0" encoding="UTF-8"?>`,
    `<gpx version="1.1" creator="Solemtrix" xmlns="http://www.topografix.com/GPX/1/1">`,
    ...wpts,
    `  <trk><name>${esc(name)}</name><trkseg>`,
    ...trkpts,
    `  </trkseg></trk>`,
    `</gpx>`,
    "",
  ].join("\n");
}
