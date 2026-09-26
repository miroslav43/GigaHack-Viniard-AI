// A 1-bit PNG encoder (RFC 2083) for binary masks: greyscale (0 black, 1 white) or a 2-entry palette with alpha.
// sharp's palette output goes through libimagequant, which picks the palette order and rounds alpha; the mask
// contract needs index 0 = fully transparent and index 1 = the exact theme colour, so the bytes are written here.
import zlib from "node:zlib";

const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const COLOUR_TYPE = { grey: 0, palette: 3 };

const chunk = (type, data) => {
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const out = Buffer.alloc(8 + body.length);
  out.writeUInt32BE(data.length, 0);
  body.copy(out, 4);
  out.writeUInt32BE(zlib.crc32(body), 4 + body.length);
  return out;
};

const header = (width, height, colourType) => {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr.set([1, colourType, 0, 0, 0], 8); // bit depth 1, deflate, adaptive filter, no interlace
  return ihdr;
};

/** Scanlines: filter byte 0, then the row's bits packed most significant first (bit set = value 1). */
const scanlines = (bits, width, height) => {
  const stride = 1 + Math.ceil(width / 8);
  const raw = Buffer.alloc(stride * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) if (bits[y * width + x]) raw[y * stride + 1 + (x >> 3)] |= 0x80 >> (x & 7);
  }
  return raw;
};

/**
 * bits: one byte per pixel, row-major (non-zero = 1). palette: null → greyscale, or [[r, g, b, a], [r, g, b, a]]
 * for index 0 and 1 (a in 0..255).
 */
export const encodeBitPng = (bits, width, height, palette = null) => {
  if (bits.length !== width * height) throw new Error(`encodeBitPng: ${bits.length} pixels for ${width}×${height}`);
  const colourType = palette ? COLOUR_TYPE.palette : COLOUR_TYPE.grey;
  const paletteChunks = palette
    ? [chunk("PLTE", Buffer.from(palette.flatMap(([r, g, b]) => [r, g, b]))), chunk("tRNS", Buffer.from(palette.map((c) => c[3])))]
    : [];
  return Buffer.concat([
    SIGNATURE,
    chunk("IHDR", header(width, height, colourType)),
    ...paletteChunks,
    chunk("IDAT", zlib.deflateSync(scanlines(bits, width, height), { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
};

/** "#D946EF" + alpha 0..1 → [217, 70, 239, 140]. */
export const rgbaOf = (hex, alpha) => {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex ?? "");
  if (!m || !(alpha >= 0 && alpha <= 1)) throw new Error(`bad mask colour ${JSON.stringify(hex)} / alpha ${alpha}`);
  return [...m.slice(1).map((h) => parseInt(h, 16)), Math.round(alpha * 255)];
};
