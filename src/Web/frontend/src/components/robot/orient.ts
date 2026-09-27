// The camera is mounted upside down: its picture is flipped (vertically, and horizontally if asked) on screen with CSS
// and in the saved photos / panoramas by redrawing the JPEG.

const QUALITY = 0.92;

export interface Orientation {
  flipV: boolean;
  flipH: boolean;
}

/** CSS transform of the live picture. */
export const orientCss = ({ flipV, flipH }: Orientation) => `scale(${flipH ? -1 : 1}, ${flipV ? -1 : 1})`;

/** The JPEG turned the same way as on screen (unchanged when there is nothing to flip). */
export async function orientJpeg(blob: Blob, { flipV, flipH }: Orientation): Promise<Blob> {
  if (!flipV && !flipH) return blob;
  const img = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d")!;
  ctx.translate(flipH ? img.width : 0, flipV ? img.height : 0);
  ctx.scale(flipH ? -1 : 1, flipV ? -1 : 1);
  ctx.drawImage(img, 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))), "image/jpeg", QUALITY));
}
