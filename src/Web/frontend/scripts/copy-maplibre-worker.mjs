// MapLibre 6 resolves its module worker relative to the bundle chunk, which Turbopack does not emit.
// Copy the worker (and the shared chunk it imports) to public/ and point setWorkerUrl() at it.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const src = path.join(root, "node_modules", "maplibre-gl", "dist");
const dst = path.join(root, "public", "maplibre");
fs.mkdirSync(dst, { recursive: true });
for (const f of ["maplibre-gl-worker.mjs", "maplibre-gl-shared.mjs"]) fs.copyFileSync(path.join(src, f), path.join(dst, f));
