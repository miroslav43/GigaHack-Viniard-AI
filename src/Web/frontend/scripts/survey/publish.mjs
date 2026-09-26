// Publishes a generated survey folder: every file is written to <root>/<id>.tmp, then swapped in with renames,
// so the site never reads a half-written survey and a failed build leaves the previous one in place.
import fs from "node:fs";
import path from "node:path";

const writeEntry = (dir, name, entry) => {
  const file = path.join(dir, name);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  if ("copy" in entry) fs.copyFileSync(entry.copy, file);
  else if ("buffer" in entry) fs.writeFileSync(file, entry.buffer);
  else fs.writeFileSync(file, "json" in entry ? JSON.stringify(entry.json) : entry.text);
};
const remove = (dir) => fs.rmSync(dir, { recursive: true, force: true });

/**
 * files: { name: { json } | { text } | { buffer } | { copy: sourcePath } } → <root>/<id>/ (replaced as a whole).
 * A name may hold one sub-folder ("masks/<tile>.png").
 */
export const publishDir = (root, id, files) => {
  const final = path.join(root, id);
  const tmp = `${final}.tmp`, old = `${final}.old`;
  remove(tmp);
  fs.mkdirSync(tmp, { recursive: true });
  try {
    for (const [name, entry] of Object.entries(files)) {
      // names come from the converter; never let one escape the survey folder
      if (path.isAbsolute(name) || path.normalize(name).startsWith("..")) throw new Error(`unsafe output name "${name}"`);
      writeEntry(tmp, name, entry);
    }
  } catch (err) {
    remove(tmp);
    throw new Error(`writing ${tmp} failed: ${err.message}`, { cause: err });
  }
  remove(old);
  const hadPrevious = fs.existsSync(final);
  if (hadPrevious) fs.renameSync(final, old);
  try {
    fs.renameSync(tmp, final);
  } catch (err) {
    if (hadPrevious) fs.renameSync(old, final);
    remove(tmp);
    throw new Error(`replacing ${final} failed (previous version kept): ${err.message}`, { cause: err });
  }
  remove(old);
  return final;
};
