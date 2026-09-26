// The assistant's system prompt (src/lib/assistant/prompt.ts) quotes UI labels as «namespace.key»; each must exist in
// every language, otherwise the /api/assistant route throws while building the prompt.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const root = new URL("../../", import.meta.url);
const prompt = readFileSync(new URL("src/lib/assistant/prompt.ts", root), "utf8");
const keys = [...new Set([...prompt.matchAll(/«([\w.]+)»/g)].map((m) => m[1]))];

test("the assistant prompt quotes labels", () => assert.ok(keys.length > 50));

for (const locale of ["ro", "en", "ru"]) {
  test(`every «label» of the assistant prompt exists in messages/${locale}.json`, () => {
    const messages = JSON.parse(readFileSync(new URL(`messages/${locale}.json`, root), "utf8"));
    const missing = keys.filter((k) => typeof k.split(".").reduce((o, x) => o?.[x], messages) !== "string");
    assert.deepEqual(missing, []);
  });
}
