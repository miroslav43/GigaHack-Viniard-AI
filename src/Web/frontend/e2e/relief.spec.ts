// The oblique 3D view over the synthetic canopy relief (scripts/build-terrain.mjs), on whichever survey the build serves.
// Skips when the relief was not generated (terrain.json absent): the toggle is then disabled, covered by the last test.
import { expect, test, type Page } from "@playwright/test";
import { SURVEY_ID } from "../src/lib/data";

const TERRAIN = `/data/${SURVEY_ID}/terrain/terrain.json`;
const MAP_LOAD_MS = 45_000;

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

/** Console errors and uncaught exceptions of the page, collected from now on. */
function collectErrors(page: Page) {
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(e.message));
  return errors;
}

const sfMarker = (page: Page) => page.locator(".maplibregl-marker").filter({ has: page.getByTitle("START / FINISH") });

test("oblique view raises the relief, keeps the map interactive, and returns to 2D", async ({ page, request }) => {
  const meta = await request.get(TERRAIN);
  test.skip(!meta.ok(), `${TERRAIN} not generated (pnpm data:terrain)`);
  const errors = collectErrors(page);
  const reliefTiles: string[] = [];
  page.on("requestfinished", (r) => r.url().includes(`/data/${SURVEY_ID}/terrain/`) && reliefTiles.push(r.url()));

  await page.goto("/harta");
  const oblique = page.getByRole("button", { name: "Vedere oblică" });
  const top = page.getByRole("button", { name: "De sus (2D)" });
  // enabled once the map fired `load` (setTerrain needs the style): slow under software WebGL with parallel workers
  await expect(oblique).toBeEnabled({ timeout: MAP_LOAD_MS });
  await expect(top).toHaveAttribute("aria-pressed", "true");
  await expect(sfMarker(page)).toHaveCount(1);

  await oblique.click();
  await expect(oblique).toHaveAttribute("aria-pressed", "true");
  const settings = page.getByRole("group", { name: "Vedere oblică" });
  await expect(settings.getByText("Relief sintetic: tile-urile au doar RGB, fără altitudine")).toBeVisible();
  await expect(settings.getByRole("slider")).toHaveValue(String((await meta.json()).height_m));
  await expect.poll(() => reliefTiles.length, { message: "relief tiles requested" }).toBeGreaterThan(0);

  // dragging still pans the camera: the S/F marker moves on screen
  const canvas = page.locator(".maplibregl-canvas");
  const before = await sfMarker(page).evaluate((el) => (el as HTMLElement).style.transform);
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2 + 60, { steps: 8 });
  await page.mouse.up();
  await expect.poll(() => sfMarker(page).evaluate((el) => (el as HTMLElement).style.transform)).not.toBe(before);

  // flat source and the slider are accepted without errors
  await settings.getByRole("radio", { name: "Plat" }).check();
  await expect(settings.getByRole("slider")).toBeDisabled();
  await settings.getByRole("radio", { name: "Adnotări" }).check();
  await settings.getByRole("slider").focus();
  await page.keyboard.press("ArrowRight");

  await top.click();
  await expect(top).toHaveAttribute("aria-pressed", "true");
  await expect(settings).toHaveCount(0);
  await page.waitForTimeout(1200); // the camera eases back to pitch 0
  expect(errors).toEqual([]);
});

test("without a generated relief the 3D toggle is disabled", async ({ page, request }) => {
  test.skip((await request.get(TERRAIN)).ok(), "this build has a relief; the enabled path is tested above");
  await page.goto("/harta");
  await expect(page.getByRole("button", { name: "Vedere oblică" })).toBeDisabled();
});
