// The farm route tool (ADR-028): a farm panel → "plan the route" → a click on the map sets the start (= finish) →
// the worker plans a closed tour through every target of the farm. Needs farms.geojson + roads.geojson (skipped on
// the mock, which ships neither).
import { expect, test, type Page } from "@playwright/test";
import { SURVEY_ID } from "../src/lib/data";
import type { SurveySummary } from "../src/lib/types";

const DATA = `/data/${SURVEY_ID}`;
const MAX_ZOOM_CLICKS = 3;

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

/** On-screen, uncovered farm labels whose farm has targets (elementFromPoint hits them). */
const pickableFarms = (page: Page, withTargets: string[]) =>
  page.evaluate((ids) => {
    const out: string[] = [];
    for (const el of Array.from(document.querySelectorAll<HTMLElement>("[data-farm-label]"))) {
      const r = el.getBoundingClientRect();
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      const id = el.dataset.farmLabel ?? "";
      if (r.width > 0 && hit && el.contains(hit) && ids.includes(id)) out.push(id);
    }
    return out;
  }, withTargets);

test("a start picked on the map gives a closed route through every target of the farm", async ({ page, request }) => {
  const ships = (await request.get(`${DATA}/farms.geojson`)).ok() && (await request.get(`${DATA}/roads.geojson`)).ok();
  test.skip(!ships, `${SURVEY_ID} ships no farms.geojson / roads.geojson`);
  // headless WebGL (swiftshader) is slow to zoom and redraw a real survey
  test.setTimeout(120_000);
  const summary = (await (await request.get(`${DATA}/summary.json`)).json()) as SurveySummary;
  const withTargets = summary.farms!.filter((x) => x.target_count > 0).map((x) => x.farm_id);

  await page.goto("/harta");
  await expect(page.getByText("Se încarcă harta…")).toHaveCount(0);
  let farmId: string | null = null;
  for (let i = 0; i <= MAX_ZOOM_CLICKS && !farmId; i++) {
    await expect.poll(() => pickableFarms(page, withTargets).then((x) => x.length), { timeout: 3000 }).toBeGreaterThan(0).catch(() => undefined);
    farmId = (await pickableFarms(page, withTargets))[0] ?? null;
    if (!farmId) await page.getByRole("button", { name: "Zoom in" }).click();
  }
  expect(farmId, "a farm label with targets on screen").not.toBeNull();
  const label = page.locator(`[data-farm-label="${farmId}"]`);
  const at = (await label.boundingBox())!;
  await label.click();
  const farm = summary.farms!.find((x) => x.farm_id === farmId)!;

  const section = page.getByTestId("farm-route");
  await section.getByTestId("farm-route-begin").click();
  // desktop: the hint is in the farm panel; phone: the panel steps aside for a banner, so the map can be clicked
  await expect(page.getByTestId("farm-route-pick")).toBeVisible();
  // the start: a little left of the farm label, on the map (below the banner on a phone)
  await page.mouse.click(Math.max(at.x - 40, 5), at.y + at.height / 2);
  await expect(page.getByTestId("farm-route-pick")).toHaveCount(0);

  const result = section.getByTestId("farm-route-result");
  await expect(result).toBeVisible({ timeout: 30_000 });
  const field = (name: string) => result.getByText(name, { exact: true }).locator("xpath=following-sibling::*[1]");
  await expect(field("Ținte vizitate")).toHaveText(String(farm.target_count));
  await expect(field("Lungime traseu")).toHaveText(/^\d[\d.,]* (m|km)$/);
  await expect(page.getByTestId("farm-route-start")).toBeVisible();

  await result.getByRole("button", { name: "Șterge traseul" }).click();
  await expect(page.getByTestId("farm-route-start")).toHaveCount(0);
  await expect(section.getByTestId("farm-route-begin")).toBeVisible();
});
