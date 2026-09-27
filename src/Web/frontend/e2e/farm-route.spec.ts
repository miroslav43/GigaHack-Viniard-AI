// The farm route tool (ADR-028): the "Traseu fermă" button → 1 choose a farm (list or map) → 2 click the start in
// the farm → 3 a closed route through every target of the farm, inside it. Needs farms.geojson + roads.geojson
// (skipped on the mock, which ships neither).
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import { SURVEY_ID } from "../src/lib/data";
import type { SurveySummary } from "../src/lib/types";

const DATA = `/data/${SURVEY_ID}`;

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

const shipsFarmsAndRoads = async (request: APIRequestContext) =>
  (await request.get(`${DATA}/farms.geojson`)).ok() && (await request.get(`${DATA}/roads.geojson`)).ok();

/** A click on the map clear of the card: upper middle on a desktop (card bottom right), lower middle on a phone (card on top). */
async function clickMap(page: Page) {
  const box = (await page.locator(".maplibregl-canvas").boundingBox())!;
  const phone = page.viewportSize()!.width < 900;
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * (phone ? 0.62 : 0.3));
}

test("the farm route tool: choose a farm from the list, click the start, get the route", async ({ page, request }) => {
  test.skip(!(await shipsFarmsAndRoads(request)), `${SURVEY_ID} ships no farms.geojson / roads.geojson`);
  // headless WebGL (swiftshader) is slow to zoom and redraw a real survey
  test.setTimeout(120_000);
  const summary = (await (await request.get(`${DATA}/summary.json`)).json()) as SurveySummary;
  // a small farm with targets: quick to plan
  const farm = [...summary.farms!].filter((x) => x.target_count > 0).sort((a, b) => a.target_count - b.target_count)[0];

  await page.goto("/harta");
  await expect(page.getByText("Se încarcă harta…")).toHaveCount(0);
  const tool = page.getByTestId("farm-route-tool");
  await expect(tool).toBeEnabled();
  await tool.click();

  const card = page.getByTestId("farm-route-card");
  await expect(card.getByText(/Click pe o fermă de pe hartă/)).toBeVisible();
  await card.getByRole("combobox", { name: "Alege ferma" }).click();
  await page.getByRole("option", { name: new RegExp(`^Ferma ${farm.farm_id} ·`) }).click();

  await expect(card.getByTestId("farm-route-pick")).toBeVisible();
  await expect(card.getByText(new RegExp(`^Ferma ${farm.farm_id} ·`))).toBeVisible();
  await page.waitForTimeout(1200); // the camera flies to the farm
  await clickMap(page);

  const result = card.getByTestId("farm-route-result");
  await expect(result).toBeVisible({ timeout: 30_000 });
  const field = (name: string) => result.getByText(name, { exact: true }).locator("xpath=following-sibling::*[1]");
  await expect(field("Ținte vizitate")).toHaveText(String(farm.target_count));
  await expect(field("Lungime traseu")).toHaveText(/^\d[\d.,]* (m|km)$/);
  await expect(page.getByTestId("farm-route-start")).toBeVisible();

  // "Altă fermă" goes back to step 1; closing removes everything
  await result.getByRole("button", { name: "Altă fermă" }).click();
  await expect(card.getByRole("combobox", { name: "Alege ferma" })).toBeVisible();
  await card.getByRole("button", { name: "Închide traseul" }).click();
  await expect(card).toHaveCount(0);
  await expect(page.getByTestId("farm-route-start")).toHaveCount(0);
});

test("a farm's panel jumps straight to step 2 of the tool", async ({ page, request }) => {
  test.skip(!(await shipsFarmsAndRoads(request)), `${SURVEY_ID} ships no farms.geojson / roads.geojson`);
  test.setTimeout(120_000);
  const summary = (await (await request.get(`${DATA}/summary.json`)).json()) as SurveySummary;
  const withTargets = summary.farms!.filter((x) => x.target_count > 0).map((x) => x.farm_id);

  await page.goto("/harta");
  await expect(page.getByText("Se încarcă harta…")).toHaveCount(0);
  // zoom in until a farm label with targets can be clicked
  let farmId: string | null = null;
  for (let i = 0; i <= 3 && !farmId; i++) {
    await page.waitForTimeout(1000);
    farmId = await page.evaluate((ids) => {
      for (const el of Array.from(document.querySelectorAll<HTMLElement>("[data-farm-label]"))) {
        const r = el.getBoundingClientRect();
        const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
        if (r.width > 0 && hit && el.contains(hit) && ids.includes(el.dataset.farmLabel ?? "")) return el.dataset.farmLabel ?? null;
      }
      return null;
    }, withTargets);
    if (!farmId) await page.getByRole("button", { name: "Zoom in" }).click();
  }
  expect(farmId, "a farm label with targets on screen").not.toBeNull();
  await page.locator(`[data-farm-label="${farmId}"]`).click();
  await page.getByTestId("farm-route-begin").click();

  const card = page.getByTestId("farm-route-card");
  await expect(card.getByTestId("farm-route-pick")).toBeVisible();
  await expect(card.getByText(new RegExp(`^Ferma ${farmId} ·`))).toBeVisible();
});
