// The optional farm and road layers (farms.geojson, roads.geojson; src/Web/CLAUDE.md §6.3), on whatever survey the
// build serves. Without the files (the mock, older bundles) the three switches are disabled, nothing is requested and
// there is no farm KPI. With them: the switches start on, the legend follows them, the KPI strip counts the farms and
// a click on a farm label opens its panel with the figures of summary.json (read at run time, never hardcoded).
import { expect, test, type Page } from "@playwright/test";
import { SURVEY_ID } from "../src/lib/data";
import { makeFormat } from "../src/lib/format";
import type { SurveySummary } from "../src/lib/types";

const DATA = `/data/${SURVEY_ID}`;
const f = makeFormat("ro");
// FarmRoadLayers.FARM_LABEL_ZOOM: the labels appear from this zoom on
const MAX_ZOOM_CLICKS = 3;

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

/** The layer panel, closed by default below the lg breakpoint (1200 px); see overlays.spec.ts. */
async function openLayers(page: Page) {
  if (page.viewportSize()!.width < 1200) await page.getByRole("button", { name: "Deschide straturile" }).click();
  await expect(page.getByText("Straturi", { exact: true })).toBeVisible();
}

const farmsSwitch = (page: Page) => page.getByRole("checkbox", { name: /^Ferme/ });
const roadsSwitch = (page: Page) => page.getByRole("checkbox", { name: /^Drumuri(?!\s*interne)/ });
const internalSwitch = (page: Page) => page.getByRole("checkbox", { name: /^Drumuri\s*interne/ });
const labels = (page: Page) => page.locator("[data-farm-label]");

/** Waits until the camera stops (the initial fit, a zoom step): the S/F marker keeps its screen position for 300 ms. */
async function mapSettled(page: Page) {
  const sf = page.locator(".maplibregl-marker").filter({ has: page.getByTitle("START / FINISH") });
  await expect(sf).toHaveCount(1);
  const position = () => sf.evaluate((el) => (el as HTMLElement).style.transform);
  await expect
    .poll(async () => {
      const before = await position();
      await page.waitForTimeout(300);
      return (await position()) === before;
    }, { message: "the map camera keeps moving" })
    .toBe(true);
}

/** A farm label that is on screen and not covered by a panel (elementFromPoint hits it), or null. */
const pickableFarm = (page: Page) =>
  page.evaluate(() => {
    for (const el of Array.from(document.querySelectorAll<HTMLElement>("[data-farm-label]"))) {
      const r = el.getBoundingClientRect();
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      if (r.width > 0 && hit && el.contains(hit)) return el.dataset.farmLabel ?? null;
    }
    return null;
  });

test("farm and road switches follow the files the survey ships; the map loads either way", async ({ page, request }) => {
  const summary = (await (await request.get(`${DATA}/summary.json`)).json()) as SurveySummary;
  const shipsFarms = (await request.get(`${DATA}/farms.geojson`)).ok();
  const shipsRoads = (await request.get(`${DATA}/roads.geojson`)).ok();
  const requested: string[] = [];
  page.on("request", (r) => {
    if (/\/(farms|roads)\.geojson$/.test(r.url())) requested.push(r.url());
  });

  await page.goto("/harta");
  await openLayers(page);
  await expect(page.getByText("Se încarcă harta…")).toHaveCount(0);
  await expect(page.getByText("Datele hărții nu s-au putut încărca")).toHaveCount(0);

  const hasInternal = shipsRoads && (summary.roads?.internal_m ?? 0) > 0;
  for (const [box, ships] of [[farmsSwitch(page), shipsFarms], [roadsSwitch(page), shipsRoads], [internalSwitch(page), hasInternal]] as const) {
    if (ships) {
      await expect(box).toBeEnabled();
      await expect(box).toBeChecked();
    } else {
      await expect(box).toBeDisabled();
      await expect(box).not.toBeChecked();
    }
  }
  const kpi = page.getByText("ferme", { exact: true });
  if (!shipsFarms && !shipsRoads) {
    expect(requested).toEqual([]);
    await expect(kpi).toHaveCount(0);
    return;
  }

  // on by default: the legend shows every layer that is on
  if (shipsFarms) {
    await expect(kpi).toBeVisible();
    await expect(kpi.locator("xpath=preceding-sibling::*[1]")).toHaveText(f.int(summary.totals.farm_count!));
    await expect(page.getByText(/^Fermă \(blocuri/)).toBeVisible();
    await farmsSwitch(page).uncheck();
    await expect(page.getByText(/^Fermă \(blocuri/)).toHaveCount(0);
    await expect(labels(page)).toHaveCount(0);
    await farmsSwitch(page).check();
    await expect(page.getByText(/^Fermă \(blocuri/)).toBeVisible();
  }
  if (shipsRoads) {
    await expect(page.getByText("Drum public", { exact: true })).toBeVisible();
    await expect(page.getByText("Drum de câmp", { exact: true })).toBeVisible();
    await roadsSwitch(page).uncheck();
    await expect(page.getByText("Drum public", { exact: true })).toHaveCount(0);
    await roadsSwitch(page).check();
    await expect(page.getByText("Drum public", { exact: true })).toBeVisible();
    if (hasInternal) {
      await expect(page.getByText("Drum intern al fermei", { exact: true })).toBeVisible();
      await internalSwitch(page).uncheck();
      await expect(page.getByText("Drum intern al fermei", { exact: true })).toHaveCount(0);
    }
    // roads are OpenStreetMap data: the attribution names it
    await expect(page.locator(".maplibregl-ctrl-attrib-inner")).toContainText("© OpenStreetMap contributors");
  }
});

test("clicking a farm label opens the farm panel with its blocks and the summed figures", async ({ page, request }) => {
  test.skip(!(await request.get(`${DATA}/farms.geojson`)).ok(), `${SURVEY_ID} ships no farms.geojson`);
  const summary = (await (await request.get(`${DATA}/summary.json`)).json()) as SurveySummary;
  await page.goto("/harta");
  await expect(page.getByText("Se încarcă harta…")).toHaveCount(0);

  // the labels appear from medium zoom on; zoom in (keeping the view centred) until one can be clicked
  let farmId: string | null = null;
  for (let i = 0; i <= MAX_ZOOM_CLICKS && !farmId; i++) {
    await mapSettled(page);
    await expect.poll(() => pickableFarm(page), { timeout: 3000 }).not.toBeNull().catch(() => undefined);
    farmId = await pickableFarm(page);
    if (!farmId) await page.getByRole("button", { name: "Zoom in" }).click();
  }
  expect(farmId, "a farm label on screen").not.toBeNull();
  await page.locator(`[data-farm-label="${farmId}"]`).click();

  const farm = summary.farms!.find((x) => x.farm_id === farmId)!;
  const panel = page.getByTestId("farm-attributes");
  await expect(page.getByRole("heading", { level: 3, name: `Ferma ${farmId}`, exact: true })).toBeVisible();
  await expect(panel.getByText(`Blocuri (${farm.n_blocks})`)).toBeVisible();
  for (const id of farm.vineyard_ids) await expect(panel.getByRole("button", { name: id, exact: true })).toBeVisible();
  const field = (label: string) => panel.getByText(label, { exact: true }).locator("xpath=following-sibling::*[1]");
  await expect(field("Rânduri")).toHaveText(f.int(farm.row_count));
  await expect(field("Plante")).toHaveText(f.int(farm.plant_count));
  await expect(field("Suprafața fermei")).toHaveText(f.area(farm.area_m2));
  await expect(field("Aria coroanelor")).toHaveText(f.area(farm.canopy_area_m2));

  // a block chip moves on to that block
  await panel.getByRole("button", { name: farm.vineyard_ids[0], exact: true }).click();
  await expect(page.getByRole("heading", { level: 3, name: `Bloc ${farm.vineyard_ids[0]}`, exact: true })).toBeVisible();
});
