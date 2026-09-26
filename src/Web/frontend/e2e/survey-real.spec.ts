// The jury flow on the real AI survey (siret3, from scripts/build-survey.mjs). Every expected figure is read
// at run time from the published files (summary.json, measurements.csv, rows.json, targets.geojson), never hardcoded.
// Run: node scripts/build-survey.mjs --survey siret3, then NEXT_PUBLIC_SURVEY_ID=siret3 in .env.local (or in the shell
//      of both commands), then pnpm build && pnpm e2e e2e/survey-real.spec.ts
//      (playwright.config.ts loads the same .env files as `next build`, so the tests see the survey the build inlined)
import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import type { FeatureCollection, LineString, Point } from "geojson";
import { SURVEY_ID } from "../src/lib/data";
import { makeFormat } from "../src/lib/format";
import type { RowRecord, SurveySummary, TargetProps } from "../src/lib/types";

const SURVEY = "siret3";
const DATA = `/data/${SURVEY}`;
// control facts (src/Web/CLAUDE.md §6.6): 311 tiles of 51.2 m = 81.5268 ha
const SURVEY_TILES = 311;
const TILE_M = 51.2;

test.skip(SURVEY_ID !== SURVEY, `build serves ${SURVEY_ID}; needs a build with NEXT_PUBLIC_SURVEY_ID=${SURVEY}`);

const f = makeFormat("ro");

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

async function getJson<T>(request: APIRequestContext, url: string): Promise<T> {
  const res = await request.get(url);
  expect(res.ok(), `${url}: HTTP ${res.status()}`).toBeTruthy();
  return (await res.json()) as T;
}

/** measurements.csv (§6.4) as header-keyed records. */
async function getMeasurements(request: APIRequestContext) {
  const res = await request.get(`${DATA}/measurements.csv`);
  expect(res.ok(), `measurements.csv: HTTP ${res.status()}`).toBeTruthy();
  const [header, ...lines] = (await res.text()).trim().split("\n").map((l) => l.split(","));
  const records = lines.map((cols) => Object.fromEntries(header.map((h, i) => [h, cols[i]])));
  const survey = records.filter((r) => r.level === "survey");
  expect(survey).toHaveLength(1);
  return { survey: survey[0], blocks: records.filter((r) => r.level === "block"), rows: records.filter((r) => r.level === "row") };
}

/** A KPI card, found by its label. */
const kpi = (page: Page, label: string) => page.locator(".MuiCard-root").filter({ has: page.getByText(label, { exact: true }) }).first();

test("summary totals equal the measurements.csv survey line", async ({ request }) => {
  const [s, csv] = await Promise.all([getJson<SurveySummary>(request, `${DATA}/summary.json`), getMeasurements(request)]);
  const tt = s.totals;
  expect(s.survey.mock).toBe(false);
  expect(tt.block_count).toBe(Number(csv.survey.block_count));
  expect(tt.row_count).toBe(Number(csv.survey.row_count));
  expect(tt.row_length_m).toBeCloseTo(Number(csv.survey.row_length_m), 2);
  expect(tt.canopy_area_m2).toBeCloseTo(Number(csv.survey.canopy_area_m2), 2);
  expect(tt.interrow_area_m2).toBeCloseTo(Number(csv.survey.interrow_area_m2), 2);
  expect(tt.plant_count).toBe(Number(csv.survey.plant_count));
  expect(csv.blocks).toHaveLength(tt.block_count);
  expect(csv.rows).toHaveLength(tt.row_count);
});

test("overview shows the survey KPIs from summary.json, without the test-data banner", async ({ page, request }) => {
  const s = await getJson<SurveySummary>(request, `${DATA}/summary.json`);
  const tt = s.totals;
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: "Primăria Sireți" })).toBeVisible();
  await expect(kpi(page, "Blocuri viticole").getByText(f.int(tt.block_count), { exact: true })).toBeVisible();
  await expect(kpi(page, "Rânduri").getByText(f.int(tt.row_count), { exact: true })).toBeVisible();
  await expect(kpi(page, "Lungime totală rânduri").getByText(f.length(tt.row_length_m), { exact: true })).toBeVisible();
  await expect(kpi(page, "Aria coroanelor").getByText(f.ha(tt.canopy_area_m2), { exact: true })).toBeVisible();
  await expect(kpi(page, "Aria inter-rândurilor").getByText(f.ha(tt.interrow_area_m2), { exact: true })).toBeVisible();
  await expect(kpi(page, "Ținte de inspecție").getByText(f.int(tt.target_count), { exact: true })).toBeVisible();
  await expect(kpi(page, "Ruta").getByText(f.length(s.route.length_m), { exact: true })).toBeVisible();
  await expect(page.getByText("Date de test.")).toHaveCount(0);
});

test("the orthophoto covers the 311 survey tiles (81.5268 ha)", async ({ page, request }) => {
  const [s, tiles] = await Promise.all([
    getJson<SurveySummary>(request, `${DATA}/summary.json`),
    getJson<{ tiles: unknown[] }>(request, "/data/tiles.json"),
  ]);
  expect(tiles.tiles).toHaveLength(SURVEY_TILES);
  expect(s.survey.tiles_total).toBe(SURVEY_TILES);
  expect(s.survey.surveyed_area_ha).toBeCloseTo((SURVEY_TILES * TILE_M ** 2) / 1e4, 4);
  await page.goto("/");
  const card = kpi(page, "Suprafață zburată");
  await expect(card.getByText(`${f.num(s.survey.surveyed_area_ha, 1)} ${f.units.ha}`, { exact: true })).toBeVisible();
  await expect(card.getByText(`${s.survey.tiles_total} tile-uri`)).toBeVisible();
});

test("map deep link selects the first row and shows its IDs and length", async ({ page, request }) => {
  const rows = await getJson<RowRecord[]>(request, `${DATA}/rows.json`);
  expect(rows.length).toBeGreaterThan(0);
  const row = rows[0];
  await page.goto(`/harta?rand=${row.row_id}`);
  const panel = page.getByRole("heading", { level: 3, name: row.row_id, exact: true });
  await expect(panel).toBeVisible({ timeout: 30_000 });
  const card = panel.locator("xpath=ancestor::div[contains(@class,'MuiPaper-root')][1]");
  await expect(card.getByText("vineyard_id")).toBeVisible();
  await expect(card.getByText(row.vineyard_id, { exact: true })).toBeVisible();
  await expect(card.getByText(f.m(row.length_m, 2), { exact: true })).toBeVisible();
});

test("map deep link to a target the route skips marks it off route", async ({ page, request }) => {
  const targets = await getJson<FeatureCollection<Point, TargetProps>>(request, `${DATA}/targets.geojson`);
  const skipped = targets.features.find((x) => x.properties.route_order == null);
  test.skip(!skipped, "every target of this survey is on the route");
  const id = skipped!.properties.target_id;
  await page.goto(`/harta?tinta=${id}`);
  const panel = page.getByRole("heading", { level: 3, name: `Ținta ${id}`, exact: true });
  await expect(panel).toBeVisible({ timeout: 30_000 });
  const card = panel.locator("xpath=ancestor::div[contains(@class,'MuiPaper-root')][1]");
  await expect(card.getByText("în afara rutei", { exact: true })).toBeVisible();
});

test("route page lists only the route stops, and the GPX and EPSG:32635 exports match", async ({ page, request }) => {
  const [s, targets] = await Promise.all([
    getJson<SurveySummary>(request, `${DATA}/summary.json`),
    getJson<FeatureCollection<Point, TargetProps>>(request, `${DATA}/targets.geojson`),
  ]);
  const stops = targets.features.filter((x) => x.properties.route_order != null).length;
  await page.goto("/ruta");
  await expect(page.getByRole("heading", { level: 1, name: "Rută de inspecție" })).toBeVisible();
  await expect(kpi(page, "Lungimea rutei").getByText(f.length(s.route.length_m), { exact: true })).toBeVisible();
  await expect(page.getByText(`${f.int(stops)} pe rută din ${f.int(targets.features.length)}`)).toBeVisible();
  await expect(page.locator("table tbody tr")).toHaveCount(stops);
  await expect(page.locator("table tbody tr").first().locator("td").first()).toHaveText("1");

  const gpx = await request.get(`${DATA}/route.gpx`);
  expect(gpx.ok()).toBeTruthy();
  const text = await gpx.text();
  expect(text).toContain("<trk>");
  expect(text.match(/<wpt /g) ?? []).toHaveLength(stops);

  const route = await getJson<FeatureCollection<LineString, { length_m: number }> & { crs: { properties: { name: string } } }>(
    request,
    `${DATA}/route_EPSG32635.geojson`,
  );
  expect(route.crs.properties.name).toBe("urn:ogc:def:crs:EPSG::32635");
  expect(route.features[0].properties.length_m).toBeCloseTo(s.route.length_m, 1);
});

test("blocks page lists every block and row of the survey", async ({ page, request }) => {
  const [s, csv] = await Promise.all([getJson<SurveySummary>(request, `${DATA}/summary.json`), getMeasurements(request)]);
  expect(s.blocks).toHaveLength(s.totals.block_count);
  expect(s.blocks.map((b) => b.vineyard_id).sort()).toEqual(csv.blocks.map((b) => b.vineyard_id).sort());
  await page.goto("/blocuri");
  await expect(page.getByRole("tab", { name: `Blocuri (${f.int(s.blocks.length)})` })).toBeVisible();
  await expect(page.getByRole("tab", { name: `Rânduri (${f.int(s.totals.row_count)})` })).toBeVisible();
});
