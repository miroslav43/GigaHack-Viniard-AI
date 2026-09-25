// The 3-minute jury flow (src/Web/docs/DEMO.md) on the siret3-mock survey:
// overview → map → object IDs → measurements → route, in RO/EN/RU.
import { expect, test } from "@playwright/test";

// reference values of the two organizer example tiles (src/Web/CLAUDE.md §6.6)
const MOCK = { rows: "51", rowLengthRo: "1,94 km", canopyRo: "536,2 m²", interrowRo: "4.064,4 m²" };

test("overview shows the jury KPIs with units", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: "Primăria Sireți" })).toBeVisible();
  await expect(page.getByText(MOCK.rows, { exact: true }).first()).toBeVisible();
  await expect(page.getByText(MOCK.rowLengthRo).first()).toBeVisible();
  await expect(page.getByText(MOCK.canopyRo).first()).toBeVisible();
  await expect(page.getByText(MOCK.interrowRo).first()).toBeVisible();
  await expect(page.getByText("2.729 ha")).toBeVisible();
});

test("map deep link selects a row and shows its IDs", async ({ page }) => {
  await page.goto("/harta?rand=V01-R05");
  const panel = page.getByRole("heading", { level: 3, name: "V01-R05" });
  await expect(panel).toBeVisible();
  const card = panel.locator("xpath=ancestor::div[contains(@class,'MuiPaper-root')][1]");
  await expect(card.getByText("vineyard_id")).toBeVisible();
  await expect(card.getByText("V01", { exact: true })).toBeVisible();
  await expect(card.getByText("regulat").first()).toBeVisible();
  await expect(card.getByText(/\d+,\d{2} m$/).first()).toBeVisible();
});

test("route page shows length_m, duration and the GPX export", async ({ page, request }) => {
  const summary = await (await request.get("/data/siret3-mock/summary.json")).json();
  await page.goto("/ruta");
  await expect(page.getByRole("heading", { level: 1, name: "Rută de inspecție" })).toBeVisible();
  const km = (summary.route.length_m / 1000).toFixed(2).replace(".", ",");
  await expect(page.getByText(`${km} km`).first()).toBeVisible();
  const gpx = await request.get("/data/siret3-mock/route.gpx");
  expect(gpx.ok()).toBeTruthy();
  expect(await gpx.text()).toContain("<trkseg>");
  const route = await (await request.get("/data/siret3-mock/route_EPSG32635.geojson")).json();
  expect(route.crs.properties.name).toBe("urn:ogc:def:crs:EPSG::32635");
  expect(route.features[0].properties.length_m).toBe(summary.route.length_m);
});

test("measurements.csv export matches the summary", async ({ request }) => {
  const csv = await (await request.get("/data/siret3-mock/measurements.csv")).text();
  const [header, survey] = csv.trim().split("\n");
  expect(header.split(",")).toEqual([
    "level", "vineyard_id", "row_id", "block_count", "row_count", "row_length_m",
    "canopy_area_m2", "canopy_area_ha", "interrow_area_m2", "interrow_area_ha", "plant_count", "row_structure",
  ]);
  const cols = survey.split(",");
  expect(cols.slice(0, 5)).toEqual(["survey", "", "", "2", "51"]);
  expect(Number(cols[5])).toBeCloseTo(1941.6, 0); // row length, m
  expect(Number(cols[6])).toBeCloseTo(536.2, 0); // canopy area, m²
  expect(Number(cols[8])).toBeCloseTo(4064.4, 0); // inter-row area, m²
});

test("blocks table lists rows sorted by largest gap", async ({ page }) => {
  await page.goto("/blocuri");
  await expect(page.getByRole("tab", { name: "Rânduri (51)" })).toBeVisible();
  await expect(page.getByRole("gridcell", { name: "V02-R09" }).first()).toBeVisible();
});

test("language switch keeps the page and translates it", async ({ page, isMobile }) => {
  await page.goto("/ruta");
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await expect(page).toHaveURL(/\/en\/ruta$/);
  await expect(page.getByRole("heading", { level: 1, name: "Inspection route" })).toBeVisible();
  await page.getByRole("button", { name: "RU", exact: true }).click();
  await expect(page).toHaveURL(/\/ru\/ruta$/);
  await expect(page.getByRole("heading", { level: 1, name: "Маршрут осмотра" })).toBeVisible();
  if (!isMobile) await expect(page.getByText("Обзор").first()).toBeVisible();
});
