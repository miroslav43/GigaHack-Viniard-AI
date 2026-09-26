// The map on the siret3-mock build (runs in CI): what a broken or partial bundle does, a deep link racing the map
// library, how targets are drawn (a canvas circle each; numbered DOM markers only for the route stops in view, close up)
// and the waste panel. Edge cases are made by rewriting responses with page.route, so no extra fixture files are needed.
import { expect, test, type Locator, type Page } from "@playwright/test";
import type { FeatureCollection, MultiLineString, Point, Polygon, Position } from "geojson";
import { SURVEY_ID } from "../src/lib/data";
import { makeFormat } from "../src/lib/format";
import type { TargetProps, WasteProps } from "../src/lib/types";

// the mock's IDs and files are used below; a build on the real survey is covered by survey-real.spec.ts
test.skip(SURVEY_ID !== "siret3-mock", `build serves ${SURVEY_ID}, not siret3-mock — see survey-real.spec.ts`);

const DATA = "/data/siret3-mock";
const TARGET = "T003";
const ROW = "V01-R05";
// MapExplorer.MAX_TARGET_LABELS
const MAX_TARGET_LABELS = 120;
// a lazily loaded script held back this long arrives well after the map data (~50 ms on localhost)
const LATE_SCRIPT_MS = 1500;

const f = makeFormat("ro");

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

const markers = (page: Page) => page.locator(".maplibregl-marker");
const panelHeading = (page: Page, name: string) => page.getByRole("heading", { level: 3, name, exact: true });
const card = (heading: Locator) => heading.locator("xpath=ancestor::div[contains(@class,'MuiPaper-root')][1]");

/** Waits until the camera stops (deep-link fly-to, default fit): the S/F marker keeps its screen position for 300 ms. */
async function mapSettled(page: Page) {
  const sf = markers(page).filter({ has: page.getByTitle("START / FINISH") });
  await expect(sf).toHaveCount(1);
  const position = () => sf.evaluate((el) => (el as HTMLElement).style.transform);
  await expect
    .poll(
      async () => {
        const before = await position();
        await page.waitForTimeout(300);
        return (await position()) === before;
      },
      { message: "the map camera keeps moving" },
    )
    .toBe(true);
}

/** Serves the mock's `file` rewritten by `edit`; every other file of the bundle is untouched. */
async function rewrite<T>(page: Page, file: string, edit: (data: T) => T) {
  await page.route(`**${DATA}/${file}`, async (route) => {
    const response = await route.fetch();
    await route.fulfill({ response, json: edit((await response.json()) as T) });
  });
}

/** Closes the deep-linked object's panel and clicks the centre of the map, where the link (`?tinta=`, `?rand=`) put it. */
async function clickMapCentre(page: Page, heading: Locator) {
  await expect(heading).toBeVisible();
  await mapSettled(page);
  await card(heading).getByRole("button", { name: "Închide", exact: true }).click();
  await expect(heading).toHaveCount(0);
  await page.locator(".maplibregl-canvas").click();
}

/** Closed ring of the bbox of `points`, grown by `pad` degrees on every side. */
function boxAround(points: Position[], pad: number): Position[] {
  const lons = points.map((p) => p[0]), lats = points.map((p) => p[1]);
  const [w, s, e, n] = [Math.min(...lons) - pad, Math.min(...lats) - pad, Math.max(...lons) + pad, Math.max(...lats) + pad];
  return [[w, s], [e, s], [e, n], [w, n], [w, s]];
}

for (const [file, status] of [["rows.geojson", 500], ["targets.geojson", 404]] as const) {
  test(`a required layer failing (${file}, HTTP ${status}) shows the load error, not "loading" forever`, async ({ page }) => {
    await page.route(`**${DATA}/${file}`, (route) => route.fulfill({ status }));
    await page.goto("/harta");
    const alert = page.getByRole("alert").filter({ hasText: "Datele hărții nu s-au putut încărca" });
    await expect(alert).toContainText(`${DATA}/${file}: HTTP ${status}`);
    await expect(page.getByText("Se încarcă harta…")).toHaveCount(0);
  });
}

test("a bundle without waste.geojson (404) loads the map with an empty waste layer", async ({ page }) => {
  await page.route(`**${DATA}/waste.geojson`, (route) => route.fulfill({ status: 404 }));
  await page.goto(`/harta?rand=${ROW}`);
  await expect(panelHeading(page, ROW)).toBeVisible();
  await expect(page.getByText("Datele hărții nu s-au putut încărca")).toHaveCount(0);
});

test("a deep link still applies when the map library loads after the data", async ({ page }) => {
  // react-maplibre imports maplibre-gl lazily: hold back every script that the /harta HTML does not list
  let listed = new Set<string>();
  await page.route((url) => url.pathname === "/harta", async (route) => {
    const response = await route.fetch();
    const html = await response.text();
    listed = new Set(html.match(/\/_next\/static\/chunks\/[\w.-]+\.js/g));
    await route.fulfill({ response, body: html });
  });
  await page.route("**/_next/static/chunks/*.js", async (route) => {
    if (!listed.has(new URL(route.request().url()).pathname)) await new Promise((resolve) => setTimeout(resolve, LATE_SCRIPT_MS));
    await route.continue();
  });
  await page.goto(`/harta?rand=${ROW}`);
  await expect(panelHeading(page, ROW)).toBeVisible();
});

test("the default view (zoom 17) draws no numbered stop, only the S/F marker", async ({ page }) => {
  await page.goto("/harta");
  await mapSettled(page);
  await expect(markers(page)).toHaveCount(1);
});

test("a deep-linked route stop gets its numbered marker close up", async ({ page, request }) => {
  const targets = (await (await request.get(`${DATA}/targets.geojson`)).json()) as FeatureCollection<Point, TargetProps>;
  const order = targets.features.find((x) => x.properties.target_id === TARGET)?.properties.route_order;
  expect(order, `${TARGET} is a route stop of the mock`).toEqual(expect.any(Number));
  await page.goto(`/harta?tinta=${TARGET}`);
  await expect(panelHeading(page, `Ținta ${TARGET}`)).toBeVisible();
  await expect(page.getByTitle(`${order}. ${TARGET}`)).toHaveText(String(order));
});

test(`at most ${MAX_TARGET_LABELS} numbered stops are drawn, however many are in view`, async ({ page }) => {
  // 200 more route stops packed within ±3.5 m of T003, all in view once the deep link flies there at zoom 20
  await rewrite<FeatureCollection<Point, TargetProps>>(page, "targets.geojson", (fc) => {
    const anchor = fc.features.find((x) => x.properties.target_id === TARGET)!;
    const [lon, lat] = anchor.geometry.coordinates;
    const packed = Array.from({ length: 200 }, (_, i) => ({
      ...anchor,
      id: 1000 + i,
      properties: { ...anchor.properties, target_id: `T${900 + i}`, route_order: 100 + i },
      geometry: { type: "Point" as const, coordinates: [lon + ((i % 20) - 10) * 4e-6, lat + (Math.floor(i / 20) - 5) * 6e-6] },
    }));
    return { ...fc, features: [...fc.features, ...packed] };
  });
  await page.goto(`/harta?tinta=${TARGET}`);
  await expect(panelHeading(page, `Ținta ${TARGET}`)).toBeVisible();
  await mapSettled(page);
  await expect(markers(page)).toHaveCount(MAX_TARGET_LABELS + 1); // + S/F
});

test("clicking a target circle on the canvas opens its panel, off route when the route skips it", async ({ page }) => {
  // T003 off the route: no numbered marker covers its circle at the map centre after the deep link
  await rewrite<FeatureCollection<Point, TargetProps>>(page, "targets.geojson", (fc) => ({
    ...fc,
    features: fc.features.map((x) => (x.properties.target_id === TARGET ? { ...x, properties: { ...x.properties, route_order: null } } : x)),
  }));
  await page.goto(`/harta?tinta=${TARGET}`);
  const heading = panelHeading(page, `Ținta ${TARGET}`);
  await clickMapCentre(page, heading);
  await expect(heading).toBeVisible();
  await expect(card(heading).getByText("în afara rutei", { exact: true })).toBeVisible();
});

test("clicking a waste polygon on the canvas opens the waste panel", async ({ page, request }) => {
  const rows = (await (await request.get(`${DATA}/rows.geojson`)).json()) as FeatureCollection<MultiLineString, { row_id: string }>;
  const row = rows.features.find((x) => x.properties.row_id === ROW);
  expect(row, `${ROW} is a row of the mock`).toBeDefined();
  const confidence = 0.87;
  const waste: WasteProps = { waste_id: "W901", vineyard_id: "V01", tile: "siret3_r021_c012", confidence };
  // a box around the row, larger than the view: the row deep link centres the map on it
  const fc: FeatureCollection<Polygon, WasteProps> = {
    type: "FeatureCollection",
    features: [{ type: "Feature", properties: waste, geometry: { type: "Polygon", coordinates: [boxAround(row!.geometry.coordinates.flat(), 5e-4)] } }],
  };
  await page.route(`**${DATA}/waste.geojson`, (route) => route.fulfill({ json: fc }));
  await page.goto(`/harta?rand=${ROW}`);
  await clickMapCentre(page, panelHeading(page, ROW));
  const heading = panelHeading(page, `Deșeul ${waste.waste_id}`);
  await expect(heading).toBeVisible();
  await expect(card(heading).getByText(waste.waste_id, { exact: true })).toBeVisible();
  await expect(card(heading).getByText(f.pct(confidence), { exact: true })).toBeVisible();
});
