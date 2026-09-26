// The optional survey overlays (tile footprints, vegetation masks; src/Web/CLAUDE.md §6.3), on whatever survey the
// build serves. The mock and older bundles do not ship tiles.geojson / masks/: both switches are disabled with a hint,
// nothing is requested and the map loads. A bundle that ships them: both switches start off, load on demand, show
// their legend and the dashboard shows the tile counts (read from summary.json, never hardcoded).
import { expect, test, type Page } from "@playwright/test";
import { SURVEY_ID } from "../src/lib/data";
import { makeFormat } from "../src/lib/format";
import type { SurveySummary } from "../src/lib/types";

const DATA = `/data/${SURVEY_ID}`;
const f = makeFormat("ro");

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

/** The layer panel, closed by default below the theme's lg breakpoint (MUI default, 1200 px). Decided on the
 *  viewport, not on what is visible: the server render has the panel open and it closes after hydration on narrow
 *  screens, so a visibility check could catch the panel open just before it closes. */
async function openLayers(page: Page) {
  if (page.viewportSize()!.width < 1200) await page.getByRole("button", { name: "Deschide straturile" }).click();
  await expect(page.getByText("Straturi", { exact: true })).toBeVisible();
}

const tilesSwitch = (page: Page) => page.getByRole("checkbox", { name: /^Tile-uri/ });
const maskSwitch = (page: Page) => page.getByRole("checkbox", { name: /^Mască vegetație/ });

test("overlay switches follow the files the survey ships; the map loads either way", async ({ page, request }) => {
  const shipsTiles = (await request.get(`${DATA}/tiles.geojson`)).ok();
  const shipsMasks = (await request.get(`${DATA}/masks/index.json`)).ok();
  const overlayRequests: string[] = [];
  page.on("request", (r) => {
    if (/\/(tiles\.geojson|masks\/)/.test(r.url())) overlayRequests.push(r.url());
  });

  await page.goto("/harta");
  await openLayers(page);
  await expect(page.getByText("Se încarcă harta…")).toHaveCount(0);
  await expect(page.getByText("Datele hărții nu s-au putut încărca")).toHaveCount(0);

  for (const [box, ships] of [[tilesSwitch(page), shipsTiles], [maskSwitch(page), shipsMasks]] as const) {
    await expect(box).not.toBeChecked();
    if (ships) await expect(box).toBeEnabled();
    else await expect(box).toBeDisabled();
  }
  const hint = page.getByText("indisponibil pentru acest survey");
  await expect(hint).toHaveCount(Number(!shipsTiles) + Number(!shipsMasks));
  // off by default: nothing is fetched until a switch is turned on
  expect(overlayRequests).toEqual([]);

  if (shipsTiles) {
    const loaded = page.waitForResponse((r) => r.url().endsWith(`${DATA}/tiles.geojson`) && r.ok());
    await tilesSwitch(page).check();
    await loaded;
    await expect(page.getByText("Tile de completat în Marcaj")).toBeVisible();
    await expect(tilesSwitch(page)).toBeChecked();
  }
});

test("dashboard: the tile line appears only when the survey ships tiles", async ({ page, request }) => {
  const summary = (await (await request.get(`${DATA}/summary.json`)).json()) as SurveySummary;
  await page.goto("/");
  await expect(page.getByText("Suprafață zburată")).toBeVisible();
  const card = page.getByText("Tile-uri analizate");
  if (!summary.tiles) {
    await expect(card).toHaveCount(0);
    return;
  }
  const t = summary.tiles;
  await expect(card).toBeVisible();
  await expect(
    page.getByText(`${f.int(t.vineyard)} cu vie · ${f.int(t.no_vineyard)} fără vie · ${f.int(t.to_complete)} de completat în Marcaj`),
  ).toBeVisible();
});
