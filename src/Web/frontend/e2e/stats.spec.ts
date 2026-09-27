// /statistici (docs/STATISTICI.md), on whatever survey the build serves: the menu link, the hero figures counting up to
// the values of summary.json / stats.json (read at run time, never hardcoded), one section per data set the survey has
// (farms and roads only when stats.json has them), charts mounting in view, the health ranking linking to the map,
// and "reduce motion" showing the final values straight away.
import { expect, test, type Page } from "@playwright/test";
import { SURVEY_ID } from "../src/lib/data";
import { makeFormat } from "../src/lib/format";
import type { SurveyStats, SurveySummary } from "../src/lib/types";

const DATA = `/data/${SURVEY_ID}`;
const f = makeFormat("ro");
const SECTION_TITLES: Record<string, string> = {
  stare: "Starea plantației",
  structura: "Structură",
  ferme: "Ferme și cadastru",
  drumuri: "Drumuri",
  ruta: "Rută",
  zbor: "Zbor și AI",
};

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

async function load(page: Page) {
  const [summary, stats] = await Promise.all([
    page.request.get(`${DATA}/summary.json`).then((r) => r.json() as Promise<SurveySummary>),
    page.request.get(`${DATA}/stats.json`).then((r) => r.json() as Promise<SurveyStats>),
  ]);
  return { summary, stats };
}

const expectedSections = (stats: SurveyStats) =>
  Object.keys(SECTION_TITLES).filter((id) => (id === "ferme" ? stats.farms != null : id === "drumuri" ? stats.roads != null : true));

test("the menu opens /statistici; the hero counts up to the survey's figures", async ({ page }) => {
  const { summary, stats } = await load(page);
  // after hydration: the mobile shell (bottom navigation) replaces the server-rendered one
  await page.goto("/", { waitUntil: "networkidle" });
  await page.getByRole("link", { name: "Statistici" }).first().click();
  await expect(page).toHaveURL(/\/statistici$/);
  await expect(page.getByRole("heading", { level: 1, name: "Statistici" })).toBeVisible();
  const hero = page.locator("dl").first();
  await expect(hero).toContainText(f.int(summary.totals.canopy_count));
  await expect(hero).toContainText(f.int(summary.totals.row_count));
  await expect(hero).toContainText(f.int(stats.targets.total));
});

test("one section per data set the survey has, each reachable from the section chips", async ({ page }) => {
  const { stats } = await load(page);
  await page.goto("/statistici", { waitUntil: "networkidle" });
  const ids = expectedSections(stats);
  for (const id of Object.keys(SECTION_TITLES)) {
    await expect(page.locator(`section#${id}`)).toHaveCount(ids.includes(id) ? 1 : 0);
  }
  const nav = page.getByRole("navigation", { name: "Secțiunile paginii" });
  await expect(nav.getByRole("link")).toHaveCount(ids.length);
  const last = ids.at(-1)!;
  await nav.getByRole("link", { name: new RegExp(SECTION_TITLES[last]) }).click();
  await expect(page).toHaveURL(new RegExp(`#${last}$`));
  await expect(page.locator(`section#${last} h2`)).toBeInViewport();
});

test("charts mount when scrolled into view; the health ranking links to the block on the map", async ({ page }) => {
  const { stats } = await load(page);
  test.skip(!stats.health.blocks.some((b) => b.score != null), "no block long enough for a health score");
  await page.goto("/statistici", { waitUntil: "networkidle" });
  const health = page.locator("section#stare");
  // the ranking is the section's last card; its bars mount only once it is on screen
  await health.locator(".MuiCard-root").last().scrollIntoViewIfNeeded();
  const blockLink = health.locator('a[href*="/harta?bloc="]').first();
  await expect(blockLink).toBeVisible();
  const href = await blockLink.getAttribute("href");
  await blockLink.click();
  await expect(page).toHaveURL(new RegExp(`${href!.replace("?", "\\?")}$`));
});

test.describe("reduce motion", () => {
  test.use({ reducedMotion: "reduce" });

  test("final values straight away, no animation", async ({ page }) => {
    const { summary } = await load(page);
    await page.goto("/statistici", { waitUntil: "networkidle" });
    await expect(page.locator("dl").first()).toContainText(f.int(summary.totals.canopy_count), { timeout: 1_000 });
    // charts are mounted without scrolling: the last section already has its chart in the DOM
    await expect(page.locator("section").last().locator("svg").first()).toBeAttached({ timeout: 2_000 });
  });
});
