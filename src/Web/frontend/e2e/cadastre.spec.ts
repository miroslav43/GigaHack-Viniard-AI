// The live cadastre layer (src/lib/cadastre.ts, AGCC WMS). Off by default: nothing goes to the external server until
// the switch is on. Switched on and zoomed in, a click on empty ground looks the parcel up (GetFeatureInfo) and shows
// it in the attribute panel. Both AGCC requests are mocked: the tests never reach geodata.gov.md.
import { expect, test, type Page } from "@playwright/test";
import { CADASTRE_ORIGIN } from "../src/lib/cadastre";

const PNG_1PX = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);
const PARCEL = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      properties: {
        codcadastral: "80371110019",
        cod_parcel: " 0019",
        landuse: "Teren pentru grădini",
        typeproperty: "NEDETERMINAT",
        aria: "0.12 ha",
        description: "<b>html-must-not-render</b>",
      },
      geometry: {
        type: "Polygon",
        coordinates: [[[28.707, 47.125], [28.7072, 47.125], [28.7072, 47.1252], [28.707, 47.1252], [28.707, 47.125]]],
      },
    },
  ],
};

async function mockCadastre(page: Page): Promise<string[]> {
  const seen: string[] = [];
  await page.route(`${CADASTRE_ORIGIN}/**`, (route) => {
    const url = route.request().url();
    seen.push(url);
    if (url.includes("GetFeatureInfo")) return route.fulfill({ json: PARCEL });
    return route.fulfill({ body: PNG_1PX, contentType: "image/png" });
  });
  return seen;
}

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

async function openLayers(page: Page) {
  if (page.viewportSize()!.width < 1200) await page.getByRole("button", { name: "Deschide straturile" }).click();
  await expect(page.getByText("Straturi", { exact: true })).toBeVisible();
}

const cadastreSwitch = (page: Page) => page.getByRole("checkbox", { name: /^Cadastru/ });

test("the cadastre switch is off by default and nothing is requested from AGCC", async ({ page }) => {
  const seen = await mockCadastre(page);
  await page.goto("/harta");
  await openLayers(page);
  await expect(cadastreSwitch(page)).not.toBeChecked();
  await page.waitForTimeout(2000);
  expect(seen).toEqual([]);
});

test("switched on and zoomed in, a click on empty ground shows the parcel", async ({ page }) => {
  const seen = await mockCadastre(page);
  await page.goto("/harta");
  await openLayers(page);
  await cadastreSwitch(page).check();
  // hide every layer of ours, so the click lands on empty ground
  for (const box of await page.getByRole("checkbox").all()) {
    const name = (await box.getAttribute("aria-label")) ?? (await box.evaluate((el) => el.closest("label")?.textContent ?? ""));
    if (/^(Cadastru|Ortofoto)/.test(name.trim()) || !(await box.isEnabled()) || !(await box.isChecked())) continue;
    await box.uncheck();
  }
  for (let i = 0; i < 3; i++) {
    await page.locator(".maplibregl-ctrl-zoom-in").click();
    await page.waitForTimeout(600);
  }
  const canvas = page.locator(".maplibregl-canvas");
  const box = (await canvas.boundingBox())!;
  await canvas.click({ position: { x: box.width * 0.75, y: box.height * 0.5 } });
  await expect(page.getByText("Parcelă cadastrală")).toBeVisible();
  await expect(page.getByTestId("cadastre-parcel")).toContainText("80371110019");
  await expect(page.getByTestId("cadastre-parcel")).toContainText("Teren pentru grădini");
  await expect(page.getByText("html-must-not-render")).toHaveCount(0);
  expect(seen.some((u) => u.includes("GetFeatureInfo"))).toBe(true);
});
