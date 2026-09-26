// Role-based flows against the real Supabase project with the demo accounts. Local only (non-destructive):
//   E2E_PASSWORD=<demo password> pnpm e2e
// Skipped in CI and whenever the password is not provided (it is never stored in the repo).
import { expect, test, type Page } from "@playwright/test";

const PASSWORD = process.env.E2E_PASSWORD;
test.skip(!PASSWORD, "set E2E_PASSWORD to run the role flows against Supabase");
test.describe.configure({ mode: "serial" });

async function login(page: Page, email: string) {
  await page.goto("/login");
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Parolă").fill(PASSWORD!);
  await page.getByRole("button", { name: "Autentificare", exact: true }).click();
  await page.waitForURL((u) => !u.pathname.startsWith("/login"));
}

test("platform admin lands in the console; municipality row opens a prefilled UAT-admin dialog", async ({ page }) => {
  await login(page, "admin@solemtrix.demo");
  await expect(page).toHaveURL(/\/super-admin/);
  await page.goto("/super-admin?tab=uat");
  await page.getByRole("row", { name: /Cojușna/ }).getByRole("link", { name: "Adaugă administrator UAT" }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("combobox", { name: "Primărie", exact: true })).toHaveText(/Cojușna/);
  await expect(dialog.getByRole("combobox", { name: "Rol", exact: true })).toHaveText(/administrator UAT/);
});

test("municipality admin sees Tasks and Team, with AI targets to turn into tasks", async ({ page }) => {
  await login(page, "primar@sireti.demo");
  await expect(page.locator("nav").getByRole("link", { name: "Sarcini" })).toBeVisible();
  await expect(page.locator("nav").getByRole("link", { name: "Echipă" })).toBeVisible();
  await page.goto("/echipa");
  await expect(page.getByRole("row", { name: /inspector@sireti\.demo/ })).toBeVisible();
  await page.goto("/sarcini");
  await expect(page.getByRole("heading", { level: 3, name: /Ținte din analiza AI/ })).toBeVisible();
});

test("inspector has Tasks but no Team (403)", async ({ page }) => {
  await login(page, "inspector@sireti.demo");
  await expect(page.locator("nav").getByRole("link", { name: "Sarcini" })).toBeVisible();
  await expect(page.locator("nav").getByRole("link", { name: "Echipă" })).toHaveCount(0);
  expect((await page.goto("/echipa"))?.status()).toBe(403);
});

test("another municipality sees none of Sireți's data", async ({ page }) => {
  await login(page, "primar@cojusna.demo");
  await expect(page.getByText("Niciun zbor de dronă în limita acestei primării")).toBeVisible();
  await page.goto("/sarcini");
  await expect(page.getByRole("heading", { level: 3, name: /Ținte din analiza AI fără sarcină \(0\)/ })).toBeVisible();
});
