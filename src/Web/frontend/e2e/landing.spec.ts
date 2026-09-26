// The presentation site: what "/" shows without a session or demo cookie, its entries into the app, the pilot form.
import { expect, test } from "@playwright/test";

test("a visitor without an account sees the presentation at /, not the login", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { level: 1, name: "Fiecare vie din comună, măsurată rând cu rând" })).toBeVisible();
  for (const name of ["Cum funcționează", "Pentru cine", "Securitate și date", "Cum începem", "Întrebări frecvente"]) {
    await expect(page.getByRole("heading", { level: 2, name })).toBeVisible();
  }
});

test("the presentation is translated and reachable at /prezentare", async ({ page }) => {
  await page.goto("/en");
  await expect(page.getByRole("heading", { level: 1, name: "Every vineyard in the commune, measured row by row" })).toBeVisible();
  await page.goto("/ru/prezentare");
  await expect(page.getByRole("heading", { level: 1, name: "Каждый виноградник коммуны, измеренный ряд за рядом" })).toBeVisible();
});

test("the demo button opens the municipality dashboard", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Vedeți demo-ul Sireți" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Primăria Sireți" })).toBeVisible();
  // with the demo cookie "/" is the app again
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1, name: "Primăria Sireți" })).toBeVisible();
});

test("sign in leads to the login page, and back", async ({ page, isMobile }) => {
  test.skip(isMobile, "the header sign-in button is in the menu drawer on phones");
  await page.goto("/");
  await page.getByRole("banner").getByRole("link", { name: "Autentificare" }).click();
  await expect(page).toHaveURL(/\/login$/);
  await page.getByRole("link", { name: "Despre Solemtrix" }).click();
  await expect(page).toHaveURL(/\/prezentare$/);
});

test("the pilot form answers without storing an instant (bot-like) submission", async ({ page }) => {
  await page.goto("/#contact");
  await page.getByLabel("Nume și prenume").fill("Test E2E");
  await page.getByLabel("Instituția").fill("Primăria Test");
  await page.getByLabel("Email").fill("e2e@example.com");
  await page.getByLabel(/Sunt de acord/).check();
  await page.getByRole("button", { name: "Trimite cererea" }).click();
  await expect(page.getByText("Mulțumim! Cererea a fost înregistrată")).toBeVisible();
});
