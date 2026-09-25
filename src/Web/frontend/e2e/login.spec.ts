// Login page and the "demo without an account" entry. Works with and without Supabase configured.
import { expect, test } from "@playwright/test";

test("login page renders the form and enters the demo", async ({ page }) => {
  await page.goto("/login");
  await expect(page.getByRole("heading", { level: 1, name: "Bine ați revenit!" })).toBeVisible();
  await expect(page.getByLabel("Email")).toBeVisible();
  await page.getByRole("button", { name: "Demo fără cont" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { level: 1, name: "Primăria Sireți" })).toBeVisible();
});

test("login page is translated", async ({ page }) => {
  await page.goto("/en/login");
  await expect(page.getByRole("heading", { level: 1, name: "Welcome back!" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Demo without an account" })).toBeVisible();
});
