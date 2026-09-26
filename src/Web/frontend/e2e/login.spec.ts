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

test("forgot password opens the reset step and returns to sign in", async ({ page }) => {
  await page.goto("/login");
  await page.getByRole("button", { name: "Ați uitat parola?" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Resetați parola" })).toBeVisible();
  await expect(page.getByLabel("Email")).toBeVisible();
  await page.getByRole("button", { name: "Înapoi la autentificare" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Bine ați revenit!" })).toBeVisible();
  // an expired reset link sends the user straight to this step
  await page.goto("/en/login?forgot=1");
  await expect(page.getByRole("heading", { level: 1, name: "Reset your password" })).toBeVisible();
});
