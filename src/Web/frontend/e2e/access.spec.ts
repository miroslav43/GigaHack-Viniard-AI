// /super-admin is reachable only by URL and only for a platform admin: everyone else gets HTTP 403.
import { expect, test } from "@playwright/test";

const DENIED = [
  ["/super-admin", "Acces interzis"],
  ["/en/super-admin", "Access denied"],
  ["/ru/super-admin", "Доступ запрещён"],
] as const;

for (const [path, title] of DENIED) {
  test(`anonymous ${path} → 403`, async ({ page }) => {
    const res = await page.goto(path);
    expect(res?.status()).toBe(403);
    await expect(page.getByRole("heading", { level: 1, name: title })).toBeVisible();
  });
}

test("demo visitor gets 403 and the menu has no super-admin link", async ({ page, context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
  const res = await page.goto("/super-admin?tab=users");
  expect(res?.status()).toBe(403);
  await page.goto("/");
  await expect(page.locator('a[href*="super-admin"]')).toHaveCount(0);
});
