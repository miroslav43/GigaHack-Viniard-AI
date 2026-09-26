// The in-app assistant: the panel, suggestions, a streamed answer with an in-app link, and the panel staying open
// after following it. The Gemini call is mocked (no key in CI; answers are not deterministic).
import { expect, test } from "@playwright/test";

const ANSWER = "1. Deschideți [Sarcini de teren](/sarcini).\n2. Bifați ținta și apăsați **Creează N sarcini**.\n\nAlt link: [extern](https://example.com)";

test("assistant answers with an in-app link and stays open across navigation", async ({ page, context }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: "http://localhost" }]);
  await page.route("**/api/assistant", async (route) => {
    const body = route.request().postDataJSON();
    expect(body.messages.at(-1).role).toBe("user");
    expect(body.locale).toBe("ro");
    await route.fulfill({ status: 200, contentType: "text/plain; charset=utf-8", body: ANSWER });
  });
  await page.goto("/harta");
  await page.getByRole("button", { name: "Asistent" }).first().click();
  const panel = page.getByRole("dialog", { name: "Asistent Solemtrix" });
  await expect(panel).toBeVisible();
  await panel.getByRole("button", { name: "Ce arată harta?" }).click();
  await expect(panel.getByRole("link", { name: "Sarcini de teren" })).toBeVisible();
  // external URLs are not rendered as links
  await expect(panel.getByRole("link", { name: "extern" })).toHaveCount(0);
  await panel.getByRole("link", { name: "Sarcini de teren" }).click();
  await expect(page).toHaveURL(/\/sarcini$/);
  await expect(panel).toBeVisible();
  await expect(panel.getByText("Ce arată harta?")).toBeVisible();
});

test("without a Gemini key the assistant explains it is not configured", async ({ page, context }) => {
  test.skip(!!process.env.GEMINI_API_KEY, "a key is configured here");
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: "http://localhost" }]);
  await page.goto("/");
  await page.getByRole("button", { name: "Asistent" }).first().click();
  const panel = page.getByRole("dialog", { name: "Asistent Solemtrix" });
  await panel.getByRole("textbox").fill("Cum export CSV?");
  await panel.getByRole("button", { name: "Trimite" }).click();
  await expect(panel.getByText("Asistentul nu este configurat")).toBeVisible();
});
