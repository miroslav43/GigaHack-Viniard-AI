// Role-based flows against the real Supabase project with the demo accounts. Local only (non-destructive):
//   E2E_PASSWORD=<demo password> pnpm e2e
// Skipped in CI and whenever the password is not provided (it is never stored in the repo).
import { expect, test, type Page } from "@playwright/test";
import { createClient } from "@supabase/supabase-js";

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
  // new members are invited by email and choose their own password: no password field in the dialog
  await page.getByRole("button", { name: "Adaugă membru" }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("button", { name: "Trimite invitația" })).toBeVisible();
  await expect(dialog.getByLabel(/Parolă/)).toHaveCount(0);
  await dialog.getByRole("button", { name: "Anulează" }).click();
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

test("assigning a task notifies the inspector live; the notification opens the task", async ({ page }) => {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL!;
  const key = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY!;
  const admin = createClient(url, key, { auth: { persistSession: false } });
  const inspector = createClient(url, key, { auth: { persistSession: false } });
  await admin.auth.signInWithPassword({ email: "primar@sireti.demo", password: PASSWORD! });
  const { data: me } = await inspector.auth.signInWithPassword({ email: "inspector@sireti.demo", password: PASSWORD! });
  const inspectorId = me.user!.id;

  await login(page, "inspector@sireti.demo");
  await page.goto("/blocuri");
  const bell = page.getByRole("button", { name: /^Notificări/ });
  await expect(bell).toBeVisible();
  await page.waitForTimeout(2000); // realtime channel joined
  const unread = async () => Number((await bell.getAttribute("aria-label"))?.match(/(\d+)/)?.[1] ?? 0);
  const before = await unread();
  // a throwaway task of the served survey, created already assigned (the trigger fires on insert)
  const { data: task, error } = await admin
    .from("task")
    .insert({
      uat_key: "sireti",
      survey_id: process.env.NEXT_PUBLIC_SURVEY_ID ?? "siret3-mock",
      kind: "other",
      title: "E2E: notificare",
      assignee: inspectorId,
      assignee_email: "inspector@sireti.demo",
    })
    .select("id")
    .single();
  expect(error).toBeNull();
  try {
    const toast = page.getByRole("alert").filter({ hasText: "V-a fost alocată o sarcină" });
    await expect(toast).toBeVisible({ timeout: 15_000 });
    await expect.poll(unread).toBe(before + 1);
    await toast.getByRole("button", { name: "Deschide" }).click();
    await expect(page).toHaveURL(new RegExp(`/sarcini\\?sarcina=${task!.id}$`));
    await expect(page.locator(`#task-${task!.id}`)).toBeVisible();
    await expect.poll(unread).toBe(before);
  } finally {
    // deleting the task also deletes its notification (on delete cascade)
    await admin.from("task").delete().eq("id", task!.id);
  }
});
