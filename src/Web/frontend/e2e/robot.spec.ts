// The field robot page (/robot): without addresses every control waits; a public address is refused; with the
// stand-in boards (scripts/robot/fake-robot.mjs, on this computer's LAN address) a photo lands in the gallery with
// the camera angles, the camera moves and a drive command goes through. The stand-in part needs a private IPv4
// address (skipped where there is none, e.g. some CI runners).
import { spawn, type ChildProcess } from "node:child_process";
import { networkInterfaces } from "node:os";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

/** each Playwright worker (desktop, mobile) runs its own stand-in boards: 3 ports from here, 10 apart per worker */
const BASE_PORT = 18081;
const lan = Object.values(networkInterfaces())
  .flat()
  .find((a) => a && a.family === "IPv4" && !a.internal && /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(a.address))?.address;

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "solemtrix_demo", value: "1", url: baseURL! }]);
});

const address = (page: Page, device: "cam" | "motors" | "drive") => page.getByTestId(`robot-url-${device}`);
/** Settings are a draft until "Salvează". */
async function saveSettings(page: Page) {
  await page.getByTestId("robot-settings-save").click();
  await expect(page.getByTestId("robot-settings-saved")).toBeVisible();
}

test("without addresses the controls wait, and a public address is refused", async ({ page }) => {
  await page.goto("/robot");
  await expect(page.getByRole("heading", { name: "Robot de teren", level: 1 })).toBeVisible();
  // the build may carry default addresses (NEXT_PUBLIC_ROBOT_*_URL): clear them
  for (const d of ["cam", "motors", "drive"] as const) await address(page, d).fill("");
  if (await page.getByTestId("robot-settings-save").isEnabled()) await saveSettings(page);
  await expect(page.getByText("Scrie adresa camerei (ESP32-CAM) în Setări.")).toBeVisible();
  await expect(page.getByTestId("robot-capture")).toBeDisabled();
  await expect(page.getByTestId("robot-camera-pad").getByRole("button", { name: "Camera în sus" })).toBeDisabled();
  await expect(page.getByTestId("robot-drive-pad").getByRole("button", { name: "Înainte" })).toBeDisabled();
  await address(page, "cam").fill("8.8.8.8");
  await expect(page.getByText("Doar o adresă din rețeaua locală, de ex. 192.168.1.50")).toBeVisible();
  await expect(page.getByTestId("robot-settings-save")).toBeDisabled();
  await expect(page.getByTestId("robot-capture")).toBeDisabled();
});

test.describe("with the stand-in boards", () => {
  test.skip(!lan, "no private IPv4 address on this machine");
  let fake: ChildProcess;
  let port = BASE_PORT;
  test.beforeAll(async ({}, info) => {
    port = BASE_PORT + info.workerIndex * 10;
    fake = spawn(process.execPath, [path.join(__dirname, "../scripts/robot/fake-robot.mjs"), "--host", lan!, "--port", String(port)], { stdio: "pipe" });
    await new Promise<void>((resolve, reject) => {
      let out = "";
      fake.stdout!.on("data", (d: Buffer) => (out += d.toString()).includes("drive") && resolve());
      fake.stderr!.on("data", (d: Buffer) => reject(new Error(`fake robot: ${d.toString().slice(0, 300)}`)));
      fake.on("exit", (code) => reject(new Error(`fake robot exited (${code})`)));
    });
  });
  test.afterAll(() => {
    fake?.kill();
  });

  /** Holds a pad button down for `ms`, then lets go. */
  async function holdButton(page: Page, pad: string, name: string, ms: number) {
    const button = page.getByTestId(pad).getByRole("button", { name, exact: true });
    await button.hover();
    await page.mouse.down();
    await page.waitForTimeout(ms);
    await page.mouse.up();
  }
  const wheels = async (): Promise<{ running: boolean }[]> => (await fetch(`http://${lan}:${port + 2}/status`)).json();

  test("hold to move: the camera turns while held, the wheels run while held and stop on release; photos keep the pose", async ({ page }) => {
    await page.goto("/robot");
    await address(page, "cam").fill(`${lan}:${port}`);
    await address(page, "motors").fill(`${lan}:${port + 1}`);
    await address(page, "drive").fill(`${lan}:${port + 2}`);
    await saveSettings(page);

    await holdButton(page, "robot-camera-pad", "Camera la dreapta", 600);
    await expect(page.getByText(/Poziție: [1-9]\d*° orizontal, înălțime 0 pași\./)).toBeVisible();
    // calibration: the turn from 0° counts as exactly 90° (the button waits for the camera's last move to end)
    await expect(page.getByTestId("robot-calibrate-90")).toBeEnabled();
    await page.getByTestId("robot-calibrate-90").click();
    await expect(page.getByTestId("robot-status")).toContainText(/Calibrat: 90° = \d+ pași/);
    await expect(page.getByText(/Poziție: 90° orizontal/)).toBeVisible();
    await holdButton(page, "robot-camera-pad", "Camera în sus", 600);
    await expect(page.getByText(/Poziție: [1-9]\d*° orizontal, înălțime [1-9]\d* pași\./)).toBeVisible();

    await page.getByTestId("robot-capture").click();
    await expect(page.getByTestId("robot-status")).toHaveText("Poza a fost salvată în galerie.");
    await expect(page.getByTestId("robot-photo")).toHaveCount(1);
    await expect(page.getByTestId("robot-photo")).toContainText(/\d+° · înălțime [1-9]\d*/);
    const download = page.waitForEvent("download");
    await page.getByTestId("robot-photo").getByRole("button", { name: "Descarcă" }).click();
    expect((await download).suggestedFilename()).toMatch(/^robot_\d{8}_\d{6}_pan\d+_h[1-9]\d*\.jpg$/);

    await expect(page.getByTestId("robot-distance")).toHaveText(/Distanță \(senzor\): 10\d\.\d cm/);
    await expect(page.getByTestId("robot-live-distance")).toHaveText(/^10\d,\d cm$/); // on the live picture, in cm

    // the wheels: running while the button is held, stopped once it is let go
    const button = page.getByTestId("robot-drive-pad").getByRole("button", { name: "Înainte", exact: true });
    await button.hover();
    await page.mouse.down();
    await expect.poll(async () => (await wheels()).every((w) => w.running), { timeout: 5000 }).toBe(true);
    await page.waitForTimeout(1200); // longer than the server's watchdog: the renewals keep them going
    expect((await wheels()).every((w) => w.running)).toBe(true);
    await page.mouse.up();
    await expect.poll(async () => (await wheels()).some((w) => w.running), { timeout: 5000 }).toBe(false);

    // the addresses are remembered after a reload
    await page.reload();
    await expect(address(page, "cam")).toHaveValue(`${lan}:${port}`);
  });

  test("panorama: each Start drives forward, takes 0° / 90° / 180°, is saved on the laptop with grape and leaf boxes", async ({ page }, info) => {
    const request = page.request; // with the page's demo cookie (the API checks it)
    // the panoramas live on the one test server, shared by both projects: this checks them from one of them
    test.skip(info.project.name !== "desktop", "server-side panoramas are checked once");
    for (const p of ((await (await request.get("/api/robot/panoramas")).json()) as { panoramas: { id: string }[] }).panoramas)
      await request.delete(`/api/robot/panoramas/${p.id}`);
    await page.goto("/robot");
    await address(page, "cam").fill(`${lan}:${port}`);
    await address(page, "motors").fill(`${lan}:${port + 1}`);
    await address(page, "drive").fill(`${lan}:${port + 2}`);
    await page.getByTestId("robot-set-recordStepMs").fill("300");
    await saveSettings(page);

    for (const n of [1, 2]) {
      await page.getByTestId("robot-record-start").click();
      await expect(page.getByTestId("robot-record-phase")).toBeVisible();
      await expect(page.getByTestId("robot-panorama")).toHaveCount(n, { timeout: 30_000 });
      await expect(page.getByTestId("robot-record-start")).toBeVisible(); // done, ready for the next
    }
    const newest = page.getByTestId("robot-panorama").first();
    await expect(newest).toContainText("Panorama 2 · la 100 cm");
    // the detection (fake model: one grape, two leaves per photo) is drawn on the three photos
    await expect(newest.getByTestId("robot-detection")).toHaveAttribute("data-status", "done", { timeout: 15_000 });
    await expect(newest.locator('rect[data-label="grape"]')).toHaveCount(3);
    await expect(newest.locator('rect[data-label="leaf"]')).toHaveCount(6);
    await expect(newest.getByTestId("robot-detection")).toContainText("3 ciorchini");
    await expect(newest.getByTestId("robot-detection")).toContainText("6 frunze");
    const download = page.waitForEvent("download");
    await newest.getByRole("button", { name: "Descarcă panorama" }).click();
    expect((await download).suggestedFilename()).toMatch(/^panorama_\d{8}_\d{6}_statia02_100cm\.jpg$/);
    expect((await wheels()).some((w) => w.running)).toBe(false);
    await expect(page.getByText(/Poziție: 0° orizontal/)).toBeVisible(); // the camera is back at 0°

    // saved: still there after a reload
    await page.reload();
    await expect(page.getByTestId("robot-panorama")).toHaveCount(2);
    await expect(page.getByTestId("robot-panorama").first().locator("rect[data-label]")).toHaveCount(9);
  });

  test("an obstacle closer than 20 cm blocks forward and stops the wheels going forward; back still works", async ({ page }) => {
    const obstacle = (cm?: number) => fetch(`http://${lan}:${port + 1}/fake/distance${cm === undefined ? "" : `?cm=${cm}`}`);
    await page.goto("/robot");
    await address(page, "cam").fill(`${lan}:${port}`);
    await address(page, "motors").fill(`${lan}:${port + 1}`);
    await address(page, "drive").fill(`${lan}:${port + 2}`);
    await saveSettings(page);
    const forward = page.getByTestId("robot-drive-pad").getByRole("button", { name: "Înainte", exact: true });
    const back = page.getByTestId("robot-drive-pad").getByRole("button", { name: "Înapoi", exact: true });

    await obstacle(12);
    await expect(page.getByTestId("robot-live-distance")).toHaveText(/12,0 cm · obstacol/);
    await expect(forward).toBeDisabled();
    await expect(page.getByTestId("robot-record-start")).toBeDisabled();
    await back.hover();
    await page.mouse.down();
    await expect.poll(async () => (await wheels()).every((w) => w.running), { timeout: 5000 }).toBe(true);
    await page.mouse.up();
    await expect.poll(async () => (await wheels()).some((w) => w.running), { timeout: 5000 }).toBe(false);

    // clear ahead: forward works again; an obstacle appearing while going forward stops the wheels
    await obstacle();
    await expect(forward).toBeEnabled();
    await forward.hover();
    await page.mouse.down();
    await expect.poll(async () => (await wheels()).every((w) => w.running), { timeout: 5000 }).toBe(true);
    await obstacle(15);
    await expect.poll(async () => (await wheels()).some((w) => w.running), { timeout: 3000 }).toBe(false);
    await page.mouse.up();
    await expect(page.getByTestId("robot-status")).toContainText("Obstacol la 15,0 cm");
    await obstacle();
  });
});
