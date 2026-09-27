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

  test("record mode: two stations give two panoramas of three photos", async ({ page }) => {
    await page.goto("/robot");
    await address(page, "cam").fill(`${lan}:${port}`);
    await address(page, "motors").fill(`${lan}:${port + 1}`);
    await address(page, "drive").fill(`${lan}:${port + 2}`);
    await saveSettings(page);
    await page.getByTestId("robot-set-recordStations").fill("2");
    await page.getByTestId("robot-set-recordStepMs").fill("300");
    await saveSettings(page);

    await page.getByTestId("robot-record-start").click();
    await expect(page.getByTestId("robot-record-phase")).toBeVisible();
    await expect(page.getByTestId("robot-panorama")).toHaveCount(2, { timeout: 30_000 });
    await expect(page.getByTestId("robot-record-start")).toBeVisible(); // done
    await expect(page.getByTestId("robot-panorama").first()).toContainText("Stația 2 · 50 cm de la start");
    const download = page.waitForEvent("download");
    await page.getByTestId("robot-panorama").first().getByRole("button", { name: "Descarcă panorama" }).click();
    expect((await download).suggestedFilename()).toMatch(/^panorama_\d{8}_\d{6}_statia02_50cm\.jpg$/);
    expect((await wheels()).some((w) => w.running)).toBe(false);
  });
});
