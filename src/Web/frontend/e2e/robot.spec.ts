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

test("without addresses the controls wait, and a public address is refused", async ({ page }) => {
  await page.goto("/robot");
  await expect(page.getByRole("heading", { name: "Robot de teren", level: 1 })).toBeVisible();
  await expect(page.getByText("Scrie adresa camerei (ESP32-CAM) în Setări.")).toBeVisible();
  await expect(page.getByTestId("robot-capture")).toBeDisabled();
  await expect(page.getByTestId("robot-camera-pad").getByRole("button", { name: "Camera în sus" })).toBeDisabled();
  await expect(page.getByTestId("robot-drive-pad").getByRole("button", { name: "Înainte" })).toBeDisabled();
  await address(page, "cam").fill("8.8.8.8");
  await expect(page.getByText("Doar o adresă din rețeaua locală, de ex. 192.168.1.50")).toBeVisible();
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

  test("a photo lands in the gallery with the camera angles; the camera and the wheels answer", async ({ page }) => {
    await page.goto("/robot");
    await address(page, "cam").fill(`${lan}:${port}`);
    await address(page, "motors").fill(`${lan}:${port + 1}`);
    await address(page, "drive").fill(`${lan}:${port + 2}`);
    const status = page.getByTestId("robot-status");

    await page.getByTestId("robot-camera-pad").getByRole("button", { name: "Camera la dreapta" }).click();
    await expect(status).toHaveText("Camera s-a mișcat.");
    await expect(page.getByText(/Poziție estimată: 14° orizontal, 0° vertical/)).toBeVisible();

    await page.getByTestId("robot-capture").click();
    await expect(status).toHaveText("Poza a fost salvată în galerie.");
    await expect(page.getByTestId("robot-photo")).toHaveCount(1);
    await expect(page.getByTestId("robot-photo")).toContainText("14° / 0°");
    const download = page.waitForEvent("download");
    await page.getByTestId("robot-photo").getByRole("button", { name: "Descarcă" }).click();
    expect((await download).suggestedFilename()).toMatch(/^robot_\d{8}_\d{6}_pan14_tilt0\.jpg$/);

    await page.getByTestId("robot-drive-pad").getByRole("button", { name: "Înainte" }).click();
    await expect(status).toHaveText("Mișcare trimisă.");

    // the addresses are remembered after a reload
    await page.reload();
    await expect(address(page, "cam")).toHaveValue(`${lan}:${port}`);
  });
});
