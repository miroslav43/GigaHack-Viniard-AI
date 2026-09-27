import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

// Demo-flow e2e against the production build (`pnpm build` first). Data: `pnpm data:fast` (siret3-mock).
// E2E_PORT: run next to another build (a second worktree) without reusing its server
const PORT = Number(process.env.E2E_PORT ?? 3100);

// The specs pick their survey from NEXT_PUBLIC_SURVEY_ID, which `next build` inlined from the shell or a .env file
// (the demo laptop sets it in .env.local). Load the same files, in Next's order for a production build; a file never
// overrides a variable that is already set, so the shell wins, as it does for Next.
for (const file of [".env.production.local", ".env.local", ".env.production", ".env"]) {
  const envPath = path.join(__dirname, file);
  if (existsSync(envPath)) process.loadEnvFile(envPath);
}

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    trace: "retain-on-failure",
    // WebGL for MapLibre in headless Chromium
    launchOptions: { args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"] },
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: {
    command: `pnpm start --port ${PORT}`,
    // the robot page's panorama detection answers fixed boxes (no model call, no credit) and saves to a scratch folder
    env: {
      ...Object.fromEntries(Object.entries(process.env).filter((e): e is [string, string] => typeof e[1] === "string")),
      ROBOT_DETECT: "fake",
      ROBOT_DATA_DIR: path.join(os.tmpdir(), `solemtrix-e2e-robot-${PORT}`),
    },
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
