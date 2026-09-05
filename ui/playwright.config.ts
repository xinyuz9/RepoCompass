import { defineConfig } from "@playwright/test";

const externalUrl = process.env.REPOCOMPASS_E2E_URL;
const baseURL = externalUrl ?? "http://127.0.0.1:8878";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  reporter: "list",
  use: {
    baseURL,
    browserName: "chromium",
    channel: process.env.REPOCOMPASS_BROWSER_CHANNEL || undefined,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 1000 } } },
    { name: "laptop", use: { viewport: { width: 1024, height: 768 } } },
    { name: "tablet", use: { viewport: { width: 768, height: 1024 } } },
    { name: "mobile", use: { viewport: { width: 390, height: 844 } } },
  ],
  webServer: externalUrl ? undefined : {
    command: "uv run --project .. repocompass-dashboard --demo --demo-turns 2 --port 8878 --log-level warning",
    url: baseURL + "/api/v1/system",
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
